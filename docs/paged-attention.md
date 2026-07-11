# Paged Attention

> Design and implementation status of Paged Attention integration.

---

## Overview

Paged Attention (Kwon et al. 2023, arXiv:2309.06180) introduces non-contiguous KV cache
allocation using page tables. Tokens belonging to the same sequence are stored in fixed-size
blocks that may be scattered across physical memory, connected via a page table indirection
layer. This enables:

- **Virtual contiguous addressing** — Sequences see logical contiguity despite physical
  fragmentation.
- **O(1) block reuse** — Free blocks are returned to a pool on sequence eviction.
- **Prefix sharing** — Multiple sequences can share physical blocks for common prefixes
  (Phase 2+).

---

## Implementation Status

### Phase 1 — Gather-Before-FA (Functional)

Phase 1 implements Paged Attention without modifying any existing Flash Attention kernels.
Instead, a new `GGML_OP_GATHER_PAGED_V` operation gathers non-contiguous V pages into a
contiguous buffer before Flash Attention executes.

**Design constraints:**

- **V cache only** — K continues to use the flat contiguous layout. V benefits most from
  paging because Flash Attention reads V by token index with a non-sequential access pattern.
- **No FA kernel changes** — The gather op produces a contiguous V buffer with uniform stride,
  so FA sees no difference from the legacy path.
- **Minimal new CUDA code** — The gather kernel is approximately 50 lines of CUDA.

**Capabilities:**

- Dynamic page allocation from a free-block pool.
- Page table indirection for V reads.
- Lazy allocation on first write.
- CPU fallback for `GATHER_PAGED_V`.
- Opt-in via `LLAMA_PAGING=1` (off by default since 2026-07-03; formerly opt-out via
  `LLAMA_NO_PAGING`).

### Phase 2 — Native Paged FA (Implemented)

Phase 2 eliminates the intermediate gather kernel by integrating page table lookup directly
inside the Flash Attention kernels (`fattn-vec.cuh`, `fattn-tile.cuh`). Instead of
materializing a contiguous V buffer, the kernel receives:

- The flat V pool pointer (same physical storage as Phase 1)
- The page table tensor via `dst->src[5]`

A new `v_paged_ptr()` device helper (defined in `fattn-common.cuh`) translates a logical
KV position `(seq, k_abs)` to a physical address:

```cuda
lpage  = k_abs / block_size;
within = k_abs % block_size;
pblock = page_table[seq * n_lpages + lpage];
return V_base + (pblock * block_size + within) * nb21;
```

**Kernel changes:**

- All `V + k*nb21` accesses replaced with `v_paged_ptr(V_paged_base, nb21, ...)` calls.
- V pointer no longer advances by `nb23*sequence` (pool has no per-sequence stride) nor by
  `blockIdx.y * nthreads * nb21` in the k-loop (page table handles positioning).
- Kernel signature extended with 3 new params: `v_ptable`, `v_ptable_ne0`, `v_block_size`.

**Graph changes (`src/llama-graph.cpp`):**

When `cparams.flash_attn && inp->self_v_page_table`:
1. Creates a 4D view of the flat V pool (shape `[head_v_eff, n_head_kv, n_kv_val, ns]`)
   with strides matching the pool layout.
2. Skips the `ggml_gather_paged_v` call entirely.
3. Calls `ggml_flash_attn_ext_set_page_table(cur, page_table)` after `build_attn_mha` to
   attach the page table to `dst->src[5]`.

**ABI compat stubs:** `fattn-mma-f16.cuh` and `fattn-wmma-f16.cu` accept the 3 new params
but mark them `GGML_UNUSED` — only the VEC and TILE kernels implement paged access.

**Performance target (not yet validated):** 10–15% latency reduction for sequences >8K tokens
by eliminating gather's global-memory write + read.

### Phase 3 — TriAttention KV Eviction (Implemented, Pending Validation)

Phase 3 adds an eviction policy on top of the paged V pool so long contexts can continue once a
fixed physical page budget is reached. The current implementation introduces:

- `--triattention-page-budget N` CLI flag, where `0` disables eviction and preserves the Phase 2
  behavior.
- Reserved physical block `0` as a dummy zero block, ensuring evicted pages can safely map to a
  known zero-filled backing page.
- `pg_score_and_evict()` in the KV cache, which scores resident pages using RoPE-inverted K
  dot-products and evicts the lowest-scoring page when the configured page budget is exhausted.

**Hypothesis H6.1:** TriAttention eviction can preserve roughly 95% of baseline quality while
using only 50% of the physical pages.

**Status:** Implemented and builds successfully, but numerical quality validation is still
pending.

---

## Architecture

### Data Structures

```cpp
struct llama_kv_cache_paged {
    // Page table: [stream][logical_page] → physical_block
    std::vector<std::vector<int32_t>> pg_page_table;

    // Free block pool (LIFO stack)
    std::vector<uint32_t> pg_free_blocks;

    // Block metadata
    uint32_t pg_block_size;    // 32 tokens (aligned to QK_TURBO3)
    uint32_t pg_n_blocks;      // total physical blocks in pool
};
```

### Tensor Layout

**Contiguous (legacy):**
```
K/V: [n_embd_gqa, kv_size, n_stream]
```

**Paged (V only):**
```
V: [n_embd_gqa, block_size * n_blocks_total, 1]
```

### Write Path

```
set_input_v_idxs:
    cell_idx = stream * kv_size + logical_cell
    lpage    = cell_idx / pg_block_size
    within   = cell_idx % pg_block_size
    pblock   = pg_page_table[stream][lpage]
    data[i]  = pblock * pg_block_size + within
```

The `ggml_set_rows` CUDA kernel sees a flat row index — no kernel changes needed.

### Read Path (Phase 1)

```
v_pool   = get_v_paged(ctx, il)                 // block pool tensor
v_ptable = get_v_page_table(ctx, il)            // INT32 page table tensor
v        = ggml_gather_paged(ctx, v_pool, v_ptable, n_kv, block_size)
kqv      = ggml_flash_attn_ext(ctx, q, k, v, mask, ...)
```

### Read Path (Phase 2)

```
v_pool   = get_v_paged(ctx, il)                 // block pool tensor (flat 2D)
v_ptable = build_input_v_page_table(ctx, ...)   // INT32 page table tensor
v        = ggml_view_4d(ctx, v_pool, ...)       // 4D view of pool (no gather)
// ... ggml_permute(v, 0, 2, 1, 3) inside build_attn_mha ...
kqv      = ggml_flash_attn_ext(ctx, q, k, v, mask, ...)
ggml_flash_attn_ext_set_page_table(kqv, v_ptable)  // attach table to dst->src[5]
```

### `v_paged_ptr()` Device Helper

```cuda
static __device__ __forceinline__ const char * v_paged_ptr(
        const char * __restrict__ V_base,
        const int64_t             nb21,
        const int32_t * __restrict__ v_ptable,
        const int32_t             seq,
        const int32_t             n0,
        const int32_t             bs,
        const int32_t             k_abs) {
    if (v_ptable) {
        const int32_t lpage  = k_abs / bs;
        const int32_t within = k_abs % bs;
        const int32_t pblock = v_ptable[seq * n0 + lpage];
        return V_base + ((int64_t)pblock * bs + within) * nb21;
    }
    return V_base + (int64_t)k_abs * nb21;
}
```

`nb21` = stride per physical slot in the pool (`n_embd_v × element_size`).
After `ggml_permute(V, 0, 2, 1, 3)`, the kernel receives `nb21 = n_embd_v × ts` — the
correct stride to index consecutive physical pool rows.

### 4D View Strides

The 4D pool view is created with shape `[head_v_eff, n_head_kv, n_kv_val, ns]` and strides
that, after `ggml_permute(0, 2, 1, 3)`, produce the correct geometry:

| Stride | Pre-permute | Post-permute | Purpose |
|--------|-------------|--------------|---------|
| `nb0`  | `ts` (inherited) | `ts` | element stride |
| `nb1`  | `head_v_eff · ts` | `n_embd_v · ts` (= `nb21`) | token stride (kernel iterates KV positions) |
| `nb2`  | `n_embd_v · ts` | `head_v_eff · ts` (= `nb22`) | head stride (kernel offsets per head group) |
| `nb3`  | `n_embd_v · n_kv_val · ts` | `0` (paged: no per-seq stride) | sequence stride |

---

## Graph Integration

### Phase 1 (gather)

```
q_cur ──►
k_cur ──► cpy_k ──► get_k ──►
v_cur ──► cpy_v ──► V pool ──► gather_paged_v ──► reshape ──► flash_attn_ext
                      ▲
page_table ───────────┘
```

### Phase 2 (native paged FA)

```
q_cur ──►
k_cur ──► cpy_k ──► get_k ──►
v_cur ──► cpy_v ──► V_pool ──► view_4d ──► permute ──► flash_attn_ext ◄── set_page_table
                      │                                              ▲
                      └─── page_table ──── build_input_v_page_table ──┘
```

---

## Configuration

| Parameter | Value | Description |
|---|---|---|
| Page block size | 32 tokens | Aligned to `QK_TURBO3` |
| Default | **Disabled** (since 2026-07-03) | Legacy contiguous layout unless explicitly opted in |
| Enable | `LLAMA_PAGING=1` | Requires `v_trans == false` (FA path) **and** the KV cache fully resident on CUDA devices; otherwise a warning is logged and paging stays off |
| TriAttention page budget | `--triattention-page-budget N` | `0` disables eviction; `N > 0` enables physical-page eviction (requires paging enabled) |

---

## Known Limitations

1. **Gather overhead (Phase 1 only)** — Phase 1 gather adds O(n_kv) memory traffic.
   **Eliminated in Phase 2** via native FA integration.

2. **No ref-counting** — `seq_rm()` scans all cells in a page range and only frees the
   physical block if every cell is empty. Explicit ref-counting is planned.

3. **No prefix sharing** — Multiple sequences with shared prefixes cannot share physical
   blocks. Planned.

4. **K cache not paged** — Only V uses paged allocation. Paging K is deferred due to
   the additional complexity in the KQ dot product path.

5. **Block-size hard-coded** — Currently fixed at 32 tokens. Dynamic block size selection
   could optimize for different sequence lengths.

6. **Phase 2 numerically validated on CUDA only** — `LLAMA_PAGING=1 test-llama-archs`
   passes with 0 failures on CUDA (2026-07-09, after the ISWA/hybrid page-table wiring
   fixes), and the paging-off suite is unchanged. ROCm/HIP validation and byte-level V
   comparison remain open (see `docs/roadmap.md`).

7. **4D view stride correctness** — The `ggml_view_4d` strides for the paged pool are
   designed to produce correct `nb21`/`nb22` values after `ggml_permute(0,2,1,3)`. Any
   change to the view creation or permute order must re-verify kernel stride invariants.

---

## Implementation Plan

### Phase 1 — Functional (Current)

| Step | Description | Status |
|---|---|---|
| 1 | Reshape V tensors to block-pool layout with identity page table | ✅ |
| 2 | Wire `set_input_v_idxs` through page table (identity) | ✅ |
| 3 | Implement `ggml_gather_paged_v` op + CUDA kernel | ✅ |
| 4 | Wire gather into compute graph | ✅ |
| 5 | Dynamic allocation: pop from free_blocks, write through page table | ✅ |
| 6 | Page-free on `seq_rm` | ✅ |
| 7 | CPU fallback for gather | ✅ |
| 8 | Paging disable gate (`LLAMA_NO_PAGING`) | ✅ |

### Phase 2 — Native Paged FA (Implemented)

| Step | Description | Status |
|---|---|---|
| 1 | Extend `fattn_kernel_t` with `v_ptable`, `v_ptable_ne0`, `v_block_size` params | ✅ |
| 2 | Implement `v_paged_ptr()` device helper in `fattn-common.cuh` | ✅ |
| 3 | Wire `launch_fattn()` to read `dst->src[5]` and pass page table to kernel | ✅ |
| 4 | Replace `V + k*nb21` with `v_paged_ptr()` in `fattn-vec.cuh` | ✅ |
| 5 | Same replacement in `fattn-tile.cuh` | ✅ |
| 6 | ABI compat stubs for `fattn-mma-f16.cuh` and `fattn-wmma-f16.cu` | ✅ |
| 7 | Add `ggml_flash_attn_ext_set_page_table()` API | ✅ |
| 8 | Wire graph to skip gather and create 4D pool view when FA + paging active | ✅ |
| 9 | Numerical validation (CUDA: `LLAMA_PAGING=1 test-llama-archs` 0 failures) | ✅ |

### Phase 3 — TriAttention KV Eviction (Implemented)

| Step | Description | Status |
|---|---|---|
| 1 | Add configurable physical page budget (`--triattention-page-budget`) | ✅ |
| 2 | Reserve dummy physical block 0 for safe zero-backed eviction targets | ✅ |
| 3 | Implement `pg_score_and_evict()` using RoPE-inverted K scoring | ✅ |
| 4 | Trigger eviction when physical budget is reached | ✅ |
| 5 | Numerical/perplexity validation for H6.1 | 🚧 |

### Future: TurboQuant-Aware Blocks

- Align block size to TurboQuant dequant tiles for zero-copy reads.
- Block size = 128 tokens (matching turbo block size).
- Eliminate all gather overhead for turbo types.

### Future: Sliding Window + Continuous Batching

- Page-level window management for sliding window attention.
- Batch-level page table compaction for continuous batching.
- Multi-sequence page sharing for prefix caching.

---

## Files

### Phase 1

| File | Purpose |
|---|---|
| `src/llama-kv-cache.h` | Page table structs, method declarations |
| `src/llama-kv-cache.cpp` | Constructor reshape, page alloc, page table lookup, seq_rm |
| `src/llama-graph.cpp` | Insert gather op in compute graph |
| `ggml/src/ggml-cuda/paged-gather.cu` | CUDA gather kernel + CPU fallback |
| `ggml/include/ggml.h` | `GGML_OP_GATHER_PAGED_V` op definition |
| `ggml/src/ggml.c` | Fallback CPU implementation for gather |

### Phase 2 Additions

| File | Change |
|---|---|
| `ggml/include/ggml.h` | `ggml_flash_attn_ext_set_page_table()` declaration |
| `ggml/src/ggml.c` | `ggml_flash_attn_ext_set_page_table()` implementation (sets `src[5]`) |
| `ggml/src/ggml-cuda/fattn-common.cuh` | `v_paged_ptr()` helper; extended `fattn_kernel_t`; `launch_fattn()` reads `src[5]` |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Paged V access via `v_paged_ptr()` |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | Paged V access via `v_paged_ptr()` |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | ABI compat stub |
| `ggml/src/ggml-cuda/fattn-wmma-f16.cu` | ABI compat stub |
| `src/llama-graph.cpp` | Phase 2 graph branch (4D view + skip gather + `set_page_table`) |

### Phase 3 Additions

| File | Change |
|---|---|
| `include/llama.h` | Public config exposure for TriAttention page budget |
| `src/llama-cparams.h` | Runtime parameter storage for page budget |
| `src/llama-context.cpp` | Context wiring for TriAttention configuration |
| `common/common.h` | CLI/common option declaration |
| `common/common.cpp` | Option defaults and propagation |
| `common/arg.cpp` | `--triattention-page-budget` parsing |
| `src/llama-kv-cache.h` | Eviction method declarations and paged-cache state |
| `src/llama-kv-cache.cpp` | Dummy block reservation, scoring, and eviction implementation |
| `src/llama-model.cpp` | Model/runtime integration for TriAttention config |

See also [docs/paged-attention-design.md](paged-attention-design.md) for the original
architecture review document.
