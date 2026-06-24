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

**Current capabilities:**

- Dynamic page allocation from a free-block pool.
- Page table indirection for V reads.
- Lazy allocation on first write.
- CPU fallback for `GATHER_PAGED_V`.
- `LLAMA_NO_PAGING` env var to disable.

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

### Read Path

```
v_pool   = get_v_paged(ctx, il)                 // block pool tensor
v_ptable = get_v_page_table(ctx, il)            // INT32 page table tensor
v        = ggml_gather_paged(ctx, v_pool, v_ptable, n_kv, block_size)
kqv      = ggml_flash_attn_ext(ctx, q, k, v, mask, ...)
```

### Gather Kernel

```cuda
// For each output row r in [0, n_kv):
int page    = r / block_size;
int off     = r % block_size;
int pblock  = page_table[page];
memcpy(out + r * row_stride,
       pool + (pblock * block_size + off) * row_stride,
       row_bytes);
```

---

## Graph Integration

```
q_cur ──►
k_cur ──► cpy_k ──► get_k ──►
v_cur ──► cpy_v ──► V pool ──► gather_paged_v ──► reshape ──► flash_attn_ext
                      ▲
page_table ───────────┘
```

---

## Configuration

| Parameter | Value | Description |
|---|---|---|
| Page block size | 32 tokens | Aligned to `QK_TURBO3` |
| Auto-enable | When `v_trans == false` (FA path) | Also requires `LLAMA_NO_PAGING` not set |
| Disable | `LLAMA_NO_PAGING=1` | Reverts to legacy contiguous layout |

---

## Known Limitations (Phase 1)

1. **Gather overhead** — The gather kernel adds O(n_kv) memory traffic per layer per forward
   pass. This will be eliminated in Phase 2 by integrating page table lookup into the FA kernel.

2. **No ref-counting** — `seq_rm()` scans all cells in a page range and only frees the
   physical block if every cell is empty. Explicit ref-counting is planned for Phase 2.

3. **No prefix sharing** — Multiple sequences with shared prefixes cannot share physical
   blocks. Planned for Phase 2.

4. **K cache not paged** — Only V uses paged allocation. Paging K is deferred to a later
   phase due to the additional complexity in the KQ dot product path.

5. **Block-size hard-coded** — Currently fixed at 32 tokens. Dynamic block size selection
   could optimize for different sequence lengths.

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

### Phase 2 — Native Paged FA (Planned)

- Replace gather + FA with a single kernel that looks up page table entries on the fly.
- New parameters to `launch_fattn()`: `const int32_t* page_table, int block_size`.
- In `fattn-vec.cuh`, replace `V + k*nb21` with `V_page + in_page*nb21`.

### Phase 3 — TurboQuant-Aware Blocks (Planned)

- Align block size to TurboQuant dequant tiles for zero-copy reads.
- Block size = 128 tokens (matching turbo block size).
- Eliminate all gather overhead for turbo types.

### Future: Sliding Window + Continuous Batching

- Page-level window management for sliding window attention.
- Batch-level page table compaction for continuous batching.
- Multi-sequence page sharing for prefix caching.

---

## Files

| File | Purpose |
|---|---|
| `src/llama-kv-cache.h` | Page table structs, method declarations |
| `src/llama-kv-cache.cpp` | Constructor reshape, page alloc, page table lookup, seq_rm |
| `src/llama-graph.cpp` | Insert gather op in compute graph |
| `ggml/src/ggml-cuda/paged-gather.cu` | CUDA gather kernel + CPU fallback |
| `ggml/include/ggml.h` | `GGML_OP_GATHER_PAGED_V` op definition |
| `ggml/src/ggml.c` | Fallback CPU implementation for gather |

See also [docs/paged-attention-design.md](paged-attention-design.md) for the original
architecture review document.
