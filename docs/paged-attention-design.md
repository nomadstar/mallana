# PagedAttention Integration Design — llama-cpp-turboquant

> Architecture review and integration plan. Read this before touching any code.

---

## 1. Current KV Cache Architecture (`llama_kv_cache`)

### Tensor Shape and Layout

Each layer allocates two 3D tensors at cache construction time:

```
k: [n_embd_k_gqa_eff, kv_size, n_stream]
v: [n_embd_v_gqa_eff, kv_size, n_stream]
```

`n_embd_*_gqa_eff` is padded to the next multiple of 128 for TurboQuant types.
`kv_size` = total token slots. `n_stream` = 1 (unified) or `n_seq_max` (multi-stream).

Token slot `i` occupies bytes `[i * ggml_row_size(type, n_embd_gqa), (i+1) * ggml_row_size(...))`.
**Tokens are contiguous in physical memory. There is no indirection.**

### Slot / Sequence System

- `v_cells[n_stream]` — tracks logical state of each physical slot (position, sequence IDs, shift)
- `seq_to_stream[seq_id]` — maps logical sequence ID → physical stream index
- `map_layer_ids` — maps model layer `il` → `kv_layer` index (supports tied layers)
- `slot_info` — allocation result for one ubatch: `strm[s]`, `idxs[s][i]` = physical cell index

`find_slot()` searches for free cells. Can place tokens non-contiguously (cells in the same sequence need not be adjacent), but the buffer itself is a flat slab.

### `get_k()` / `get_v()`

`get_k()` returns a 4D ggml view:
```
shape:   [head_k_eff, n_head_kv, n_kv, ns]
nb[1] = ggml_row_size(type, head_k_eff)           — stride per head
nb[2] = ggml_row_size(type, n_embd_k_gqa)         — stride per token slot
nb[3] = ggml_row_size(type, n_embd_k_gqa*kv_size) — stride per stream
```

`get_v()` returns a 4D view in one of two layouts:
- `v_trans=false` (FA path): `[head_v_eff, n_head_kv, n_kv, ns]` — V as [embd × token] rows
- `v_trans=true` (SDPA path): `[n_kv, n_head_kv, head_v_eff, ns]` — token dim outermost

### `cpy_k()` / `cpy_v()`

Zero-pads head dim to 128 (TurboQuant), then calls:
```cpp
ggml_set_rows(ctx, k, k_cur, k_idxs)
```
`k_idxs` carries flat global row indices (`stream * kv_size + cell`) computed in
`set_input_k_idxs`. The CUDA `set_rows` kernel maps `row_index → byte_offset` using a
uniform stride. **No per-token pointer indirection.**

---

## 2. Current FlashAttention Integration

`launch_fattn()` (fattn-common.cuh:1180) extracts three byte strides from the V tensor:

```
nb21 = V->nb[1]  — bytes between consecutive token rows (THE critical stride)
nb22 = V->nb[2]  — bytes between KV-head slabs
nb23 = V->nb[3]  — bytes between streams
```

In the VEC kernel (fattn-vec.cuh), per-token access is:

```cuda
// At kernel entry:
V += nb23*sequence + nb22*(head / gqa_ratio);   // advance to correct head slab

// At tile loop entry:
V += blockIdx.y * nthreads * nb21;               // advance to tile start

// Inner loop:
dequantize_V(V + k*nb21, tmp, element_offset);   // access token k
```

**`nb21` is a single fixed scalar.** The kernel computes `V + k*nb21` — pure arithmetic
progression. There is no page table lookup, no pointer indirection, no branching on block
boundaries.

---

## 3. PagedAttention Concept (Kwon et al. 2023, arXiv:2309.06180)

A **page** (block) is a fixed-size group of token slots allocated and freed as a unit.
A **page table** maps a sequence's logical pages → physical blocks in a shared pool.

```
Logical view (sequence A):  [page 0][page 1][page 2]...
Physical pool:               block 7  block 3  block 12 ...  ← non-contiguous
```

Key properties:
- **Block size** (16–64 tokens): smaller = less fragmentation, more table overhead
- **Prefix sharing**: multiple sequences point to the same physical blocks for shared prefixes
- **O(1) free**: return blocks to pool on sequence eviction

---

## 4. Integration Points

| Component | File | What changes | Complexity |
|-----------|------|-------------|------------|
| Page table data structure | `src/llama-kv-cache.h` | Add `block_pool_meta`, `page_table[stream][lpage]→pblock`, `free_blocks` stack | Medium |
| Physical page pool | `src/llama-kv-cache.cpp` | Reshape K/V tensors to `[n_embd_gqa, block_size*n_blocks, 1]` at construction | High |
| `set_input_k/v_idxs` | `src/llama-kv-cache.cpp` | Translate logical cell → `phys_block*B + within_block` via page table | Medium |
| `get_v()` gather | `src/llama-kv-cache.cpp` + new op | Emit `GGML_OP_GATHER_PAGED_V` before FA instead of returning direct view | Medium |
| Page allocation | `src/llama-kv-cache.cpp` | In `apply_ubatch`: pop block from free list on first token of new logical page | Medium |
| Sequence free/copy | `src/llama-kv-cache.cpp` | `seq_rm`: ref-count decrement + free push; `seq_cp`: page table clone | Medium |
| Compute graph | `src/llama-graph.cpp` | Insert gather op between `get_v()` and `ggml_flash_attn_ext` | Medium |
| New CUDA gather kernel | `ggml/src/ggml-cuda/` (new file) | `GGML_OP_GATHER_PAGED_V`: ~50 lines, indexed row gather | Medium |
| FA kernels (Phase 1) | `fattn-vec.cuh`, `fattn-common.cuh` | **No changes needed** (gather produces contiguous V) | None |
| FA kernels (Phase 2) | `fattn-vec.cuh`, `fattn-common.cuh` | Replace `V + k*nb21` with page-table-aware pointer | High |
| TurboQuant set_rows | `ggml-cuda/set-rows.cu` | **No changes needed** (physical row index passed in k_idxs) | None |
| `include/llama.h` | Public API | No changes for Phase 1 | None |

---

## 5. Critical Constraints

### Can `ggml_flash_attn_ext` work with non-contiguous V without kernel changes?

**No.** The kernel computes `V + k*nb21` where `nb21` is a single scalar read once at
kernel launch. If token `k` and token `k+1` are in different physical pages, no single
scalar stride can express their actual distance in memory.

The FP16 conversion path also assumes uniform strides (it reads `V->nb[1]` to rescale).

### Minimum new CUDA code for Phase 1

A gather kernel (~50 lines):
```cuda
// For each output row r in [0, n_kv):
int page  = r / block_size;
int off   = r % block_size;
int pblock = page_table[page];
memcpy(out + r * row_stride,
       block_pool + (pblock * block_size + off) * row_stride,
       row_bytes);
```

This runs once per layer per forward pass, before FA. No changes to `fattn-vec.cuh`.

### Does TurboQuant's `set_rows` kernel need changes?

**No.** `k_idxs[i]` already carries a flat integer row index. With paging, `set_input_k_idxs`
computes `phys_block * block_size + within_block_offset` and writes that into `k_idxs[i]`.
The CUDA kernel sees the same flat row index it always did. Zero-padding (head dim → 128)
is orthogonal to paging (token-slot axis).

### Can per-sequence page tables live purely in C++ without GPU kernel changes?

**For writes (cpy_k/cpy_v):** Yes — `set_input_k_idxs` runs on CPU, translates through
the page table, produces physical row indices. CUDA unchanged.

**For reads (get_v → FA):** No — a gather kernel on the GPU is required. It is new CUDA
code, but trivial compared to modifying the FA kernel itself.

---

## 6. Recommended Architecture

### Phase 1 — Functional (no FA kernel changes)

**Data structures** added to `llama_kv_cache`:
```cpp
struct paged_block { uint32_t ref_count; };

uint32_t block_size   = 32;      // tokens per block
uint32_t n_blocks;               // total blocks in pool
std::vector<paged_block>          block_pool_meta;
std::vector<uint32_t>             free_blocks;   // stack
std::vector<std::vector<uint32_t>> page_table;   // [stream][lpage] → pblock
```

**K/V tensor layout** at construction:
```
Old: [n_embd_gqa, kv_size, n_stream]
New: [n_embd_gqa, block_size * n_blocks_total, 1]
     where n_blocks_total = ceil(kv_size/block_size) * n_stream
```

**Write path** (`set_input_k_idxs`):
```
logical cell c in stream s:
  lpage  = c / block_size
  within = c % block_size
  pblock = page_table[s][lpage]
  k_idxs[i] = pblock * block_size + within
```

**Read path** — compute graph becomes:
```
v_pool   = get_v_paged(ctx, il)      // view of block pool tensor
v_ptable = get_v_page_table(ctx, il) // INT32 tensor [n_logical_pages, ns]
v        = ggml_gather_paged(ctx, v_pool, v_ptable, n_kv, block_size)
kqv      = ggml_flash_attn_ext(ctx, q, k, v, mask, ...)
```

**Validation sequence** (each step testable independently):
1. Reshape K/V tensors to block-pool layout with **identity page table** (page `p` → block `p`). Output must be bit-exact with current behavior.
2. Wire `set_input_k_idxs` through page table (still identity). Still bit-exact.
3. Implement and test `ggml_gather_paged_v`. With identity table, output is bit-exact.
4. Wire gather into graph. Verify inference matches baseline.
5. Add dynamic allocation: pop from `free_blocks` on new page; return on `seq_rm`.
6. Test with multiple sequences to verify page reuse.

### Phase 2 — Performance (native paged FA) — Implemented

**Status:** Code complete. Awaits numerical validation.

Replace gather+FA with a single kernel. In `fattn-vec.cuh`, the hot loop now calls:

```cuda
dequantize_V(v_paged_ptr(V_paged_base, nb21, v_ptable, sequence, v_ptable_ne0, v_block_size, k_VKQ_0 + k), tmp, ...);
```

Where `v_paged_ptr()` (defined in `fattn-common.cuh`) computes:

```cuda
lpage  = k_abs / bs;
within = k_abs % bs;
pblock = v_ptable[seq * n0 + lpage];
return V_base + ((int64_t)pblock * bs + within) * nb21;
```

**New parameters to `launch_fattn()`:** `const int32_t* v_ptable_data, int32_t v_ptable_ne0, int32_t v_block_size`.

**Graph integration:** When flash_attn + paging are active, `src/llama-graph.cpp` builds a
4D view of the flat V pool (`ggml_view_4d`) and attaches the page table via
`ggml_flash_attn_ext_set_page_table(kqv, page_table)` which stores it in `kqv->src[5]`.
The `launch_fattn()` entry point reads `dst->src[5]` to retrieve the page table data.

**Stride invariant:** The 4D view uses strides that, after `ggml_permute(0, 2, 1, 3)`,
yield `nb21 = n_embd_v × ts` (correct stride between physical pool rows) and
`nb22 = head_v_eff × ts` (correct stride between heads within a row).

### Phase 3 — TurboQuant-Aware Paged Blocks

No changes to quantization logic needed. Block size constraint: none from TurboQuant
(WHT operates within head dimension, not across token slots). Choose `block_size` that
is a multiple of `QK_TURBO3=32` for alignment: 32 or 64 recommended.

---

## 7. Full File Modification Table

| File | What changes | Complexity | Phase |
|------|-------------|------------|-------|
| `src/llama-kv-cache.h` | Add page table structs, new method declarations | Medium | 1 |
| `src/llama-kv-cache.cpp` | Constructor reshape; `apply_ubatch` page alloc; `set_input_k/v_idxs` page lookup; `seq_rm` ref-count; `seq_cp` page clone; `get_v_paged/get_v_page_table` | High | 1 |
| `src/llama-graph.cpp` | Insert `ggml_gather_paged` before `ggml_flash_attn_ext` | Medium | 1 |
| `ggml/src/ggml-cuda/paged-gather.cu` (new) | `GGML_OP_GATHER_PAGED_V` CUDA kernel + CPU fallback | Medium | 1 |
| `ggml/src/ggml.h` + `ggml.c` | Register new op type | Low | 1 |
| `src/llama-context.h` / `.cpp` | Forwarders for new `get_v_paged` interface | Low | 1 |
| `ggml/include/ggml.h` | `ggml_flash_attn_ext_set_page_table()` declaration | Low | 2 |
| `ggml/src/ggml.c` | `ggml_flash_attn_ext_set_page_table()` implementation | Low | 2 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Add `v_ptable`/`v_ptable_ne0`/`v_block_size` to `fattn_kernel_t` and `launch_fattn()`; `v_paged_ptr()` helper | High | 2 |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Replace `V + k*nb21` with `v_paged_ptr()` | High | 2 |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | Same paged pointer change | High | 2 |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | ABI compat: accept new params, mark GGML_UNUSED | Low | 2 |
| `ggml/src/ggml-cuda/fattn-wmma-f16.cu` | ABI compat stub | Low | 2 |
| `src/llama-graph.cpp` | Phase 2 graph branch: 4D view + skip gather + `set_page_table` | Medium | 2 |
| `include/llama.h` | No changes | None | — |
| `ggml-cuda/set-rows.cu` | No changes | None | — |

---

## 8. Verdict

**Is minimal PagedAttention compatible with TurboQuant + FlashAttention?**

**Yes.** Phase 1 (gather-before-FA) required:
- No changes to any CUDA kernel in `fattn-vec.cuh`, `fattn-common.cuh`, or `set-rows.cu`
- One new trivial gather kernel (~50 lines CUDA)
- C++-only changes to index computation and tensor layout

Phase 2 (native paged FA) extends the approach:
- Adds `v_paged_ptr()` device helper shared by VEC and TILE kernels
- Extends all kernel signatures with 3 new page-table parameters
- Plumbs page table through `dst->src[5]` via `ggml_flash_attn_ext_set_page_table()`
- Replaces gather + FA with a single kernel call per layer
- Leaves MMA-f16 and WMMA-f16 kernels unchanged (ABI compat stubs)

The existing TurboQuant zero-padding, WHT rotation, and block encoding remain orthogonal
to paging and required zero modification.

**Validation note:** Phase 2 code compiles but awaits numerical parity testing against
Phase 1 and the non-paged baseline.

**Smallest working unit**: Steps 1–4 of the Phase 1 validation sequence produce a
functionally correct system with identity page tables. Phase 2 builds on this foundation,
eliminating the gather step entirely.
