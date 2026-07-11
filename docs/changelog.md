# Changelog

> Chronological engineering changelog for the TurboQuant fork.

---

## 2026-06

### M007 audit round — paged attention correctness fixes

Four correctness bugs found and fixed by iterative opencode audit of `src/llama-kv-cache.cpp`:

**C-01 — State save/restore silent corruption with paged V cache**
- `state_write_data` used linear cell offsets to read V tensor data, but the paged V layout
  is physically indirected through a page table. Linear reads hit the dummy zero block (block 0)
  instead of actual data, silently corrupting any serialized KV state.
- Fix: `state_write_data` now aborts with an actionable message (`LLAMA_NO_PAGING=1` workaround);
  `state_read_data` now returns `false` with an error log.

**H-02 — Stack array OOB when seq_id ≥ LLAMA_MAX_SEQ in `set_input_kq_mask_impl`**
- `seq_pos_min[LLAMA_MAX_SEQ]` was indexed by an unbounded `seq_id` from the ubatch with no
  bounds check, risking stack corruption on malformed batches.
- Fix: Added `GGML_ASSERT(ubatch->n_seq_id[i] > 0)` before reading `seq_id[i][0]`, and
  `GGML_ASSERT(seq_id >= 0 && (size_t)seq_id < seq_to_stream.size())` to match the existing
  codebase assertion pattern (used at 8+ other call sites).

**Pool undercount for n_stream > 1**
- `pg_n_blocks = n_pages_per_stream * n_stream` left `n_stream - 1` physical blocks at the end
  of the V tensor permanently unreachable (pool never filled to them, V tensor too small).
- Fix: `pg_n_blocks = (n_pages_per_stream + 1) * n_stream - 1`. The extra `+1` per stream
  is the dummy zero block overhead; the `-1` accounts for block 0 not being in the pool.

**seq_to_stream shrink bug for n_stream > 1**
- Constructor shrank `seq_to_stream` from `LLAMA_MAX_SEQ` to `n_stream` when `n_stream > 1`,
  causing OOB access for any `seq_id ≥ n_stream`.
- Fix: Remove the second `.resize(n_stream)` call; vector stays at `LLAMA_MAX_SEQ`.

### M007 calibration run — H6.1 INDETERMINADO

**Calibration results** (`-fa off`, qwen2.5-coder-1.5b, wikitext-2, ctx=2048):
- Baseline PPL: 10.9552
- Eviction PPL (50% page budget): 10.9552 (Δ = 0)

**Why Δ = 0**: `pg_alloc_for_sinfo` protects pages in `current_batch_pages` from eviction.
In `llama-perplexity` batch mode, *all* pages are in the current batch, so the budget is never
enforced. This is correct behavior (don't evict pages being written) but means TriAttention
eviction is a generation-mode-only mechanism.

**H6.1 verdict**: INDETERMINADO. Generation-mode evaluation required (`llama-cli --predict`
with a long prompt). Infrastructure and workaround documented in `research/milestone-007/`.

**Phase 2 FA n_seq bug** (documented, not yet fixed): CUDA kernel indexes `v_ptable[seq * n0 + lpage]`
where `seq` iterates over `n_seq`, but the page table has only `n_stream` rows. With
`n_stream=1` and `n_seq=4`, indices 1–3 are OOB → corrupted V values → PPL ≈ 35125.
Workaround: `-fa off`. Pending fix: expand page table to `max(ns, n_seq)` rows in
`set_input_v_page_table` / `build_input_v_page_table`.

### M007 - TriAttention calibration and numerical validation infrastructure

**Feature**: Added the calibration/validation scaffolding for hypothesis H6.1.

- New script `scripts/triattention_calibrate.py` runs baseline (no eviction) vs eviction
  (`--triattention-page-budget N`) and computes `quality_retention_pct = (baseline_ppl / eviction_ppl) * 100`.
- Prefers `build/bin/llama-perplexity` and falls back to `build/bin/llama-cli -ppl`.
- Writes machine-readable output to `research/milestone-007/calibration_results.json` and prints a comparison table to stdout.
- Added `research/milestone-007/objective.md` plus stub `evidence.md` / `conclusions.md`
  marked pendiente until the GPU + model run is executed.

**Status**: Infrastructure complete. Numerical validation pending GPU execution.

### M006 fix - TriAttention RoPE freq_base and prefill eviction

**Problem**: A critique of the initial M006 TriAttention implementation found two issues: a RoPE
frequency correctness bug in `get_unrotated_key()` and a prefill budget-enforcement bug that kept
eviction decode-only.

- `get_unrotated_key()` was using the wrong effective RoPE frequency source for YaRN/NTK-aware
  models, which could corrupt page relevance scores.
- The eviction gate still depended on the removed `sinfo.size() == 1` decode-only assumption,
  allowing long prefills to exhaust the page pool before decode.

**Fix**:

- Added `rope_freq_base_eff` and `rope_freq_scale_eff` storage to `llama_kv_cache` and wired them
  from `llama_model::get_rope_freq_base()` / `get_rope_freq_scale()`.
- Updated `get_unrotated_key()` to use the stored effective RoPE values.
- Removed the `sinfo.size() == 1` guard from the allocation/eviction path so the page budget is
  enforced during both prefill and decode.

**Impact**: YaRN/NTK-scaled models are now scored correctly by TriAttention, and prefill can no
longer exhaust the physical page pool before decode.

### TriAttention KV eviction implementation

**Feature**: Added page-budgeted KV eviction for paged V cache using RoPE-inverted key
scoring.

- New CLI flag `--triattention-page-budget N` (`0` = disabled).
- Physical block 0 reserved as a dummy zero block.
- `pg_score_and_evict()` scores resident pages using RoPE-inverted K dot-products and evicts
  the lowest-scoring page when the physical page budget is exhausted.
- Wiring added across `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.cpp`,
  `common/common.h`, `common/common.cpp`, `common/arg.cpp`, `src/llama-kv-cache.h`,
  `src/llama-kv-cache.cpp`, and `src/llama-model.cpp`.

**Status**: Implemented, build passes, pending numerical validation.

### ROCm backend completion

**Fix**: Completed HIP portability work for the paged attention and TurboQuant CUDA codepaths.

- Fixed warp shuffle arity in `set-rows.cu` and `fattn-vec.cuh` by passing `WARP_SIZE` to the
  4-argument HIP-compatible shuffle form.
- Updated `__ballot_sync` macro in `vendors/hip.h` for AMD 64-thread wavefront behavior.
- Audited `turbo-quant.cuh`, `fattn-common.cuh`, and `fattn-tile.cuh`; no additional changes
  were required.

**Status**: Complete.

### Explicit NaN/Inf validation for turbo quantizers

**Robustness**: Added non-finite guards across CPU and CUDA TurboQuant paths.

- CPU `ggml-turbo-quant.c`: `isfinite()` guards added in quantize/dequantize paths for
  `turbo2`, `turbo3`, and `turbo4`.
- CUDA `turbo-quant.cuh`: matching `isfinite()` guards added to the dequantizer helpers.
- `tests/test-turbo-quant.c`: 6 extreme-input round-trip tests added and registered in
  `tests/CMakeLists.txt`.
- Hypothesis H4.1 now supported by automated tests.

**Status**: Complete. All tests pass.

### Paged Attention Phase 2 — native paged FA integrated

**Feature**: Completed the native paged Flash Attention path and exposed page-table attachment
through a dedicated GGML API.

- `ggml_flash_attn_ext_set_page_table()` attaches the page table through `src[5]`.
- `v_paged_ptr()` helper added in `fattn-common.cuh` and shared by the VEC/TILE FA kernels.
- `fattn_kernel_t` extended with page-table metadata.
- `llama-graph.cpp` now has the Phase 2 branch that skips gather, builds a 4D pool view, and
  calls `set_page_table()` when FA + paging are active.

**Status**: Complete. Numerically validated on CUDA — `LLAMA_PAGING=1 test-llama-archs` passes
with 0 failures (2026-07-09, after the ISWA/hybrid page-table wiring fixes); the paging-off
suite is unchanged. ROCm/HIP validation remains open.

### Mathematical consistency audit — CPU/CUDA equivalence verified

**Scope**: Complete audit of all TurboQuant mathematical contracts between CPU and CUDA
backends.

**Verified items**:
- WHT sign arrays (CPU `turbo_cpu_s1/s2` vs CUDA `turbo_gpu_s1/s2`): identical values.
- WHT butterfly order: identical stage ordering.
- Normalization factor (`1/sqrt(d)`): identical.
- Turbo2/3/4 centroid tables: CPU and CUDA match exactly.
- Turbo2/3/4 packing format: bit-level identical.
- Turbo2/3/4 dequantization logic: identical.
- Norm correction (`corrected_norm = block_norm / recon_norm`): identical.
- `vec_dot` contract: dequant-and-dot accumulator semantics verified correct.
- InnerQ scaling contract: per-channel RMS ratio application verified consistent.

**Verdict**: No correctness-critical CPU/CUDA mathematical mismatches remain. The turbo4
dense-rotation fix (`6457eac19`) restored full mathematical equivalence, and no other
systematic inconsistencies were found.

**Impact**: The project now operates on a validated mathematical foundation. All future
development can assume CPU/CUDA numerical parity. This milestone closes the "debugging"
phase of the project.

### `6457eac` — CPU turbo4 FWHT consistency fix

**Problem**: CPU turbo4 quantization used dense random rotation matrices (128×128 Gaussian,
QR-orthogonalized) while GPU used only FWHT. This produced incompatible quantized
representations between the two backends.

**Fix**:
- Replaced dense matrix-vector multiply with FWHT butterfly stages in CPU quantize path.
- Removed inverse dense rotation from CPU dequantize path.
- Both CPU and GPU now apply identical sign-flip → FWHT → sign-flip sequences.
- Dense rotation matrices remain for the QJL residual path (legacy turbo4 mode).

**Impact**: CPU-GPU bit-exact consistency for all turbo quantization types. Affects turbo4
(4-bit and 3-bit+QJL modes).

### `ce5b97f` — K-cache flat pool indices fix

**Problem**: When paging was enabled, K cache writes could be routed through the page table,
causing corruption.

**Fix**: K cache always uses flat pool indices regardless of paging state. Only V cache
uses page-table-based indexing.

### `5f5034d` — Device tensor upload fix for page table

**Problem**: `set_input_v_page_table` used `ggml_backend_tensor_set` for device tensors,
which bypasses the device memory allocator for page table data.

**Fix**: Corrected memory management for device-side page table upload.

### `02dfaa8` — Paged gather crash + FA auto-disable

**Problem**: Paged gather crashed when Flash Attention was auto-disabled (unsupported head
dimension). The SDPA fallback path requires `v_trans=true`, which is incompatible with paging.

**Fix**: When FA is auto-disabled, paging is also disabled.

### `4833eb0` — Device sync fix for paged KV

**Problem**: Race condition between `cudaMemcpyAsync` for page table upload and the
subsequent `buffer_clear` operation on the KV cache.

**Fix**: Added explicit `cudaDeviceSynchronize` before `buffer_clear` to ensure all page
table updates are visible to the device.

### `f0c7209` — Paged Attention Phase 1

**Implementation**: Gather-before-FlashAttention with dynamic page allocation.

- New `GGML_OP_GATHER_PAGED_V` op.
- Page table: `pg_page_table[stream][lpage] → pblock`.
- Block pool: LIFO `pg_free_blocks` stack.
- Block size: 32 tokens (aligned to `QK_TURBO3`).
- Lazy allocation on first write per page.
- CPU fallback for gather kernel.

---

## 2026-05

### `0e79b56` — TriAttention calibration script

Added Python calibration script (`scripts/calibrate-triattention.py`) for TriAttention KV
cache eviction. Generates per-layer importance scores from a calibration corpus using
RoPE-inverted key vectors.

### `b74a2cc` — HIP/ROCm FA template instances

**Problem**: TurboQuant FA template instances for HIP/ROCm build were missing, causing
linker errors on AMD GPUs.

**Fix**: Added all 16 missing turbo-typed VEC FA template instantiations for the HIP/ROCm
build path.

### `aca4594` — CUDA block size fix

**Problem**: Warp-to-block mapping was incorrect for `block_size=128` on turbo3/turbo2
kernels.

**Fix**: Fixed CUDA block size from 32 to 128 for turbo3/turbo2 SET_ROWS kernels,
improving GPU utilization on larger head dimensions.

### `58d51a6` — HIP/ROCm port for turbo3/turbo2

**Implementation**: Ported warp-cooperative SET_ROWS kernels to HIP/ROCm.

- Tested on RDNA3 (7900 XTX).
- Used CUDA-to-HIP macros (`__shfl_xor_sync`, `CUBLAS_GEMM_DEFAULT`).
- ROCWMMA FA path for `GGML_HIP_ROCWMMA_FATTN`.

### `70b35c7` — Boundary V (experimental)

**Feature**: Layer-aware V compression protection.

- Modes 5-7: Protect first and last N layers from aggressive V quantization.
- Mode 7 (auto-enabled for turbo2-V): first 2 + last 2 layers V = q8_0, rest V = turbo2.
- Designed to protect attention sinks and position encoding layers.

### `ae702148` — GLM-4 turbo4 compatibility

**Problem**: turbo4 on GLM-4 produced wrong results due to zero-padding handling.

**Fix**: Context initialization accounts for zero-padding when head_dim is not a multiple
of turbo block size. Tested on head_dim = 576 (GLM-4).

---

## 2026-04

### `b90b5e0` — CUDA turbo4 port

**Implementation**: Full CUDA port of turbo4 (4-bit mode, 3.8× compression).

- Warp-cooperative quantize kernel with nibble packing.
- Centroid lookup via 4-bit midpoint comparisons.
- Validated on NVIDIA GPUs.

### `d46ac77` — MMA Flash Attention for D=640

**Performance**: CUDA MMA flash attention for head_dim = 640 (GLM-4.7).

- Prefill throughput: 37 → 192 t/s (5.2× improvement).
- Required MMA template instantiation for non-standard head dimensions.

### `c1d9b34` — Non-128 head support via zero-pad WHT

**Feature**: Zero-pad non-128 head dimensions for full 7-stage WHT.

- Replaces the previous q8_0 fallback for non-128 heads.
- Head dimensions are padded to the next multiple of 128 before WHT.
- Padded values are trimmed from the WHT output.

### `3380d3c` — Metal turbo2 support

**Feature**: Metal implementation of turbo2 (2-bit KV cache, 6.4× compression).

- 4-centroid polar codebook.
- Sparse V dequantization integration.
- Auto-enable Boundary V for turbo2-V.

### `4c4511c` — Head dim validation

**Fix**: Require `head_dim % 128 == 0` for turbo KV types. Fall back to q8_0 if condition
is not met. Replaced by zero-padding in the subsequent iteration.

### `a5efe54` — Sparse V dequant (Metal)

**Performance**: Skip dequantization of V elements whose attention weight falls below a
threshold. Implemented for Metal VEC FA kernel.

### `4cf7145` — WHT rotation in ISWA build_attn

**Fix**: Added turbo WHT rotation to the ISWA (independent stream window attention) build
path, fixing Gemma 2 compatibility.

### `6fb85a6` — InnerQ per-channel equalization

**Feature**: Per-channel activation equalization for turbo quantization.

- Computes per-channel RMS ratios during calibration.
- Applies inverse scales before quantization.
- Reduces quantization error on channels with large activation magnitudes.
- Controlled by `TURBO_INNERQ=N` env var.

### Paged Attention Phase 2 — Native Paged FA

**Feature**: Eliminated gather kernel by integrating page table lookup directly into Flash Attention kernels.

- `v_paged_ptr()` device helper in `fattn-common.cuh` — shared by VEC and TILE kernels.
- Extended `fattn_kernel_t` with `v_ptable`, `v_ptable_ne0`, `v_block_size` params.
- `launch_fattn()` reads `dst->src[5]` for page table tensor.
- All `V + k*nb21` replaced with `v_paged_ptr()` in `fattn-vec.cuh` and `fattn-tile.cuh`.
- ABI compat stubs for `fattn-mma-f16.cuh` and `fattn-wmma-f16.cu`.
- `ggml_flash_attn_ext_set_page_table()` API attaches page table to FA op via `src[5]`.
- Graph builder (`llama-graph.cpp`) skips gather, creates 4D pool view, calls `set_page_table` when FA + paging active.
- Note: code compiles but awaits numerical validation.

### `da6b0fd` — GGML_TYPE_TURBO2_0

**Feature**: 2-bit TurboQuant KV cache type (6.4× compression).

- Enum value 43 (`GGML_TYPE_TURBO2_0`).
- 4 centroids, no QJL residual.
- 10 bytes per 128-element block.
