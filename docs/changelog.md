# Changelog

> Chronological engineering changelog for the TurboQuant fork.

---

## 2026-06

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

### `da6b0fd` — GGML_TYPE_TURBO2_0

**Feature**: 2-bit TurboQuant KV cache type (6.4× compression).

- Enum value 43 (`GGML_TYPE_TURBO2_0`).
- 4 centroids, no QJL residual.
- 10 bytes per 128-element block.
