# Architecture

> High-level architecture of the TurboQuant fork, describing the relationship between
> quantization types, backends, Flash Attention, and KV cache management.

---

## Overview

This fork adds a TurboQuant quantization layer on top of llama.cpp's existing KV cache
infrastructure. The key architectural components and their interactions are described below.

```
┌──────────────────────────────────────────────────────────────────┐
│                         llama.cpp Core                           │
│  (model loading, graph building, scheduling, inference loop)     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      KV Cache (per layer)                        │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐  │
│  │    Key Cache        │    │        Value Cache              │  │
│  │  (q8_0 / f16 /      │    │  (turbo2/3/4 / q8_0 / f16)     │  │
│  │   turbo2/3/4)       │    │                                 │  │
│  └─────────┬───────────┘    └────────────┬────────────────────┘  │
│            │                             │                       │
│            ▼                             ▼                       │
│  ┌──────────────────┐          ┌──────────────────┐              │
│  │  WHT Rotation    │          │  WHT Rotation    │              │
│  │  (fwht + signs)  │          │  (fwht + signs)  │              │
│  └────────┬─────────┘          └────────┬─────────┘              │
│           │                             │                        │
│           ▼                             ▼                        │
│  ┌──────────────────┐          ┌──────────────────┐              │
│  │  Quantize        │          │  Quantize        │              │
│  │  (polar codebook)│          │  (polar codebook)│              │
│  └──────────────────┘          └──────────────────┘              │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Flash Attention                             │
│                                                                  │
│  ┌──────────────┐  ┌─────────────────────────────────────────┐   │
│  │  Pre-rotate Q│  │  FA Kernel (VEC/TILE/MMA)               │   │
│  │  (fwht)      │  │  - On-the-fly K dequant → KQ dot        │   │
│  └──────────────┘  │  - On-the-fly V dequant → VEC           │   │
│                    │  - Sparse V dequant (Metal)              │   │
│                    └─────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────┐            │
│  │  Post-rotate V output (inverse fwht)              │            │
│  └──────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Backend Execution                           │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │  CPU     │  │  CUDA    │  │  Metal   │  │  Future: Vulkan │  │
│  │  (ggml-  │  │  (ggml-  │  │  (ggml-  │  │  SYCL, WebGPU   │  │
│  │  turbo-  │  │  cuda/)  │  │  metal/) │  │                 │  │
│  │  quant.c)│  │          │  │          │  │                 │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## TurboQuant

TurboQuant is a family of KV cache quantization codecs consisting of three types:

- **turbo2** (2-bit, 6.4× compression) — Four centroids, no QJL residual.
- **turbo3** (3-bit, 4.6× compression) — Eight centroids, 3-bit PolarQuant.
- **turbo4** (4-bit, 3.8× compression) — Sixteen centroids, 4-bit PolarQuant (default) or
  legacy 3-bit + QJL residual.

All types share a common structure:

1. **Per-block L2 norm** — Each 128-element block stores its L2 norm as fp16.
2. **WHT rotation** — Forward FWHT with sign flips Gaussianizes the data distribution.
3. **Polar codebook quantization** — Scalar quantization using Lloyd-Max optimal centroids
  for the rotated distribution N(0, 1/128).

See [turboquant.md](turboquant.md) for the complete algorithmic description.

---

## GGML Integration

TurboQuant is integrated into GGML as three new types (enum values 41–43):

- `GGML_TYPE_TURBO3_0 = 41`
- `GGML_TYPE_TURBO4_0 = 42`
- `GGML_TYPE_TURBO2_0 = 43`

The `GGML_OP_TURBO_WHT` operation handles forward and inverse WHT rotation. The
`GGML_OP_SET_ROWS` operation handles quantized row insertion into the KV cache tensor.

### Graph Building

The attention computation graph (`llama-graph.cpp`) is modified as follows:

1. **Pre-rotate Q**: Before Flash Attention, the Q tensor is rotated via `ggml_turbo_wht()`
   with direction = forward (0). This applies the same WHT transform that was applied during
   KV cache quantization, ensuring compatibility between Q and the quantized K cache.

2. **Flash Attention**: The FA kernel dequantizes K and V on the fly during the KQ dot
   product and V aggregation stages.

3. **Post-rotate V output**: After attention computes the output, it is rotated back via
   `ggml_turbo_wht()` with direction = inverse (1). The padded head dimensions are trimmed.

---

## CPU Path

The CPU implementation lives in `ggml/src/ggml-turbo-quant.c`:

- **Quantize**: Forward WHT → polar codebook quantization → corrected norm.
- **Dequantize**: Centroid lookup × norm.
- **SET_ROWS handler**: Communicates WHT group size (64 or 128) to the quantize function.
- **WHT forward op**: In `ggml/src/ggml-cpu/ops.cpp`, the `ggml_compute_forward_turbo_wht_f32`
  function applies FWHT with sign flips, supporting both forward and inverse directions.

The dense rotation matrices (128×128) are generated via QR decomposition of a seeded random
Gaussian matrix (seed=42) and stored as static arrays. These are used for the QJL residual
path (legacy turbo4). The primary rotation path uses sign arrays + FWHT.

---

## CUDA Path

The CUDA implementation spans several files:

- `ggml/src/ggml-cuda/turbo-quant.cuh` — Device functions for WHT, centroid lookup, quantize,
  and dequantize. The WHT is implemented as in-place butterfly stages with `__syncthreads()`.
- `ggml/src/ggml-cuda/turbo-wht.cu` — WHT kernel (`k_turbo_wht_f32`) with dispatch for
  direction and group size. One block per group, `group_size` threads per block.
- `ggml/src/ggml-cuda/set-rows.cu` — Warp-cooperative SET_ROWS kernels for turbo2, turbo3,
  and turbo4. These include InnerQ calibration for per-channel equalization.
- `ggml/src/ggml-cuda/fattn-common.cuh` — Flash Attention integration with on-the-fly
  dequantization for turbo types in both KQ dot and V aggregation.

The warp-cooperative SET_ROWS kernels are a key optimization: they compute L2 norms in
parallel using warp shuffle reductions, apply WHT rotation, quantize, and pack the result
in a single kernel launch.

---

## Flash Attention Integration

TurboQuant forces Flash Attention on when the KV cache uses turbo types. This is implemented
in `src/llama-context.cpp` (line ~2988).

Three FA kernel variants support turbo dequant:

- **VEC kernel**: On-the-fly dequant of K during KQ dot product and V during aggregation.
  Supports all turbo types via template instantiations (16 turbo-related VEC instances).
- **TILE/MMA kernels**: Used for non-turbo types. Fall back when turbo is detected.
- **Sparse V dequant (Metal)**: Attention weights below threshold skip V dequantization
  entirely, reducing memory bandwidth.

Turbo-specific LUT scoring optimizes the attention-weight computation path based on the
turbo type (`turbo2`/`turbo3`/`turbo4`), balancing dequantization throughput with
attention-matrix arithmetic.

---

## KV Cache Management

The KV cache (`llama-kv-cache.cpp`) implements:

- **Layer-adaptive quantization** — Controlled by `TURBO_LAYER_ADAPTIVE` env var:
  - Mode 0: All layers same type.
  - Mode 1: First + last 2 layers use `q8_0`, middle layers use turbo.
  - Mode 2: First + last 8 layers use `q8_0`.
  - Mode 5: Boundary V — first 2 + last 2 layers V = turbo4, rest V = turbo2.
  - Mode 6: V-only — last 8 V = turbo4, rest V = turbo2.
  - Mode 7 (default for turbo2-V): Boundary V — first 2 + last 2 V = `q8_0`, rest V = turbo2.

- **Head dimension padding** — If `head_dim` is not a multiple of 128, K and V are zero-padded
  to the next multiple of 128. Padding is stripped after inverse WHT on the attention output.

- **Rotation matrices** — Pre-computed WHT-based 128×128 R and R^T matrices loaded into the
  KV cache buffer from `turbo-rotation-data.h`.

---

## Paged Attention Integration

Phase 1 (gather-before-FA) is implemented. See [paged-attention.md](paged-attention.md) for
the full design.

The `GGML_OP_GATHER_PAGED_V` op gathers non-contiguous V pages into a contiguous buffer
before Flash Attention runs. The V cache uses a page table (`pg_page_table[stream][lpage]`)
and a block pool with LIFO free list. Page block size is 32 tokens, aligned to
`QK_TURBO3 = 32`.

---

## Backend Portability

| Backend | Quantization | WHT Rotation | Flash Attention | Notes |
|---|---|---|---|---|
| CPU | ✅ Full | ✅ Full | ✅ (via dequant) | Reference implementation |
| CUDA | ✅ Full | ✅ Full | ✅ (VEC + LUT) | Primary target |
| Metal | ✅ turbo2/3/4 | ✅ Full | ✅ (sparse V) | Validated on Apple Silicon |
| HIP/ROCm | ✅ turbo3/turbo2 | ✅ Full | ✅ (WMMA FA) | RDNA3/4, CDNA3/4 |
| Vulkan | ❌ None | ❌ None | ❌ None | Not yet implemented |
| SYCL | ❌ None | ❌ None | ❌ None | Not yet implemented |
| WebGPU | ❌ None | ❌ None | ❌ None | Not yet implemented |
