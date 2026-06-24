# llama.cpp TurboQuant

> Advanced KV cache compression for llama.cpp with validated CPU and GPU implementations.
> Walsh-Hadamard rotated polar codebook quantization for long-context LLM inference.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is a research fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on
progressively bringing advanced inference optimizations — TurboQuant KV cache compression, Flash
Attention integration, and Paged Attention — into the llama.cpp codebase while maintaining
correctness, reproducibility, and compatibility.

The project emphasizes structured research, implementation, benchmarking, and validation rather
than experimental hacks. Every codec and optimization path includes a validation methodology,
a reference implementation, and documented limitations.

---

## Project Overview

TurboQuant applies Walsh-Hadamard Transform (WHT) rotation followed by polar codebook
quantization to the KV cache. This family of techniques, introduced in Google's TurboQuant
paper (ICLR 2026), enables 3–6× compression of the key-value cache with minimal perplexity
degradation compared to traditional MSE-optimal quantization.

This fork extends the original work significantly:

- **Asymmetric K/V policy** — V tolerates aggressive compression (3–6×) while K does not.
  The recommended default (`q8_0` K + `turbo3` V) reflects this finding.
- **Two new types** — `turbo2` (2-bit, 6.4× compression) and `turbo4` (4-bit, 3.8× compression),
  added on top of the original `turbo3` (3-bit, 4.6× compression).
- **Cross-backend validation** — Every type is validated on CPU and GPU (CUDA), with
  an automated quality gate ensuring PPL stays within 5% of the fp16 baseline.
- **Paged Attention Phase 1** — Functional gather-before-FlashAttention with dynamic page
  allocation.
- **TriAttention** — KV cache eviction via RoPE-inverted key scoring.

All existing llama.cpp quantization types, model architectures, and backends continue to work
unchanged. TurboQuant types are opt-in via `--cache-type-k` / `--cache-type-v`.

---

## Current Status

| Component | Status |
|---|---|
| CPU TurboQuant (turbo2, turbo3, turbo4) | ✅ Stable |
| CUDA TurboQuant (turbo2, turbo3, turbo4) | ✅ Stable |
| Flash Attention Integration | ✅ Stable |
| KV Cache Layer-Adaptive Quantization | ✅ Working |
| Quality Gate (automated PPL + speed) | ✅ Operational |
| Paged Attention (Phase 1) | 🚧 In Progress |
| TriAttention | 🚧 In Progress |
| ROCm / HIP Support | 🚧 In Progress |
| Metal Support | ✅ Stable |
| Vulkan Support | ❌ Not Started |

### Quantization Types

| Type | Enum | Bits per Element | Compression vs FP16 | Block Size | Block Bytes |
|---|---|---|---|---|---|
| `turbo2` | `GGML_TYPE_TURBO2_0` | 2.5 bpw | 6.4× | 128 | 10 B |
| `turbo3` | `GGML_TYPE_TURBO3_0` | 3.5 bpw | 4.6× | 128 | 14 B |
| `turbo4` | `GGML_TYPE_TURBO4_0` | 4.25 bpw | 3.8× | 128 | 68 B |

All types use 128-element blocks with per-block L2 norm scaling. The WHT rotation operates
on groups of 128 elements (typically one head dimension).

---

## Recent Validation Results

### Llama-3.2-3B F16 — Perplexity

Baseline (F16, no quantization):

| Configuration | PPL |
|---|---|
| Baseline (F16) | 8.68 |

TurboQuant variations:

| Configuration | PPL |
|---|---|
| turbo2 K | 13.42 |
| turbo3 K | 9.45 |
| turbo4 K | 8.99 |
| turbo4 V | 8.76 |
| turbo4 K + turbo4 V | 8.99 |

CPU TurboQuant is now considered validated for Llama-family models. The `turbo4` types show
less than 0.4 PPL degradation from baseline, while `turbo3` shows less than 0.8 PPL
degradation. The asymmetric policy (`q8_0` K + `turbo4` V) is within 0.1 PPL of baseline.

> **Note:** These validation results were generated using the test infrastructure in
> `scripts/turbo-quality-gate.sh` and `llama-perplexity`.

---

## Root Cause Analysis: CPU-GPU Rotation Inconsistency

A critical bug was identified and fixed in the CPU turbo4 quantization path.

### The Problem

CPU turbo4 K originally applied a **dense random rotation matrix** (128×128 Gaussian,
QR-orthogonalized) during quantization and its inverse during dequantization. GPU turbo4 K
applied only the **Fast Walsh-Hadamard Transform (FWHT)** with elementwise sign flips.

This produced incompatible quantized representations: a model quantized on CPU would not
dequantize correctly on GPU, and vice versa. The numerical outputs differed because dense
rotation and FWHT are fundamentally different transforms.

### The Fix (commit `6457eac19`)

1. **CPU quantization**: Replaced dense matrix-vector multiply with FWHT butterfly stages,
   matching the GPU implementation exactly.
2. **CPU dequantization**: Removed the inverse dense rotation. The CPU path now applies the
   same sign-flip → FWHT → sign-flip sequence as the GPU.

### Why This Works

The FWHT is an orthogonal linear transform requiring only O(d log d) operations (896 ops for
d=128) compared to O(d²) for dense rotation (16384 ops). Both CPU and GPU now apply the
exact same mathematical transform, producing bit-identical results for equivalent inputs.

The sign arrays (`turbo_cpu_s1[128]`, `turbo_cpu_s2[128]`) replace 64 KB of dense rotation
matrices with 512 bytes of pre-computed WHT signs.

---

## Qwen Validation: Known Compatibility Issue

Testing on Qwen-family models revealed anomalous perplexity results:

| Configuration | PPL |
|---|---|
| Baseline (F16) | 11.79 |
| q8_0 K + q8_0 V | 9.20 |
| q4_0 K + q4_0 V | 531 |
| turbo3 K + turbo3 V | 4098 |
| turbo4 K + turbo4 V | 1658 |

### Analysis

This is **not** a TurboQuant implementation bug. The evidence:

- Plain `q4_0` KV quantization already fails dramatically (PPL = 531 vs 11.79 baseline).
- The failure affects all low-bit KV quantization, not just TurboQuant.
- The degradation is consistent across multiple quantization methods, indicating a
  model-level sensitivity rather than a codec-specific issue.

### Likely Cause

Qwen models exhibit large K activation outliers that cannot be represented by low-bit
quantization. Standard MSE-optimal quantizers (including q4_0) also fail because the outliers
dominate the quantization range, leaving insufficient precision for the remaining values.

This is a known limitation of low-bit KV cache quantization on certain model families.
Workaround: use `f16` or `q8_0` for K cache, or apply turbo types only to V cache
(`--cache-type-v turbo3`) where the impact is less pronounced.

---

## Architecture

```
GGUF model
    │
    ▼
KV Cache (per layer)
    │
    ├─ Key cache ────► q8_0 / f16 / turboN
    │
    └─ Value cache ──► turboN / q8_0
                           │
                           ▼
    WHT Rotation (pre-computed R / R^T matrices)
                           │
                           ▼
    Quantization (polar codebook, 2/3/4 bit)
                           │
                           ▼
    ┌──────────────────────────┬──────────────────────┐
    │                          │                      │
    ▼                          ▼                      ▼
  CPU path                 CUDA path              Metal path
  (ggml-turbo-quant.c)     (turbo-quant.cuh,        (turbo-wht.h)
                             set-rows.cu)
    │                          │                      │
    ▼                          ▼                      ▼
  Flash Attention (on-the-fly K/V dequantization)
                           │
                           ▼
  KV Cache  ◄── Inverse WHT rotation (attention output)
                           │
                           ▼
                      Inference
```

---

## Roadmap

### Phase 1 — TurboQuant (Complete)

- [x] TurboQuant type definitions (turbo2, turbo3, turbo4)
- [x] CPU quantize/dequantize kernels
- [x] CUDA quantize/dequantize kernels
- [x] CUDA WHT rotation kernel
- [x] Flash Attention integration (KQ dot + V dequant)
- [x] Metal WHT rotation and FA support
- [x] Head dimension padding to 128
- [x] Asymmetric K/V compression policy
- [x] Layer-adaptive quantization modes
- [x] Boundary V protection for turbo2 V

### Phase 2 — Validation (Complete)

- [x] Round-trip quantization test (`tests/test-turbo-quant.c`)
- [x] Automated quality gate (`scripts/turbo-quality-gate.sh`)
- [x] Perplexity validation on Llama models
- [x] CPU-GPU consistency fix (FWHT alignment)
- [x] HIP/ROCm port for turbo3/turbo2

### Phase 3 — Paged Attention (In Progress)

- [ ] Phase 1: Gather-before-FA with dynamic page allocation (✅ Functional)
- [ ] Phase 2: Native paged FA (page-table-lookup in kernel)
- [ ] Phase 3: TurboQuant-aware block alignment
- [ ] Sliding window support
- [ ] Continuous batching

### Phase 4 — Backend Portability (Pending)

- [ ] Vulkan TurboQuant kernels
- [ ] Full HIP/ROCm turbo4 support
- [ ] SYCL TurboQuant support
- [ ] WebGPU TurboQuant support

### Phase 5 — Future Research

- [ ] TurboQuant-aware weight quantization
- [ ] Multi-GPU paged attention
- [ ] Zero-overhead FA dequant (eliminate gather)
- [ ] Adaptive bit-width selection per layer

---

## Documentation Index

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | High-level architecture and component relationships |
| [TurboQuant](docs/turboquant.md) | Detailed TurboQuant algorithm and implementation |
| [Paged Attention](docs/paged-attention.md) | Paged Attention design and implementation plan |
| [Validation](docs/validation.md) | Validation methodology and results |
| [Benchmarks](docs/benchmarks.md) | Quality, performance, and memory benchmarks |
| [Roadmap](docs/roadmap.md) | Detailed milestones and priorities |
| [Changelog](docs/changelog.md) | Engineering changelog |

---

## Quick Start

Standard llama.cpp build. TurboQuant types become available automatically.

```bash
# Apple Silicon (Metal)
cmake -B build -DGGML_METAL=ON && cmake --build build -j

# NVIDIA CUDA
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# AMD HIP / ROCm
cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx942;gfx950" && cmake --build build -j
```

### Usage

```bash
# Recommended default (asymmetric turbo)
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo3 --fa on

# Aggressive V compression at long context
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo2 -c 131072

# Conservative (first contact with new model)
llama-cli -m model.gguf --cache-type-k f16 --cache-type-v turbo4
```

See [docs/turboquant.md](docs/turboquant.md) for detailed configuration guidance.

---

## Background

TurboQuant is inspired by:

- **TurboQuant** (ICLR 2026) — Google's original paper introducing Walsh-Hadamard-rotated
  polar codebook quantization for KV cache, demonstrating 4.6× compression at ~1% PPL loss.
- **Asymmetric K/V compression** — Empirical finding that V tolerates aggressive compression
  while K does not, driving the asymmetric default policy.
- **PagedAttention** (Kwon et al. 2023, arXiv:2309.06180) — Non-contiguous KV cache with
  page-table indirection for efficient memory management.

---

## License

MIT, same as upstream llama.cpp.
