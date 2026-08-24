<p align="center">
  <img src="assets/mallana-logo.png" alt="Mallana" width="360">
</p>

# Mallana — Building tomorrow’s llama.cpp.

> Advanced KV cache compression for llama.cpp with CPU and GPU implementations and documented validation.
> Walsh-Hadamard rotated polar codebook quantization for long-context LLM inference.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is a research fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on
progressively bringing advanced inference optimizations — TurboQuant KV cache compression, Flash
Attention integration, and Paged Attention — into the llama.cpp codebase while maintaining
correctness, reproducibility, and compatibility.

The project emphasizes structured research, implementation, benchmarking, and validation rather
than experimental hacks. Implemented optimization paths are expected to carry a validation
methodology, a reference or comparison point where applicable, and documented limitations.

> **The mission —** *how much useful intelligence can we run on the hardware people already own?*
> Instead of scaling hardware to fit larger models, Mallana works to reduce the **cost of
> intelligence itself** — every byte of memory, every cache line, every tensor op, every watt — so
> existing machines can run larger and better models. TurboQuant is the first step down that path;
> quantization, attention, KV-cache systems, decoding, and execution are all in scope. Read the
> **[Manifesto](MANIFESTO.md)** for the full vision and the principles that gate every optimization.

### Claims, evidence, and scope

The manifesto describes Mallana's **research direction and intended destination**. It is not a
statement that every part of that destination has already been achieved.

The README uses a narrower standard for technical claims:

- **Implemented** means the code path exists and can be exercised.
- **Validated** means the repository contains documented tests or experiments supporting the
  stated behavior under the configurations that were actually tested.
- **Experimental / pending validation** means implementation exists, but the evidence is not yet
  sufficient to make a stronger claim about quality, performance, or generality.
- **Roadmap / future research** describes intended work, not current capability.

Benchmark numbers in this document are measurements from the hardware, models, parameters, and
run counts stated beside them. They are evidence for those tested conditions, not universal
performance guarantees. The implementation, test infrastructure, regression history, negative
results, and documented limitations are intentionally preserved so that other people and systems
working on Mallana have a reproducible technical record to inspect.

---

## 🧪 Evaluator's Guide — Evaluate Current Claims in Minutes

Everything below is designed so you can reproduce the documented behavior yourself, from a
short smoke test to the numerical validation suite. A passing test establishes evidence for the
configuration it exercises; broader conclusions require broader testing.

### Option A — Docker (fastest, CPU-only)

A prebuilt server image is published to GHCR:

```bash
docker pull ghcr.io/nomadstar/mallana:server-cpu

# Serve any GGUF model straight from Hugging Face, with TurboQuant V-cache compression:
docker run -p 8080:8080 ghcr.io/nomadstar/mallana:server-cpu \
    -hf ggml-org/gemma-3-1b-it-GGUF \
    --cache-type-k q8_0 --cache-type-v turbo3 -fa on

# Or mount a local model:
docker run -p 8080:8080 -v "$HOME/models:/models" ghcr.io/nomadstar/mallana:server-cpu \
    -m /models/your-model.gguf --cache-type-v turbo3 -fa on
```

Then open `http://localhost:8080` for the built-in web UI, or hit the OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### Option B — Build from source (unlocks CUDA + PagedAttention)

```bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON && cmake --build build -j
```

(See [Quick Start](#quick-start) for Metal / ROCm variants.)

### Option C — AMD GPU (ROCm / HIP) 🔴

Validated on the documented real-hardware configuration — **gfx1100 (RDNA3), ROCm 7.2.4** — with
`test-llama-archs` matching the CPU reference at NMSE 1e-8–1e-12 across the architectures covered
by that validation run. Other AMD architectures should be treated as separate validation targets.

**Native build** (fastest path on an AMD box):

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
  && cmake --build build -j$(nproc)

# Exercise it end-to-end on the GPU:
./build/bin/test-turbo-quant
./build/bin/test-llama-archs
```

Set `AMDGPU_TARGETS` to your GPU's arch (`gfx1100` RX 7900 / W7900, `gfx942` MI300,
`gfx1201` RX 9070, etc.) — run `rocminfo | grep gfx` to find it.

#### 📊 ROCm Compatibility & Validation Matrix

Tested on **gfx1100 (RDNA3), ROCm 7.2.4** using Gemma 3 4B:

| Configuration | pp2048 (t/s) | tg128 (t/s) | Status | Note |
|---|---|---|---|---|
| K=F16, V=F16 (Baseline) | 5279.61 ± 2.96 | 116.03 ± 0.00 | ✅ OK | Baseline |
| K=q8_0, V=turbo3 | ~5200 | ~75 | ✅ OK | 4.6× V compression, ~0.5% prefill cost |
| K=F16, V=turbo3 | 5174.49 ± 55.86 | 84.34 ± 0.08 | ✅ OK | Prefill within 2% of baseline |
| K=q8_0, V=turbo2 | ~5100 | ~76 | ✅ OK | 6.4× V compression |
| K=q8_0, V=q8_0 | ~5200 | ~80 | ✅ OK | Standard quantization baseline |
| K=F16, V=F16 (FA=off) | ~5100 | ~70 | ✅ OK | Non-FA path validated |

> [!NOTE]
> In the tested RDNA3 configuration, the listed KV cache configurations work with Flash Attention.
> The double inverse WHT bug (fixed 2026-07-13) previously caused turbo V types to produce garbage
> output during decode. Use any combination of `ctk`/`ctv` with `-fa on`, and validate new hardware
> or model combinations independently.

**Docker with GPU passthrough:**

```bash
docker build -f .devops/rocm.Dockerfile --target server -t mallana:server-rocm .

docker run --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined --group-add video \
    -p 8080:8080 mallana:server-rocm \
    -hf ggml-org/gemma-3-1b-it-GGUF --cache-type-v turbo3 -fa on -ngl 99
```

The `--device /dev/kfd --device /dev/dri --group-add video` flags are what expose the AMD GPU
to the container. Then run with `--cache-type-v turbo3 -ngl 99` to exercise TurboQuant on the GPU.
On some hosts `/dev/dri/renderD*` is owned by the `render` group — add `--group-add render`
too if the container can't see the GPU.

### Feature test matrix

| Feature | How to exercise it | What to look for |
|---|---|---|
| **TurboQuant KV compression** | `llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo3 -fa on` | Coherent output; KV buffer size in the load log shrinks ~4.6× for V |
| **Aggressive long-context compression** | `--cache-type-v turbo2 -c 32768` | KV memory 6.4× smaller vs f16 at the same context length |
| **PagedAttention (native paged FA)** | `LLAMA_PAGING=1 llama-cli -m model.gguf -ngl 99 -fa on ...` (CUDA build, KV fully on GPU) | Compare output vs `LLAMA_PAGING=0`; log line confirming paging is active |
| **TriAttention KV eviction** | See [docs/paged-attention.md](docs/paged-attention.md) — experimental, off by default (`triattention_page_budget = 0`) | Research feature; calibration status in `research/milestone-007/` |
| **Correctness suite** | `bash scripts/validate.sh` | Incremental build + full ctest (`-L main`), all green |
| **Per-architecture regression** | `LLAMA_PAGING=1 ./build/bin/test-llama-archs` | Reproduce the documented CUDA regression result; investigate any divergence from the CPU reference |
| **Perplexity quality gate** | `MODEL=/path/model.gguf WIKI=/path/wiki.test.raw bash scripts/turbo-quality-gate.sh` | Check TurboQuant PPL against the gate's configured reference and threshold |
| **Throughput benchmark** | `python3 scripts/benchmark.py` | Prompt/generation t/s per model (table below measured on an RTX 2050) |
| **Multi-agent Research OS** | `python3 scripts/multiswarm.py --audit` | opencode audits the current diff and writes `.multiswarm_audit.md` |

### Notes for evaluators

- **PagedAttention is opt-in** (`LLAMA_PAGING=1`) and currently requires the KV cache to be
  fully resident on CUDA devices — the loader verifies this and falls back with a warning
  otherwise. This gate exists because paging is validated on CUDA only (see
  [docs/paged-attention.md](docs/paged-attention.md) for the design and status).
- **TurboQuant types are opt-in** via `--cache-type-k` / `--cache-type-v`; standard upstream
  quantization paths are intended to remain unchanged. Regressions in those paths should be
  treated as bugs rather than assumed compatibility.
- **Known limitation to not trip over**: Qwen-family models in the documented tests degrade with
  low-bit K-cache quantization (including upstream `q4_0`) — use `q8_0`/`f16` for K there and
  validate additional models independently (details [below](#qwen-compatibility)).

---

## Project Overview

TurboQuant applies Walsh-Hadamard Transform (WHT) rotation followed by polar codebook
quantization to the KV cache. The TurboQuant paper (ICLR 2026) reports 3–6× KV-cache compression
with limited perplexity degradation under its evaluated conditions; Mallana implements and tests
this family of techniques inside llama.cpp.

This fork currently includes:

- **Asymmetric K/V policy** — current Mallana experiments show that V tolerates more aggressive
  compression than K on the tested models. The recommended default (`q8_0` K + `turbo3` V)
  reflects that evidence.
- **Two new types** — `turbo2` (2-bit, 6.4× compression) and `turbo4` (4-bit, 3.8× compression),
  added on top of the original `turbo3` (3-bit, 4.6× compression).
- **Cross-backend implementation and validation** — Turbo types have CPU and CUDA validation,
  with additional backend-specific validation documented below. The automated quality gate is a
  regression tool for its configured model, reference value, corpus, threshold, and environment;
  it should not be read as a universal quality guarantee.
- **Paged Attention Phase 1** — Functional gather-before-FlashAttention with dynamic page
  allocation.
- **TriAttention** — KV cache eviction via RoPE-inverted key scoring; implemented and still
  pending quality validation.

Standard llama.cpp paths are intended to remain compatible when TurboQuant is not enabled.
TurboQuant types are opt-in via `--cache-type-k` / `--cache-type-v`. Current implementation and
validation coverage includes CPU, CUDA, HIP/ROCm, and Metal to different degrees described in this
README and the validation docs. Other backends such as Vulkan, SYCL, and WebGPU do not yet support
TurboQuant and may reject or fall back when TurboQuant is enabled.

### Why this exists

This repository is not simply a TurboQuant fork — it is a research platform built around one
question: **how much useful intelligence can we run on the hardware people already own?** Rather
than asking how to make models larger, we ask how to make existing hardware capable of running
larger and better models, by reducing the *cost of intelligence itself* — every byte of memory,
every cache line, every tensor op, every watt. TurboQuant is the first technique down that path;
quantization, attention, KV-cache systems, decoding, and execution are all in scope, provided they
compose cleanly, preserve correctness across supported backends, and stay responsive on the
hardware real users own. The full vision and the principles that gate every optimization live in
the **[Manifesto](MANIFESTO.md)**.

---

## Current Status

Status labels below describe the repository's present engineering state. A component marked
**Validated** is validated only to the extent documented by its tests and recorded environments;
it is not a claim of exhaustive validation across all models and hardware.

| Component | Status |
|---|---|
| CPU TurboQuant (turbo2, turbo3, turbo4) | ✅ Validated |
| CUDA TurboQuant (turbo2, turbo3, turbo4) | ✅ Validated |
| CPU/CUDA Mathematical Equivalence Audit | ✅ Complete |
| Flash Attention Integration | ✅ Stable on validated paths |
| KV Cache Layer-Adaptive Quantization | ✅ Working |
| Quality Gate (automated PPL + speed) | ✅ Operational regression gate |
| Paged Attention (Phase 1) | ✅ Functional |
| Paged Attention (Phase 2) | ✅ Validated on CUDA 2026-07-09 — `LLAMA_PAGING=1 test-llama-archs` 0 failures in recorded run; opt-in via `LLAMA_PAGING=1` |
| TriAttention | 🚧 Implemented — Pending Validation |
| TriAttention Calibration (M007) | 🔄 H6.1 INDETERMINADO — batch mode prevents eviction; generation-mode eval needed |
| ROCm / HIP (turbo2/3/4 + Flash Attention) | ✅ Validated on gfx1100 (RDNA3, ROCm 7.2.4) 2026-07-13 — all recorded KV configs pass (`amd-validate.sh` 6/6); PagedAttention (`LLAMA_PAGING=1`) is still CUDA-only |
| Metal Support | ✅ Implemented / validated on documented paths |
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

## Benchmarks — RTX 2050 (4 GB VRAM)

Measured with `scripts/benchmark.py`, full GPU offload (`-ngl 99`), 32 tokens, 2 runs. These
numbers characterize that recorded setup; the small run count makes them useful as engineering
measurements, not as population-level performance estimates.

| Model | Size | Prompt (t/s) | Generation (t/s) |
|---|---|---|---|
| qwen2.5-coder-1.5b | 1.1 GB | 2921 | 97.4 |
| qwen2.5-coder-3b | 1.8 GB | 1736 | 59.1 |
| llama3.2-3b | 1.9 GB | 1833 | 50.2 |
| llama3.1-8b Q2_K | 3.0 GB | 715 | 29.7 |

The listed sub-4 GB quantized models ran successfully in this setup. IQ-family quants (IQ2_S,
IQ3_XS) are not yet supported (require importance-based quantization kernels from upstream
llama.cpp).

---

## Benchmarks — AMD Radeon (gfx1100 / RDNA3, ROCm 7.2.4)

TurboQuant KV compression on real AMD hardware. Model: Qwen2.5-Coder-7B-Q8_0,
`llama-bench -fa 1 -ngl 99 -p 2048 -n 128 -r 3`, K cache held at `q8_0`.

| KV config | V compression | Prompt (t/s) | Generation (t/s) |
|---|---|---|---|
| `f16` K / `f16` V (baseline) | 1× | 2823.7 | 80.0 |
| `q8_0` K / `turbo3` V | 4.6× | 2808.6 | 75.0 |
| `q8_0` K / `turbo2` V | 6.4× | 2797.4 | 75.7 |

In this RDNA3 benchmark, TurboQuant measured roughly 0.5% lower prompt throughput and roughly 6%
lower generation throughput while reducing the value-cache representation by 4.6–6.4×. The
memory reduction is the demonstrated result; whether it translates into a useful long-context
advantage depends on model, context, backend, available VRAM, and workload and should be measured
explicitly for those scenarios.

---

## Validation Results

### CPU/CUDA Mathematical Consistency Audit

The implementation has undergone a mathematical consistency audit of the listed CPU and CUDA
contracts:

| Checked Item | Status |
|---|---|
| WHT sign arrays (CPU vs CUDA) | ✅ Identical |
| WHT butterfly order (CPU vs CUDA) | ✅ Identical |
| WHT normalization | ✅ Identical |
| Turbo2/3/4 centroids | ✅ Match |
| Turbo2/3/4 packing format | ✅ Match |
| Turbo2/3/4 dequantization | ✅ Match |
| Norm correction after quantize | ✅ Match |
| `vec_dot` contract | ✅ Correct |
| InnerQ scaling contract | ✅ Correct |
| WHT-only rotation (post-turbo4-fix) | ✅ Shared contract |

**Engineering verdict for the audited paths:**

> No correctness-critical CPU/CUDA mathematical mismatch was found in the contracts listed above
> after the documented fixes.

This provides a validated foundation for those audited paths while leaving new backends, models,
features, and future changes subject to their own regression and validation work.

### Llama-3.2-3B F16 — Perplexity

| Configuration | PPL |
|---|---|
| Baseline (F16) | 8.68 |
| turbo2 K | 13.42 |
| turbo3 K | 9.45 |
| turbo4 K | 8.99 |
| turbo4 V | 8.76 |
| turbo4 K + turbo4 V | 8.99 |

For this Llama-3.2-3B evaluation, `turbo4` shows less than 0.4 PPL degradation from the recorded
baseline and `turbo3` less than 0.8 PPL. The tested asymmetric configuration is within 0.1 PPL of
the recorded baseline. These figures characterize this experiment and should not be generalized
to other model families without measurement.

> **Note:** These results were generated using `llama-perplexity` on wikitext-2 with the
> infrastructure documented in [docs/validation.md](docs/validation.md).

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
same mathematical transform in the audited implementation path.

The sign arrays (`turbo_cpu_s1[128]`, `turbo_cpu_s2[128]`) replace 64 KB of dense rotation
matrices with 512 bytes of pre-computed WHT signs.

---

## Root Cause Analysis: Double Inverse WHT on ROCm Decode

A critical bug caused turbo V types (turbo2/3/4) to produce garbage output during decode
on the affected GPU path before the documented fix.

### The Problem

The VEC flash attention kernel (`fattn-vec.cuh`) accumulated VKQ in the WHT domain for turbo V
types, then applied an inverse WHT internally before writing to `dst`. The graph-side
(`build_attn_mha` in `llama-graph.cpp`) then applied `ggml_turbo_wht(..., direction=1, ...)` —
a *second* inverse WHT — on the same tensor. Two inverses compose to a forward WHT, producing
output in the WHT domain instead of the original domain.

The VEC kernel's inverse was also incomplete: it did not apply the InnerQ `scale_inv`
correction that the graph-side `ggml_turbo_wht` includes.

Symptom observed in the affected path: first 1–2 tokens correct (prefill reads original f16 V),
then catastrophic corruption (decode reads from turbo-quantized KV cache).

### The Fix (commit `1beb7e1`)

Removed the in-kernel inverse WHT from `fattn-vec.cuh` (lines 689–777). The graph-side
`ggml_turbo_wht` is now the single source of truth for the inverse transform in the documented
paths, including the InnerQ scale correction.

### Why This Works

- **VEC kernel** (decode, ncols=1): outputs VKQ in WHT domain → graph-side inverse corrects.
- **MMA/tile kernels** (prefill, ncols>1): output in WHT domain → graph-side inverse corrects.
- **Non-FA path** (`ggml_mul_mat`): graph-side inverse handles this independently.

The validated paths converge on a single, complete inverse WHT with InnerQ scaling.

## Qwen Compatibility

Testing on Qwen-family models revealed anomalous perplexity results across the tested low-bit KV
cache quantization methods:

| Configuration | PPL |
|---|---|
| Baseline (F16) | 11.79 |
| q8_0 K + q8_0 V | 9.20 |
| q4_0 K + q4_0 V | 531 |
| turbo3 K + turbo3 V | 4098 |
| turbo4 K + turbo4 V | 1658 |

### Current Evidence

The current evidence is consistent with a model/quantizer compatibility issue rather than, by
itself, isolating a TurboQuant implementation defect:

- Plain `q4_0` KV quantization already fails dramatically in this experiment (PPL = 531 vs
  11.79 baseline).
- The failure pattern is present across the low-bit quantizers tested in this experiment.
- Models that fail with q4_0 K also fail with turbo3 K and turbo4 K in the recorded tests.

**Candidate explanation**: Large K activation outliers may be difficult to represent with these
low-bit quantization schemes. This remains a hypothesis until isolated by further experiments.

> **Note**: This conclusion represents current evidence, not absolute fact. The investigation
> remains open if new data emerges.

**Workaround for the affected tested models**: Use `f16` or `q8_0` for K cache, apply turbo types
only to V cache (`--cache-type-v turbo3`).

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
    WHT Rotation (sign-flip → FWHT → sign-flip)
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
  KV Cache  ◄── Graph-side inverse WHT (sign-flip → FWHT → sign-flip + InnerQ scale)
                           │
                           ▼
                      Inference
```

---

## Milestones

### Completed

| Milestone | Status |
|---|---|
| TurboQuant CPU correctness on documented validation suite | ✅ |
| TurboQuant CUDA correctness on documented validation suite | ✅ |
| CPU/CUDA mathematical equivalence audit | ✅ |
| Llama validation (recorded PPL within 0.4 for turbo4) | ✅ |
| Initial ROCm portability audit | ✅ |

### Upcoming

| Milestone | Priority |
|---|---|
| Re-enable PagedAttention by default (CUDA divergences fixed; owner decision) | P1 |
| ROCm/HIP PagedAttention validation | P2 |
| TriAttention calibration and numerical validation | P4 |
| Large-scale benchmarks (multi-GPU, multi-model) | P2 |
| Commodity-hardware efficiency (FFAI-inspired): kernel warmup, dispatch overhead, AURA-vs-TurboQuant KV | P4 |
| Upstream synchronization | P3 |
| Vulkan TurboQuant kernels | P3 |

The completed milestones provide an engineering foundation for the tested paths; broader claims
remain tied to future validation on additional workloads and hardware.

---

## Roadmap

### Phase 1 — TurboQuant (Implementation Complete)

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

### Phase 2 — Validation Infrastructure / Recorded Validation (Complete for listed scope)

- [x] Round-trip quantization test (`tests/test-turbo-quant.c`)
- [x] Automated quality gate (`scripts/turbo-quality-gate.sh`)
- [x] Perplexity validation on documented Llama model(s)
- [x] CPU-GPU consistency fix (FWHT alignment)
- [x] HIP/ROCm port for turbo3/turbo2

### Phase 3 — Paged Attention (In Progress)

- [x] Phase 1: Gather-before-FA with dynamic page allocation (✅ Functional)
- [x] Phase 2: Native paged FA (page-table-lookup in kernel) (✅ Validated on CUDA — `LLAMA_PAGING=1 test-llama-archs` 0 failures in recorded run, 2026-07-09)
- [x] Phase 3: TriAttention KV eviction via RoPE-inverted key scoring (🚧 Implemented, Pending Validation)
- [x] M007: TriAttention calibration infrastructure (`scripts/triattention_calibrate.py`, milestone stubs, calibration run complete — H6.1 INDETERMINADO; generation-mode eval required)
- [ ] Sliding window support
- [ ] Continuous batching

### Research Scripts

- `scripts/turbo-quality-gate.sh` — automated PPL + speed regression gate
- `scripts/triattention_calibrate.py` — baseline vs eviction calibration runner for H6.1; writes `research/milestone-007/calibration_results.json`
- `scripts/multiswarm.py` — multi-agent task runner; `--audit` / `--audit-scope` mode runs opencode as a code auditor and writes `.multiswarm_audit.md`

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
- [ ] Layer-wise inference / weight streaming (AirLLM-style) — investigate very-large-model inference on highly constrained VRAM without weight quantization
- [ ] Prefetch-compute overlap for layer streaming
- [ ] codebase-memory-mcp integration for Research OS agent navigation
- [ ] **Commodity-hardware efficiency** ([FFAI](https://github.com/thewafflehaus/FFAI)-inspired) — the *efficient-on-hardware-you-own* half of the mission, not just *lighter*: precompiled-kernel warmup (cut Metal cold-start / time-to-first-token), a CPU↔GPU dispatch-overhead audit (ties into the single-GPU sync debt), and an AURA-vs-TurboQuant KV-compression benchmark. Ideas only — mallana stays a portable multi-backend fork; measurement first. See [docs/roadmap.md](docs/roadmap.md#commodity-hardware-efficiency-ffai-inspired).

---

## Documentation Index

<details open>
<summary><b>🧭 Vision &amp; Process</b></summary>

| Document | Description |
|---|---|
| [Manifesto](MANIFESTO.md) | General vision, optimization principles, and targets |
| [Research Automation](RESEARCH.md) | Research OS framework, roles, and automated workflows |
| [Roadmap](docs/roadmap.md) | Detailed milestones and priorities |
| [Changelog](docs/changelog.md) | Engineering changelog |

</details>

<details>
<summary><b>🏗️ Architecture &amp; Algorithms</b></summary>

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | High-level architecture and component relationships |
| [TurboQuant](docs/turboquant.md) | Detailed TurboQuant algorithm and implementation |
| [Paged Attention](docs/paged-attention.md) | Paged Attention design and implementation plan |

</details>

<details>
<summary><b>📊 Evidence</b></summary>

| Document | Description |
|---|---|
| [Validation](docs/validation.md) | Validation methodology and results |
| [Benchmarks](docs/benchmarks.md) | Quality, performance, and memory benchmarks |

</details>

---

## Research Automation (Research OS)

This repository features an integrated **Research OS** designed to coordinate human-supervised AI collaboration. If you are an AI assistant working on this repository:

1. **Get Up to Speed**: Run the resume script to display the research dashboard:
   ```bash
   bash scripts/resume.sh
   ```
2. **Consult Roles & Rules**: See [RESEARCH.md](RESEARCH.md) for execution workflows and find role-specific instructions under `prompts/`.
3. **Execute Safely**: Make minimal changes based on the active milestone in `research/state.md`. Verify them using `bash scripts/validate.sh` and benchmark using `bash scripts/benchmark.sh`.

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
# Recommended default based on current asymmetric-policy evidence
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo3 --fa on

# Aggressive V compression at long context — validate quality for your model
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo2 -c 131072

# Conservative first contact with a new model
llama-cli -m model.gguf --cache-type-k f16 --cache-type-v turbo4
```

See [docs/turboquant.md](docs/turboquant.md) for detailed configuration guidance.

---

## Background

TurboQuant is inspired by:

- **TurboQuant** (ICLR 2026) — Google's original paper introducing Walsh-Hadamard-rotated
  polar codebook quantization for KV cache, reporting 4.6× compression at ~1% PPL loss in the
  paper's evaluated conditions.
- **Asymmetric K/V compression** — Mallana's empirical observation that V tolerated more
  aggressive compression than K in the recorded experiments, driving the current asymmetric
  default policy.
- **PagedAttention** (Kwon et al. 2023, arXiv:2309.06180) — Non-contiguous KV cache with
  page-table indirection for efficient memory management.

---

## License

MIT, same as upstream llama.cpp.