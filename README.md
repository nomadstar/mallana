# llama.cpp — TurboQuant + TriAttention + PagedAttention

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-atomicmilkshake%2Fllama--cpp--turboquant-blue?logo=github)](https://github.com/atomicmilkshake/llama-cpp-turboquant)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-binaries-yellow)](https://huggingface.co/atomicmilkshake/llama-cpp-turboquant-binaries)

*Read this in [English](README.md) | Leer en [Español](README_es.md)*

A heavily optimized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) designed to maximize LLM inference performance and memory efficiency on NVIDIA GPUs. We've introduced three major architectural additions:

- **PagedAttention** — vLLM-style block tables for dynamic, non-contiguous KV cache memory management, eliminating fragmentation and enabling massive concurrency.
- **TurboQuant** — custom low-bit quantization formats (turbo2, turbo3, turbo4) with hardware-optimised CUDA kernels for faster inference with a smaller memory footprint.
- **TriAttention** — GPU-accelerated KV cache pruning ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921)) that scores token importance using RoPE-inverted key vectors and evicts low-value tokens, keeping long-context inference within a strict VRAM budget.

---

## 🔥 Status Matrix

This fork is actively pushing the frontier of local LLM inference.

| Feature | Status |
|---------|--------|
| TurboQuant CUDA kernels (turbo2/3/4, SM75+) | ✅ **Live** |
| TriAttention GPU KV-cache pruning | ✅ **Live** |
| PagedAttention block table | ✅ **Live** (`feature/paged-attention`) |
| ROCm / HIP backend for AMD | ⏳ Partial — compilation works, pending full validation |
| Vulkan backend support | 🗺️ Planned |
| Arch Linux Package (`llama-cpp-turboquant-git`) | 📦 **Ready** |

> ⚠️ **Hardware notice:** Current TurboQuant kernels are optimized for **NVIDIA GPUs (CUDA, SM75+)**. Full ROCm (AMD) and Vulkan compatibility is **pending** and will land in upcoming releases.

## Pre-built Windows Binaries

Download the latest Release build (Windows x64, CUDA 13, RTX 2000+) from Hugging Face:

**[🤗 atomicmilkshake/llama-cpp-turboquant-binaries](https://huggingface.co/atomicmilkshake/llama-cpp-turboquant-binaries)**

> Requires CUDA 13.x runtime (`cublasLt64_13.dll`). Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) or the CUDA runtime redistributable if you don't have it.

---

## 🧠 Core Technologies

### 1. PagedAttention (Concurrency & Batching)
By default, standard `llama.cpp` allocates continuous memory blocks for the KV cache, leading to severe fragmentation and out-of-memory (OOM) errors under high concurrency. Our PagedAttention implementation pages the KV cache into fixed-size virtual blocks, allocating memory dynamically just like an operating system.

**Benefits:**
- **Zero Fragmentation:** Enables processing significantly larger batch sizes on consumer GPUs.
- **Extreme Throughput:** In our Llama 3 (8B) benchmarks on a modest RTX 2050 (4GB VRAM), enabling 4 concurrent requests with PagedAttention yielded a **108% increase in aggregate throughput** (from 8.4 t/s to 17.6 t/s) and reduced Time-To-First-Token (TTFT) by **12x** by eliminating queue starvation.

### 2. TriAttention (KV Cache Pruning)
TriAttention keeps your KV cache within a fixed token budget by periodically scoring all cached tokens and evicting the least important ones. Scoring uses the geometric structure of RoPE-encoded key vectors — no additional model weights or fine-tuning required.

**Performance (Qwen3-8B Q4_K_M, RTX 3080, `-c 512`)**
| Mode | Prune overhead | Generation speed |
|------|---------------|-----------------|
| No budget limit | — | 17.5 tok/s |
| CPU scoring | ~5,900 ms/event | 17.5 tok/s |
| **GPU scoring** | **~4–9 ms/event** | **75.0 tok/s** |

GPU scoring is ~1,000× faster than CPU. The 4.3× generation speedup comes from keeping the KV cache within VRAM budget (no eviction stalls, consistent flash-attention batch sizes).

### 3. TurboQuant
TurboQuant provides three custom quantization formats that outperform standard GGUF quants at equivalent bit widths:

| Format | Bits/weight | Notes |
|--------|------------|-------|
| `turbo4` | ~4.0 | Drop-in replacement for `q4_0`, with rotation-based clustering |
| `turbo3` | ~3.0 | Sub-byte with Hadamard pre-rotation |
| `turbo2` | ~2.0 | Aggressive compression with WHT-space centroids |

All formats feature highly optimized CUDA kernels for Turing+ (SM75) and Ampere (SM80/86) architectures.

---

## ⚡ Building from Source

### Requirements
- Windows 10/11 or Linux (Arch, Ubuntu, Fedora, etc.)
- CUDA Toolkit 12.x or 13.x
- Visual Studio 2022+ with C++ workload (Windows) or GCC 11+ (Linux)
- CMake 3.21+

### 🐧 Linux (CUDA)
Works on any distro with GCC 11+ and the CUDA Toolkit installed.
```bash
git clone https://github.com/atomicmilkshake/llama-cpp-turboquant
cd llama-cpp-turboquant

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server -j$(nproc)
```

### 🎯 Arch Linux (Native Packaging)
We provide native `PKGBUILD` support that handles CUDA weak stubs, systemd service configuration, and clean package metadata. Use your favorite AUR helper:
```bash
# Available soon on AUR
yay -S llama-cpp-turboquant-git
```

### 🪟 Windows (CUDA)
Requires Visual Studio 2022 and the CUDA Toolkit.
```powershell
git clone https://github.com/atomicmilkshake/llama-cpp-turboquant
cd llama-cpp-turboquant

cmake -B build -G "Visual Studio 18 2022" -A x64 `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121"

cmake --build build --config Release --target llama-server -j
```

### 🍎 macOS (Metal)
> ⚠️ TurboQuant quantization formats and TriAttention GPU scoring are **CUDA-only** and will not be available on macOS Metal builds.

You can still build and use the base llama.cpp functionality with Metal acceleration:
```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j$(sysctl -n hw.ncpu)
```

---

## 🚀 Quick Start (Ollama Integration)

This fork serves as the perfect engine for [Ollama](https://github.com/ollama/ollama). By building Ollama against this fork, you unlock PagedAttention and TurboQuant seamlessly through Ollama's API. 

Make sure to run Ollama with parallel processing enabled to take full advantage of PagedAttention batching:
```bash
OLLAMA_NUM_PARALLEL=4 ./ollama serve
```

## Branches

| Branch | Description |
|--------|-------------|
| `main` | **Default** — Includes PagedAttention (latest) |
| `feature/triattention` | TurboQuant + TriAttention base |
| `master` | Upstream llama.cpp base |

---

## Credits & Upstream

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov and the incredible ggml contributors.
- [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) — Original TurboQuant fork.
- TriAttention algorithm — [arXiv 2604.04921](https://arxiv.org/abs/2604.04921)
- PagedAttention architecture inspired by vLLM — [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)
- GPU integration, KV cache implementation, and Ollama integration — [@atomicmilkshake](https://github.com/atomicmilkshake) and contributors.

*For comprehensive documentation on the original `llama.cpp` API, models, and UIs, please refer to the [upstream documentation](https://github.com/ggml-org/llama.cpp).*
