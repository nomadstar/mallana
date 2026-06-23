# llama.cpp — TurboQuant + TriAttention

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-nomadstar%2Fllama--cpp--turboquant-blue?logo=github)](https://github.com/nomadstar/llama-cpp-turboquant)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-binaries-yellow)](https://huggingface.co/atomicmilkshake/llama-cpp-turboquant-binaries)

[🇬🇧 English](#english) | [🇪🇸 Español](#español)

---

<a name="english"></a>
## English

A personal fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) combining three major additions:

- **TurboQuant** — custom low-bit quantization formats (`turbo2`, `turbo3`, `turbo4`) with hardware-optimised CUDA kernels using Walsh-Hadamard Transform (WHT) polar coding for smaller memory footprint
- **TriAttention** — GPU-accelerated KV cache pruning ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921)) that scores token importance using RoPE-inverted key vectors and evicts low-value tokens, enabling long-context inference within a fixed memory budget
- **PagedAttention** — block-table KV memory management ([arXiv 2309.06180](https://arxiv.org/abs/2309.06180)) — engine present, CLI flags pending

> **Hardware note:** TurboQuant+ KV compression shows meaningful throughput gains at long context (≥16k tokens) on GPUs with ≥16 GB VRAM. On smaller GPUs (≤8 GB), the KV cache fits in memory regardless of quantization at typical context lengths, so the improvement is negligible.

### Pre-built Windows Binaries

Download the latest Release build (Windows x64, CUDA 13, RTX 2000+) from Hugging Face:

**[🤗 atomicmilkshake/llama-cpp-turboquant-binaries](https://huggingface.co/atomicmilkshake/llama-cpp-turboquant-binaries)**

> Requires CUDA 13.x runtime (`cublasLt64_13.dll`). Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) or the CUDA runtime redistributable if you don't have it.

### TriAttention

TriAttention keeps your KV cache within a fixed token budget by periodically scoring all cached tokens and evicting the least important ones. Scoring uses the geometric structure of RoPE-encoded key vectors — no additional model weights or fine-tuning required.

#### Performance (Qwen3-8B Q4\_K\_M, RTX 3080, `-c 512`)

| Mode | Prune overhead | Generation speed |
|------|---------------|-----------------|
| No budget limit | — | 17.5 tok/s |
| CPU scoring | ~5,900 ms/event | 17.5 tok/s |
| **GPU scoring** | **~4–9 ms/event** | **75.0 tok/s** |

GPU scoring is ~1,000× faster than CPU. The 4.3× generation speedup comes from keeping the KV cache within VRAM budget (no eviction stalls, consistent flash-attention batch sizes).

#### Quick start

```bash
llama-server -m YourModel.gguf -c 32768 -ngl 99 --port 8080 \
  --triattention-stats model.triattention \
  --triattention-budget 4096 \
  --triattention-window 256 \
  --triattention-log
```

A `.triattention` calibration file is required. Generate one from a HuggingFace model and a plain-text corpus:

```bash
python3 scripts/calibrate-triattention.py \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --corpus corpus.txt \
  --output model.triattention \
  --vram-gb 3 --ram-gb 20
```

#### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--triattention-stats <file>` | *(none)* | Calibration file — **required to enable TriAttention** |
| `--triattention-budget <n>` | `512` | Maximum KV tokens to retain after each prune |
| `--triattention-window <n>` | `64` | Most-recent N tokens always protected from eviction |
| `--triattention-trigger <mode>` | `slack` | When to prune: `slack` (budget+window), `interval`, `fill` |
| `--triattention-log` | off | Print a line for each prune event |
| `--triattention-no-protect-prefill` | off | Allow evicting prompt (prefill) tokens |

#### How it works

1. When occupied KV cells exceed `budget + window` (SLACK mode), a prune is triggered
2. The most recent `window` positions and all prefix/prompt tokens are protected
3. For each sampled `(layer, head)` pair, key vectors are read from the KV cache, RoPE rotation is inverted, and a geometric offset score is computed on the GPU
4. The top-`budget` tokens by importance score are kept; the rest are evicted
5. Position gaps left by evicted tokens are harmless — RoPE handles non-contiguous positions natively

### TurboQuant

TurboQuant provides three custom quantization formats that apply WHT-based rotation before quantization:

| Format | Bits/weight | Notes |
|--------|------------|-------|
| `turbo4_0` | ~4.0 | Drop-in replacement for `q4_0`, with rotation-based clustering |
| `turbo3_0` | ~3.0 | Sub-byte with Hadamard pre-rotation |
| `turbo2_0` | ~2.0 | Aggressive compression with WHT-space centroids |

All formats have CUDA kernels optimised for Turing+ (SM75) and Ampere (SM80/86) architectures. AMD GPUs are supported via HIP (`ggml_cuda_dp4a` portable path) for RDNA3/RDNA4/CDNA3/CDNA4.

### Building from source

#### Requirements

- Windows 10/11 or Linux
- CUDA Toolkit 12.x or 13.x (or ROCm 5.x+ for AMD)
- Visual Studio 2022+ with C++ workload (Windows) or GCC 11+ (Linux)
- CMake 3.21+
- ccache (recommended — `bench.sh` uses it to skip unchanged rebuilds)

#### Linux — NVIDIA (CUDA)

```bash
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121"
cmake --build build --config Release --parallel $(nproc)
```

#### Linux — AMD (ROCm/HIP)

```bash
cmake -B build -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1100"   # adjust to your GPU
cmake --build build --config Release --parallel $(nproc)
```

#### Windows (CUDA)

```powershell
cmake -B build -G "Visual Studio 18 2022" -A x64 `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121"
cmake --build build --config Release --target llama-server -j
```

#### Using bench.sh (recommended)

`bench.sh` auto-detects your GPU (NVIDIA or AMD), skips compilation if no source changes, and benchmarks all Ollama models automatically:

```bash
./bench.sh
# Override GPU flags manually if needed:
GPU_FLAGS="-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942" ./bench.sh
```

### Branches

| Branch | Description |
|--------|-------------|
| `feature/supermerge` | **Main** — TurboQuant + TriAttention + PagedAttention |
| `feature/triattention-paged` | TriAttention + PagedAttention (without TurboQuant KV) |
| `master` | Upstream llama.cpp base |

### Known issues / Current status

> This section documents the real state of the project, including confirmed bugs and unverified functionality.

#### 🔴 Critical — TurboQuant produces degenerate output

The CUDA dispatch for `turbo2_0` / `turbo3_0` / `turbo4_0` applies a **flat** Hadamard to the Q vector, while K/V use WHT with random rotation (`TURBO_ROTATION_R`). The asymmetry produces degenerate attention scores. Observed symptoms: word repetition, then repeated `UNK`/`?` tokens. Active suspicion: incorrect initialization of `TURBO_ROTATION_R` or a transposed argument in the dequant kernel. **There is no evidence that these formats produce coherent text.**

The 75 tok/s figure in the TriAttention table is with standard `Q4_K_M` + TriAttention — **not** with TurboQuant formats.

#### 🔴 Stack buffer overflow — CPU dequant at head_dim=256

A stack buffer overflow was identified in the CPU dequantization path for `head_dim=256` models (e.g. Llama-3 70B, Qwen2.5 72B). Not confirmed whether it was patched in `feature/supermerge`.

#### 🟡 No regression tests for turbo2/3/4

No test in `/tests` covers dequant correctness for `turbo2_0`, `turbo3_0`, `turbo4_0`. The existing `test-quantize-fns` does not exercise these types.

#### 🟡 PagedAttention — engine present, CLI not connected

The PagedAttention engine exists in the code but CLI flags are not implemented. Not usable from `llama-server` or `llama-cli`.

#### 🟡 Ollama fork — integration unverified

`nomadstar/ollama` is on `main` with no dedicated branch for TurboQuant. It is not confirmed whether the fork's `CMakeLists` points to the correct branch of this repo.

#### ⚪ HIP/ROCm — no dedicated branch, untested on real AMD hardware

Kernels use `ggml_cuda_dp4a` with AMD path in `common.cuh`, but the ROCm port has never been tested on real hardware.

### Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov and contributors
- [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) — original TurboQuant fork by Tom
- TriAttention algorithm — [arXiv 2604.04921](https://arxiv.org/abs/2604.04921)
- GPU integration and KV cache implementation — [@atomicmilkshake](https://github.com/atomicmilkshake)
- `feature/supermerge`, calibration tooling, AMD detection — [@nomadstar](https://github.com/nomadstar)

---

<a name="español"></a>
## Español

Fork personal de [llama.cpp](https://github.com/ggml-org/llama.cpp) que combina tres adiciones principales:

- **TurboQuant** — formatos de cuantización de baja precisión (`turbo2`, `turbo3`, `turbo4`) con kernels CUDA optimizados que usan Transformada de Walsh-Hadamard (WHT) para codificación polar, reduciendo el footprint de memoria
- **TriAttention** — poda de caché KV acelerada por GPU ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921)) que puntúa la importancia de los tokens usando vectores clave con rotación RoPE invertida y evicta los menos relevantes, permitiendo inferencia de contexto largo dentro de un presupuesto fijo de memoria
- **PagedAttention** — gestión de memoria KV por tabla de bloques ([arXiv 2309.06180](https://arxiv.org/abs/2309.06180)) — motor presente, flags CLI pendientes

> **Nota de hardware:** La compresión KV de TurboQuant+ muestra mejoras de rendimiento significativas en contexto largo (≥16k tokens) en GPUs con ≥16 GB de VRAM. En GPUs más pequeñas (≤8 GB), la caché KV cabe en memoria independientemente de la cuantización en longitudes de contexto típicas, por lo que la mejora es insignificante.

### Binarios precompilados para Windows

Descarga el último build de release (Windows x64, CUDA 13, RTX 2000+) desde Hugging Face:

**[🤗 atomicmilkshake/llama-cpp-turboquant-binaries](https://huggingface.co/atomicmilkshake/llama-cpp-turboquant-binaries)**

> Requiere el runtime de CUDA 13.x (`cublasLt64_13.dll`). Instala el [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) o el redistribuible del runtime de CUDA si no lo tienes.

### TriAttention

TriAttention mantiene tu caché KV dentro de un presupuesto fijo de tokens puntuando periódicamente todos los tokens en caché y evictando los menos importantes. La puntuación usa la estructura geométrica de los vectores clave codificados con RoPE — no requiere pesos adicionales ni fine-tuning.

#### Rendimiento (Qwen3-8B Q4\_K\_M, RTX 3080, `-c 512`)

| Modo | Overhead de poda | Velocidad de generación |
|------|-----------------|------------------------|
| Sin límite de presupuesto | — | 17.5 tok/s |
| Puntuación CPU | ~5.900 ms/evento | 17.5 tok/s |
| **Puntuación GPU** | **~4–9 ms/evento** | **75,0 tok/s** |

La puntuación GPU es ~1.000× más rápida que CPU. La mejora de 4.3× en velocidad de generación viene de mantener la caché KV dentro del presupuesto de VRAM (sin pausas por evicción, tamaños de lote consistentes en flash-attention).

#### Inicio rápido

```bash
llama-server -m TuModelo.gguf -c 32768 -ngl 99 --port 8080 \
  --triattention-stats modelo.triattention \
  --triattention-budget 4096 \
  --triattention-window 256 \
  --triattention-log
```

Se requiere un archivo de calibración `.triattention`. Genera uno desde un modelo HuggingFace y un corpus de texto plano:

```bash
python3 scripts/calibrate-triattention.py \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --corpus corpus.txt \
  --output modelo.triattention \
  --vram-gb 3 --ram-gb 20
```

#### Flags CLI

| Flag | Por defecto | Descripción |
|------|-------------|-------------|
| `--triattention-stats <archivo>` | *(ninguno)* | Archivo de calibración — **requerido para activar TriAttention** |
| `--triattention-budget <n>` | `512` | Máximo de tokens KV a retener tras cada poda |
| `--triattention-window <n>` | `64` | Los N tokens más recientes siempre protegidos de evicción |
| `--triattention-trigger <modo>` | `slack` | Cuándo podar: `slack` (presupuesto+ventana), `interval`, `fill` |
| `--triattention-log` | desactivado | Imprime una línea por cada evento de poda |
| `--triattention-no-protect-prefill` | desactivado | Permite evictar tokens del prompt (prefill) |

#### Cómo funciona

1. Cuando las celdas KV ocupadas superan `presupuesto + ventana` (modo SLACK), se dispara una poda
2. Las `ventana` posiciones más recientes y todos los tokens del prefijo/prompt quedan protegidos
3. Para cada par `(capa, cabeza)` muestreado, los vectores clave se leen de la caché KV, se invierte la rotación RoPE y se calcula una puntuación de desplazamiento geométrico en la GPU
4. Los `presupuesto` tokens de mayor puntuación se conservan; el resto se evictan
5. Los huecos de posición dejados por los tokens evictados son inofensivos — RoPE maneja posiciones no contiguas de forma nativa

### TurboQuant

TurboQuant proporciona tres formatos de cuantización personalizados que aplican rotación WHT antes de cuantizar:

| Formato | Bits/peso | Notas |
|---------|-----------|-------|
| `turbo4_0` | ~4.0 | Reemplazo directo de `q4_0`, con agrupamiento basado en rotación |
| `turbo3_0` | ~3.0 | Sub-byte con pre-rotación de Hadamard |
| `turbo2_0` | ~2.0 | Compresión agresiva con centroides en espacio WHT |

Todos los formatos tienen kernels CUDA optimizados para arquitecturas Turing+ (SM75) y Ampere (SM80/86). Las GPUs AMD son compatibles vía HIP (path portable `ggml_cuda_dp4a`) para RDNA3/RDNA4/CDNA3/CDNA4.

### Compilar desde el código fuente

#### Requisitos

- Windows 10/11 o Linux
- CUDA Toolkit 12.x o 13.x (o ROCm 5.x+ para AMD)
- Visual Studio 2022+ con workload C++ (Windows) o GCC 11+ (Linux)
- CMake 3.21+
- ccache (recomendado — `bench.sh` lo usa para evitar recompilaciones innecesarias)

#### Linux — NVIDIA (CUDA)

```bash
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121"
cmake --build build --config Release --parallel $(nproc)
```

#### Linux — AMD (ROCm/HIP)

```bash
cmake -B build -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1100"   # ajusta a tu GPU
cmake --build build --config Release --parallel $(nproc)
```

#### Windows (CUDA)

```powershell
cmake -B build -G "Visual Studio 18 2022" -A x64 `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121"
cmake --build build --config Release --target llama-server -j
```

#### Usando bench.sh (recomendado)

`bench.sh` detecta automáticamente tu GPU (NVIDIA o AMD), omite la compilación si no hay cambios en el código fuente, y hace benchmark de todos los modelos de Ollama automáticamente:

```bash
./bench.sh
# Sobrescribir los flags de GPU manualmente si es necesario:
GPU_FLAGS="-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942" ./bench.sh
```

### Ramas

| Rama | Descripción |
|------|-------------|
| `feature/supermerge` | **Principal** — TurboQuant + TriAttention + PagedAttention |
| `feature/triattention-paged` | TriAttention + PagedAttention (sin TurboQuant KV) |
| `master` | Base de llama.cpp upstream |

### Issues conocidos / Estado actual

> Esta sección documenta el estado real del proyecto, incluyendo bugs confirmados y funcionalidad no verificada end-to-end.

#### 🔴 Crítico — TurboQuant genera texto degenerado

El dispatch CUDA de los formatos `turbo2_0` / `turbo3_0` / `turbo4_0` aplica un Hadamard **plano** al vector Q, mientras K/V usan WHT con rotación aleatoria (`TURBO_ROTATION_R`). La asimetría produce attention scores degenerados. Síntomas observados: repetición de palabras, luego tokens `UNK`/`?` repetidos. Sospecha activa: inicialización incorrecta del tensor `TURBO_ROTATION_R` o transposición de argumento en el kernel de dequant. **No hay evidencia de que estos formatos generen texto coherente.**

Los 75 tok/s de la tabla de TriAttention son con `Q4_K_M` estándar + TriAttention — **no** con los formatos TurboQuant propios.

#### 🔴 Stack buffer overflow — CPU dequant con head_dim=256

Identificado un stack buffer overflow en el path de dequantización CPU para `head_dim=256` (ej. Llama-3 70B, Qwen2.5 72B). No está confirmado si quedó parcheado en `feature/supermerge`.

#### 🟡 Sin tests de regresión para turbo2/3/4

No existe ningún test en `/tests` que cubra correctitud de dequant para los tipos `turbo2_0`, `turbo3_0`, `turbo4_0`. El `test-quantize-fns` existente no los ejercita.

#### 🟡 PagedAttention — engine presente, CLI no conectado

El engine de PagedAttention existe en el código pero los flags CLI no están implementados. No es usable desde `llama-server` ni `llama-cli`.

#### 🟡 Fork de Ollama — integración sin verificar

`nomadstar/ollama` está en `main` sin rama específica para TurboQuant. No está confirmado si el `CMakeLists` del fork apunta al branch correcto de este repo.

#### ⚪ HIP/ROCm — sin rama dedicada ni prueba en hardware AMD real

Los kernels usan `ggml_cuda_dp4a` con path AMD en `common.cuh`, pero el port ROCm nunca fue probado en hardware real.

### Créditos

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov y colaboradores
- [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) — fork original de TurboQuant por Tom
- Algoritmo TriAttention — [arXiv 2604.04921](https://arxiv.org/abs/2604.04921)
- Integración GPU e implementación del caché KV — [@atomicmilkshake](https://github.com/atomicmilkshake)
- `feature/supermerge`, herramienta de calibración, detección AMD — [@nomadstar](https://github.com/nomadstar)

---

*Para la documentación original de llama.cpp, ver [docs/](docs/) o [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).*

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from Hugging Face Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. The `MODEL_ENDPOINT` must point to a Hugging Face compatible API endpoint.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
