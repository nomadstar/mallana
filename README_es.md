# llama.cpp — TurboQuant + TriAttention + PagedAttention

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-atomicmilkshake%2Fllama--cpp--turboquant-blue?logo=github)](https://github.com/atomicmilkshake/llama-cpp-turboquant)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-binaries-yellow)](https://huggingface.co/atomicmilkshake/llama-cpp-turboquant-binaries)

*Leer en [Español](README_es.md) | Read this in [English](README.md)*

Un fork altamente optimizado de [llama.cpp](https://github.com/ggml-org/llama.cpp) diseñado para maximizar el rendimiento de inferencia de LLMs y la eficiencia de memoria en GPUs NVIDIA. Hemos introducido tres grandes adiciones arquitectónicas:

- **PagedAttention** — Tablas de bloques al estilo vLLM para la gestión de memoria dinámica y no contigua del caché KV, eliminando la fragmentación y permitiendo una concurrencia masiva.
- **TurboQuant** — Formatos de cuantización personalizados de bajos bits (turbo2, turbo3, turbo4) con kernels CUDA optimizados por hardware para una inferencia más rápida con un menor uso de memoria.
- **TriAttention** — Poda (pruning) del caché KV acelerada por GPU ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921)) que evalúa la importancia de los tokens utilizando vectores clave invertidos por RoPE y elimina los tokens de bajo valor, manteniendo la inferencia de contexto largo dentro de un presupuesto estricto de VRAM.

---

## 🔥 Matriz de Estado

Este fork está empujando activamente la frontera de la inferencia local de LLMs.

| Característica | Estado |
|---------|--------|
| Kernels CUDA de TurboQuant (turbo2/3/4, SM75+) | ✅ **En vivo** |
| Poda de caché KV con TriAttention en GPU | ✅ **En vivo** |
| Tabla de bloques PagedAttention | ✅ **En vivo** (`feature/paged-attention`) |
| Backend ROCm / HIP para AMD | ⏳ Parcial — compila correctamente, validación completa pendiente |
| Soporte de backend Vulkan | 🗺️ Planeado |
| Paquete para Arch Linux (`llama-cpp-turboquant-git`) | 📦 **Listo** |

> ⚠️ **Aviso de hardware:** Los kernels actuales de TurboQuant están optimizados para **GPUs NVIDIA (CUDA, SM75+)**. La compatibilidad total con ROCm (AMD) y Vulkan está **pendiente** y llegará en próximas versiones.

## Binarios Pre-compilados para Windows

Descarga la última versión (Windows x64, CUDA 13, RTX 2000+) desde Hugging Face:

**[🤗 atomicmilkshake/llama-cpp-turboquant-binaries](https://huggingface.co/atomicmilkshake/llama-cpp-turboquant-binaries)**

> Requiere el entorno de ejecución de CUDA 13.x (`cublasLt64_13.dll`). Instala el [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) o los redistribuibles de CUDA si no lo tienes.

---

## 🧠 Tecnologías Principales

### 1. PagedAttention (Concurrencia y Batching)
Por defecto, el `llama.cpp` estándar asigna bloques de memoria continuos para el caché KV, lo que lleva a una severa fragmentación y errores de falta de memoria (OOM) bajo alta concurrencia. Nuestra implementación de PagedAttention pagina el caché KV en bloques virtuales de tamaño fijo, asignando memoria dinámicamente tal como lo hace un sistema operativo.

**Beneficios:**
- **Cero Fragmentación:** Permite procesar tamaños de lote (batch sizes) significativamente mayores en GPUs de consumo.
- **Throughput Extremo:** En nuestras pruebas con Llama 3 (8B) en una modesta RTX 2050 (4GB VRAM), activar PagedAttention con 4 peticiones concurrentes produjo un **aumento del 108% en el throughput agregado** (de 8.4 t/s a 17.6 t/s) y redujo el Time-To-First-Token (TTFT) en **12x** al eliminar la inanición de cola (queue starvation).

### 2. TriAttention (Poda del Caché KV)
TriAttention mantiene tu caché KV dentro de un presupuesto de tokens fijo evaluando periódicamente todos los tokens almacenados en caché y desalojando los menos importantes. La puntuación utiliza la estructura geométrica de los vectores clave codificados con RoPE — sin requerir pesos de modelo adicionales ni fine-tuning.

**Rendimiento (Qwen3-8B Q4_K_M, RTX 3080, `-c 512`)**
| Modo | Sobrecarga de poda | Velocidad de generación |
|------|---------------|-----------------|
| Sin límite de presupuesto | — | 17.5 tok/s |
| Puntuación en CPU | ~5,900 ms/evento | 17.5 tok/s |
| **Puntuación en GPU** | **~4–9 ms/evento** | **75.0 tok/s** |

La puntuación en GPU es ~1,000× más rápida que en CPU. El aumento de velocidad de generación de 4.3× proviene de mantener el caché KV dentro del presupuesto de VRAM (evitando estancamientos por desalojo y manteniendo tamaños de lote consistentes en flash-attention).

### 3. TurboQuant
TurboQuant proporciona tres formatos de cuantización personalizados que superan a los GGUF estándar en anchos de bit equivalentes:

| Formato | Bits/peso | Notas |
|--------|------------|-------|
| `turbo4` | ~4.0 | Reemplazo directo para `q4_0`, con clustering basado en rotación |
| `turbo3` | ~3.0 | Sub-byte con pre-rotación de Hadamard |
| `turbo2` | ~2.0 | Compresión agresiva con centroides en espacio WHT |

Todos los formatos cuentan con kernels CUDA altamente optimizados para las arquitecturas Turing+ (SM75) y Ampere (SM80/86).

---

### 🛠️ Auditoría de Junio 2026: Unificación de Caché SWA y Refactorización de TurboQuant

Recientemente realizamos una auditoría de código Ponytail completa y una refactorización para brindar estabilidad de nivel de producción a los contextos de sliding window attention (SWA) y cuantizaciones de bajos bits:

1. **Caché Dividido de SWA Unificado**:
   - Reemplazamos las clases redundantes de gestión de secuencias y mapeo de punteros de tokens en la ruta de caché dividido SWA con **delegación unificada** dentro de [llama-kv-cache.cpp](file:///home/ignatus/GitHub/llama-cpp-turboquant/src/llama-kv-cache.cpp) y [llama-kv-cache.h](file:///home/ignatus/GitHub/llama-cpp-turboquant/src/llama-kv-cache.h).
   - Eliminamos más de 140 líneas de código duplicado de secuencias, resolviendo fallos de sincronización y crashes de indexación de memoria incorrecta (como en `test-llama-archs`) al ejecutar arquitecturas con SWA (ej. Gemma-3).
2. **Correcciones del Decuantizador de Referencia de TurboQuant**:
   - Corregimos los decuantizadores de referencia de CPU para `turbo3` y `turbo2` en [ggml-turbo-quant.c](file:///home/ignatus/GitHub/llama-cpp-turboquant/ggml/src/ggml-turbo-quant.c) mediante la implementación simétrica de la **Transformada Inversa de Walsh-Hadamard (IWHT)** (`turbo_cpu_iwht`). Anteriormente, los stubs dejaban los vectores reconstruidos en el espacio rotado, lo que fallaba las validaciones.
3. **Seguridad de Heap en Pruebas Unitarias**:
   - Parcheamos [test-quantize-fns.cpp](file:///home/ignatus/GitHub/llama-cpp-turboquant/tests/test-quantize-fns.cpp) para asignar correctamente memoria de heap para productos punto de vectores `F32`, resolviendo un crash crítico de corrupción de heap (`double free or corruption`).
   - Integramos umbrales de error absoluto y producto punto personalizados para formatos de bajos bits (`turbo2`, `turbo3`, `turbo4`), logrando una **validación del 100% en la suite de pruebas (0 pruebas falladas)** tanto en compilaciones de CPU como de CUDA.

---

## ⚡ Compilando desde el Código Fuente

### Requisitos
- Windows 10/11 o Linux (Arch, Ubuntu, Fedora, etc.)
- CUDA Toolkit 12.x o 13.x
- Visual Studio 2022+ con carga de trabajo C++ (Windows) o GCC 11+ (Linux)
- CMake 3.21+

### 🐧 Linux (CUDA)
Funciona en cualquier distribución con GCC 11+ y el CUDA Toolkit instalado.
```bash
git clone https://github.com/atomicmilkshake/llama-cpp-turboquant
cd llama-cpp-turboquant

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server -j$(nproc)
```

### 🎯 Arch Linux (Empaquetado Nativo)
Proveemos soporte nativo de `PKGBUILD` que maneja stubs débiles (weak stubs) de CUDA, configuración de servicios systemd y metadatos limpios de paquetes. Usa tu helper AUR favorito:
```bash
# Disponible pronto en AUR
yay -S llama-cpp-turboquant-git
```

### 🪟 Windows (CUDA)
Requiere Visual Studio 2022 y el CUDA Toolkit.
```powershell
git clone https://github.com/atomicmilkshake/llama-cpp-turboquant
cd llama-cpp-turboquant

cmake -B build -G "Visual Studio 18 2022" -A x64 `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121"

cmake --build build --config Release --target llama-server -j
```

### 🍎 macOS (Metal)
> ⚠️ Los formatos de cuantización de TurboQuant y la puntuación GPU de TriAttention son **exclusivos de CUDA** y no estarán disponibles en compilaciones Metal de macOS.

Aún puedes compilar y usar la funcionalidad base de llama.cpp con aceleración Metal:
```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j$(sysctl -n hw.ncpu)
```

---

## 🚀 Inicio Rápido (Integración con Ollama)

Este fork sirve como el motor perfecto para [Ollama](https://github.com/ollama/ollama). Al compilar Ollama contra este fork, desbloqueas PagedAttention y TurboQuant sin problemas a través de la API de Ollama.

Asegúrate de ejecutar Ollama con el procesamiento paralelo activado para aprovechar al máximo el batching de PagedAttention:
```bash
OLLAMA_NUM_PARALLEL=4 ./ollama serve
```

## Ramas (Branches)

| Rama | Descripción |
|--------|-------------|
| `main` | **Por defecto** — Incluye PagedAttention (más reciente) |
| `feature/triattention` | Base de TurboQuant + TriAttention |
| `master` | Base del llama.cpp upstream |

---

## Créditos y Upstream

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov y los increíbles colaboradores de ggml.
- [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) — Fork original de TurboQuant.
- Algoritmo TriAttention — [arXiv 2604.04921](https://arxiv.org/abs/2604.04921)
- Arquitectura PagedAttention inspirada en vLLM — [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)
- Integración GPU, implementación del caché KV, e integración con Ollama — [@atomicmilkshake](https://github.com/atomicmilkshake) y colaboradores.

*Para documentación completa sobre la API original de `llama.cpp`, modelos y UIs, por favor consulta la [documentación original](https://github.com/ggml-org/llama.cpp).*
