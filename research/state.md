# Research State (Estado de la Investigación)

*Última actualización: 2026-06-26*

Este archivo representa el estado vivo del conocimiento en este repositorio. Cualquier sistema de IA debe leer esto antes de proponer hipótesis o escribir código, y actualizarlo al finalizar un experimento.

---

## 📌 Hito Actual (Active Milestone)
- **Hito**: `milestone-006-triattention-kv-eviction`
- **Objetivo**: Implementar desalojo físico de páginas KV mediante scoring TriAttention con claves K invertidas por RoPE, más presupuesto configurable vía `--triattention-page-budget`.
- **Estado**: Implementado. Build OK. Pendiente validación numérica/calibración.

---

## ✅ Caminos Validados (Validated Paths / Known Good)

Las siguientes técnicas e implementaciones están demostradas, optimizadas y bajo control de calidad estricto:

### 1. Cuantización Turbo (turbo2, turbo3, turbo4)
- **Fórmula**: Rotación Fast Walsh-Hadamard Transform (FWHT) + Polar Codebook Quantization en bloques de 128 elementos.
- **Bit-width**: `turbo2` (2.5 bpw), `turbo3` (3.5 bpw), `turbo4` (4.25 bpw).
- **Consistencia CPU/GPU**: Verificada bit a bit en el Hito 001. Comparten tablas de centroides, signos de WHT y factor de normalización `1/sqrt(d)`.
- **Compatibilidad**: Funciona en CPU, CUDA, HIP/ROCm (2/3) y Metal (2).

### 2. Flash Attention con De-cuantización On-the-fly
- **Desempeño**: Acceso directo a V decodificado dentro de los hilos de CUDA sin almacenamiento global temporal de matrices FP16.
- **Kernels**: Plantillas `VEC` optimizadas con `ldmatrix` y operaciones de MMA (Matrix Multiply-Accumulate) para KQ dot.

### 3. Política Asimétrica K/V (Asymmetric K/V Policy)
- **Hallazgo**: La clave (K) es altamente sensible a la cuantización de bajo bit; el valor (V) tolera una compresión más agresiva (2.5–3.5 bpw).
- **Default Recomendado**: K = `q8_0` o `f16` + V = `turbo3`. Mantiene la perplexidad a menos de 0.1 PPL de distancia del baseline.

### 4. Paged Attention Fase 1 (Gather-before-FA)
- **Implementación**: Asignador dinámico de páginas (bloques de 32 tokens) y kernel de gather rápido `GGML_OP_GATHER_PAGED_V` para reconstruir un tensor contiguo de V antes de entrar a Flash Attention.
- **Resultado**: Correctitud funcional demostrada para contextos dinámicos no contiguos sin tocar los kernels de FA.

### 5. Paged Attention Fase 2 (Native Paged FA)
- **Implementación**: Integración de búsqueda en tabla de páginas directamente dentro de los kernels VEC y TILE de Flash Attention mediante `v_paged_ptr()`. Elimina el gather intermedio. API `ggml_flash_attn_ext_set_page_table()` para adjuntar tabla de páginas vía `src[5]`. Stubs ABI para kernels MMA-f16 y WMMA-f16.
- **Resultado**: Código compilado. Pendiente de validación numérica y benchmark de latencia.

### 6. ROCm/HIP Backend Completion
- **Implementación**: Compatibilidad HIP completada para shuffles de warp y ballot en los caminos de SET_ROWS y Flash Attention. `__shfl_xor_sync` ajustado al formulario de 4 argumentos con `WARP_SIZE`; `__ballot_sync` actualizado para wavefront AMD de 64 hilos.
- **Resultado**: Build HIP/ROCm corregido. Auditoría de `turbo-quant.cuh`, `fattn-common.cuh` y `fattn-tile.cuh` sin cambios adicionales.

### 7. TriAttention KV Eviction
- **Implementación**: Presupuesto configurable de páginas físicas (`--triattention-page-budget`), bloque físico 0 reservado como dummy zero block y `pg_score_and_evict()` para desalojar la página de menor score usando productos punto sobre K con RoPE inverso.
- **Resultado**: Implementado y compila. Pendiente de validación numérica de calidad/perplexity.

---

## ❌ Caminos Rechazados o Problemáticos (Known Bad / Rejected)

**¡NO volver a investigar ni intentar implementar estas soluciones!**

### 1. Rotación Densa QR en CPU para `turbo4`
- **Rechazado porque**: Generaba inconsistencia numérica con el kernel de GPU que usaba FWHT por motivos de rendimiento.
- **Decisión**: Se reemplazó por FWHT en CPU (Commit `6457eac19`). La rotación densa en CPU está obsoleta.

### 2. Cuantización Directa de Bajo Bit para K en la Familia Qwen (sin protección)
- **Rechazado porque**: Qwen presenta outliers masivos de activación en ciertas dimensiones de las claves (K). La cuantización simétrica de bajo bit aplasta la precisión de los demás canales, disparando la perplexidad (PPL > 500).
- **Decisión**: Mantener K en `q8_0`/`f16`, o usar ecualización `InnerQ` para atenuar outliers.

### 3. Modificación del Flujo de K en Paged Attention mediante Page Tables
- **Rechazado porque**: El cálculo y la estructura de K deben permanecer en un pool plano y lineal rápido para evitar la sobrecarga de indirection en el prefill. K siempre usa índices planos (`flat pool indices`).

### 4. Paging y fallback SDPA en simultáneo
- **Rechazado porque**: Si Flash Attention se auto-desactiva (por dimensiones de cabezal incompatibles), el fallback de SDPA requiere que `v_trans=true`, lo cual es incompatible con el layout del pool de páginas.
- **Decisión**: Si FA se desactiva, la paginación se desactiva automáticamente para evitar crashes.

---

## 🔬 Hipótesis Abiertas (Open Hypotheses)

1. **Hipótesis H3.1**: "La integración de la tabla de páginas en el kernel de Flash Attention (Fase 2) reducirá el tiempo de atención en un 12-18% en contextos largos (>8K tokens) al eliminar la sobrecarga de escritura en memoria global del kernel de gather".
2. **Hipótesis H4.1**: "La validación explícita de NaN/Inf en el de-cuantizador de `turbo4` evitará desbordamientos aritméticos catastróficos durante inferencias prolongadas con rangos dinámicos extremos".
3. **Hipótesis H6.1**: "La remoción adaptativa de páginas usando scoring RoPE (TriAttention) permite mantener el contexto efectivo al 95% de la calidad de perplexidad original usando sólo el 50% de las páginas físicas".

---

## 📋 Lista de Tareas Pendientes (TODO)

- [x] Completar Hito 003 Fase 2 (Native Paged FA — implementación)
- [ ] Validación GPU numérica Hito 003 (comparación Phase 2 vs Phase 1 vs baseline no-paged)
- [ ] Benchmark de latencia Hito 003 (contextos >8K tokens)
- [x] Ejecutar validación de NaN en `turbo4` (Hito 004)
- [x] Portar los kernels de `turbo4` a HIP/ROCm
- [x] Implementar el scoring y desalojo físico en TriAttention
- [ ] Validación numérica Hito 006 (TriAttention KV eviction)
