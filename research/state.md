# Research State (Estado de la Investigación)

*Última actualización: 2026-07-02*

Este archivo representa el estado vivo del conocimiento en este repositorio. Cualquier sistema de IA debe leer esto antes de proponer hipótesis o escribir código, y actualizarlo al finalizar un experimento.

---

## 📌 Hito Actual (Active Milestone)
- **Hito**: `milestone-007-triattention-calibration`
- **Objetivo**: Calibrar el scoring de páginas de TriAttention sobre corpus representativos y realizar la validación numérica de la calidad del contexto y perplejidad del modelo bajo presupuestos de páginas físicas.
- **Estado**: Infraestructura de calibración completa; validación numérica en progreso (Estado: PENDIENTE GPU + modelo).

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
- **Resultado**: ✅ Corregido el 2026-07-02. Se resolvieron dos bugs independientes de correctitud de perplejidad (PPL): (1) bug de direccionamiento/despacho en kernels FA (MMA_F16/WMMA_F16 ignoraban la page table y leían direcciones contiguas; corregido enrutando a VEC/TILE cuando hay page table adjunta en `fattn.cu`), y (2) race condition por sincronización omitida en `llama_context::process_ubatch` (el scheduler reusaba la tabla de páginas sobreescrita por el ubatch activo antes de que el kernel asíncrono anterior terminara de leer; corregido haciendo la sincronización del backend incondicional antes del reset del grafo).
- **Brecha de validación pendiente**: Falta realizar la comparación byte a byte (byte-level V-pool comparison) para `sequence >= 1` entre paged (`-fa on`) y gather (`-fa off`), aunque la perplejidad (PPL) ya coincide plenamente con el baseline.

### 6. ROCm/HIP Backend Completion
- **Implementación**: Compatibilidad HIP completada para shuffles de warp y ballot en los caminos de SET_ROWS y Flash Attention. `__shfl_xor_sync` ajustado al formulario de 4 argumentos con `WARP_SIZE`; `__ballot_sync` actualizado para wavefront AMD de 64 hilos.
- **Resultado**: Build HIP/ROCm corregido. Auditoría de `turbo-quant.cuh`, `fattn-common.cuh` y `fattn-tile.cuh` sin cambios adicionales.

### 7. TriAttention KV Eviction
- **Implementación**: Presupuesto configurable de páginas físicas (`--triattention-page-budget`), bloque físico 0 reservado como dummy zero block y `pg_score_and_evict()` para desalojar la página de menor score usando productos punto sobre K con RoPE inverso. La fix pass de M006 corrigió el uso de `rope_freq_base`/`rope_freq_scale` efectivos en `get_unrotated_key()` y habilitó enforcement del presupuesto también durante prefill.
- **Resultado**: Implementado, corregido frente a los issues P1/P3 de la crítica y compatible con modelos YaRN/NTK-aware en escenarios single-sequence.
- **Recomendación de validación**: No alterar la implementación hasta cerrar la calibración y validación del hito M007 usando un test de evaluación en modo de generación (generation-mode). La métrica mínima exigida es comparar baseline vs un budget del 50%, manteniendo la misma semilla y el mismo prompt largo, para evaluar la perplejidad y la retención de calidad (generation retention).

### 8. Guarda de Serialización de Estado bajo Paged Attention (`pg_enabled`)
- **Problema**: El layout de V paginado está indirectamente referenciado a través de la tabla de páginas; una lectura lineal (como la que usa `state_write_data()`/`state_read_data()`) corrompería el estado serializado.
- **Implementación**: Capas de defensa simétricas:
  1. `llama_kv_cache::state_write()` (`src/llama-kv-cache.cpp:2134`) retorna con `LLAMA_LOG_ERROR` cuando `pg_enabled==true`, sin escribir ningún byte al IO.
  2. `llama_kv_cache::state_read()` (`src/llama-kv-cache.cpp:2190`) tiene la misma guarda simétrica: retorna con `LLAMA_LOG_ERROR` antes de intentar leer del IO (evita parsear un stream vacío/inválido cuando `state_write` no escribió nada).
  3. `state_write_data()` y `state_read_data()` repiten la guarda como defensa en profundidad en caso de invocación directa que saltee los wrappers.
- **Flujo de `get_size`**: `llama_state_seq_get_size_ext()` llama internamente a `state_write()` con un IO dummy. Si `state_write()` retorna tempranamente (0 bytes escritos), `get_size` devuelve `0`.
- **Server-side**: `server_slot::prompt_save()` (`tools/server/server-context.cpp:105`) verifica `cur_size == 0` antes de llamar `llama_state_seq_get_data_ext()` y omite el guardado en caché con un log de advertencia.
- **Workaround CI**: Las líneas de `ci/run.sh` que prueban `llama-save-load-state -fa on` anteponen `LLAMA_NO_PAGING=1` para deshabilitar paging y verificar la ruta de serialización con flash attention activo.
- **Resultado**: La serialización de estado es una ruta **no soportada** bajo paged attention. Es una ruta **soportada y validada en CI** con `LLAMA_NO_PAGING=1` o con Flash Attention apagado.

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

## 🚦 Estado de CI y Problemas Conocidos (CI Status / Known Issues)

- **`ci/run.sh` — `llama-save-load-state`**: Las dos invocaciones con `-fa on` (`-ngl 10` y `-ngl 99`) se ejecutan con `LLAMA_NO_PAGING=1` porque el paging se activa automáticamente con Flash Attention, y la serialización de estado no es compatible con el layout paginado (ver [Sección 8](#8-guarda-de-serialización-de-estado-bajo-paged-attention-pg_enabled)). Las dos invocaciones con `-fa off` no necesitan la variable porque el paging nunca se activa sin FA.
  - **Causa raíz**: layout de V paginado indirecto vía tabla de páginas; lecturas/escrituras lineales en `state_write_data()`/`state_read_data()` asumen un pool contiguo.
  - **Estado**: Mitigado vía guarda explícita (no crash, no estado corrupto silencioso) + override de entorno para CI.

---

## 🔍 Trabajo de Revisión Pendiente Antes del Próximo Hito

- [ ] Evaluar si exponer `pg_enabled` (o un helper equivalente) en la API pública de `llama.h`/`llama_memory_*` para que clientes como `tools/server` puedan detectar el modo paginado sin depender del side-channel `size == 0` de `llama_state_seq_get_size_ext()`.
- [ ] Decidir si `llama_context::state_seq_get_size()`/`get_data()` deberían distinguir explícitamente "tamaño 0 por paging" de "tamaño 0 por error real" (hoy ambos casos colapsan al mismo valor de retorno).
- [ ] Revisar otros llamadores de `state_write()`/`state_read()` fuera de `tools/server` (p. ej. `llama-save-load-state`, bindings externos) para confirmar que todos manejan con gracia el caso `pg_enabled=true`.

---

## 🔬 Hipótesis Abiertas (Open Hypotheses)

1. **Hipótesis H3.1**: "La integración de la tabla de páginas en el kernel de Flash Attention (Fase 2) reducirá el tiempo de atención en un 12-18% en contextos largos (>8K tokens) al eliminar la sobrecarga de escritura en memoria global del kernel de gather".
2. **Hipótesis H4.1**: "La validación explícita de NaN/Inf en el de-cuantizador de `turbo4` evitará desbordamientos aritméticos catastróficos durante inferencias prolongadas con rangos dinámicos extremos".
3. **Hipótesis H6.1**: "La remoción adaptativa de páginas usando scoring RoPE (TriAttention) permite mantener el contexto efectivo al 95% de la calidad de perplexidad original usando sólo el 50% de las páginas físicas".

---

## 📋 Lista de Tareas Pendientes (TODO)

- [x] Completar Hito 003 Fase 2 (Native Paged FA — implementación)
- [x] Validación GPU numérica Hito 003 (PPL correcto verificado en VEC/TILE; pendiente el gap de comparación byte-level)
- [ ] Benchmark de latencia Hito 003 (contextos >8K tokens)
- [x] Ejecutar validación de NaN en `turbo4` (Hito 004)
- [x] Portar los kernels de `turbo4` a HIP/ROCm
- [x] Implementar el scoring y desalojo físico en TriAttention
- [x] Preparar infraestructura M007 para calibración/validación de TriAttention (`scripts/triattention_calibrate.py`, `research/milestone-007/`)
- [ ] Ejecutar validación GPU de M007 (H6.1: baseline vs eviction con `--triattention-page-budget` sobre modelo real)
- [ ] Completar `research/milestone-007/evidence.md` con resultados de `calibration_results.json`
- [ ] Completar `research/milestone-007/conclusions.md` con veredicto sobre retención >= 95% a 50% de page budget
