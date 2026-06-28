# Milestone 007: Evidence

**Fecha**: 2026-06-28  
**Hardware**: NVIDIA RTX 2050 (4 GB VRAM), Linux  
**Modelo**: qwen2.5-coder-1.5b-bf16.gguf  
**Corpus**: data/wikitext2-test.txt (wikitext-2 test split, 1,287,655 bytes)

---

## 1. Crash Fix: Off-by-One in Page Pool

### Síntoma
`GGML_ASSERT(!pg_free_blocks.empty())` en `pg_alloc_for_sinfo` al usar `llama-perplexity`
con n_seq=4 (modo por defecto).

### Root Cause
`pg_n_blocks` se usaba como tamaño de pool, pero el bloque 0 es el "dummy zero block"
(pre-zeroed, nunca en el pool libre). El pool tenía `pg_n_blocks - 1` bloques físicos
disponibles para `pg_n_blocks` páginas lógicas. Con n_seq=4 y ctx=512 (16 páginas por stream),
las 4 secuencias necesitaban 64 páginas simultáneamente → pool de 63 bloques → crash.

### Fix aplicado (`src/llama-kv-cache.cpp`)

1. **V tensor size** (línea ~262): Para el path paged (`!v_trans`), se agrega 1 fila extra:
   ```cpp
   const uint32_t v_rows = (!v_trans) ? kv_size + pg_block_size : kv_size;
   ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_gqa_eff, v_rows, n_stream) : nullptr;
   ```

2. **Pool init y clear** (~línea 385, ~418): Pool ahora tiene `pg_n_blocks` bloques (1..pg_n_blocks),
   todos usables:
   ```cpp
   pg_free_blocks.resize(pg_n_blocks);
   for (uint32_t i = 0; i < pg_n_blocks; ++i) {
       pg_free_blocks[i] = pg_n_blocks - i;  // LIFO: top=1, bottom=pg_n_blocks
   }
   ```

3. **Reshape V** (~línea 1574): Usa `v->ne[1]` directamente en lugar de `kv_size`:
   ```cpp
   v = ggml_reshape_2d(ctx, v, n_embd_gqa, v->ne[1]*n_stream);
   ```

**Resultado**: `llama-perplexity` con n_seq=4 ya no crashea.

---

## 2. Bug Documentado: Phase 2 FA Page Table (n_seq vs ns)

### Síntoma
Con el crash fix aplicado, `llama-perplexity` da PPL=35125 (vs baseline 9.72) cuando se usa
Flash Attention nativo (Phase 2, `-fa on` o auto).

### Root Cause
El kernel CUDA (`fattn-common.cuh`) accede `v_ptable[seq * n0 + lpage]` donde `seq` itera
sobre `n_seq` (secuencias en el batch). Pero la tabla de páginas se construye con `ns` filas
(número de streams, típicamente 1). Con n_stream=1 y n_seq=4, los índices `seq=1,2,3`
acceden a memoria fuera de la tabla → valores de pblock inválidos → V values corruptos → PPL inválido.

### Workaround para M007
`-fa off` fuerza el path Phase 1 (gather-before-FA), que lee V values correctamente
independiente del n_seq. PPL con `-fa off` = 9.7094 (correcto, dentro del 0.2% del baseline
LLAMA_NO_PAGING=1 = 9.7214).

### Fix Pendiente
En `set_input_v_page_table` / `build_input_v_page_table`, expandir la tabla a `max(ns, n_seq)`
filas y replicar las filas de streams compartidos. Requiere acceso a n_seq en las funciones
de setup del KV cache.

---

## 3. Calibración M007: Resultados de PPL

### Baseline (sin evicción)
```
LLAMA_NO_PAGING=1 -fa off -c 512 --chunks 5  →  PPL = 9.7214
-fa off -c 2048 --chunks 3                    →  PPL = 10.8322  (n_seq=1)
-fa off -c 2048 --chunks 5                    →  PPL = 10.9552  (calibración oficial)
```

### Eviction (--triattention-page-budget 32, 50% budget, ctx=2048)
```
PPL = 10.9552  (Δ = 0.0000)
```

### Análisis del Delta Zero

El resultado `Δ=0` es **correcto pero no evidencia suficiente para H6.1**. La razón:

`pg_alloc_for_sinfo` tiene protección explícita contra evictar páginas que están en uso en el
batch actual (`current_batch_pages`). En modo batch de `llama-perplexity`, **todas las páginas
del contexto son parte del batch actual** → `current_batch_pages` contiene todas las páginas
→ la condición de evicción nunca se cumple → `--triattention-page-budget` no tiene efecto.

Este es comportamiento correcto: no se deben evictar páginas que se van a escribir en el
batch actual. La implicación es que **TriAttention eviction está diseñado para modo
generación autoregresivo**, no para evaluación de perplexity en batch.

---

## 4. Infraestructura M007 (Completada)

| Componente | Estado |
|---|---|
| `scripts/triattention_calibrate.py` | ✅ Completo, sanity checks, soporte wikitext-2 |
| `data/wikitext2-test.txt` | ✅ Corpus real (1.2 MB wikitext-2 test split) |
| `research/milestone-007/calibration_results.json` | ✅ Generado automáticamente |
| Crash fix (`pg_free_blocks` off-by-one) | ✅ Corregido |
| Phase 2 FA n_seq bug | 📋 Documentado, workaround disponible (`-fa off`) |

---

## 5. Metodología de Validación Requerida para H6.1

Para validar H6.1 correctamente se requiere **modo generación autoregresivo**:

1. Prompt largo (≥ budget × pg_block_size tokens) para llenar el KV cache hasta el límite
2. Generación token a token (`llama-cli --predict N`)
3. Con `--triattention-page-budget B`, cada nuevo token evicta un bloque anterior
4. Medición: log-probabilidad de los tokens generados vs baseline (token condicional PPL)

Esta metodología garantiza que `current_batch_pages` solo contiene la página del token
actual, permitiendo que la evicción afecte el contexto efectivo.

**Estimado de implementación**: 1-2 horas adicionales de script (`scripts/triattention_generation_eval.py`).
