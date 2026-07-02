# Phase 2 FA Debug Handoff

**Fecha:** 2026-06-30
**Estado:** Bloqueado — PPL=35125 con `-fa on` + paging activo
**Próximo responsable:** agy

---

## Resumen del problema

Phase 2 paged FA produce PPL=35125 en lugar de ~9.7 (baseline).
Phase 1 (gather, `-fa off`) funciona correctamente: PPL=9.7094 ✓
El bug es específico del path CUDA de Flash Attention con page table.

---

## Lo que se hizo (en esta sesión y la anterior)

### Fixes ya commiteados

| Commit | Fix |
|--------|-----|
| `f82d02b9` | Phase 2 n_seq OOB: expand page table to `max(ns, n_seq)` rows |
| `9fe216c4` | Múltiples fixes de Phase 2 FA + diagnóstico + generation eval script |

### Cambios específicos en `9fe216c4` (`Cuda Fix`)

**`src/llama-kv-cache.cpp`** — `build_input_v_page_table`:
- `op_params[0]` = `pg_block_size` (ya estaba)
- `op_params[1]` = `ns_phys` (conteo de streams físicos, **distinto** de `n_seq` = filas del page table)

**`src/llama-graph.cpp`** — `build_attn`:
- View 4D usa `ns_phys` (de `op_params[1]`) en vez de `v_ptable->ne[1]`
- Strides **antes** del permute: `nb[1]=head_v_eff*elem`, `nb[2]=n_embd_v*elem`
- Después del `permute(0,2,1,3)`: `new_nb[1]=n_embd_v*elem` (slot stride = nb21), `new_nb[2]=head_v_eff*elem` (head stride = nb22)

**`ggml/src/ggml-cuda/fattn-common.cuh`** — Diagnóstico:
- Bloque `#if defined(TURBO_DIAG_FA_PAGED)` añadido después del kernel launch (~línea 1647)
- Imprime: `v_ptable_ne0`, `v_block_size`, `nb21`, `nb22`, primeras 8 entradas del page table, `V->ne[]`, `V->nb[]`

**`scripts/triattention_generation_eval.py`** — nuevo script para H6.1 (generación)

---

## Estado actual del código

```
git log --oneline -3:
9fe216c4c Cuda Fix              ← último commit, todo limpio
f82d02b97 fix: Phase 2 FA n_seq OOB
301b82a49 docs: sync roadmap
```

Working tree: limpio (`git status` = nada)

---

## El problema: análisis técnico

### Pool layout (correcto, ya verificado)

```
pool tensor: [n_embd_v=256, v_rows=544, n_stream=1]
  Block 0: slots  0..31  = dummy zeros (sentinel)
  Block 1: slots 32..63  = real data (page 0)
  ...
  Block 16: slots 512..543 = real data (page 15)
  Total slots: 17 × 32 = 544 ✓
```

### View 4D que se pasa a FA (correcto en teoría)

```cpp
v = ggml_view_4d(ctx0, v_pool,
    128,    // head_v_eff = n_embd_v/n_head_kv = 256/2
    2,      // n_head_kv
    512,    // n_kv_val
    1,      // ns_phys
    256,    // nb[1] = head_v_eff * 2 bytes
    512,    // nb[2] = n_embd_v  * 2 bytes
    262144, // nb[3] = n_embd_v * n_kv_val * 2 bytes
    0);     // offset = 0
```

Después del `permute(0,2,1,3)`:
```
nb21 = new_nb[1] = old_nb[2] = 512  (slot stride)   ✓
nb22 = new_nb[2] = old_nb[1] = 256  (head stride)    ✓
v_trans = (old_nb[1] > old_nb[2]) = (256 > 512) = false ✓  (no extra transpose)
```

### Kernel addressing (correcto en teoría)

```c
// v_paged_ptr:
V_base + (pblock * 32 + within) * nb21
= pool_ptr + head*256 + (pblock*32+within)*512
```

Para token p=0 (lp=0, block=1, within=0):
→ slot 32 → offset = 32 × 512 = 16384 bytes ✓

Para token p=511 (lp=15, block=16, within=31):
→ slot 543 → offset = 543 × 512 = 278016 bytes (< pool_size=278528) ✓

### Math es correcta, pero PPL=35125

Si el kernel leyera del pool sin page table (modo no-paged):
- Leería slots 0..511 directamente (sin offset del bloque dummy)
- Slot 0 = dummy zeros → atención errónea → PPL≈35125 ✓

**Hipótesis principal: la page table NO está llegando al kernel** (v_ptable_data=nullptr en la línea 1243 de fattn-common.cuh).

---

## Hipótesis ordenadas por probabilidad

### H1 (más probable): v_ptable_data=nullptr

En `fattn-common.cuh:1243`:
```cpp
const int32_t * v_ptable_data = v_ptable_tensor ? (const int32_t *)v_ptable_tensor->data : nullptr;
```

Si `dst->src[5] = nullptr`, el kernel corre en modo no-paged y lee slots consecutivos (con dummy block = zeros) → PPL exactamente como el observado.

**¿Por qué src[5] sería null?** En `src/llama-graph.cpp:2185-2196`:
```cpp
ggml_tensor * fa_tensor = cur;
while (fa_tensor && fa_tensor->op != GGML_OP_FLASH_ATTN_EXT) {
    if (op == RESHAPE || VIEW || TRANSPOSE || TURBO_WHT)
        fa_tensor = fa_tensor->src[0];
    else break;
}
if (fa_tensor && fa_tensor->op == GGML_OP_FLASH_ATTN_EXT) {
    ggml_flash_attn_ext_set_page_table(fa_tensor, inp->self_v_page_table);
} else {
    ggml_flash_attn_ext_set_page_table(cur, inp->self_v_page_table);  // ← ASSERT si cur no es FA
}
```

Si el while-loop falla (porque hay un op en la cadena que no es RESHAPE/VIEW/TRANSPOSE/TURBO_WHT), cae al `else` y llama `set_page_table(cur, ...)`. En **release build** (NDEBUG), el `GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT)` es NO-OP → silently sets wrong tensor's src[5], FA tensor's src[5] stays NULL.

Verificar: agregar `fprintf(stderr, "fa_tensor->op=%d cur->op=%d\n", fa_tensor->op, cur->op)` antes del if.

### H2 (menos probable): El build no recompila fattn-common.cuh

El diagnóstico en `#if defined(TURBO_DIAG_FA_PAGED)` no tiene errores de sintaxis, pero compile con `-DTURBO_DIAG_FA_PAGED` para activarlo.

### H3 (poco probable): El page table tensor tiene data=nullptr por orden de inicialización

Si `inp->self_v_page_table->data` es nullptr cuando se lee en el kernel (¿antes de ggml_backend_tensor_set?), retorna garbage. Pero ggml scheduler debería asignar memoria antes de ejecutar.

### H4 (descartada): Stride swap en v_trans

Ya fixeada y verificada matemáticamente. `v_trans = false` con los strides actuales.

---

## Diagnóstico recomendado (para agy)

### Paso 1: Activar diagnóstico sin CUDA_FLAGS

La forma más rápida — cambiar el `#if defined(TURBO_DIAG_FA_PAGED)` a `#if 1` en `ggml/src/ggml-cuda/fattn-common.cuh` (~línea 1647), recompilar solo ese archivo, y ejecutar:

```bash
# Recompile (solo CUDA, ~5-10 min en RTX 2050)
cmake --build build --target llama-perplexity -j$(nproc) 2>&1 | tail -5

# Ejecutar diagnóstico
build/bin/llama-perplexity \
    -m qwen2.5-coder-1.5b-bf16.gguf \
    -f data/wikitext2-test.txt \
    -c 512 --chunks 5 \
    -ngl 99 -fa on \
    --triattention-page-budget 16 \
    2>&1 | grep -E "PAGED_FA_DIAG|Final estimate"
```

**Interpretar resultado:**
- Si `[PAGED_FA_DIAG]` NO aparece → `v_ptable_data=nullptr` → H1 confirmada
- Si aparece con `ptable[0..7]: 0 0 0 0 ...` → page table es dummy block para todo → H3 o bug en set_input_v_page_table
- Si aparece con `ptable[0..7]: 1 2 3 4 5 6 7 8` → page table OK → bug en strides o kernel
- `nb21` debería ser 512, `nb22` debería ser 256

### Paso 2: Si H1 confirmada — arreglar el while-loop

Agregar log para ver qué op tiene `cur` y qué ops hay en la cadena:

```cpp
// En build_attn, antes del while-loop:
fprintf(stderr, "[FA_SEARCH] cur->op=%d\n", (int)cur->op);
ggml_tensor * dbg = cur;
for (int i = 0; i < 5 && dbg; i++, dbg = dbg->src[0])
    fprintf(stderr, "  chain[%d]->op=%d\n", i, (int)dbg->op);
```

Una vez identificado el op que corta el loop, agregar su case al while:
```cpp
while (fa_tensor && fa_tensor->op != GGML_OP_FLASH_ATTN_EXT) {
    auto op = fa_tensor->op;
    if (op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_TRANSPOSE ||
        op == GGML_OP_TURBO_WHT || op == GGML_OP_CONT || /* el op que falta */) {
        fa_tensor = fa_tensor->src[0];
    } else break;
}
```

### Paso 3: Verificar PPL tras el fix

```bash
build/bin/llama-perplexity \
    -m qwen2.5-coder-1.5b-bf16.gguf \
    -f data/wikitext2-test.txt \
    -c 512 --chunks 10 \
    -ngl 99 -fa on \
    --triattention-page-budget 16
```

Esperado: PPL ≈ 9.7 (igual que baseline y Phase 1)

---

## Comandos de referencia

```bash
# Baseline (sin paging):
LLAMA_NO_PAGING=1 build/bin/llama-perplexity -m qwen2.5-coder-1.5b-bf16.gguf \
    -f data/wikitext2-test.txt -c 512 --chunks 5 -ngl 99 -fa on
# → PPL ≈ 9.7214

# Phase 1 (paged, gather, -fa off):
build/bin/llama-perplexity -m qwen2.5-coder-1.5b-bf16.gguf \
    -f data/wikitext2-test.txt -c 512 --chunks 5 -ngl 99 -fa off \
    --triattention-page-budget 16
# → PPL = 9.7094 ✓

# Phase 2 (paged, FA on) — BUGGY:
build/bin/llama-perplexity -m qwen2.5-coder-1.5b-bf16.gguf \
    -f data/wikitext2-test.txt -c 512 --chunks 5 -ngl 99 -fa on \
    --triattention-page-budget 16
# → PPL = 35125 ✗ (esperado: ~9.7)
```

---

## Archivos clave

| Archivo | Qué contiene |
|---------|-------------|
| `src/llama-graph.cpp:2114-2196` | Phase 2 FA path: view 4D + while-loop para set_page_table |
| `src/llama-kv-cache.cpp:1467-1510` | `set_input_v_page_table`: llena page table CPU→GPU |
| `ggml/src/ggml-cuda/fattn-common.cuh:1242-1247` | Extrae v_ptable_data del tensor |
| `ggml/src/ggml-cuda/fattn-common.cuh:1647-1666` | Diagnóstico `TURBO_DIAG_FA_PAGED` |
| `ggml/src/ggml-cuda/fattn-vec.cuh:870` | `need_f16_V = type_V == GGML_TYPE_F16` |
| `ggml/src/ggml-cuda/fattn-vec.cuh:54-69` | `v_paged_ptr`: addressing físico por page table |

---

## Tareas pendientes (en orden)

1. **[CRÍTICO]** Fix Phase 2 FA PPL — diagnóstico + fix según H1/H2/H3 arriba
2. **[P2]** Benchmarks completos una vez validado Phase 2:
   - baseline vs FA vs FA+paging vs FA+TriAttention
   - context lengths: 512, 1024, 2048
   - Guardar en `research/milestone-008/benchmarks.json`
3. **[P4]** Correr `scripts/triattention_generation_eval.py` para H6.1:
   ```bash
   python3 scripts/triattention_generation_eval.py \
       --model qwen2.5-coder-1.5b-bf16.gguf \
       --context-len 512 --n-predict 64 \
       --page-budgets 4 8 16 --runs 2 \
       --extra-args "-ngl 99 -fa on" \
       --output research/milestone-008/generation_eval.json
   ```
4. **[P3]** Commit final con Phase 2 validado + benchmarks + H6.1 resultado

---

## Notas de hardware

- **GPU**: RTX 2050, 3767 MiB VRAM
- **Modelo**: `qwen2.5-coder-1.5b-bf16.gguf` = ~2.88 GiB (cabe justo)
- **OOM**: Usar contexto ≤512 y budget ≤16. Con `-c 2048 --triattention-page-budget 64` → OOM
- **Compile tiempo**: CUDA recompile ~5-10 min (no interrumpir)
- **Procesos paralelos**: NO correr 2 llama-* simultáneos → OOM

## Notas de seguridad (AGENTS.md)

- **NUNCA** commitear API keys, tokens, passwords
- Credenciales solo via env vars o archivos en .gitignore
- Revisar archivos modificados antes de `git commit`

---

## Update — 2026-07-01

**Estado:** Bloqueado — PPL sigue en ~35000-38000 con `-fa on` + paging activo. H1 **descartada** con evidencia empírica. Bug acotado a otra parte del path.

### H1 descartada (con prueba, no solo razonamiento)

Se reemplazó el while-loop frágil de búsqueda de `GGML_OP_FLASH_ATTN_EXT` por una
búsqueda DFS recursiva acotada (`find_flash_attn_ext`, commit `e0e3486dc`,
`src/llama-graph.cpp`), que además aborta explícitamente si no encuentra el
tensor (en vez de fallar silenciosamente en release). El diagnóstico
`TURBO_DIAG_FA_PAGED` confirma que la page table SÍ llega correctamente al
kernel: `v_ptable_ne0=8`, `nb21=512`, `nb22=256`, `ptable[0..7]=1,2,3,4,0,0,0,0`
— todo consistente con lo esperado. **PPL no cambió tras el fix (sigue en
35125.27)**, confirmando que H1 no era la causa raíz.

### Nuevas hipótesis descartadas (esta ronda, con prueba)

1. **Colisión multi-secuencia**: descartada. El bug reproduce incluso con
   `--chunks 1` (fuerza `n_seqs=1`, una sola secuencia activa) → PPL=8285.60
   (roto igual, sin secuencias concurrentes de por medio).
2. **Desacuerdo write/read de fila física**: descartada. Se agregó
   instrumentación `TURBO_DIAG_PAGE_ROWS` (opt-in, commit `d57d50028`) tanto en
   el lado de escritura (`set_input_v_idxs`, `src/llama-kv-cache.cpp`) como en
   el de lectura (`v_paged_ptr`, `ggml/src/ggml-cuda/fattn-common.cuh`). Para
   la página lógica 0, ambos lados resuelven independientemente `pblock=1` →
   `phys_row=32`. Idénticos.
3. **Datos corruptos/no escritos en el pool**: descartada. Se volcaron los
   bytes crudos del pool en `phys_row=32` tras el kernel launch: valores half
   reales y variados (ej. `0.084, 0.29, -0.009...`), no ceros ni basura. La
   fila 0 (bloque dummy/sentinela) sí lee todo-cero correctamente. La
   escritura ocurre de verdad, en la dirección exacta que el kernel lee.

### Conclusión de esta ronda

La corrupción **no está** en el wiring del tensor-graph, ni en el cálculo de
direcciones físicas, ni en que falte escribir datos reales. Está en otra parte
del compute path del kernel FA paginado. Pistas concretas sin verificar aún:

- `flash_attn_mask_to_KV_max` en `ggml/src/ggml-cuda/fattn-common.cuh` — deriva
  el límite de iteración KV del kernel escaneando la mask; nunca se verificó
  específicamente contra el branch paginado de V.
- Manejo de `gqa_ratio` / stride de la mask en el branch paginado de
  `llm_graph_context::build_attn` (`src/llama-graph.cpp` ~líneas 2130-2183).

### Validación actual (sin regresión en los paths que ya funcionaban)

```
Paged, -fa on, --triattention-page-budget 16, --chunks 10:
  PPL = 38849.78   (roto, esperado ~9.7)
Paged, -fa on, --chunks 1 (fuerza n_seqs=1):
  PPL = 8285.60    (roto igual, descarta colisión multi-secuencia)
Paged, -fa off (gather path), --chunks 5:
  PPL = 9.7094     (correcto, sin cambios)
LLAMA_NO_PAGING=1, -fa on, --chunks 5:
  PPL = 9.7214     (correcto, sin cambios)
```

### Commits de esta ronda

| Commit | Contenido |
|--------|-----------|
| `e0e3486dc` | Reemplaza while-loop frágil por DFS recursivo + abort explícito (H1 descartada) |
| `d57d50028` | Instrumentación `TURBO_DIAG_PAGE_ROWS` (opt-in, sin cambio de comportamiento) |

### Próximo paso recomendado

Instrumentar `flash_attn_mask_to_KV_max` y el branch paginado de `build_attn`
para el manejo de `gqa_ratio`/mask-stride, comparando contra el path
`-fa off` (que sí funciona) para el mismo prompt. Repetir la metodología de
esta ronda: probar con evidencia empírica (valores reales impresos/volcados),
no solo lectura estática del código — la lectura estática ya descartó dos
hipótesis plausibles que resultaron correctas en el papel pero no en la
práctica.
