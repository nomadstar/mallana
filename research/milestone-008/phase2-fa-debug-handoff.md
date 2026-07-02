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

---

## Update — 2026-07-02: `flash_attn_mask_to_KV_max` descartada (con prueba)

**Estado:** Bloqueado — PPL sigue en ~38849 con `-fa on` + paging activo, `--chunks
10`. Se descarta empíricamente la hipótesis del fast-path `KV_max` como causa
(única o compuesta) del bug. Bug sigue acotado al resto del compute path
paginado del kernel FA.

Nota aparte (trace estático, sesión previa a esta): el cálculo de addressing
de `gqa_ratio` en sí (`fattn-vec.cuh:124-128`, offset de head aplicado una
sola vez sobre `V_paged_base` antes de `v_paged_ptr`, `gqa_ratio = ne02/ne12`
tomado de `K->ne[2]`, no de la vista paginada de V) ya fue revisado
línea por línea vía el grafo de código y es matemáticamente consistente con
lo que `TURBO_DIAG_FA_PAGED` había verificado empíricamente (`nb21=512,
nb22=256`). Eso descarta la aritmética de `gqa_ratio` puntualmente. Lo que
queda sin verificar del lead original es el manejo de **stride de la mask**
en el branch paginado (no `gqa_ratio` en sí) — ver más abajo.

### Nota de entorno: repo detach rompió el build dir

Antes de correr el experimento, `cmake --build` falló:

```
CMake Error: The current CMakeCache.txt directory ... is different than the
directory /home/ignatus/GitHub/llama-cpp-turboquant/build where CMakeCache.txt
was created. The source directory "/home/ignatus/GitHub/llama-cpp-turboquant"
does not exist.
```

Confirma el "repo detach" ya anotado en el commit `95ab89c01` (rename
`llama-cpp-turboquant` → `mallana`). Workaround aplicado (fuera del repo git,
no versionado): `ln -s /home/ignatus/GitHub/mallana
/home/ignatus/GitHub/llama-cpp-turboquant`. Con el symlink en su lugar,
`cmake --build build` funciona sin reconfigurar ni recompilar desde cero. El
symlink queda vivo en el filesystem del host para que sesiones futuras no
tropiecen con lo mismo; no requiere ninguna acción en git.

### Experimento: deshabilitar el fast-path `KV_max`

Se gateó la condición de `ggml/src/ggml-cuda/fattn-common.cuh:1455`
(`flash_attn_mask_to_KV_max`) detrás de una macro diagnóstica opt-in, en el
mismo estilo que `TURBO_DIAG_FA_PAGED`/`TURBO_DIAG_PAGE_ROWS` ya existentes:

```cpp
#if defined(TURBO_DIAG_DISABLE_KV_MAX)
if (false && mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
#else
if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
#endif
```

Sin la macro definida (comportamiento default, sin cambios), el fast-path
sigue activo tal como estaba. Este bloque queda en el árbol de trabajo
(no commiteado) tal como los otros `TURBO_DIAG_*`.

### Resultados (build `qwen2.5-coder-1.5b-bf16.gguf`, `-c 512 --chunks 10 -ngl 99 --triattention-page-budget 16`)

```
LLAMA_NO_PAGING=1, -fa on:
  PPL = 12.2762   (correcto — chunks 10; consistente con ~9.7 reportado para chunks 5)

Paged, -fa off (gather path):
  PPL = 12.2763   (correcto, sin regresión frente a LLAMA_NO_PAGING)

Paged, -fa on (buggy, ANTES del cambio — KV_max fast-path activo, default):
  PPL = 38849.7815
  per-chunk: 23181.78,39044.76,39236.25,48435.47,39480.73,37305.15,32432.57,35913.84,35592.74,38849.78

Paged, -fa on (KV_max fast-path DESHABILITADO vía TURBO_DIAG_DISABLE_KV_MAX):
  PPL = 38849.7815
  per-chunk: 23181.78,39044.76,39236.25,48435.47,39480.73,37305.15,32432.57,35913.84,35592.74,38849.78
```

Los números son **idénticos byte a byte**, per-chunk y en el estimado final,
con y sin el fast-path `KV_max`. El diagnóstico `TURBO_DIAG_FA_PAGED` (activo
en ambas corridas) confirma además que `Q->ne[3]=4` (4 streams paralelos, como
predecía el plan — `--chunks 10` sí produce `Q->ne[3] > 1`), es decir el
fast-path **sí se estaba activando** en la corrida "buggy" (no es que la
condición nunca se cumpliera) y aun así deshabilitarlo no cambió nada.

### Conclusión

**`flash_attn_mask_to_KV_max` queda descartada como causa del bug**, tanto en
su forma "bug único" como en la forma "segundo bug compuesto sobre el bug
base". El fast-path se ejecuta (`Q->ne[3]=4 > 1`) en la corrida rota, y
deshabilitarlo por completo no altera ni un solo valor de PPL por chunk. La
corrupción sigue localizada en el resto del compute path paginado del kernel
(lectura/cómputo de V vía page table dentro del kernel FA en sí, no en el
paso previo de determinar el límite KV a iterar).

Del lead original de la ronda anterior ("manejo de `gqa_ratio`/stride de la
mask en el branch paginado de `build_attn`"), la parte de `gqa_ratio` ya fue
descartada por trace estático (ver nota arriba) — el candidato más
prometedor que queda sin verificar es específicamente el **stride de la
mask** en ese mismo branch paginado (`src/llama-graph.cpp` ~líneas
2130-2183), comparado explícitamente contra el path `-fa off` que sí
funciona.

### Estado del árbol de trabajo al cerrar esta ronda

- `ggml/src/ggml-cuda/fattn-common.cuh`: bloque `TURBO_DIAG_DISABLE_KV_MAX`
  agregado (opt-in, sin macro definida por defecto — comportamiento idéntico
  al de antes de esta ronda cuando se compila sin flags extra).
- Build actual (`build/`) compilado **sin** `TURBO_DIAG_DISABLE_KV_MAX`
  definida (`CMAKE_CUDA_FLAGS = -DTURBO_DIAG_FA_PAGED -DTURBO_DIAG_PAGE_ROWS`,
  igual que al inicio de la sesión) — el fast-path `KV_max` está activo por
  defecto, como en producción.
- Nada comiteado. `git status` debería mostrar solo el cambio en
  `fattn-common.cuh` (más lo que ya estuviera sucio de antes, ej.
  `scripts/multiswarm.py`, no tocado en esta ronda).
- Symlink de host `~/GitHub/llama-cpp-turboquant → ~/GitHub/mallana`: no es
  parte del repo git, pero es necesario para que `cmake --build build`
  funcione sin reconfigurar desde cero. Ver nota de "repo detach" arriba.

---

## Update — 2026-07-02 (ronda 2): mask-stride en el branch paginado también descartada (con prueba)

**Estado:** Bloqueado — PPL sigue en 38849.7815 con `-fa on` + paging, `--chunks 10`
(sin cambios respecto a la ronda anterior). Se descarta empíricamente el único
lead restante del plan original ("manejo de stride de la mask en el branch
paginado de `build_attn`"). Bug sigue acotado a otra parte del compute path
paginado del kernel FA, aún no identificada.

### Contexto de esta ronda

Al retomar la sesión, el árbol de trabajo ya contenía instrumentación
`TURBO_DIAG_MASK_PAGED` parcialmente escrita en `ggml/src/ggml-cuda/softmax.cu`
(dump del lado `-fa off`, path `ggml_cuda_op_soft_max`) y en
`ggml/src/ggml-cuda/fattn-common.cuh` (dump del lado `-fa on`, path
`launch_fattn`, justo antes del kernel launch), y el build activo (`build/`)
ya tenía `CMAKE_CUDA_FLAGS=-DTURBO_DIAG_MASK_PAGED`. Aparentemente quedó a
medio terminar de un intento anterior (no documentado en este handoff). Se
verificó el código, se refinó el filtro de captura (ver abajo) y se completó
el experimento.

### Hallazgo estructural: `build_attn` usa el MISMO tensor de mask para ambos paths

Antes de instrumentar, se confirmó por lectura del grafo
(`src/llama-graph.cpp:2119, 2201` y `src/llama-graph.h:271-312`) que el
branch paginado de `build_attn` (líneas ~2130-2183) construye la vista 4D de
`v` (pool + page table) de forma condicional a `cparams.flash_attn`, pero
**`kq_mask` se obtiene una sola vez, antes de esa rama, vía
`inp->get_kq_mask()`, y se pasa sin modificación a `build_attn_mha`** tanto
en el branch FA (`ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, ...)`, línea
1836) como en el branch softmax (`ggml_soft_max_ext(ctx0, kq, kq_mask, ...)`,
línea 1910). El único tratamiento diferencial es el cast a F16
(`self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(..., F16) : ...`,
`llama-graph.cpp:2055`). Es decir: **no existe, en el código actual, una
rama de "stride de mask específica del path paginado"** — la hipótesis del
plan asumía una bifurcación de stride que no está presente; la única
diferencia estructural posible es la conversión F32→F16.

### Experimento: dump de `ne[]`/`nb[]`/valores reales del mask en ambos paths

Se refinó el filtro de captura en ambas sondas (`s_mask_diag`) para que
solo dispare cuando `mask->ne[3] > 1` (multi-stream, el caso donde ocurre el
bug — con `--chunks 1` no hay streams paralelos), en vez de capturar
indiscriminadamente las primeras 3 invocaciones (que con `--chunks 10`
correspondían al primer chunk single-stream, no representativo). Recompilado
`llama-perplexity` y corridos ambos paths para el mismo modelo/prompt:

```
-fa off (gather, funciona, PPL=12.2763):
  [MASK_DIAG_SOFTMAX] mask.ne=[256,128,1,4] mask.nb=[4,1024,131072,131072] type=f32
  [MASK_DIAG_SOFTMAX] mask[0..7]: 0 -inf -inf -inf -inf -inf -inf -inf

-fa on (paged FA, roto, PPL=38849.7815):
  [MASK_DIAG_FA] mask.ne=[256,128,1,4] mask.nb=[2,512,65536,65536] type=f16
  [MASK_DIAG_FA] mask[0..7]: 0 -inf -inf -inf -inf -inf -inf -inf
```

**Análisis:** `ne[]` es idéntico byte a byte entre ambos paths:
`[256,128,1,4]` = `[n_kv, n_batch/n_stream, 1, n_stream]` en los dos casos.
Los strides (`nb[]`) son consistentes con un tensor contiguo estándar,
escalado exactamente 2× por el tamaño de elemento (F32→F16):
`nb01_f32=4→nb01_f16=2`, `nb1_f32=1024→nb1_f16=512`,
`nb2/nb3_f32=131072→nb2/nb3_f16=65536` — es decir `nb[i]_f16 = nb[i]_f32 / 2`
en todas las dimensiones, exactamente lo esperado de un `ggml_cast` a F16 sin
ninguna otra transformación de layout. Los primeros 8 valores (posición
causal-diagonal, columna 0 = atendible, columnas 1-7 = enmascaradas) son
idénticos en ambos paths.

Con esto, el `ne[]`/`nb[]` del tensor de mask y sus primeros 8 valores
(offset 0, `sequence=0, ic0=0`) son idénticos entre ambos paths. Esto por sí
solo **no** prueba todavía que el direccionamiento del kernel CUDA
(`maskh = mask + nb33*(sequence % ne33) + nb31*ic0`, `fattn-vec.cuh:131`)
lea la misma fila lógica que `ggml_soft_max_ext` para `sequence > 0` o
`ic0 > 0` — solo se comparó el offset base. Instrumentación adicional
(`[MASK_DIAG_FA]` / `[MASK_DIAG_SOFTMAX]` con lectura en coordenadas
`(sequence, query, key)` emparejadas vía la fórmula de indexado de cada
kernel, y `[GRAPH_DIAG_MASK]` en `build_attn_mha` confirmando que ambos
paths reciben el mismo objeto `kq_mask` en tiempo de construcción del grafo)
se añadió para cerrar esta brecha; los valores dumped en esas coordenadas
para `sequence > 0` / `ic0 > 0` aún deben ejecutarse y compararse antes de
descartar el lead.

### Conclusión (provisional)

**El lead de "stride de la mask en el branch paginado" NO puede darse aún
por descartado con evidencia empírica completa.** Lo verificado hasta ahora
(mismo `ne[]`, mismo layout salvo escala de tipo F32→F16, mismos valores en
el offset base) solo descarta diferencias groseras de forma/stride
contiguo y corrupción de valores en la fila base — no descarta una
desalineación específica de paging en filas/tiles distintos de cero. Con
`gqa_ratio` (ronda 1) y `flash_attn_mask_to_KV_max` (ronda 1) descartados
con prueba, y mask-stride pendiente de la verificación en coordenadas
emparejadas descrita arriba, el candidato más probable sigue siendo el
cómputo interno del kernel paginado (lectura/dequantización de V vía
`v_paged_ptr` + acumulación KQV dentro de `fattn-vec.cuh`), pero mask-stride
permanece abierto hasta correr la instrumentación nueva.

Pista concreta sin verificar aún, visible en `fattn-vec.cuh:122-128`:

```cpp
const int sequence = blockIdx.z / ne02;
...
V += (v_ptable ? (int64_t)0 : nb23*sequence) + nb22*(head / gqa_ratio);
```

Cuando `v_ptable` está activo, el offset `nb23*sequence` se omite
deliberadamente (la página física ya codifica la secuencia vía
`v_ptable[seq*n0+lpage]` dentro de `v_paged_ptr`). Esto fue revisado
estáticamente y es consistente en el papel, pero **nunca se verificó con
valores reales que, para `sequence` > 0 (streams 1..3 del batch de 4
streams paralelos de esta corrida), el page table efectivamente resuelva
filas físicas distintas y correctas** — el diagnóstico `TURBO_DIAG_PAGE_ROWS`
de la ronda anterior sólo confirmó `pblock`/`phys_row` para la página lógica
0 sin especificar para qué `seq`. Dado que el bug persiste incluso con
`--chunks 1` (`n_seqs=1`, sin streams paralelos — ver ronda 1), esta pista
por sí sola no explica el bug base, pero no se ha verificado si hay una
segunda ruta de corrupción independiente aquí. Recomendado para la próxima
ronda: instrumentar `v_paged_ptr` (ya tiene el gancho `TURBO_DIAG_PAGE_ROWS`)
para imprimir `seq` explícitamente y correr con `--chunks 1` comparando
valores reales de V leídos por el kernel contra los mismos valores leídos
por el path `-fa off` (gather), byte a byte, en vez de sólo direcciones
calculadas — la ronda 1 ya verificó que los *bytes en el pool* son reales y
no basura, pero no verificó que el kernel FA en sí los combine/pondere
correctamente en la acumulación softmax-QKV interna.

### Estado del árbol de trabajo al cerrar esta ronda

- `ggml/src/ggml-cuda/softmax.cu` y `ggml/src/ggml-cuda/fattn-common.cuh`:
  instrumentación `TURBO_DIAG_MASK_PAGED` completa en ambos archivos
  (opt-in, sin macro definida por defecto). Se ajustó el filtro de captura
  en ambos (`s_mask_diag`) para disparar sólo cuando `mask->ne[3] > 1`
  (caso multi-stream, representativo del escenario roto) en vez de las
  primeras 3 invocaciones sin filtrar.
- Build actual (`build/`) compilado **con** `-DTURBO_DIAG_MASK_PAGED` (además
  de `TURBO_DIAG_FA_PAGED` y `TURBO_DIAG_PAGE_ROWS` de rondas anteriores) —
  no representa el build de producción; recompilar sin esas macros antes de
  cualquier benchmark que no sea diagnóstico.
- `roadmap.md` (`research/roadmap.md`): sin cambios — Milestone 003 ya
  refleja correctamente el estado real (`✅ IMPLEMENTADO — pendiente
  validación numérica GPU`), no se marca como completado.
- Nada comiteado. `git status` debería mostrar los cambios en
  `fattn-common.cuh`, `softmax.cu`, este archivo, y lo que ya estuviera sucio
  de antes (`scripts/multiswarm.py`, no tocado en esta ronda).
- Symlink de host `~/GitHub/llama-cpp-turboquant → ~/GitHub/mallana` sigue
  vivo y necesario (ver nota de ronda anterior).

---

## Update — 2026-07-02 (ronda 3)

### Instrumentación de coordenadas emparejadas

La ronda 2 sólo comparó `ne[]`/`nb[]`/tipo y los primeros 8 valores desde el
offset 0 del tensor de mask — insuficiente para probar que el *indexado
lógico* del kernel (que depende de `sequence % ne33` y del tile de query
`ic0`) lee la misma fila en ambos paths para `sequence > 0` o `ic0 > 0`.
Se reescribió la instrumentación en ambos archivos para calcular el
`byte_offset` con la fórmula real de cada kernel y leer el valor en 4
coordenadas `(sequence, query, key)` emparejadas: `(0,0,0)`, `(0,16,16)`,
`(1,16,16)`, `(1,32,32)`:

- **FA (`fattn-common.cuh`):** `mask + nb33*(sequence % ne33) + j*nb31 + ic0*nb30`
- **Softmax (`softmax.cu`):** `src1_d + (s % ne13)*nb13 + (h % ne12)*nb12 + j*nb11 + ic0*nb10` (h=0)

También se añadió `[GRAPH_DIAG_MASK]` en `build_attn_mha`
(`src/llama-graph.cpp`), que imprime el puntero/`ne[]`/`nb[]`/tipo del
`kq_mask` recibido en cada capa, en tiempo de construcción del grafo.

### Resultado

Build con `-DTURBO_DIAG_MASK_PAGED` (CXX y CUDA), corrida con
`-c 512 --chunks 5 -ngl 99 --triattention-page-budget 16`:

- `-fa off`: `[MASK_DIAG_SOFTMAX] mask.ne=[256,128,1,4] mask.nb=[4,1024,131072,131072] type=f32`.
  Las 4 coordenadas dan `val=0` con offsets `0, 16448, 147520, 163968`.
- `-fa on`: `[MASK_DIAG_FA] mask.ne=[256,128,1,4] mask.nb=[2,512,65536,65536] type=f16`.
  Las 4 coordenadas dan `val=0` con offsets `0, 8224, 73760, 81984` — exactamente
  la mitad de los offsets del path F32 (consistente con el cast a F16), y
  **el mismo valor lógico (0 = posición atendible)** en cada una de las 4
  coordenadas emparejadas, incluyendo `sequence=1` con `ic0=16` y `ic0=32`.

**Conclusión: el lead de "stride/indexado de la mask en el branch paginado"
queda descartado de forma concluyente.** No sólo la forma/stride/valores en
offset 0 coinciden (ronda 2): ahora se probó, con la fórmula de indexado
real de cada kernel, que ambos paths leen el mismo valor lógico en
coordenadas `(sequence, query, key)` no triviales, incluyendo `sequence=1`.
`[GRAPH_DIAG_MASK]` confirma además que ambos paths comparten el mismo
puntero `kq_mask` en tiempo de construcción del grafo (mismo `il`, mismo
`kq_mask_ptr` en cada capa) — no hay una copia o transformación adicional
introducida sólo para el path paginado antes del cast a F16.

### El fix de `find_flash_attn_ext` (DFS) no resuelve el bug

El plan de esta ronda asumía que el bug era el walk-back loop frágil en
`build_attn` (`src/llama-graph.cpp`) que no atravesaba `GGML_OP_CONT` y caía
al fallback de llamar `ggml_flash_attn_ext_set_page_table` sobre el tensor
equivocado. Ese fix (DFS acotada + `GGML_ABORT` si no encuentra el tensor)
**ya estaba commiteado** (`e0e3486dc — fix: replace fragile FA-tensor
while-loop with recursive DFS + hard abort`) antes de esta ronda. Se
verificó que sigue activo (`find_flash_attn_ext` en
`src/llama-graph.cpp:2069`, usado en `build_attn` línea 2203) y que el build
no aborta (si el DFS no encontrara el tensor `GGML_OP_FLASH_ATTN_EXT`, el
`GGML_ABORT` haría fallar la corrida inmediatamente — no fue el caso).

A pesar de que este fix está activo y funcionando (encuentra y setea la
page table sobre el tensor `GGML_FLASH_ATTN_EXT` correcto), **la corrida de
validación de esta ronda sigue dando `PPL = 35125.2709` con `-fa on`**,
idéntico a las rondas anteriores. Esto confirma que el walk-back del tensor
FA **no era la causa raíz** — sólo un bug real pero independiente que ya fue
corregido sin efecto sobre el síntoma principal.

### Estado y próxima pista

Con `gqa_ratio` (ronda 1), `flash_attn_mask_to_KV_max` (ronda 1),
mask-stride/indexado (rondas 2-3, ahora con prueba concluyente) y el walk
del tensor FA (esta ronda, ya commiteado sin efecto) descartados, el bug
**no está en ninguno de los inputs/parámetros que llegan al kernel FA ni en
el tensor sobre el que se setea la page table** — el fallback bug real
existía pero no es la causa de PPL=35125. Queda acotado al cómputo interno
del kernel paginado en sí: lectura/dequantización de V vía `v_paged_ptr` y
acumulación KQV dentro de `fattn-vec.cuh`, en código que sólo se ejecuta
cuando `v_ptable` está activo. La pista concreta sin verificar de la ronda
2 (`V += (v_ptable ? (int64_t)0 : nb23*sequence) + ...` en
`fattn-vec.cuh:122-128`, y la resolución de `v_paged_ptr[seq]` para
`sequence > 0`) sigue siendo la recomendación principal para la próxima
ronda — instrumentar `TURBO_DIAG_PAGE_ROWS` para imprimir `seq` explícito y
comparar valores de V leídos por el kernel FA paginado, byte a byte, contra
los del path gather (`-fa off`) para las mismas coordenadas `(seq, lpage,
head, dim)`.

---

## Update — 2026-07-02 (ronda 4): root cause del kernel-dispatch encontrado y verificado; nuevo bug de memoria expuesto (NO resuelto todavía)

**Estado:** Root cause de la corrupción de PPL **identificado y verificado
matemáticamente** (no sólo por lectura estática): el kernel FA paginado (VEC,
`fattn-vec.cuh`) **nunca se estaba ejecutando** para este workload — el
dispatcher elegía el kernel MMA_F16 (`fattn-mma-f16.cuh`), que ignora
por completo la page table. Se aplicó un fix de ruteo (`fattn.cu`) que
fuerza el kernel correcto, y **se validó que corrige el PPL** bajo
`CUDA_LAUNCH_BLOCKING=1` (PPL=12.2909, prácticamente idéntico al baseline).
Pero el mismo cambio expone un **segundo bug, independiente y aún sin
arreglar**: un acceso de memoria fuera de rango real (confirmado con
`compute-sanitizer`) dentro de `flash_attn_ext_vec` cuando se combina con
paginación, que crashea el proceso en ejecución normal (sin
`CUDA_LAUNCH_BLOCKING=1`). **El bug de PPL no puede darse por resuelto
todavía** — ver criterios de aceptación al final de esta entrada.

### Parte 1: se corrigió el experimento de coordenadas de mask (tarea 1 del plan)

Se reemplazaron los `test_coords` causal-diagonales (ronda 3, siempre
`val=0`, insuficientes según la crítica) por 8 coordenadas — una posición
enmascarada (`key > query`, se espera `-inf`) y una posición permitida no
trivial (`key < query`, se espera `0`) por cada una de las 4 secuencias —
en `ggml/src/ggml-cuda/softmax.cu` y `ggml/src/ggml-cuda/fattn-common.cuh`
(mismas coordenadas en ambos, para diff 1:1). Corrida con
`-c 512 --chunks 10 -ngl 99 --triattention-page-budget 16` (chunks=10 tal
como exige la crítica, no chunks=5):

```
-fa off (gather... ver Parte 2, en realidad NO es el path paginado):
  (s=0..3, j=16, ic0=32): val=-inf  (masked, correcto)
  (s=0..3, j=32, ic0=16): val=0     (allowed, correcto)
Final estimate: PPL = 12.2763
```

Todas las 8 coordenadas, para las 4 secuencias, dieron el valor lógico
esperado. **El lead de "stride/indexado de la mask" queda descartado de
forma aún más concluyente que en la ronda 3** — pero ver Parte 2: esta
comparación en particular resultó ser menos informativa de lo que se
pensaba, porque el lado "-fa off" nunca ejerce el código de paginación.

### Parte 2: hallazgo estructural — "-fa off" (Phase 1 "gather") nunca ejecutó el código de paging, en NINGUNA ronda anterior

Al instrumentar la escritura de la page table (`TURBO_DIAG_V_READS` en
`llama_kv_cache::set_input_v_page_table`, `src/llama-kv-cache.cpp`) para
verificar filas 1-3 (pendiente de rondas anteriores), el diagnóstico
**nunca se disparó en la corrida `-fa off`** — la función ni siquiera se
llamó. Investigando la causa (lectura del código, no instrumentación):

```
llama-model.cpp:8201,8219:  attn_v_trans = !cparams.flash_attn
llama-kv-cache.cpp:380:     pg_enabled  = !v_trans && !getenv("LLAMA_NO_PAGING")
llama-kv-cache.cpp:2834:    is_paged()  = pg_enabled
llama-graph.cpp:2061-2063:  self_v_page_table sólo se construye si is_paged()
```

`cparams.flash_attn` es el mismo objeto/valor fijado por el flag `-fa` de
línea de comandos para **todo el proceso** (se fija una vez al construir el
KV-cache y no cambia durante la corrida salvo por un hook de auto-disable
no relevante aquí). Con `-fa off`: `v_trans=true` ⟹ `pg_enabled=false`
⟹ `is_paged()=false` ⟹ `self_v_page_table` nunca se construye ⟹
`build_attn` toma la rama `v = mctx_cur->get_v(ctx0, il)` (línea 2198-2199,
completamente ajena al pool/page-table) — el kernel gather
(`ggml_gather_paged_v`, `ggml/src/ggml-cuda/paged-gather.cu`) **jamás se
invoca**.

**Consecuencia:** todas las comparaciones "`-fa on` (roto) vs `-fa off`
(bueno)" hechas en las rondas 1-3 de este handoff — incluida la
verificación de mask de esta misma ronda arriba — comparaban el path
roto contra un path **completamente distinto y ajeno al bug** (atención
clásica no paginada), no contra el path de paginación real. Esto no
invalida las conclusiones puntuales (la mask, en efecto, no depende de
`v_trans`/paginación, así que esa comparación seguía siendo válida por
casualidad), pero explica por qué 3 rondas de comparación cuidadosa
"-fa on vs -fa off" nunca acorralaron el bug: **nunca estaban comparando
dos ejecuciones del mismo código**.

Además, esto implica que `ggml_gather_paged_v` / `paged-gather.cu` (el
fallback "Phase 1" en `src/llama-graph.cpp:2184-2197`, rama
`else` de `if (cparams.flash_attn)`) es **código muerto bajo el wiring
actual**: `is_paged()`/`pg_enabled` sólo es `true` si `cparams.flash_attn`
era `true` al construir el KV-cache, y esa misma variable es la que decide,
dentro de `build_attn`, si se toma la rama Phase 2 (nativa) o la rama
`else` Phase 1 (gather) — ambas lecturas del mismo valor, nunca
discrepan. La rama Phase 1 nunca se alcanza con ninguna combinación de
flags de CLI actual. (Se dejó instrumentado con `TURBO_DIAG_V_READS` de
todos modos, gateado y sin efecto, para si en el futuro se rehabilita esa
rama.)

La comparación válida — mismo kernel, mismo `cparams.flash_attn=true`,
única diferencia es paginación sí/no — es `-fa on` con
`--triattention-page-budget` vs `LLAMA_NO_PAGING=1 -fa on`. Ambas usan
`v_trans=false` y (si hay page table) el mismo kernel FA; la única
diferencia es `v_ptable == nullptr` vs no. Se usó esta comparación en la
Parte 3.

### Parte 3: root cause encontrado — el dispatcher de kernels FA ignora la page table para este workload

Se instrumentó `fattn-vec.cuh` (`TURBO_DIAG_V_READS`, dump de bytes V
crudos leídos vía `v_paged_ptr` en `seq=1,2,3, k_abs=0`) esperando ver
output en la corrida `-fa on` paginada — **no apareció ningún
`[V_READ_FA]`**, pese a que el string sí está en el binario
(`strings libggml-cuda.so | grep V_READ_FA` lo confirma). Se investigó por
qué el kernel `flash_attn_ext_vec` (`fattn-vec.cuh`) no se ejecuta:

`ggml_cuda_flash_attn_ext` (`ggml/src/ggml-cuda/fattn.cu:571`) llama a
`ggml_cuda_get_best_fattn_kernel`, que — para esta GPU (RTX 2050, cc=8.6,
`turing_mma_available(cc)=true`) y este workload (`Q->ne[1]=128`, batch de
prefill, no decode de un token) — cae en la rama de
`turing_mma_available` (línea 463) y, como `Q->ne[1] != 1` y no cumple
ninguna de las condiciones de `can_use_vector_kernel` con `Q->ne[1]<=2`,
**retorna `BEST_FATTN_KERNEL_MMA_F16`** (línea 491), nunca `VEC`.

Se verificó `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (el kernel MMA
realmente ejecutado): recibe `v_ptable` como parámetro pero lo descarta
explícitamente —

```cpp
// fattn-mma-f16.cuh:1886-1887
GGML_UNUSED(v_ptable);
GGML_UNUSED(v_ptable_ne0);
```

— y direcciona V de forma **no paginada**, incondicionalmente:

```cpp
// fattn-mma-f16.cuh:1970, 2016
const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
```

Esto es exactamente el síntoma descrito como "hipótesis original" al
principio de este handoff (2026-06-30): un kernel que lee V sin traducir
por la page table, aplicando `nb23*sequence` sobre un pool que — cuando
está paginado — no tiene stride por secuencia válido (ver el comentario en
`fattn-vec.cuh:127`: *"the pool has no per-sequence stride"*), leyendo
bytes de la posición física equivocada (probablemente el bloque sentinela
cero, u otra secuencia) para casi todo `K->ne[1]=256` × 4 secuencias ×
28 capas → acumulación de error catastrófica → PPL≈35000-38000.
`fattn-tile.cuh` (el otro kernel candidato) **sí** implementa
`v_paged_ptr` correctamente (confirmado leyendo el código: líneas 486,
546, 816, 931, 1057 etc.); `fattn-wmma-f16.cuh` no menciona `v_ptable` en
absoluto (tampoco soportado).

### Fix aplicado: forzar un kernel compatible con paginación cuando hay page table

`ggml/src/ggml-cuda/fattn.cu`, dentro de `ggml_cuda_get_best_fattn_kernel`,
justo después de calcular `can_use_vector_kernel` y antes de cualquier
rama de tensor-cores:

```cpp
if (dst->src[5] != nullptr) {   // page table adjunta (ggml_flash_attn_ext_set_page_table)
    if (can_use_vector_kernel) {
        return BEST_FATTN_KERNEL_VEC;
    }
    return BEST_FATTN_KERNEL_TILE;
}
```

(`dst->src[5]` es exactamente el slot que usa
`ggml_flash_attn_ext_set_page_table`, ver `ggml/src/ggml.c:5402-5409`.)
Para este modelo/config, `can_use_vector_kernel` es `true`
(`Q->ne[0]=128<=256`, `128%64==0`, `K->ne[1]=256 % FATTN_KQ_STRIDE(256)==0`),
así que se selecciona `BEST_FATTN_KERNEL_VEC` (`fattn-vec.cuh`), que sí
implementa `v_paged_ptr`.

### Validación: PPL correcto bajo `CUDA_LAUNCH_BLOCKING=1`

Build limpio (sin macros `TURBO_DIAG_*`, `CMAKE_CUDA_FLAGS`/`CMAKE_CXX_FLAGS`
vacíos, `GGML_CUDA_GRAPHS=ON` por defecto — se probó también con
`GGML_CUDA_GRAPHS=OFF`, ver más abajo, mismo resultado):

```
CUDA_LAUNCH_BLOCKING=1, -fa on, paged, --chunks 10, --triattention-page-budget 16:
  PPL = 12.2909 +/- 0.67798
  per-chunk: 6.8668, 9.4321, 9.1124, 9.5137, 9.7254, 10.1348, 10.4772, 11.2055, 11.7714, 12.2909

Comparar contra baseline (misma corrida, misma ronda):
  LLAMA_NO_PAGING=1 -fa on, --chunks 10:  PPL = 12.2762
  -fa off (legacy, no paginado, ver Parte 2), --chunks 10: PPL = 12.2763
```

**12.2909 vs 12.2762/12.2763 — dentro del rango esperado (~9.7-12.3), a
menos de 0.15 de diferencia relativa.** Esto confirma que, una vez que el
kernel correcto (VEC) realmente ejecuta la lógica de `v_paged_ptr`,
`nb21`/`nb22`, `gqa_ratio` y el cast de mask a F16 — toda esa lógica
matemática, revisada y re-revisada en las rondas 1-3, **era correcta**. El
bug nunca estuvo en el direccionamiento paginado en sí; estuvo en que ese
código de direccionamiento correcto nunca se ejecutaba.

Diagnóstico adicional (`TURBO_DIAG_V_READS` en una build de instrumentación
aparte) confirmó, para `seq=1,2,3`, que `[V_READ_FA]` resuelve
`pblock=5,9,13` para `lpage=0` — coincide exactamente con lo que
`[V_PT_WRITE]` había escrito (`row 1 (strm=1): 5 6 7 8...`, `row 2
(strm=2): 9 10 11 12...`, `row 3 (strm=3): 13 14 15 16...`), cerrando
definitivamente la verificación de filas 1-3 del page table que quedaba
pendiente desde la ronda 3 (antes sólo se había verificado la fila/página
0).

### PERO: el mismo fix crashea en ejecución normal (sin `CUDA_LAUNCH_BLOCKING=1`) — bug nuevo, sin resolver

Sin `CUDA_LAUNCH_BLOCKING=1` (el modo de ejecución normal/por defecto), la
misma build crashea **de forma 100% reproducible** (probado 3+ veces,
`--chunks 2` y `--chunks 10`, con y sin `--no-warmup`, con
`GGML_CUDA_GRAPHS=ON` y `=OFF`):

```
/home/ignatus/GitHub/mallana/ggml/src/ggml-cuda/ggml-cuda.cu:100: CUDA error
CUDA error: an internal operation failed
  current device: 0, in function ggml_cuda_op_mul_mat_cublas at .../ggml-cuda.cu:1334
  cublasSetStream_v2(ctx.cublas_handle(id), stream)
```

Este error en `cublasSetStream` es un **síntoma downstream** (el contexto
CUDA queda corrupto por un error asíncrono anterior no capturado hasta la
siguiente llamada a la API). Se usó `compute-sanitizer --tool memcheck`
para encontrar el error real, y confirma un acceso de memoria
genuinamente fuera de rango, no un falso positivo:

```
========= Invalid __global__ read of size 16 bytes
=========     at void flash_attn_ext_vec<(int)128, (int)2, (ggml_type)1, (ggml_type)1, (bool)0>(...)+0x63b0
=========     by thread (0,3,0) in block (1,0,0)
=========     Access to 0x8f76b8700000 is out of bounds
=========     and is 17.324.374.491.137 bytes after the nearest allocation ...
```

Un offset de ~17 billones de bytes más allá de la asignación válida no es
un simple off-by-one — sugiere un puntero o índice completamente
corrupto/no inicializado dentro del kernel (por ejemplo, un `pblock`
basura si algo lee la page table fuera de sus límites, o una dirección
mal calculada en otra parte del kernel VEC), no una desalineación menor
del cálculo de `v_paged_ptr` ya validado matemáticamente arriba. Se
descartaron dos hipótesis simples con evidencia:
- **No es específico de CUDA graphs**: se recompiló con
  `-DGGML_CUDA_GRAPHS=OFF` (deshabilita `GGML_CUDA_USE_GRAPHS` a nivel de
  compilación) y el crash persiste idéntico.
- **No es específico del warmup**: se corrió con `--no-warmup` y el crash
  persiste idéntico (ocurre igual en el primer chunk real).

No se identificó la causa exacta de este segundo bug en el tiempo
disponible de esta ronda — requiere profiling adicional con
`compute-sanitizer --tool memcheck --show-backtrace=host` correlacionado
con líneas fuente (compilar con `-lineinfo`) dentro de
`flash_attn_ext_vec` (revisar en particular `dequantize_V`/`vec_dot_KQ`
para la combinatoria `D=128, ncols=2, K=V=F16`, y el puntero de mask
`maskh` para `ncols=2` — el diagnóstico de mask de la Parte 1 sólo
verificó offsets con `j` fijo por coordenada, no el acceso real que hace
el kernel iterando `ic0+0..ncols-1` en paralelo).

### Conclusión de esta ronda

**Root cause del bug original de PPL: conclusivamente identificado y
verificado matemáticamente** — el dispatcher de kernels FA
(`ggml_cuda_get_best_fattn_kernel`, `fattn.cu`) selecciona el kernel
MMA_F16 para este workload de prefill batched, y ese kernel ignora la
page table por completo, leyendo V sin traducir. El fix de ruteo
(forzar VEC/TILE cuando hay page table) corrige el PPL matemáticamente
(validado bajo `CUDA_LAUNCH_BLOCKING=1`: 12.29 vs baseline 12.28), pero
**expone un segundo bug, real y reproducible, de acceso a memoria fuera
de rango dentro de `flash_attn_ext_vec` bajo paginación**, que no existía
como problema observable antes porque ese kernel nunca se ejecutaba con
`v_ptable` activo en este workload. **El bug de PPL NO puede darse por
resuelto todavía** — no se cumple el criterio de aceptación del plan
("validar que el PPL baja al rango baseline... antes de reclamar que está
arreglado") bajo ejecución normal, sólo bajo `CUDA_LAUNCH_BLOCKING=1`, que
no es representativo de uso real.

### Estado del árbol de trabajo al cerrar esta ronda

- `ggml/src/ggml-cuda/fattn.cu`: **fix real** — fuerza `BEST_FATTN_KERNEL_VEC`
  o `_TILE` cuando `dst->src[5]` (page table) está presente. Necesario pero
  no suficiente (ver bug nuevo arriba). Recomendado dejarlo — no regresiona
  ningún caso sin paginación (sólo cambia el ruteo cuando `v_ptable` ya
  estaba adjunta, que antes producía PPL basura de todos modos).
- `ggml/src/ggml-cuda/fattn-vec.cuh`: instrumentación `TURBO_DIAG_V_READS`
  (opt-in) que dumpea `pblock`/bytes crudos de V para `seq=1,2,3, k_abs=0`,
  funciona tanto con `v_ptable` presente como ausente (para diffear
  paginado vs `LLAMA_NO_PAGING=1`, la comparación válida — ver Parte 2).
- `ggml/src/ggml-cuda/fattn-common.cuh`, `ggml/src/ggml-cuda/softmax.cu`:
  `test_coords` ampliado de 4 a 8 coordenadas (masked + allowed
  off-diagonal × 4 secuencias) bajo `TURBO_DIAG_MASK_PAGED` (opt-in, sin
  cambio de comportamiento por defecto).
- `src/llama-kv-cache.cpp`: instrumentación `TURBO_DIAG_V_READS` (opt-in)
  en `set_input_v_page_table`, dumpea el mapeo fila→stream→pblock escrito
  para las primeras 2 invocaciones.
- `ggml/src/ggml-cuda/paged-gather.cu`: instrumentación `TURBO_DIAG_V_READS`
  (opt-in) agregada por completitud del plan, pero **confirmada como código
  muerto bajo el wiring actual** (ver Parte 2) — nunca se disparará con
  ninguna combinación de flags de CLI hasta que se repare/rehabilite la
  rama Phase 1 de `build_attn`.
- Build activo (`build/`) en estado limpio: `CMAKE_CUDA_FLAGS`/
  `CMAKE_CXX_FLAGS` vacíos (sin macros `TURBO_DIAG_*`), `GGML_CUDA_GRAPHS=ON`
  (default). Con este build, `-fa on --triattention-page-budget 16`
  **crashea** en ejecución normal (ver arriba) — no usar este build para
  demos hasta resolver el bug de memoria del kernel VEC.
- `docs/roadmap.md`: **sin cambios** — el bug de Phase 2 FA sigue sin
  poder marcarse como resuelto (ver criterios de aceptación arriba).
- Nada comiteado, como en rondas anteriores. `git status` debe mostrar
  cambios en: `ggml/src/ggml-cuda/fattn.cu`, `fattn-vec.cuh`,
  `fattn-common.cuh`, `softmax.cu`, `paged-gather.cu`,
  `src/llama-kv-cache.cpp`, y este archivo.

### Próximo paso recomendado

1. Recompilar con `-lineinfo` (o usar `cuda-gdb`) y correr
   `compute-sanitizer --tool memcheck` sobre `flash_attn_ext_vec` para
   ubicar la línea exacta del acceso fuera de rango dentro del kernel VEC
   bajo paginación (candidatos: `dequantize_V`, `vec_dot_KQ`, o el puntero
   de mask `maskh` para el caso `ncols=2`).
2. Una vez arreglado ese segundo bug, re-validar PPL **sin**
   `CUDA_LAUNCH_BLOCKING=1` (condición de ejecución normal) antes de
   marcar el bug de Phase 2 FA como resuelto en `docs/roadmap.md`.
3. Considerar además si `BEST_FATTN_KERNEL_TILE` (la otra opción del fix
   de ruteo, para cuando `can_use_vector_kernel` es falso) tiene el mismo
   problema o no — no se probó en esta ronda porque este workload siempre
   cae en la rama VEC.
