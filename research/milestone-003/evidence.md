# Milestone 003: Evidence

*Estado: Hito en progreso — código implementado, pendiente de validación numérica.*

## Implementación Completada

### Archivos Modificados

| Archivo | Cambio |
|---|---|
| `ggml/include/ggml.h` | `ggml_flash_attn_ext_set_page_table()` declaration (L2355) |
| `ggml/src/ggml.c` | `ggml_flash_attn_ext_set_page_table()` — sets `a->src[5] = page_table` (L5402) |
| `ggml/src/ggml-cuda/fattn-common.cuh` | `v_paged_ptr()` device helper (L54); `fattn_kernel_t` extended con 3 params page-table (L45–47); `launch_fattn()` lee `dst->src[5]` y pasa datos al kernel (L1242–1248, L1641) |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Firma extendida; V base ajustado (paged: nb23*sequence=0) (L128); V loop skip (L309, L313); todos los accesos `V + k*nb21` reemplazados con `v_paged_ptr()` (L459, L466, L503, L538, L571, L608) |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | Mismo patrón: `v_paged_ptr()` en load helpers (L486, L546, L816); V base sin seq stride (L954) |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | 3 nuevos params, `GGML_UNUSED` (compat ABI) (L1883–1887) |
| `ggml/src/ggml-cuda/fattn-wmma-f16.cu` | Mismo stub ABI (L48–52) |
| `src/llama-graph.cpp` | Rama Phase 2: cuando `cparams.flash_attn && self_v_page_table`, salta gather, crea 4D view del pool, llama `ggml_flash_attn_ext_set_page_table` (L2120–2135, L2168–2170); ruta gather preservada como fallback no-FA |

### Diseño del `v_paged_ptr`

```
Entrada: (V_base, nb21, v_ptable, seq, n0, bs, k_abs)
  lpage  = k_abs / bs
  within = k_abs % bs
  pblock = v_ptable[seq * n0 + lpage]
  retorna V_base + (pblock * bs + within) * nb21
  (si v_ptable==NULL, retorna V_base + k_abs * nb21 — ruta legacy)
```

### Invariante de Strides

La `ggml_view_4d` del pool produce strides que después de `ggml_permute(0, 2, 1, 3)` generan:

- `nb21 = n_embd_v · ts` — stride correcto entre filas físicas del pool
- `nb22 = head_v_eff · ts` — stride correcto entre cabezales dentro de una fila
- `nb23 = 0` — sin stride entre secuencias (pool plano)

Esto es correcto porque el kernel nunca recorre V linealmente por secuencia/posición;
la tabla de páginas resuelve la fila física para cada `(seq, k_abs)`.

## Pendiente

- [ ] Test de paridad numérica: comparar salida de inferencia Phase 2 vs Phase 1 (gather) para secuencias cortas y largas
- [ ] Benchmark de latencia: medir mejora 10–15% esperada en contextos >8K tokens
- [ ] Verificar que `v_ptable` con todos los mapeos identity (pblock == lpage) produce salida bit-exacta con ruta no-paged
