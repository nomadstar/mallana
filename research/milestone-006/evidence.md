# Milestone 006: Evidence

*Estado: IMPLEMENTADO — build OK, pendiente validación GPU/numerical.*

## Cambios por Archivo

| Archivo | Cambio |
|---|---|
| `include/llama.h` | Exposición pública del presupuesto de páginas TriAttention |
| `src/llama-cparams.h` | Campo de parámetros para `triattention_page_budget` |
| `src/llama-context.cpp` | Cableado de parámetros de contexto |
| `common/common.h` | Declaración de opción CLI/común |
| `common/common.cpp` | Default y propagación del presupuesto |
| `common/arg.cpp` | Parsing de `--triattention-page-budget` |
| `src/llama-kv-cache.h` | Declaraciones de estado y API de eviction |
| `src/llama-kv-cache.cpp` | Reserva de bloque 0, `pg_score_and_evict()`, desalojo bajo presupuesto |
| `src/llama-model.cpp` | Integración de configuración en runtime/modelo |

## Resumen de Evidencia

- Total de archivos modificados: 9
- Delta reportado: 244 líneas
- Feature flag: `--triattention-page-budget N`
- Build: pasa

## Pendiente

- [ ] Validación numérica GPU
- [ ] Calibración de scores TriAttention
- [ ] Verificación de hipótesis H6.1 frente al baseline
