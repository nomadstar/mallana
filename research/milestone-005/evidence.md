# Milestone 005: Evidence

*Estado: COMPLETADO.*

## Cambios por Archivo

| Archivo | Cambio |
|---|---|
| `ggml/src/ggml-cuda/set-rows.cu` | Ajuste de aridad de warp shuffle: 3 argumentos -> 4 argumentos con `WARP_SIZE` |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Mismo ajuste de aridad para compatibilidad HIP en Flash Attention VEC |
| `ggml/src/ggml-cuda/vendors/hip.h` | Corrección de macro `__ballot_sync` para wavefront AMD de 64 hilos |

## Resumen de Evidencia

- Total de archivos modificados: 3
- Delta reportado: +22 líneas
- Auditoría adicional: `turbo-quant.cuh`, `fattn-common.cuh`, `fattn-tile.cuh` revisados sin
  cambios requeridos
- Compatibilidad: alineado con `hip-quality-check`

## Resultado

- Objetivo de portabilidad HIP/ROCm: cumplido
- Estado de build: pasa
