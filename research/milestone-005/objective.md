# Milestone 005: ROCm Backend Completion

## Objetivo

Cerrar las diferencias de compatibilidad restantes entre CUDA y HIP/ROCm para los kernels de
TurboQuant y Flash Attention paginado, asegurando compilación correcta en GPUs AMD.

## Metas Específicas

1. Corregir la aridad de warp shuffle en los caminos HIP para usar la forma de 4 argumentos con
   `WARP_SIZE`.
2. Alinear `__ballot_sync` con el comportamiento real de wavefront AMD de 64 hilos.
3. Auditar los kernels relacionados (`turbo-quant.cuh`, `fattn-common.cuh`, `fattn-tile.cuh`)
   para confirmar si existen divergencias CUDA/HIP adicionales.

## Criterios de Éxito

1. `set-rows.cu` y `fattn-vec.cuh` compilan bajo HIP sin errores de firma para shuffle.
2. `vendors/hip.h` refleja el ancho de wavefront correcto para `__ballot_sync`.
3. La revisión del resto de kernels no revela fixes adicionales necesarios.

## Archivos a Modificar

- `ggml/src/ggml-cuda/set-rows.cu`
- `ggml/src/ggml-cuda/fattn-vec.cuh`
- `ggml/src/ggml-cuda/vendors/hip.h`
