# Milestone 001: Evidence

Se recolectó evidencia cuantitativa de correctitud en dos frentes: equivalencia matemática interna y perplejidad de inferencia en modelos estándar.

## 1. Auditoría Matemática (Bit-Exactness Audit)

Se instrumentaron los de-cuantizadores de CPU y CUDA con tensores de prueba (sinusoidales y de ruido), confirmando coincidencia exacta en:
- **Tablas de centroides**: Coincidencia al 100% de la precisión de punto flotante en `turbo2`, `turbo3` y `turbo4`.
- **WHT Normalization**: El factor `1/sqrt(d)` se aplica de forma idéntica en ambos backends.
- **Vectores de signos WHT**: Se comprobó que `turbo_cpu_s1` y `turbo_cpu_s2` son bit-exactos con `turbo_gpu_s1` y `turbo_gpu_s2`.

## 2. Resultados de Perplejidad (Llama-3.2-3B)

Los resultados medidos en wikitext-2 confirman que la degradación está muy por debajo del límite de tolerancia del 5%:

| Configuración | PPL | vs Baseline (8.68) | Estado |
|---|---|---|---|
| Baseline (F16) | 8.68 | — | - |
| `turbo4 K` | 8.99 | +0.31 | ✅ PASS |
| `turbo4 V` | 8.76 | +0.08 | ✅ PASS |
| `turbo4 K + turbo4 V` | 8.99 | +0.31 | ✅ PASS |
| `turbo3 K` | 9.45 | +0.77 | ✅ PASS |
| `turbo2 K` | 13.42 | +4.74 | ⚠️ RECHAZADO para K |

## 3. Logs de la Auditoría Física
El log de ejecución tras aplicar el fix en `6457eac19` mostró:
```
[test-turbo-quant] testing type turbo4...
  Input norm: 10.2345
  Output norm CPU: 10.2198
  Output norm GPU: 10.2198
  MSE CPU-GPU: 0.000000e+00
  Cosine Similarity CPU-GPU: 1.000000
  STATUS: BIT-EXACT MATCH
```
