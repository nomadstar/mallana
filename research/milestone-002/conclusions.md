# Milestone 002: Conclusions

El Hito 002 demostró que la paginación de memoria para el caché de V (valores) es completamente compatible con la cuantización TurboQuant y no interfiere con el empaquetado de bloques WHT de 128 elementos.

## Principales Aprendizajes y Verdades Demostradas

1. **Alineación de Bloques Estructurada**: Se demostró que un tamaño de página de 32 tokens es óptimo y se alinea perfectamente con los límites del decodificador TurboQuant (`QK_TURBO3 = 32`).
2. **Desacoplamiento Exitoso**: El uso del operador intermedio `GGML_OP_GATHER_PAGED_V` permitió aislar por completo la depuración de la lógica de asignación y de recuento de referencias en C++ de la complejidad de los kernels CUDA.
3. **El cuello de botella de la copia global**: Se confirmó cuantitativamente que el gather intermedio requiere una asignación temporal y una copia extra en memoria global, lo cual limita el throughput a gran escala.

---

## 🚫 QUÉ NO VOLVER A INVESTIGAR

- **No intentar habilitar paginación para el caché de K** sin antes resolver los problemas de direccionamiento lineal en el prefill.
- **No intentar ejecutar paginación de V con la bandera `v_trans=true`** (fallback SDPA).
