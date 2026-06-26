# Milestone 003: Conclusions

*Estado: Implementado — pendiente de validación numérica.*

## Logros

1. **`v_paged_ptr()` helper unificado** — Implementado en `fattn-common.cuh`, compartido por los kernels VEC y TILE. Reemplaza la lógica duplicada que inicialmente existía en ambos archivos.

2. **Integración nativa page-table en FA** — Los kernels VEC y TILE ahora resuelven direcciones físicas de V mediante la tabla de páginas en cada acceso, eliminando la necesidad del kernel de gather intermedio.

3. **ABI compatibilidad hacia atrás** — Los kernels MMA-f16 y WMMA-f16 aceptan los nuevos parámetros pero no activan el paginado, manteniendo compatibilidad sin cambios en su lógica interna.

4. **API `ggml_flash_attn_ext_set_page_table`** — Proporciona un mecanismo limpio para adjuntar la tabla de páginas a un tensor `GGML_OP_FLASH_ATTN_EXT` post-creación, usando `src[5]`.

## Hipótesis Pendientes

- **H3.1** (reducción de latencia 12–18%): No verificada. Requiere benchmark cuantitativo.
- **Paridad numérica**: No verificada. Requiere test de integración con y sin paginado.

## Riesgos

- Los strides de `ggml_view_4d` son correctos solo para el orden de permutación actual (0, 2, 1, 3). Cualquier cambio en el pipeline de construcción del grafo de atención podría romper el invariante.
- La corrección del paginado depende de que `v_paged_ptr` reciba `nb21` = stride de fila física del pool. Si el formato del pool cambia (e.g., cuantización por bloques), `nb21` debe recalcularse.
