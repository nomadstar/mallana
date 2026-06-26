# Milestone 002: Objective

El objetivo de este hito fue implementar la primera fase de **Paged Attention** (atención paginada) en el repositorio, permitiendo memoria física no contigua para el KV cache y asignación dinámica bajo demanda.

## Metas Específicas
1. **Layout de Bloques**: Definir un pool físico de bloques en memoria compartida (tamaño de bloque = 32 tokens, alineado con `QK_TURBO3`) para evitar la fragmentación interna.
2. **Tabla de Páginas**: Diseñar e implementar la estructura de datos que mapea páginas lógicas de secuencias a bloques físicos.
3. **Integración con Flash Attention sin alterar kernels**: Diseñar un operador de Gather intermedio (`GGML_OP_GATHER_PAGED_V`) que se ejecute en GPU para materializar las páginas no contiguas en un búfer contiguo temporal antes de llamar al kernel de Flash Attention. Esto desacopla la lógica de asignación de memoria de las optimizaciones matemáticas del kernel FA.
4. **Resguardo de caídas (Fallback)**: Asegurar que si el soporte de Flash Attention se desactiva (por ejemplo, dimensiones de cabeza no soportadas), el sistema desactive la paginación limpiamente en lugar de causar una falla de segmentación.
