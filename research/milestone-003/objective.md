# Milestone 003: Objective

El objetivo actual de este hito es **eliminar el kernel de gather intermedio** y la asignación temporal en memoria global de la Fase 1, integrando la búsqueda en la tabla de páginas directamente dentro de los kernels de Flash Attention (`fattn-vec.cuh` y `fattn-tile.cuh`).

## Metas Específicas
1. **Reducción de Latencia**: Lograr una mejora del 10% al 15% en el tiempo de procesamiento de atención en secuencias largas (>8K tokens) al eliminar la sobrecarga de copias globales de gather.
2. **Direccionamiento en el Kernel**: Modificar la aritmética de cálculo de índices del tensor de V en CUDA. En lugar de avanzar linealmente mediante `V + k * nb21`, calcular dinámicamente la página física (`phys_block`) usando el índice de la tabla de páginas y la posición dentro del bloque.
3. **Paridad de Resultados**: Asegurar que la implementación nativa paginada produzca resultados numéricamente idénticos y mantenga la misma perplexidad que la Fase 1 y el baseline sin paginación.
