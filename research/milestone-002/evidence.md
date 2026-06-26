# Milestone 002: Evidence

La implementación de Paged Attention Fase 1 se validó demostrando correctitud funcional en pruebas multi-secuencia y midiendo la degradación por el gather intermedio.

## 1. Cambios Estructurales
Se introdujeron los siguientes componentes funcionales:
- **Operador GGML**: `GGML_OP_GATHER_PAGED_V` para recolectar páginas.
- **Estructura C++**:
  - `block_pool_meta` con contadores de referencia.
  - Pila LIFO `pg_free_blocks` para reciclaje rápido de bloques con costo O(1).
  - Tabla de páginas `pg_page_table[stream][lpage] → pblock` en `llama_kv_cache`.
- **Sincronización CUDA**: Se añadió un dispositivo de sincronización explícito (`cudaDeviceSynchronize`) antes del borrado del búfer para evitar colisiones asíncronas con la carga de la tabla de páginas.

## 2. Pruebas de Correctitud
- **Modelos**: Validados en Qwen3.5-35B-A3B-Q8_0 y Llama-3.2-3B.
- **Comportamiento**: Inferencia correcta y outputs estables al habilitar paginación mediante variables de entorno.
- **Bit-Exactness**: Con tabla de páginas de identidad, las respuestas generadas con paginación activa son idénticas a las del pipeline lineal clásico.
- **Mecanismo de Desactivación Automática**: Se verificó que al forzar un head_dim no compatible en Flash Attention, el código desactiva la paginación con éxito y continúa la inferencia normal.

## 3. Penalización de Rendimiento
El gather intermedio añade una pequeña sobrecarga por la copia intermedia en memoria global:
- **Costo de Gather**: ~3% a 5% del tiempo de ejecución de atención en contextos cortos.
- **Conclusión**: Esta sobrecarga motiva directamente el paso a la Fase 2 (Hito 003: Native Paged FA) para fusionar el gather dentro del kernel de Flash Attention.
