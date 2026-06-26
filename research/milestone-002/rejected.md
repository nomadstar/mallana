# Milestone 002: Rejected Paths

Durante el diseño de la infraestructura de páginas se descartaron las siguientes alternativas:

## 1. Paginación Contigua y Dinámica para el Caché de Claves (K)
- **Idea**: Aplicar paginación no contigua e indexación por tabla de páginas tanto a K como a V.
- **Por qué se rechazó**: Las claves (K) requieren una lectura secuencial rápida y regular durante el prefill y la atención. Introducir indirección de punteros a nivel de bloque en K añade latencia crítica en el path de prefill sin otorgar beneficios adicionales de fragmentación. 
- **Decisión**: El caché de K se mantiene con índices planos lineales (`flat pool indices`), mientras que sólo V utiliza el direccionamiento por tabla de páginas.

## 2. Modificación directa de los kernels de Flash Attention en la Fase 1
- **Idea**: Escribir direccionamiento de páginas directamente en `fattn-vec.cuh` en el primer intento.
- **Por qué se rechazó**: Depurar la tabla de páginas, la pila de bloques libres, las colisiones de hilos CUDA y la lógica de asignación dinámica de C++ al mismo tiempo que la aritmética de warp de Flash Attention era una estrategia de alto riesgo. Separar el problema en dos fases (Fase 1: Gather intermedio, Fase 2: Kernel nativo) permitió estabilizar la asignación de memoria primero.

## 3. Fallback a SDPA manteniendo la paginación activa
- **Idea**: Si Flash Attention no es compatible, usar el kernel de atención SDPA de GGML mapeando las páginas al vuelo.
- **Por qué se rechazó**: SDPA requiere la bandera `v_trans=true` (la matriz de V transpuesta). Esto rompe la estructura del pool físico de páginas.
- **Decisión**: Si FA se apaga, la paginación se apaga por completo.
