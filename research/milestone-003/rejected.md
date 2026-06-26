# Milestone 003: Rejected Paths

*Estado: Implementado — pendiente de validación numérica.*

## Rechazados durante implementación

### 1. Duplicación de `v_paged_ptr` en VEC y TILE

**Problema:** Inicialmente `v_paged_ptr` fue implementado como función `__device__` separada
dentro de `fattn-vec.cuh` y `fattn-tile.cuh`, duplicando el código.

**Decisión:** Se movió a `fattn-common.cuh` (incluido por ambos) y se eliminaron las copias
locales. Esto asegura que cualquier corrección al helper beneficie a ambos kernels
simultáneamente.

### 2. Paso de tabla de páginas como tensor separado vs parámetro de op

**Problema:** Se consideró almacenar la tabla de páginas en `op_params` del tensor FA.

**Decisión:** Se usó `src[5]` en su lugar porque `op_params` tiene tamaño limitado (fijo a
64 bytes en ggml) y la tabla de páginas puede ser grande. `src[5]` permite que ggml maneje
automáticamente la gestión del device buffer y las transferencias.

### 3. Modificación directa de los kernels MMA-f16 y WMMA-f16

**Problema:** Activar paginado en MMA-f16 requeriría modificar la lógica de carga de V en
tile (usando `cp.async` con página física).

**Decisión:** Se optó por stubs ABI-compatibles (`GGML_UNUSED`) para estos kernels. El
paginado nativo solo se activa en los kernels VEC y TILE, que son los que se usan para
modelos con cabezales de dimensión arbitraria y tipos turbo, respectivamente.
