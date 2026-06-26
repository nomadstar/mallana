# Research Roadmap (Plan de Investigación)

Este documento detalla los hitos estructurados del proyecto, ordenados por prioridad técnica y dependencias lógicas. Cada hito se asocia con un directorio de evidencias en `research/milestone-XXX/`.

---

## 🗺️ Mapa de Hitos (Milestone Map)

```mermaid
gantt
    title Roadmap de Investigación TurboQuant
    dateFormat  YYYY-MM-DD
    section Fase 1
    Milestone 001 - Auditoría CPU/CUDA           :done,    m1, 2026-04-01, 2026-06-10
    section Fase 2
    Milestone 002 - Paged Attention Fase 1       :done,    m2, 2026-06-11, 2026-06-20
    Milestone 003 - Paged Attention Native FA   :active,  m3, 2026-06-21, 2026-07-15
    section Fase 3
    Milestone 004 - Validación NaN turbo4        :ref,     m4, after m3, 10d
    Milestone 005 - ROCm Backend Completo        :         m5, after m4, 15d
    Milestone 006 - TriAttention Eviction        :         m6, after m5, 20d
```

---

## 🎯 Detalle de Hitos

### [Milestone 001: CPU/CUDA Equivalence Audit & WHT Alignment](file:///home/ignatus/GitHub/llama-cpp-turboquant/research/milestone-001/)
- **Estado**: ✅ **COMPLETADO**
- **Prioridad**: P1 (Correctitud)
- **Objetivos**:
  - Eliminar la inconsistencia de rotación densa en CPU para `turbo4`.
  - Garantizar equivalencia bit a bit de de-cuantización y WHT entre CPU y GPU (CUDA).
- **Evidencia**: Perplejidades alineadas en Llama-3.2-3B (< 0.4 PPL incremento).

### [Milestone 002: Paged Attention Phase 1 — Gather-before-FA](file:///home/ignatus/GitHub/llama-cpp-turboquant/research/milestone-002/)
- **Estado**: ✅ **COMPLETADO**
- **Prioridad**: P2 (Arquitectura)
- **Objetivos**:
  - Implementar pool de bloques de memoria no contiguos (bloques de 32 tokens).
  - Diseñar la tabla de páginas dinámica (`page_table[stream][lpage] → pblock`).
  - Kernel CUDA `GGML_OP_GATHER_PAGED_V` para reconstruir un tensor temporal continuo para Flash Attention.
- **Evidencia**: Ejecución correcta con paginación activa y caída elegante si FA se deshabilita.

### [Milestone 003: Paged Attention Phase 2 — Native Paged FA](file:///home/ignatus/GitHub/llama-cpp-turboquant/research/milestone-003/)
- **Estado**: 🚧 **EN PROGRESO** (Activo)
- **Prioridad**: P2 (Rendimiento)
- **Objetivos**:
  - Eliminar el kernel de gather intermedio y la asignación temporal en memoria global.
  - Pasar la tabla de páginas y el `block_size` directamente a `launch_fattn()`.
  - Reemplazar direccionamiento contiguo en `fattn-vec.cuh` y `fattn-tile.cuh` con indexación dinámica de páginas.

### [Milestone 004: Explicit turbo4 NaN Validation](file:///home/ignatus/GitHub/llama-cpp-turboquant/research/milestone-004/)
- **Estado**: 📅 **PLANIFICADO**
- **Prioridad**: P1 (Robustez)
- **Objetivos**:
  - Diseñar suite de test con distribuciones extremas (Gaussianas, Laplace, Uniformes con outliers masivos).
  - Asegurar que ningún de-cuantizador de `turbo4` retorne NaN o Inf bajo ninguna circunstancia.

### [Milestone 005: ROCm Backend Completion](file:///home/ignatus/GitHub/llama-cpp-turboquant/research/milestone-005/)
- **Estado**: 📅 **PLANIFICADO**
- **Prioridad**: P3 (Portabilidad)
- **Objetivos**:
  - Resolver diferencias mínimas de API entre CUDA y HIP.
  - Compilar y validar soporte completo de `turbo4` en GPUs AMD (RDNA3/CDNA).

### [Milestone 006: TriAttention KV Eviction](file:///home/ignatus/GitHub/llama-cpp-turboquant/research/milestone-006/)
- **Estado**: 📅 **PLANIFICADO**
- **Prioridad**: P4 (Investigación)
- **Objetivos**:
  - Integrar scoring de claves (K) proyectadas con RoPE inverso.
  - Implementar desalojo físico de páginas con menor relevancia en el pool.
