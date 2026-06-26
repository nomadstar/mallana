# Process: Milestone Complete (Cierre de Milestone)

Este prompt/proceso se activa de forma automática cuando un hito del roadmap ha sido completado con éxito (es decir, el experimento ha sido diseñado, implementado, validado cuantitativamente y revisado sin vetos).

Tu tarea como agente que ejecuta este proceso es generar o actualizar la documentación histórica en `/research/milestone-XXX/`.

---

## 📋 Estructura del Reporte de Cierre de Milestone

Debes responder generando los siguientes 4 archivos dentro de `research/milestone-XXX/`:

### 1. `objective.md`
- **Contenido**: Declaración corta y concisa del objetivo original de la investigación.
- **Formato**: Markdown estructurado con títulos claros.

### 2. `evidence.md`
- **Contenido**: Evidencia empírica ineludible recopilada por el Validador.
- **Elementos**:
  - Benchmarks de velocidad (TPS antes y después del cambio).
  - Resultados exactos de perplejidad en wikitext-2.
  - Trazas de logs o salidas de tests unitarios que verifiquen el éxito.

### 3. `rejected.md`
- **Contenido**: Enfoques y soluciones que se intentaron y se descartaron durante el hito.
- **Razón**: Detalla por qué falló cada idea y por qué no debe repetirse.

### 4. `conclusions.md`
- **Contenido**: Las conclusiones generales, las "verdades demostradas" y las decisiones de diseño consolidadas.
- **Sección Mandatoria - QUÉ NO VOLVER A INVESTIGAR**:
  - Una lista clara de cosas que el proyecto ha demostrado que no funcionan o que tienen limitaciones intrínsecas, previniendo que futuras IAs pierdan tiempo volviendo a recorrer caminos descartados.

---

## 🔄 Pasos Finales de Cierre
1. Crea el nuevo directorio `research/milestone-XXX/` con los 4 archivos descritos.
2. Actualiza [research/state.md](file:///home/ignatus/GitHub/llama-cpp-turboquant/research/state.md):
   - Mueve el hito completado de la sección "Hito Actual" a la sección "Caminos Validados" con un resumen breve.
   - Si surgieron verdades definitivas de lo que no funciona, regístralas en la sección "Caminos Rechazados".
   - Promueve el siguiente hito planificado de [research/roadmap.md](file:///home/ignatus/GitHub/llama-cpp-turboquant/research/roadmap.md) a "Hito Actual".
