# Milestone 008: Latency and Throughput Benchmark Infrastructure

## Objetivo

Establecer la infraestructura de benchmarks de latencia y throughput para validar las mejoras de rendimiento de Native Paged Flash Attention y la eficiencia de TurboQuant.

## Hipótesis

- **H3.1**: Native Paged FA reduce la latencia de atención un 12-18% en comparación con gather para contextos mayores a 8K tokens (>8K context).
- **H8.1**: TurboQuant logra una mayor inteligencia por gigabyte (intelligence/GB) que q4_0/q8_0 a una perplejidad equivalente.

## Criterios de Éxito

1. Implementar un script de benchmark robusto `scripts/benchmark.py` que permita medir tokens por segundo, tiempo al primer token (time to first token) y tiempo total de generación en tres configuraciones: baseline, native-paged y triattention.
2. Generar resultados estructurados en un archivo JSON (`research/milestone-008/benchmark_results.json`).
3. Imprimir una tabla comparativa en formato Markdown en stdout detallando la mejora porcentual respecto a la configuración baseline.

## Archivos a Modificar / Crear

- `scripts/benchmark.py` (Crear)
- `research/milestone-008/objective.md` (Crear)
- `research/milestone-008/evidence.md` (Crear stub)
- `research/milestone-008/conclusions.md` (Crear stub)
