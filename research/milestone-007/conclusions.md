# Milestone 007: Conclusions

**Fecha de cierre parcial**: 2026-06-28  
**Estado**: PARCIALMENTE COMPLETO — Infraestructura lista, H6.1 pendiente de validación en generación

---

## Veredicto H6.1

**H6.1: INDETERMINADO** — No refutada, pero tampoco confirmada.

La hipótesis "TriAttention page eviction mantiene ≥95% PPL al 50% de páginas" no puede
verificarse con evaluación de perplexity en batch (modo `llama-perplexity`). La razón es
estructural: la protección de páginas en-uso impide que la evicción dispare durante batch
evaluation. Se requiere validación en modo generación autoregresivo.

---

## Hallazgos Confirmados

### 1. Bug Crítico Corregido (off-by-one en pool de páginas)
El pool de bloques físicos tenía `pg_n_blocks - 1` bloques disponibles para `pg_n_blocks`
páginas lógicas. Crash en llama-perplexity con n_seq=4 (modo por defecto). **Corregido**.

### 2. Bug Phase 2 FA Documentado (n_seq vs ns en page table)
El kernel CUDA de Flash Attention accede índices OOB en la page table cuando n_seq > n_stream.
**Workaround**: `-fa off` (Phase 1 gather). **Fix pendiente** en `set_input_v_page_table`.

### 3. Infraestructura de Calibración Funcional
- Script `scripts/triattention_calibrate.py`: validado, sanity checks, corpus correcto
- Corpus `data/wikitext2-test.txt`: PPL ~10.9 con qwen2.5-coder-1.5b (correcto)
- JSON de resultados generado automáticamente

### 4. TriAttention No Crashea con Corpus Correcto
Con `-fa off` y `--triattention-page-budget 32` (ctx=2048): no crash, PPL estable.
La infraestructura de evicción está funcional; solo falta la validación de calidad.

---

## Trabajo Pendiente para Cierre Completo de M007

1. **Fix Phase 2 FA** (`set_input_v_page_table`): expandir page table a `max(ns, n_seq)` filas
2. **Script de generación** (`scripts/triattention_generation_eval.py`): validar evicción en
   modo autoregresivo con prompt largo + medición de PPL condicional
3. **Validación H6.1 real**: ejecutar script de generación con budget=50% y medir degradación

---

## Lecciones Aprendidas

1. **Perplexity batch ≠ generación**: El modo de evaluación importa para testear evicción.
   La evicción es un mecanismo de runtime autoregresivo, no se puede medir off-line en batch.

2. **Corpus importa enormemente**: README.md como corpus → PPL=38805 (inútil). wikitext-2
   → PPL~10.9 (correcto). El script ahora detecta y rechaza corpus tipo Markdown.

3. **Off-by-one en pools de memoria**: Con bloques dummy/sentinel que consumen una entrada,
   el tamaño del pool debe compensar explícitamente. El comentario del fix explica el invariante.

4. **n_seq vs n_stream**: La distinción entre "número de secuencias en el batch" y "número
   de streams de KV cache" es una fuente de bugs. El paged FA asume 1:1 pero en perplexity
   mode n_seq puede ser 4× n_stream.
