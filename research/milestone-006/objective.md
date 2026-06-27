# Milestone 006: TriAttention KV Eviction

## Objetivo

Implementar un mecanismo de desalojo físico de páginas KV basado en scoring TriAttention para
mantener contextos largos bajo un presupuesto fijo de páginas físicas.

## Hipótesis

**H6.1**: La remoción adaptativa de páginas usando scoring RoPE (TriAttention) permite mantener
el contexto efectivo al 95% de la calidad de perplexidad original usando sólo el 50% de las
páginas físicas.

## Metas Específicas

1. Exponer un presupuesto configurable mediante `--triattention-page-budget`.
2. Reservar el bloque físico 0 como dummy zero block para remapeo seguro tras eviction.
3. Implementar `pg_score_and_evict()` usando productos punto sobre claves K invertidas por RoPE.
4. Integrar el flujo en la gestión del KV cache sin romper el camino paginado existente.

## Archivos a Modificar

- `include/llama.h`
- `src/llama-cparams.h`
- `src/llama-context.cpp`
- `common/common.h`
- `common/common.cpp`
- `common/arg.cpp`
- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`
- `src/llama-model.cpp`
