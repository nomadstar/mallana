# Milestone 001: Objective

El objetivo principal de este hito fue asegurar la **correctitud y consistencia matemática absoluta** de la implementación de TurboQuant a través de los backends de CPU y GPU (CUDA).

## Metas Específicas
1. **Auditar la equivalencia numérica**: Verificar que para cualquier tensor de entrada, la cuantización y posterior de-cuantización produzcan los mismos valores exactos tanto en la ejecución en CPU (implementación de referencia) como en GPU (CUDA).
2. **Resolver la discrepancia en turbo4**: Investigar y corregir por qué el de-cuantizador de `turbo4` difería significativamente en sus salidas entre CPU y GPU, lo que rompía la interoperabilidad de modelos cargados o cuantizados en distintas plataformas.
3. **Validar métricas de calidad**: Asegurar que la perplexidad (PPL) en modelos Llama-3.2-3B no se degradara más de 0.4 puntos frente al baseline FP16 al usar `turbo4`.
