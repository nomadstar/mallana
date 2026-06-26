# Milestone 001: Rejected Paths

Durante la auditoría y corrección de consistencia se evaluaron y descartaron las siguientes soluciones:

## 1. Mantener la matriz de rotación densa aleatoria QR en CPU
- **Idea**: Usar una transformación densa ortogonal mediante multiplicación matriz-vector en CPU para simular la rotación de polarización de `turbo4`.
- **Por qué se rechazó**: Multiplicar por una matriz de 128x128 en CPU requiere $O(d^2)$ (16,384 operaciones) en lugar de las $O(d \log d)$ (896 operaciones) del FWHT. Además, el kernel de GPU de `turbo4` ya estaba optimizado para FWHT usando sumas/restas de mariposa. Mantener la rotación densa creaba una divergencia matemática insalvable en rendimiento y en valores de salida.
- **Acción**: Se eliminó por completo la rotación densa en CPU para el modo estándar y se alineó a FWHT.

## 2. Realizar de-cuantización en precisión FP64
- **Idea**: Usar acumuladores de doble precisión para la suma de mariposa del WHT inverso para evitar deriva numérica de punto flotante.
- **Por qué se rechazó**: El incremento de tiempo de cómputo en CUDA y CPU era prohibitivo y no aportaba diferencias significativas de perplejidad. La precisión estándar de FP32 es lo suficientemente estable.

## 3. Cuantizar K con `turbo2`
- **Idea**: Utilizar compresión máxima de 2 bits tanto en K como en V.
- **Por qué se rechazó**: La perplejidad de `turbo2` aplicada a K saltó a **13.42** (un incremento de casi 5 puntos frente al baseline). Las llaves (K) son muy sensibles a la pérdida de información debido a que preservan los patrones posicionales finos de atención.
- **Acción**: Se restringió el uso de `turbo2` únicamente para el caché de V (valores).
