# Milestone 001: Conclusions

El Hito 001 cerró con éxito la fase de depuración de correctitud del motor de cuantización TurboQuant.

## Principales Aprendizajes y Verdades Demostradas

1. **La consistencia bit-exacta es viable**: Demostrado que CPU y GPU pueden coincidir exactamente en operaciones FWHT y des-empaquetado de polar-codebooks si se sincronizan las mariposas y los signos. No hay excusa para aceptar discrepancias en la decodificación.
2. **FWHT es superior a la Rotación Ortogonal QR**: La estructura de mariposa O(d log d) provee la misma dispersión de energía que la rotación aleatoria dense QR, pero a una fracción del costo computacional (896 vs 16,384 flops).
3. **Regla de Oro de Asimetría**: K y V deben tratarse como canales distintos con tolerancias al ruido disímiles. V acepta bajo bit (`turbo3`/`turbo2`), mientras que K requiere precisión (`q8_0` o superior).

---

## 🚫 QUÉ NO VOLVER A INVESTIGAR

- **No re-introducir rotaciones densas (multiplicaciones matriciales completas) en el pipeline de WHT**.
- **No intentar aplicar cuantización de 2 bits a las llaves (K)** en entornos de producción sin técnicas avanzadas de detección de outliers (como ecualización dinámica activa por canal).
