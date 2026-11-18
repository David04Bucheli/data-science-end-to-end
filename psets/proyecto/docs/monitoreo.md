# Monitoreo del modelo de ventas en producción

El modelo se sirve por lotes (cada noche, 7 días hacia adelante) y también por la API de `servicio/` para consultas puntuales; ver `despliegue.md`. Lo que se vigila es lo mismo en los dos casos: que el servicio responda bien (sistema) y que el modelo acierte (modelo). Que el servicio responda en milisegundos y devuelva 200 no dice nada sobre si las unidades estimadas son buenas.

## Las métricas

| Métrica | Tipo | De dónde sale | Cuándo se conoce | Umbral de alerta |
|---|---|---|---|---|
| Disponibilidad | sistema | `GET /salud` | al instante | no responde, o `version_modelo` distinta de la esperada |
| Latencia p50 y p95 | sistema | `GET /metricas` (middleware de `servicio/api.py`) | al instante, por petición | p95 > 200 ms |
| Errores de servidor (5xx) | sistema | `GET /metricas` | al instante | cualquiera: `errores_servidor` > 0 |
| Errores de validación (422) | sistema (contrato con quien llama) | `GET /metricas` | al instante | más del 5% de las peticiones |
| Fracción de peticiones sin precio | modelo, indirecta (calidad de la entrada) | `monitoreo/predicciones.jsonl` (`sell_price` vacío), vía `monitoreo/desempeno.py` | el mismo día | una semana con más del doble que la anterior |
| Promedio de `unidades_estimadas` por semana | modelo, indirecta (deriva de la salida) | `monitoreo/predicciones.jsonl`, vía `monitoreo/desempeno.py` | el mismo día | se aleja más del 30% de la semana anterior |
| RMSE semanal, por versión | modelo (desempeño) | predicciones unidas con las ventas reales (`UNITS`), `monitoreo/desempeno.py` | cuando cierra el día predicho: de 1 a 8 días después de la predicción | > 1.25 × 2.93 (la referencia de prueba), es decir, > 3.67 |
| Costo de negocio semanal, por versión (faltante 3, sobrante 1) | modelo (desempeño, la métrica del negocio) | igual que el RMSE | igual que el RMSE | > 1.25 × 2.29 (promedio de la validación temporal), es decir, > 2.86 |

## Qué dice cada grupo, y qué no

- **Sistema.** Dice si el servicio está vivo, rápido y recibe peticiones bien formadas. No dice nada de la calidad de las estimaciones: un modelo que ya no acierta sigue respondiendo 200 en 10 ms.
- **Indirectas (sin la verdad).** Se calculan el mismo día y avisan que cambió lo que el modelo recibe o responde: más productos sin precio, o estimaciones que suben o bajan de golpe. No dicen cuánto se equivoca; un cambio puede ser real (una promoción) y el modelo seguir acertando.
- **Desempeño (con la verdad).** Es la única que mide el error de verdad, y llega con retraso: las ventas de un día se conocen al cerrarlo, así que una semana solo tiene RMSE completo cuando termina su último día. Mientras tanto, solo se pueden mirar las indirectas.

## Cómo se calcula

- `GET /metricas` devuelve `peticiones`, `errores_validacion`, `errores_servidor`, `latencia_p50_ms` y `latencia_p95_ms`. Están en memoria: se reinician cuando se reinicia el servicio.
- Cada predicción queda en `monitoreo/predicciones.jsonl` con `id_prediccion`, `momento`, `version_modelo`, la entrada (incluida `fecha`, el día que se predice) y `unidades_estimadas`. La unión con las ventas reales es por producto, tienda y fecha: por eso `fecha` es parte de la petición aunque el modelo no la use.
- `python -m monitoreo.desempeno` (desde `psets/proyecto/`) imprime el error por semana y versión, las señales sin la verdad y las alertas.