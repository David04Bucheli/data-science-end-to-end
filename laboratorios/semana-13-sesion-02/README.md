# Semana 13, Sesión 2: Monitoreo en producción I: métricas de rendimiento

> Objetivo de la sesión: definir las métricas de monitoreo del modelo de tu proyecto en producción, distinguiendo métricas de sistema (latencia, disponibilidad) de métricas de modelo (desempeño).

**Lectura previa de esta sesión:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 8: Data Distribution Shifts and Monitoring.

## Requisitos antes de esta sesión

Nada nuevo que instalar. Los laboratorios de hoy son scripts. El Ejercicio 1 no necesita levantar uvicorn: `simular_noviembre.py` le manda las peticiones al servicio con `TestClient`, como las pruebas.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿por qué no alcanza con vigilar la salud y el volumen de peticiones del servicio? ¿Qué hay que medir además?**

El artículo plantea que las métricas de infraestructura (que el servicio esté vivo, el volumen de peticiones, la latencia) solo dicen que el sistema funciona, no que el modelo acierte: un modelo puede responder rápido, sin errores, y estar equivocándose. Por eso hay que medir además el **desempeño funcional del modelo**: su calidad comparando las predicciones con la verdad (ground truth), más señales indirectas sobre los datos de entrada y las predicciones (deriva de datos y de predicciones) y la salud del pipeline que prepara los datos.

**El artículo dice que la verdad (ground truth) a veces llega con demora, o no llega. ¿Qué propone medir mientras tanto?**

Propone medir mientras tanto **métricas indirectas (proxy)**: cómo cambia la distribución de los datos de entrada y la de las predicciones respecto de lo que el modelo vio al entrenar (deriva de datos y de predicciones), y vigilar que el pipeline de datos no esté fallando. Y, cuando la verdad llega, unirla con cada predicción guardada y calcular las métricas reales (por eso cada predicción tiene que guardarse con lo necesario para unirla después). Las indirectas avisan de que *algo cambió*, pero no dicen cuánto se equivoca el modelo.

## Pregunta de apertura

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Que el modelo se degrada en silencio. El servicio sigue respondiendo en milisegundos, `/salud` dice «ok» y no hay errores, pero las estimaciones son cada vez peores porque el mundo cambió (otros precios, otros hábitos, otra demanda). Ninguna métrica de sistema lo ve. Me enteraría solo si guardo cada predicción y, cuando llega la verdad, calculo el error por semana y lo comparo con el de prueba (eso es el Ejercicio 1: en noviembre el error se multiplica por 3.7 y el servicio sigue «perfecto»); mientras la verdad no llega, con señales indirectas, como el promedio de lo que predice el modelo.

## Ejercicio 1: la tarifa, instrumentada

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa con la versión 1 del modelo (la red de 64 neuronas), ahora con dos agregados. Un middleware (una función que corre alrededor de cada petición) mide cuánto tarda cada predicción y con qué código responde; `/metricas` resume esas mediciones. Y cada predicción queda guardada en `predicciones.jsonl`, con un identificador que la app guarda junto al viaje. [`simular_noviembre.py`](./starter/ejercicio1/simular_noviembre.py) manda al servicio los 1156 viajes de noviembre y escribe `cobros.csv`, lo que el sistema de cobros anotó al terminar cada viaje: el identificador de la predicción y la tarifa cobrada (no lo modifiques). [`desempeno.py`](./starter/ejercicio1/desempeno.py) une las dos cosas y calcula el error por semana. Los 4 `TODO`: medir la latencia, resumir las métricas, guardar cada predicción y calcular el error por semana.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python simular_noviembre.py
python desempeno.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** los 1156 viajes recibieron su tarifa. ¿Qué dicen `/metricas` y `desempeno.py`? ¿Qué podrías afirmar sobre el servicio en noviembre?

Antes de los TODO, `/metricas` devuelve todo en cero y las latencias vacías (`peticiones: 0`, `errores_validacion: 0`, `errores_servidor: 0`, `latencia_p50_ms` y `latencia_p95_ms` en `None`), y `desempeno.py` termina con «No hay predicciones.jsonl: el servicio no esta guardando sus predicciones». Del servicio en noviembre solo podría afirmar que **respondió** a las peticiones (los 1156 viajes recibieron tarifa). No podría decir cuánto tardó, si falló alguna petición (hubo 3 mal formadas que nadie contó), ni, sobre todo, si las tarifas estaban bien: sin el registro no hay con qué comparar la tarifa cobrada.

**Respuesta 2 (TODO 1 y 2):** antes de correr la simulación, ¿qué latencia p95 esperas, y cuántos errores de validación? ¿Esperas que la latencia cambie entre la primera y la última semana del mes?

Espero una latencia p95 de unas pocas decenas de milisegundos como máximo (un modelo pequeño, sin red, con `TestClient`) y **3 errores de validación**, las 3 peticiones mal formadas que manda la simulación, y 0 errores de servidor. La latencia **no** debería cambiar entre la primera y la última semana: el modelo, el tamaño de la entrada y el servicio son los mismos todo el mes (lo que cambia en noviembre no es la velocidad).

Resultado al correrlo: 1159 peticiones (1156 + 3), 3 errores de validación, 0 de servidor, p50 6.52 ms y p95 10.41 ms (los milisegundos dependen de la máquina).

**Respuesta 3 (TODO 3):** el servicio devuelve `id_prediccion` y lo guarda en el registro. ¿Para qué lo necesita el sistema de cobros? ¿Qué pasaría si solo se guardaran la fecha y la tarifa estimada?

El sistema de cobros guarda el `id_prediccion` junto al viaje para poder decir **exactamente cuál predicción** corresponde a cada tarifa cobrada. Con ese id, el cobro se une con la fila del registro (versión del modelo, entrada, tarifa estimada) y se calcula el error de esa predicción.

Si solo se guardaran la fecha y la tarifa estimada, la unión sería ambigua o imposible: muchos viajes comparten el mismo minuto (en `viajes_noviembre.csv` hay varios con la misma hora de inicio) y la misma tarifa estimada; si una petición se repite o se reintenta habría dos predicciones para un solo viaje; y no se sabría qué versión del modelo la calculó. El error se asignaría a los viajes equivocados y se mediría mal.

**Respuesta 4 (TODO 4):** el MAE en prueba de la versión 1 fue 1.17. Antes de correr `desempeno.py`, ¿qué esperas para cada semana de noviembre?

Si nada hubiera cambiado, esperaría un MAE cercano al de prueba (1.17) en las cuatro semanas, con algo de variación (±0.1–0.2) porque cada semana es una muestra distinta. Resultado:

| Semana | Viajes | MAE | Veces la referencia (1.17) |
|---|---|---|---|
| 2 de noviembre | 291 | 1.1792 | 1.01 |
| 9 de noviembre | 318 | 1.3403 | 1.15 |
| 16 de noviembre | 308 | 4.2916 | 3.67 |
| 23 de noviembre | 239 | 4.3042 | 3.68 |

Las dos primeras semanas están dentro de lo esperado. Desde el 16 de noviembre el error se multiplica por 3.7 y se queda ahí.

**Lo que mostró el monitoreo:**

- **Latencia:** p50 6.52 ms, **p95 10.41 ms** (varía según la máquina), igual de estable todo el mes.
- **Errores:** 3 de validación (las peticiones mal formadas), 0 de servidor; 1159 peticiones en total.
- **MAE por semana:** 1.18 (2 nov), 1.34 (9 nov), **4.29 (16 nov)** y **4.30 (23 nov)**, contra 1.17 en prueba.

Todas las métricas del sistema estaban bien durante todo noviembre, y el modelo empeoró casi cuatro veces: solo las métricas de modelo (error con la verdad) lo mostraron.


## Ejercicio 2: el fraude, cuando la verdad tarda tres semanas

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de fraude guardó los 5874 pagos que evaluó en noviembre en [`predicciones_fraude.csv`](./starter/ejercicio2/predicciones_fraude.csv): la probabilidad, si mandó el pago a revisión y cuánto tardó. Un fraude se confirma cuando el titular de la tarjeta reclama el cobro, entre 2 y 21 días después; los confirmados están en [`contracargos.csv`](./starter/ejercicio2/contracargos.csv). Hoy es 14 de diciembre. [`monitoreo_fraude.py`](./starter/ejercicio2/monitoreo_fraude.py) arma el reporte del mes. Los pasos 1 a 3 tienen `TODO`: la latencia p95 por día, la tasa de alertas por semana y los fraudes confirmados hasta hoy. Los pasos 4 (el recall de las semanas cuyos fraudes ya se confirmaron) y 5 (las alertas) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python monitoreo_fraude.py`.

**Respuesta 1:** el reporte da tres tipos de alerta. ¿Cuáles son de sistema y cuáles de modelo? ¿El día de la latencia alta coincide con alguna semana de recall bajo?

**De sistema:** la de latencia («el 2026-11-10 la latencia p95 fue 78.2 ms», sobre el máximo de 50 ms). **De modelo:** las otras dos: la tasa de alertas («modelo (indirecta)»: las semanas del 16 y del 23 de noviembre mandaron a revisión 5.0% y 4.6% de los pagos, bajo el 75% de la referencia de 7.58%) y el recall («la semana del 16 de noviembre detectó 0.276 de los fraudes»).

**No coincide:** el día lento (10 de noviembre) cae en la semana del 9, cuyo recall fue 0.736 (normal); y la semana de recall bajo (la del 16) no tuvo ningún día lento. La latencia fue un problema aislado, de sistema, sin relación con la caída del modelo.

**Respuesta 2:** la tasa de alertas se puede calcular el mismo día; el recall, tres semanas después. ¿Qué semana mostró primero el problema la tasa de alertas, y cuándo lo confirmó el recall? ¿Por qué la tasa de alertas no alcanza para decir que el modelo empeoró?

La tasa de alertas mostró el problema **primero en la semana del 16 de noviembre** (5.0% contra 7.1% y 6.9% de las semanas anteriores), una información que ya se tenía el mismo día. El recall lo **confirmó recién el 13 de diciembre** (el domingo 22 de noviembre más 21 días; el reporte del 14 de diciembre ya lo trae: 0.276 contra 0.767 y 0.736), unas tres semanas después.

La tasa de alertas no alcanza porque mide **cuántos pagos marca** el modelo, no **cuántos fraudes atrapa**. Puede bajar porque llegan menos pagos sospechosos (menos fraude, otra temporada, otro tipo de cliente) y el modelo siga igual de bueno, o bajar porque el modelo dejó de reconocer fraudes. Solo el recall, que necesita saber cuáles pagos eran fraude, distingue los dos casos. La tasa sirve como aviso temprano, no como veredicto.

**Respuesta 3:** ¿por qué el reporte no muestra el recall de la semana del 23 de noviembre, si ya hay fraudes confirmados de esa semana?

Porque a la fecha del reporte (14 de diciembre) **todavía no se han confirmado todos los fraudes de esa semana**: su domingo (29 de nov) más 21 días es el 20 de diciembre, y un fraude puede confirmarse hasta 21 días después. Con lo confirmado hasta hoy hay 66 fraudes en esa semana, y el recall saldría 0.394; pero faltan por confirmar fraudes (al terminar son 82, con recall 0.402). Un recall calculado solo con los fraudes que ya reclamó el titular tiene el denominador incompleto: es un número provisional y sesgado hacia lo que se confirma más rápido, que podría dar una falsa tranquilidad o una falsa alarma. El reporte solo muestra semanas «maduras».

**Si terminas antes:** cambia `HOY` a `"2026-11-30"` y vuelve a correr. ¿Qué semanas tienen recall? ¿Desde qué día el reporte puede mostrar la caída del recall de la semana del 16?

Con `HOY = "2026-11-30"` solo tiene recall la **semana del 2 de noviembre** (0.767): su domingo (8 de nov) más 21 días es el 29 de nov. La del 9 madura el 6 de diciembre, la del 16 el 13 y la del 23 el 20. Por eso la **caída del recall de la semana del 16 recién se puede mostrar desde el 13 de diciembre**. El 30 de noviembre el reporte solo trae la alerta de latencia y las dos alertas indirectas de la tasa de alertas (semanas del 16 y 23), pero ninguna alerta de recall: la caída ya había ocurrido y todavía no se podía confirmar.

## Ejercicio 3: el monitoreo de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio de las dos sesiones anteriores. Esta vez no hay plantilla.

### Entregable 1: el servicio, instrumentado

- Un middleware que mida la latencia y el código de cada predicción, y una ruta `/metricas` con, al menos, la cantidad de peticiones, los errores de validación, los errores de servidor y la latencia p50 y p95.
- Cada predicción queda en un registro (un archivo JSONL en `monitoreo/`, por ejemplo) con su identificador, el momento, la versión del modelo, la entrada y la salida. Lo que haga falta para unirla después con la verdad tiene que estar en ese registro.
- Pruebas en `servicio/test_api.py` que comprueben que una predicción queda registrada y que `/metricas` separa los errores de validación de los de servidor, con un registro temporal (`tmp_path`).

### Entregable 2: `monitoreo/desempeno.py`

Une el registro con la verdad de tu proyecto y calcula la métrica de la Semana 11 por período (día o semana) y por versión. Si en tu proyecto la verdad no llega, o llega muy tarde, define una métrica que se pueda calcular antes y explica qué te dice y qué no.

### Entregable 3: `docs/monitoreo.md`

Una tabla con cada métrica que vigilas: si es de sistema o de modelo, de dónde sale, cuándo se conoce y con qué umbral alerta.

**¿Cuánto tarda en llegar la verdad en tu proyecto, y qué métrica te avisa mientras tanto?**

La verdad son las ventas reales (`UNITS`) de cada producto, tienda y día, y se conocen cuando cierra el día predicho. Como el modelo se sirve por lotes (cada noche, 7 días hacia adelante), la verdad de cada predicción llega entre **1 y 8 días después**. Mientras tanto me avisan las métricas indirectas del mismo día: la fracción de peticiones sin precio, el promedio de `unidades_estimadas` por semana y los errores de validación (422), que cambian si cambió lo que llega o lo que el modelo responde. No dicen cuánto se equivoca el modelo; solo el error con las ventas reales (`monitoreo/desempeno.py`) lo dice.

**¿Qué falla de tu modelo no vería ninguna de tus métricas de sistema?**

Casi cualquier falla del modelo: ninguna métrica de sistema (latencia, 5xx, 422, disponibilidad) lo vería. Ejemplos concretos de mi proyecto: (1) una promoción, un evento o un cambio de precios que cambia la demanda y el modelo sigue estimando como antes: responde 200 en milisegundos con estimaciones equivocadas; (2) quien llama empieza a mandar siempre `sell_price` vacío: la API lo acepta, la preparación imputa la mediana y el modelo predice peor, sin ningún error; (3) productos o tiendas nuevos: `handle_unknown="ignore"` y el codificador del producto los dejan pasar y devuelven una estimación genérica, sin error. En los tres casos solo se nota comparando con las ventas reales, o con las señales indirectas.

## Antes de la próxima sesión

**Lectura:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe el servicio son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

Cambió **la relación entre el viaje y su tarifa**, no los viajes. Los viajes que llegan tienen el mismo perfil (distancia media de 7.8 a 8.8 km y 1.6–1.7 pasajeros todas las semanas, mismas zonas y horas), pero desde el 16 de noviembre la tarifa cobrada es unas 15–20% mayor para el mismo tipo de viaje (promedio cobrado de 31.4 contra 27.2 estimado, un sesgo de −4.2 dólares que se repite en la semana del 23, y también en los viajes de tarifa fija al JFK). Eso es **concept drift**: el modelo aprendió una función de las entradas a la tarifa, y esa función cambió (por ejemplo, un aumento de tarifas o un nuevo recargo).

**Mirando solo los viajes que llegan, no lo habría detectado**: las entradas no cambiaron, así que ninguna alerta de deriva de datos habría saltado, y las métricas de sistema tampoco. Se ve comparando las estimaciones con la tarifa cobrada (el MAE por semana, o el sesgo promedio, que acá es de signo constante).