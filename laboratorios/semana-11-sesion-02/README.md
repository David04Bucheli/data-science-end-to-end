# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

Que un modelo puede tener una accuracy muy alta y aun así ser inútil cuando las clases están desbalanceadas. El ejemplo del artículo: una enfermedad que tiene 1 de cada 100 personas; un modelo que dice que todos están sanos tiene 99% de accuracy y no detecta a ningún enfermo. Recomienda precision, recall y F1.

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". El artículo da el ejemplo de una agencia de noticias que puso como meta subir 20% la duración de la sesión: el equipo alargó las introducciones y la duración subió, pero bajó la calidad. Si elijo el modelo por una sola métrica, puedo quedarme con el que mejor se ajusta a esa métrica y no con el que sirve de verdad, así que conviene mirar varias métricas.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Cambiaría bastante. Lo vi en el registro: el mismo gradient boosting dio 1.1860 con una división y 1.2750 con otra. En el Ejercicio 1, además, los cinco pliegues al azar van de 1.16 a 1.28.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Sí, parecidos pero no iguales. Esperaba una variación chica, de unas centésimas. Salieron 1.28, 1.24, 1.16, 1.24 y 1.23: promedio 1.2285 y desviación de 0.0359.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Mayor. En la temporal el modelo predice el futuro con el pasado y los patrones pueden cambiar; con pliegues al azar el modelo ve viajes de las mismas fechas que valida. Salió 1.3094 de promedio, con desviación 0.0686 y peor pliegue de 1.39.

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El de las zonas más raras, como JFK Airport, que tiene tarifa fija y no se parece a las demás. Salió así: el quinto pliegue (Brooklyn y JFK Airport) dio 6.77, contra entre 1.19 y 1.31 en los otros.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El porcentaje de viajes subestimados en más de 20%, porque es justo lo que causa los reclamos. El MAE y el RMSE miden el tamaño del error sin importar si fue por arriba o por abajo. En este modelo: MAE 1.2750, RMSE 1.9188 y 4.37% de viajes subestimados en más de 20%.

**Lo que confirmó la tabla, esquema por esquema:**

Aleatorio con gradient boosting: promedio 1.2285, desviación 0.0359, peor pliegue 1.28. Aleatorio con bosque: 1.2887, 0.0306, peor 1.34. Temporal con gradient boosting: 1.3094, 0.0686, peor 1.39. Por zona con gradient boosting: 2.3677, 2.2010, peor 6.77. Mientras más difícil la validación, más alto el error y más variación.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy de 0.9508 y recall de 0: nunca predice fraude, así que acierta en los pagos normales y no detecta ninguno.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

La accuracy sube de 0.9508 a 0.9712, apenas 2 puntos; el recall sube de 0 a 0.4610. El recall describe mejor lo que gana la empresa, porque es la proporción de fraudes que detecta.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con 0.5 deja pasar 159 de 295 fraudes (y solo 14 falsas alarmas). El paso 5 eligió el umbral 0.10: con ese pasan 122 fraudes y hay 222 falsas alarmas, el menor costo de los probados (5546).

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

Sube. Con la falsa alarma en 20 dólares el mejor umbral pasó a 0.5 (costo 6640, contra 9320 con 0.10), porque ahora cada falsa alarma sale más cara.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

El temporal. El modelo predice ventas de días que todavía no ocurrieron, así que cada pliegue tiene que entrenar con el pasado y validar con lo que viene después: usé `TimeSeriesSplit` con 5 pliegues sobre el entrenamiento ordenado por fecha. La validación al azar mezcla días de todas las épocas, y el modelo ve en el entrenamiento ventas del mismo mes y de los mismos productos que luego le piden predecir.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

El nivel del error sí cambió; el orden de los modelos no. En promedio de los 5 pliegues:

| Modelo | RMSE al azar | RMSE temporal | Costo del negocio al azar | Costo del negocio temporal |
|---|---|---|---|---|
| bosque `min_samples_leaf=20` | 2.9957 | 3.2488 | 2.0448 | 2.2891 |
| red neuronal (32, 16) | 2.9643 | 3.2350 | 2.0797 | 2.3561 |

La validación al azar es optimista: subestima el RMSE del bosque en un 8% (2.9957 contra 3.2488) y el costo en un 11% (2.0448 contra 2.2891). Con cualquiera de los dos esquemas, el RMSE de la red es un poco menor y el costo del negocio del bosque es un poco menor. El RMSE de prueba del bosque (2.9332 en MLflow) no se puede comparar directo con el promedio temporal (3.2488): son conjuntos distintos, y el RMSE depende mucho de unas pocas ventas grandes (la desviación entre pliegues es de 0.39).

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

No. La diferencia de RMSE entre la red y el bosque es de 0.014 en promedio con el esquema temporal (por pliegue: -0.002, +0.014, -0.009, -0.047, -0.025), y la desviación entre pliegues es de 0.39 y 0.40. En RMSE son un empate: la red gana en 4 de los 5 pliegues temporales, pero por muy poco, y el orden incluso cambia con el error de prueba según cómo se ordenen las filas (el bosque gana por 0.024 en MLflow y la red por 0.019 en el notebook de validación).

Sigo con el bosque, porque con la métrica del negocio gana de forma consistente: en la validación temporal (2.2891 contra 2.3561), al azar (2.0448 contra 2.0797) y en la prueba (2.5982 contra 2.6476). Además es más fácil de explicar.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

Un costo asimétrico: cada unidad que falta (venta perdida) cuesta 3 veces lo que una unidad que sobra (inventario parado). Es una aproximación, porque no tengo los costos reales. El costo por fila es `3 × unidades que faltaron + 1 × unidades que sobraron`.

Dice dos cosas que el RMSE no decía. Primero, que el modelo se equivoca más caro de lo que parece: sobre la prueba, el bosque se queda corto por 26,189 unidades y se pasa por 25,263, casi lo mismo en cantidad, pero las faltantes pesan el triple, así que representan el 76% del costo. Segundo, que el modelo, que predice promedios, subestima justo en los productos que más venden, así que para ahorrar costo convendría que predijera un poco más alto de lo que predice hoy (un cuantil alto en vez del promedio). Con esta métrica el bosque mejora un 34% a la línea base (costo 2.5982 contra 3.9459), más que el 23% que mostraba el RMSE.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa se podría precalcular para los viajes ya reservados, pero no para todos los casos posibles: hay demasiadas combinaciones de zonas, distancias y horas. El de fraude tiene que responder en el momento del pago, en menos de un segundo, porque depende de ese pago y el pasajero espera.
