# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada, y tu `pipeline_proyecto.py`.

Como la sesión pasada, los laboratorios son notebooks que importan la preparación de un script (`preparacion_taxis.py` en los Ejercicios 1 y 2, tu `pipeline_proyecto.py` en el 3).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

El aprendizaje automático clásico conviene con datos tabulares (columnas numéricas y categóricas) cuando quieres una predicción numérica precisa, barata y que se pueda explicar. Un LLM conviene con texto o datos no estructurados y con tareas abiertas (resumir, redactar, responder preguntas), donde no hay una tabla de entrenamiento.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

El costo. Mi proyecto es tabular y hay que predecir unas 213,430 filas cada noche; con un LLM se pagaría por token en cada predicción, mientras que un modelo clásico entrenado una vez predice casi gratis.

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Mediría cuánto cuesta cada predicción, cuánto tarda en responder (latencia), cuánto tarda en entrenar y cada cuánto hay que reentrenar, si da siempre el mismo número para el mismo viaje y si se puede explicar por qué dio esa tarifa.

## Ejercicio 1: una red neuronal frente al ensamblaje

Notebook: [`starter/ejercicio1_red_neuronal.ipynb`](./starter/ejercicio1_red_neuronal.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). Ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división, importadas de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Que la red dejó de entrenar porque llegó al máximo de 200 iteraciones sin haber convergido: el error todavía estaba bajando. Por eso sale peor (en prueba 1.6854 contra 1.2750 del ensamblaje). Con 1000 iteraciones espero que converja y que el error baje bastante, a un nivel parecido o mejor que el del ensamblaje.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

Primera capa: 25 × 64 pesos + 64 sesgos = 1,664. Capa de salida: 64 pesos + 1 sesgo = 65. En total 1,729 números.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

Espero que `distancia_km`, porque la tarifa depende sobre todo de cuánto se recorre. Para explicarle al conductor por qué son 70 dólares me sirve el ensamblaje, porque puedo ver qué columnas pesaron; la red solo tiene pesos que no significan nada por separado.

**Lo que confirmó la salida, TODO por TODO:**

Gradient boosting: error de 1.0705 en entrenamiento y 1.2750 en prueba, 0.28 s. Red neuronal (con 1000 iteraciones): 1.0882 y 1.1695, 2.15 s (sin resolver daba 1.6854 en prueba y el `ConvergenceWarning`). La red aprendió 1,729 parámetros y se detuvo en 692 iteraciones. Columnas más importantes del ensamblaje: `distancia_km` (0.98), `es_tarifa_fija` (0.012) y `es_hora_pico` (0.003).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Notebook: [`starter/ejercicio2_tres_soluciones.ipynb`](./starter/ejercicio2_tres_soluciones.ipynb), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre las celdas hasta la que escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y corre sus celdas y la de la comprobación hasta que todas digan `OK`. Los modelos ya están entrenados en el kernel: no hace falta volver a correr las celdas de arriba. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.8957 | 0.24 s | casi nulo | Sí: importancia de cada columna |
| Red neuronal | 0.9637 | 2.57 s | casi nulo | No: pesos sin significado individual |
| LLM | 1.4355 | no se entrena | 0.73 USD | Da una explicación, pero no se puede comprobar |

Usé Claude Sonnet 5.5. El error es el MAE sobre los 20 viajes. El costo del LLM sale de un prompt de unos 2,900 caracteres (≈725 tokens) por cada 20 viajes y un precio de 2 USD por millón de tokens de entrada (precio que tomé de un sitio de terceros, no de la página del proveedor).

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

No lo corrí otra vez en una conversación nueva, pero espero que no me dé los mismos 20 números: un LLM no es determinista, así que el mismo viaje puede salir con tarifas distintas. Para una app que cobra es un problema: dos pasajeros con el mismo viaje pagarían distinto, no se podría auditar ni responder a un reclamo.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

El viaje 354: de JFK Airport a JFK Airport, de solo 1.01 km. La tarifa real fue 11.46 y el LLM dijo 7 (error de 4.46). Es raro porque la tarifa no sigue la distancia (el aeropuerto tiene tarifa mínima o fija). El ensamblaje sí lo aprende: en entrenamiento hay 7 viajes JFK a JFK con tarifas entre 12.39 y 18.67, y para este viaje predijo 12.74.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

Porque cada predicción del LLM cuesta tokens y habría que armar prompts y pegar las respuestas a mano. Y comparar sobre 20 no es muy confiable: un solo viaje mal predicho (el de 4.46) mueve bastante el error promedio, así que sirve de orientación, no para decidir.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí tu preparación y tu división, y entrene el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

Modelos sobre el conjunto de prueba completo (39,962 filas) y el LLM sobre 10 filas. El costo del LLM es estimado: el prompt de 10 casos y 30 ejemplos pesa unos 2,000 tokens, así que los 213,430 casos de una noche (ver `docs/despliegue.md`) serían del orden de 40 millones de tokens por noche, frente a un costo casi nulo con los otros dos.

| | Bosque (`min_samples_leaf=20`) | Red neuronal (32, 16) | LLM con prompt |
|---|---|---|---|
| RMSE en prueba (39,962 filas) | 2.9332 | 2.9572 | no se midió sobre todas |
| RMSE en las mismas 10 filas | 0.8132 | 0.6805 | 0.7746 |
| Segundos de entrenamiento | 26.1 | 20.6 | no se entrena |
| Parámetros | no aplica | 1,409 | no aplica |
| Costo de predecir | casi nulo | casi nulo | alto, se paga por token en cada predicción |
| Explicabilidad | alta: importancia de cada columna | baja: pesos sin significado individual | da una explicación, pero no se puede comprobar que sea la razón real |

La línea base da 1.0488 en esas 10 filas.

Usaría el bosque. Es el mejor en la prueba completa (2.9332 contra 2.9572 de la red y 3.8348 de la línea base, un 23.5% mejor que la línea base), predice casi gratis y se puede explicar con la importancia de sus columnas. La red queda a 0.024 de RMSE del bosque y entrena un poco más rápido, pero esa diferencia es muy chica y no gano nada a cambio, porque explicarla es mucho más difícil. Con el LLM la comparación es débil: sobre 10 filas, de las cuales 7 vendieron 0 unidades, un solo acierto o error cambia el RMSE, así que su 0.7746 no se puede distinguir del 0.8132 del bosque ni del 0.6805 de la red. Lo que sí es claro es el costo: predecir las 213,430 filas de cada noche con un LLM costaría del orden de 40 millones de tokens, y no hay garantía de que dé el mismo número dos veces. Para un problema con tantos casos y que se repite todos los días, un modelo clásico entrenado una vez es la opción práctica.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

La versión de scikit-learn y de las otras librerías, los hiperparámetros del modelo (n_estimators, learning_rate, etc.), la semilla (`random_state`), cómo dividieron los datos, qué archivo de datos usaron (su huella), qué columnas y qué preparación aplicaron, y la versión del código.
