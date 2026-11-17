# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

Código, datos y modelo. El código se versiona con Git; para los datos y el modelo el autor usa herramientas aparte (creo que algo tipo DVC para los datos y un registro de modelos), porque Git no está hecho para eso.

**¿Por qué no alcanza con Git para versionar los datos?**

Porque los datos suelen ser archivos grandes o binarios: cada cambio guarda una copia completa, el repositorio se infla y se vuelve lento, y Git no puede mostrar qué cambió dentro de un archivo así. Por eso conviene guardar los datos aparte y en Git solo una referencia o huella.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Los hiperparámetros de cada corrida (n_estimators, learning_rate, semilla), cómo dividieron los datos (la semilla de la división), qué archivo de datos usaron (su huella) y las versiones de las librerías y del código.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Ninguno tiene la razón: el gradient boosting con división 42 dio 1.2750 y el del compañero 1.1860. La salida solo muestra un error por modelo; no dice con qué semilla ni con qué datos se calculó, así que no puedo saber por qué difieren ni cuál es mejor.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La semilla de la división de los datos (42 contra 7). Esperaba ver `semilla_division` distinta en la tabla y lo demás igual (misma huella de datos y mismo `n_estimators`).

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Para poder volver a cargarlo y usarlo o reproducir su error sin tener que reentrenar. Con solo los parámetros y las métricas no tengo el modelo, y reentrenar podría dar otro resultado.

**Lo que mostró la tabla del paso 5:**

Bosque aleatorio (división 42): error de prueba 1.2993. Gradient boosting (división 42): 1.2750. Gradient boosting del compañero (división 7): 1.1860. La huella de los datos fue la misma en las tres (`4800204d50c0`), o sea que usaron el mismo archivo y lo único que cambió fue la semilla.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

Quedó primero la red neuronal con 1.1695 de error en prueba; le gana al gradient boosting (1.2750) por 0.1055, un 8% aproximadamente. Después vienen el bosque (1.2993), ridge (2.5565) y la línea base (12.5308).

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Que es el mismo modelo guardado en el registro y los mismos datos de prueba, así que da el mismo error (reproduje 1.169505). No garantiza que funcione igual con datos nuevos ni con otra versión de las librerías: solo confirma el número sobre esos 595 viajes.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Por la huella de los datos: esa corrida tendría un `huella_datos` distinto al `4800204d50c0` de las anteriores, y seguramente también otro error.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

En la interfaz puedo comparar las corridas lado a lado con sus parámetros, métricas y tiempos juntos, sin copiarlos a mano. De la red guarda sus hiperparámetros (capas, iteraciones, semilla), los segundos y el modelo, y todo con fecha y con la huella de los datos, cosas que en una tabla escrita a mano se pierden o se copian mal.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

Elegí el bosque con `min_samples_leaf=20`, `run_id` `2abf63a785674d0db32122197a396661`. Del registro de MLflow:

| Corrida | RMSE entrenamiento | RMSE prueba | Segundos |
|---|---|---|---|
| bosque `min_samples_leaf=20` (final) | 2.8233 | 2.9332 | 26.1 |
| red neuronal (32, 16) | 2.8186 | 2.9572 | 20.5 |
| ridge `alpha=10` | 3.0879 | 3.0188 | 0.4 |
| línea base | 3.8749 | 3.8348 | 0.3 |

Es la de menor error en prueba. Cargué su modelo desde `runs:/2abf63a785674d0db32122197a396661/modelo` y reprodujo el error registrado (2.933216). La ventaja sobre la red es de 0.024, así que la decisión es ajustada; la Semana 11, Sesión 2 revisa si esa diferencia es mayor que la variación entre pliegues.

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita el mismo archivo de datos, el mismo código de preparación, la misma división y las mismas semillas, y las mismas versiones de las librerías.

**Quedó registrado en MLflow:** los hiperparámetros de cada modelo, la fecha de corte (30 de abril de 2015), la semilla (42), la huella SHA-256 del archivo de datos (para comprobar que es el mismo archivo), las métricas, los segundos y el pipeline entrenado como artefacto.

**No quedó registrado:** el archivo de datos en sí (no está en el repositorio, solo su huella), la versión del código de `pipeline_proyecto.py` con que se corrió (no registré el commit de git), y las versiones de scikit-learn, pandas y numpy. Con otra versión de scikit-learn, el mismo código puede dar un número ligeramente distinto. Las semillas sí están fijas, incluida la del `TargetEncoder`, que sin semilla daba un error distinto en cada corrida.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Cambiaría bastante. Lo vi en el registro: el mismo gradient boosting dio 1.1860 con la división 7 y 1.2750 con la 42, una diferencia de casi 0.09, parecida a la que separa a varios modelos.
