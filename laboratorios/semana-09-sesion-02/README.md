# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano el script de features de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

Es cuando información que el modelo no debería tener (del conjunto de prueba, o que solo se conoce después del evento) se cuela en el entrenamiento. Las métricas salen infladas, y en producción esa información no existe, así que el modelo falla.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

La de variables que no están disponibles al momento de predecir: la duración solo se conoce cuando el viaje termina.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Con todos, el conjunto de prueba influye en el escalado, que es una fuga. Con muchos datos el cambio es pequeño, pero la evaluación deja de ser honesta.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

En la línea que imputa y escala con todo el dataset, antes de dividir en entrenamiento y prueba.

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

Esperaría un cambio muy pequeño (al correrlo fue de 0,0001). Que casi no cambie no significa que no hubiera fuga: con casi 3.000 filas, el 20 % de prueba mueve muy poco la mediana y la media, pero el error de procedimiento sigue ahí.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

_(tu respuesta)_

**Lo que confirmó la salida, TODO por TODO:**

_(anota las filas que vio cada escalador y los dos errores)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Quedan 25 columnas: 2 numéricas (distancia y pasajeros) + 18 del one-hot (8 de zona_origen, 8 de zona_destino y 2 de metodo_pago) + 5 que pasan sin cambios.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de entrenamiento (6,94 y 1), no con las de todo el dataset (6,93 y 1). Eso demuestra que el imputador aprendió solo con entrenamiento.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

Ninguna columna de zona_origen queda en 1: con handle_unknown="ignore" toda la fila queda en ceros y no hay error. Con pd.get_dummies sobre ese viaje solo saldría una columna origen_Bronx, y no coincidiría con las columnas del modelo.

## Ejercicio 3: el pipeline de tu proyecto

Usa tu archivo con las 2 features de la sesión pasada y la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

### Paso 1: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

- Objetivo: UNITS
- Numéricas: SELL_PRICE, precio_relativo
- Categóricas: CAT_ID, STORE_ID
- Ya listas: en_descuento, snap_fin_de_semana
- Fuera: REVENUE, IS_SALES_SPIKE e IS_SUSPECTED_STOCKOUT_*, que no existen al predecir.

### Paso 2: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha; si no, al azar. Justifica.

Divido por fecha. El modelo predice ventas futuras a partir de filas con fecha, así que entreno con el 80 % más antiguo (hasta el 29-04-2015) y pruebo con el 20 % más reciente. Un corte al azar mezclaría días futuros en el entrenamiento.

### Paso 3: construye y corre el pipeline (15 minutos)

Completa la plantilla y córrela. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

Fecha de corte: 2015-04-29
Entrenamiento: (159871, 17) Prueba: (39962, 17)
['numericas__SELL_PRICE', 'numericas__precio_relativo', 'categoricas__CAT_ID_FOODS', 'categoricas__CAT_ID_HOBBIES', 'categoricas__CAT_ID_HOUSEHOLD', 'categoricas__STORE_ID_CA_1', 'categoricas__STORE_ID_CA_2', 'categoricas__STORE_ID_CA_3', 'categoricas__STORE_ID_CA_4', 'categoricas__STORE_ID_TX_1', 'categoricas__STORE_ID_TX_2', 'categoricas__STORE_ID_TX_3', 'categoricas__STORE_ID_WI_1', 'categoricas__STORE_ID_WI_2', 'categoricas__STORE_ID_WI_3', 'ya_listas__en_descuento', 'ya_listas__snap_fin_de_semana']

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Está sobreajustando (overfitting): memorizó el entrenamiento, incluido el ruido, y no generaliza a datos nuevos.
