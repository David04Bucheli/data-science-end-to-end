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

Ten a mano tu notebook de features de la sesión pasada. Los tres ejercicios de hoy son scripts, no notebooks: un pipeline de preparación es código de producción, que tiene que correr igual cada vez y que otros archivos importan. En el Ejercicio 3, la preparación de tu proyecto pasa del notebook donde la probaste a ese script.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

Es cuando información del conjunto de prueba, o que no existirá al predecir, entra al entrenamiento. El modelo se ve excelente en la evaluación porque ya "vio" parte de lo que se evalúa, y en producción esa información no está y falla.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

La fuga por la variable objetivo (target leakage): la duración del viaje solo se conoce al terminar, cuando la tarifa ya está determinada, y no existe al pedir el taxi. Un modelo que la usara se vería muy bien en la prueba y fallaría en producción.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Sí cambia: si uso todos, la media incluye viajes de prueba y es una fuga. El número cambia poco, pero la evaluación deja de ser honesta.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

En el `fit_transform` del imputador y del escalador, que se hace sobre todo `X` antes de dividir. Lo delata que el escalador vio 2975 filas (todas) y no las 2380 de entrenamiento.

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

Nada útil: mide el error sobre las mismas filas con que entrenó, así que no es un error de prueba y sale optimista. No dice cómo se comporta con viajes nuevos.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Esperaba un cambio muy pequeño, porque con dos columnas y un modelo lineal la media y la desviación casi no cambian. Salió 2.5375 antes y 2.5373 después. Que casi no cambie no significa que no hubiera fuga: el proceso sigue mal y con otros datos o modelos podría pesar bastante.

**Lo que confirmó la salida, TODO por TODO:**

Versión del compañero: el escalador vio 2975 filas, media de distancia 8.5034, error 2.5375. Versión correcta: el escalador vio 2380 filas, media 8.5576, 595 filas de prueba, error 2.5373.

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

25 columnas: 2 numéricas (distancia y pasajeros, escaladas) + 18 de one-hot (8 de `zona_origen`, 8 de `zona_destino` y 2 de `metodo_pago`) + 5 que pasan sin cambios (`YA_LISTAS`).

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de entrenamiento (6.94 y 1.0); las de todo el dataset son 6.93 y 1.0. Demuestra que el imputador aprendió solo de entrenamiento, aunque la diferencia sea de una centésima.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

Con `handle_unknown="ignore"` las columnas de `zona_origen` quedan todas en 0 y el viaje no da error. Con `pd.get_dummies` por separado solo se crearía `origen_Bronx`, con columnas distintas a las del entrenamiento, y el modelo fallaría o recibiría las columnas desalineadas.

## Ejercicio 3: el pipeline de tu proyecto, como script

Copia la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py) a `psets/proyecto/modelado/`, al lado de tu notebook de features. Desde la próxima sesión, tus notebooks de modelos importan este archivo en vez de copiar su código:

```python
from pipeline_proyecto import cargar_y_dividir, construir_preparacion
```

### Paso 1: pasa tus features a `agregar_features` (5 minutos)

Copia de tu notebook de la sesión pasada solo las líneas que construyen tus 2 features, dentro de la función `agregar_features`. Sin gráficos, sin `print` y sin las celdas que usaste para probar ideas: esas se quedan en el notebook, como evidencia de por qué elegiste cada feature.

### Paso 2: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

- **Objetivo:** `UNITS`.
- **Numéricas:** `PRECIO_LOG` (el log de `SELL_PRICE`) y `WDAY`.
- **Categóricas:** `DEPT_ID` y `STORE_ID`, con one-hot. `ITEM_ID` va aparte con `TargetEncoder`, porque tiene 3049 valores y casi todos aparecen muy pocas veces.
- **Ya listas:** `SEMANA_SIN`, `SEMANA_COS`, `IS_SNAP`, `IS_WEEKEND`, `IS_EVENT_DAY` e `IS_PRE_LAUNCH`.
- **Fuera:** `REVENUE`, `IS_SALES_SPIKE`, `IS_SUSPECTED_STOCKOUT_CONSERVATIVE` e `IS_SUSPECTED_STOCKOUT_NB`, porque se calculan con las ventas del propio día; y las fechas y los identificadores que no aportan.

### Paso 3: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha dentro de `cargar_y_dividir`; si no, al azar. Justifica.

Sí, el modelo predice ventas futuras a partir de filas con fecha, así que divido por fecha: entrenamiento antes del 30 de abril de 2015 (159,871 filas) y prueba desde esa fecha (39,962 filas, un 20%). Una división al azar mezclaría días futuros en el entrenamiento, y el modelo parecería mejor de lo que va a ser en producción.

### Paso 4: corre el script (10 minutos)

Desde una terminal, en `psets/proyecto/modelado/`:

```
python3 pipeline_proyecto.py
```

Si corre de principio a fin en una terminal nueva, sin el notebook abierto, otro código ya lo puede importar. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

```
Entrenamiento: (159871, 26) Prueba: (39962, 26)
['numericas__PRECIO_LOG', 'numericas__WDAY', 'categoricas__DEPT_ID_FOODS_1', ..., 'categoricas__STORE_ID_WI_3', 'producto__ITEM_ID', 'ya_listas__SEMANA_SIN', 'ya_listas__SEMANA_COS', 'ya_listas__IS_SNAP', 'ya_listas__IS_WEEKEND', 'ya_listas__IS_EVENT_DAY', 'ya_listas__IS_PRE_LAUNCH']
```
Son 26 columnas: 2 numéricas, 7 de departamento y 10 de tienda (one-hot), 1 del producto y 6 ya listas.

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: memorizó el entrenamiento, ruido incluido, y no aprendió el patrón general, así que falla con datos nuevos.
