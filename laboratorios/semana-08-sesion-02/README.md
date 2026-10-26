# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. Ábrelas desde el explorador de archivos de VS Code. También puedes correr el mismo código en un notebook de Jupyter (en el Codespace, crea un archivo `.ipynb` y elige el kernel de Python 3.11), donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

Según el artículo, el EDA es la tarea más importante al inicio de un proyecto de ciencia de datos y responde cuatro preguntas:

- ¿Con qué estoy trabajando? Cuántas filas y variables hay, de qué tipo son, qué porcentaje de datos faltan y cuántas filas están duplicadas.
- ¿Qué variables son relevantes y qué preparación necesitan? Si hay valores fuera de rango, asimetrías, o si hay que estandarizar o codificar.
- ¿Qué relaciones existen entre variables? Correlaciones, redundancias, qué variables discriminan mejor al objetivo y posibles fugas de datos (data leakage).
- ¿Qué problemas de calidad tienen los datos? Faltantes, desbalance, columnas constantes, duplicados, ruido y exceso de ceros.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

- Una sola variable (univariado): estadísticos de resumen (media, desviación estándar, asimetría, curtosis, cuantiles) con df.describe(), y los histogramas para ver la distribución de las numéricas. Para las categóricas, tablas de frecuencia y gráficos de barras.
- Relación entre variables (multivariado): la matriz de correlación, mostrada como mapa de calor (heatmap). El artículo menciona Spearman para numérica–numérica y V de Cramér para las que involucran categóricas, y propone ydata-profiling para automatizarlo.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

Con 15 de 3000 viajes en 999.99 el modelo tiene que ajustarse a valores que son mucho mayores a los de la tarifa mediana. Un modelo que minimiza el error cuadrático hace un esfuerzo desproporcionado por acercarse a esos puntos, porque el error se eleva al cuadrado, y eso distorsiona las predicciones de todos los viajes normales.

Mirando solo el resultado final probablemente no lo notaría, vería un RMSE alto o inestable y podría atribuirlo a que el modelo es malo, a falta de variables o a ruido, sin sospechar que hay un error de captura. Por eso hay que explorar los datos antes de entrenar.

## Ejercicio 1: resumen, distribución y atípicos

Archivo: [`starter/ejercicio1_explorar.py`](./starter/ejercicio1_explorar.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York, con fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. El script ya lee el archivo, arma los gráficos y los guarda en `figuras/`; faltan 4 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-08-sesion-02/starter
python3 ejercicio1_explorar.py
```

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

1. tarifa máximo = 999.99: está muy lejos del percentil 75 (32.55) y es un número "redondo" típico de un código de error.
2. duracion_min mínimo = -74: una duración negativa es imposible.

**Respuesta 2 (TODO 2):** abre `figuras/hist_tarifa.png` después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

No, no todos son errores. El IQR solo marca lo que está lejos de la mayoría, no lo que es incorrecto.

Para averiguarlo puedo revisar cuáles valores se repiten exactamente, porque un valor repetido suele ser un código o una regla de negocio; cruzar los atípicos con otras columnas para ver si tienen una explicación; y validar con el dueño de los datos o una regla de negocio. Solo se eliminan los que se confirman como error.

**Respuesta 3 (TODO 3 y 4):** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

<<<<<<< HEAD
Script: starter/EDA.py. Pasos que hace:

1. Carga el CSV con pd.read_csv, y revisa su estructura con info(), dtypes, isna().sum() y duplicated().sum().
2. describe() para las estadísticas de las columnas numéricas.
3. Histograma (50 bins) y boxplot de tarifa, guardados en figuras/.
4. Atípicos con la regla del IQR. Separa atipicos y sin_atipicos con query().
5.Dispersión de tarifa contra distancia, con y sin atípicos.
=======
_(tu respuesta)_

**Lo que confirmó la salida, TODO por TODO:**

_(anota el límite superior, cuántos atípicos marcó y cómo se reparten)_
>>>>>>> f9c474b3a2c2e5c9177d6177a998d779a6788b94

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Archivo: [`starter/ejercicio2_relaciones.py`](./starter/ejercicio2_relaciones.py), sobre el mismo `viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0. Al final, el script imprime una comprobación por TODO: completa los 5 hasta que todas digan `OK`.

```
python3 ejercicio2_relaciones.py
```

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

Además de la tendencia lineal (correlación 0.96), el gráfico muestra una línea horizontal en 70 USD. Hay 300 viajes con tarifa exactamente 70.0, y todos tienen a JFK Airport como origen o destino. Es una tarifa plana: sin importar la distancia, que va de 13 a 45 km, la tarifa no sube. De los 373 viajes de JFK, 300 (80%) tienen esa tarifa. También hay 25 viajes con distancia alrededor de 0 y tarifa positiva (mediana 25 USD), lo que probablemente es un GPS sin registrar la distancia.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Creo que describe al sistema que registra los datos y no a los pasajeros. Una división tan perfecta (100% contra 0%) no es comportamiento humano. Lo más probable es que la propina en efectivo no se capture, y por eso no se puede concluir que quienes pagan en efectivo no dejan propina.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La correlación de duración con tarifa es 0.956, casi igual a la de distancia. Pero la duración solo se conoce cuando el viaje termina. Si el modelo estima la tarifa en el momento del pedido, usarla sería data leakage. La distancia sí se puede estimar con origen y destino. Lo mismo aplica a la propina (correlación 0.63), que ocurre después del viaje. 

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar, y la plantilla [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py).

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.

Trabajé con una muestra aleatoria del CSV completo (~59 millones de filas), así que todas las cifras son estimaciones. La muestra tiene 199,833 filas y 40 columnas. El objetivo es UNITS.

Faltantes por columna:

- SELL_PRICE y REVENUE: 41,600 cada una (20.8%).
- EVENT_NAME_1 y EVENT_TYPE_1: 183,611 (91.9%). EVENT_NAME_2 y EVENT_TYPE_2: 199,414 (99.8%). Son nulos porque casi ningún día tiene evento, no por error de captura.
- DATE_KEY_LY_364: 37,620 (18.8%). Corresponde a los días del primer año, que no tienen un día equivalente un año antes (364 de 1,941 días es 18.8%).
- Las otras 33 columnas no tienen faltantes.

Atípicos: la regla del IQR marcó 23,716 filas (11.9%) en UNITS. Con Q1 = 0 y Q3 = 1 el IQR vale 1 y el límite superior es 2.5, así que marca toda venta de 3 unidades o más.

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

La variable de entrada que esperaba que más influyera es SELL_PRICE: a mayor precio, esperaba menos unidades vendidas.

Consulta 1: datos.corr(numeric_only=True)["UNITS"].sort_values(). La correlación de SELL_PRICE con UNITS es -0.15: el signo es el esperado, pero es débil. Las variables de calendario y eventos (mes, trimestre, día de la semana, IS_EVENT_DAY, IS_SNAP, etc.) quedan entre -0.03 y 0.04, es decir casi sin relación lineal. La única correlación alta es REVENUE (0.71), y no se puede usar como entrada (Hallazgo 2).

Consulta 2: datos["UNITS"].value_counts().head() dio 0: 135,873 (68.0%), 1: 26,715, 2: 13,529, 3: 7,272 y 4: 4,483. El objetivo está dominado por ceros, y una correlación lineal es una mala medida para un objetivo así. Por eso las correlaciones salen tan bajas aunque haya relaciones reales.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1:**

UNITS es un conteo con 68% de ceros y cola larga, y el IQR marca como atípico algo que no es un error.

La mediana es 0, la media 1.11, la desviación estándar 3.70 y el máximo 301. 135,873 de 199,833 filas (68.0%) tienen 0 unidades. El IQR marca 23,716 filas (11.9%), pero solo porque Q1 y Q3 son 0 y 1, así que toda venta de 3 unidades o más queda como "atípica". Son ventas normales de productos que se venden más, no errores de captura

Qué cambia en el modelo: no quito esas 23,716 filas, porque eliminaría justo las ventas que el modelo tiene que aprender a predecir. Solo reviso el máximo (301) contra su producto, tienda y fecha para ver si es un pico real.

**Hallazgo 2:**

Hay columnas que contienen la respuesta o se calculan con ella.

REVENUE tiene correlación 0.71 con UNITS, la más alta con diferencia, y el ingreso se calcula como unidades por precio. Tiene exactamente el mismo número de faltantes que SELL_PRICE (41,600), lo que concuerda con REVENUE = UNITS * SELL_PRICE.

Qué cambia en el modelo: saco esas cuatro columnas de las variables de entrada, porque al momento de predecir no se conocen.

**Hallazgo 3:**

SELL_PRICE tiene 20.8% de faltantes y no parecen al azar.

41,600 filas sin precio. IS_PRE_LAUNCH tiene casi la misma correlación negativa con UNITS (-0.15) que SELL_PRICE (-0.15), lo que sugiere que el precio falta cuando el producto todavía no se vendía en esa tienda y, en esas filas, UNITS sería 0.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Hay que crear columnas nuevas (feature engineering) a partir de las que ya existen, como una tarifa fija de JFK o la hora del día.