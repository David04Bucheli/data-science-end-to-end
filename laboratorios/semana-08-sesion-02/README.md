# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas ya lo tienes desde la Semana 4. Instala matplotlib:

```
pip install matplotlib
```

Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. También puedes correr el mismo código en un notebook de Jupyter, donde el gráfico aparece debajo de la celda.

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

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv) tiene 3000 viajes en taxi de Nueva York: fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Primera mirada: qué contiene, cómo se distribuye la tarifa y qué valores están fuera de lo esperable.

**Respuesta 1, antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

1. tarifa máximo = 999.99: está muy lejos del percentil 75 (32.55) y es un número "redondo" típico de un código de error.
2. duracion_min mínimo = -74: una duración negativa es imposible.

**Respuesta 2:** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No, no todos son errores. El IQR solo marca lo que está lejos de la mayoría, no lo que es incorrecto.

Para averiguarlo puedo revisar cuáles valores se repiten exactamente, porque un valor repetido suele ser un código o una regla de negocio; cruzar los atípicos con otras columnas para ver si tienen una explicación; y validar con el dueño de los datos o una regla de negocio. Solo se eliminan los que se confirman como error.

**La formulación completa, tal como la resuelvo yo:**

Script: starter/EDA.py. Pasos que hace:

1. Carga el CSV con pd.read_csv, y revisa su estructura con info(), dtypes, isna().sum() y duplicated().sum().
2. describe() para las estadísticas de las columnas numéricas.
3. Histograma (50 bins) y boxplot de tarifa, guardados en figuras/.
4. Atípicos con la regla del IQR. Separa atipicos y sin_atipicos con query().
5.Dispersión de tarifa contra distancia, con y sin atípicos.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/explorar_relaciones.py`](./plantilla-ejercicio2/explorar_relaciones.py) y córrela desde su carpeta (`cd plantilla-ejercicio2`, `python3 explorar_relaciones.py`): lee el mismo `starter/viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0.

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

_(filas, columnas, faltantes por columna, y cuántos atípicos marcó el IQR en tu objetivo)_

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

_(qué gráfico o consulta hiciste, y qué mostró)_

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1:**

_(tu hallazgo)_

**Hallazgo 2:**

_(tu hallazgo)_

**Hallazgo 3:**

_(tu hallazgo)_

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Hay que crear columnas nuevas (feature engineering) a partir de las que ya existen, como una tarifa fija de JFK o la hora del día.