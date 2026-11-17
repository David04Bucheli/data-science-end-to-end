# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los laboratorios de hoy son notebooks de Jupyter, como el diagnóstico de la Semana 4: cada gráfico aparece debajo de la celda que lo dibuja. Una exploración se lee para decidir qué hacer con los datos; por eso va en un notebook y no en un script.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

Responde qué datos tengo (filas, columnas, tipos, faltantes, duplicados), si hay anomalías o valores fuera de rango, qué preparación necesitan (imputar, codificar, escalar, corregir distribuciones sesgadas), qué problemas de calidad hay (desbalance, duplicados, posibles fugas) y qué variables se relacionan con lo que quiero predecir.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Para una sola variable: estadísticas como `describe()` e histogramas (barras si es categórica). Para la relación entre dos: gráficos por pares y correlaciones, con una matriz o mapa de calor.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

El modelo trataría esas 15 tarifas como reales. Jalarían el promedio hacia arriba y, si el error penaliza más los errores grandes, el modelo intentaría ajustarlas y se distorsionaría todo lo demás. Mirando solo el resultado final casi no lo notaría: vería un error alto o raro y no sabría por qué. Se ve fácil con `describe()` o un histograma.

## Ejercicio 1: resumen, distribución y atípicos

Notebook: [`starter/ejercicio1_explorar.ipynb`](./starter/ejercicio1_explorar.ipynb), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York, con fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Ya lee el archivo y arma los gráficos; faltan 4 `TODO`. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

La tarifa máxima, 999.99, que está muy lejos del percentil 75 (32.55), y la duración mínima, −74 minutos, que es imposible.

**Respuesta 2 (TODO 2):** mira el histograma después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

Porque unos pocos valores de 999.99 estiran el eje hasta casi 1000, y el resto de los viajes (casi todos por debajo de 60) caben en una o dos barras pegadas a la izquierda.

**Respuesta 3 (TODO 3 y 4):** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No. La regla del IQR marca lo que está lejos del resto, no lo que está mal. Lo averiguaría viendo qué valores se repiten y qué viajes son (zona, distancia). Salió que de 334 atípicos, 302 tienen tarifa exacta de 70.00 (y no son errores) y solo 15 son errores de captura.

**Lo que confirmó la salida, TODO por TODO:**

Límite superior: 60.40 (Q1 = 13.99, Q3 = 32.55, IQR = 18.56). Viajes sobre el límite: 334. De esos, 302 tienen tarifa exacta de 70.00, 15 tienen 999.99 y los otros 17 son tarifas altas normales. Además hay 10 duraciones negativas.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Notebook: [`starter/ejercicio2_relaciones.ipynb`](./starter/ejercicio2_relaciones.ipynb), sobre el mismo `viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0. La última celda imprime una comprobación por TODO: completa los 5 hasta que todas digan `OK`.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

Además de la tendencia (más kilómetros, más tarifa; correlación de 0.96), hay una fila horizontal de viajes que cuestan 70 dólares sin importar la distancia (300 viajes, los de JFK a Manhattan y al revés), y 25 viajes con distancia 0 pero tarifa de entre 7.23 y 70.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Los 888 viajes en efectivo tienen propina de 0, y los 2027 viajes con tarjeta tienen una propina promedio de 4.73 y ninguno con 0. Creo que describe al sistema: la propina en efectivo no se registra, no es que nadie dé propina en efectivo.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La duración solo se conoce cuando termina el viaje, no al pedir el taxi. Usarla sería una fuga, aunque su correlación con la tarifa sea tan alta como la de la distancia (0.96).

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar. Copia la plantilla [`plantilla-eda/explorar_proyecto.ipynb`](./plantilla-eda/explorar_proyecto.ipynb) a `psets/proyecto/modelado/`: el notebook es tu entregable, con el código, lo que mostró y lo que concluyes en el mismo lugar.

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3: filas, columnas, faltantes por columna, la distribución del objetivo y cuántos atípicos marca el IQR.

### Paso 2: relaciones (10 minutos)

Completa el paso 4: la relación entre tu objetivo y la variable que esperas que más influya.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

En el paso 5 del notebook, para cada hallazgo: una celda de código con el número o el gráfico que lo confirma, y debajo, en la celda de texto, qué encontraste y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar). Antes de hacer commit, reinicia el kernel y corre todo.

**Ruta de tu notebook de exploración en tu fork:**

`psets/proyecto/modelado/explorar_proyecto.ipynb`

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Las convertiría en columnas nuevas: `es_tarifa_fija`, que vale 1 si el viaje va de JFK a una zona de Manhattan o al revés, y la hora (y si es hora pico) sacada de `fecha_hora_inicio`.
