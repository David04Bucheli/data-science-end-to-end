# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tus 3 hallazgos de la exploración de la Semana 8, Sesión 2: las features de hoy salen de ahí.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio representa algo real que se puede explicar a las personas del negocio (una que solo mejora una métrica puede subir el rendimiento sin que nadie sepa justificarla). El artículo pide mejorar el poder predictivo manteniendo las features explicables.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

Los ratios: 5 transacciones online significan cosas distintas si el cliente tiene 100 en total (usa poco el canal) o 5 en total (es 100 % online). El conocimiento es que importa la proporción, no el conteo

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Convirtiendo cada patrón en una columna. La tarifa fija sale de origen y destino (es_tarifa_fija), y el costo por hora sale de fecha_hora_inicio (hora, es_hora_pico, hora_sin y hora_cos)

## Ejercicio 1: features de fecha y hora

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada, sin los errores confirmados. A partir de `fecha_hora_inicio`, construir hora, día de la semana, fin de semana, hora pico y la hora como ciclo, y comprobar si la hora pico aporta algo a la tarifa.

**Respuesta 1, antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como número, la hora 23 queda a 23 unidades de la 0, la distancia máxima, aunque en el reloj están a una hora. Con hora_sin y hora_cos la distancia 23 a 0 es 0,261, igual que de 0 a 1.

**Respuesta 2:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

Puede traer el efecto del tráfico típico de esa hora: los viajes en hora pico duran en promedio 27,9 min contra 22,7 fuera de ella. No trae lo propio de cada viaje, como un accidente.

**La formulación completa, tal como la resuelvo yo:**



## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/features_de_zona.py`](./plantilla-ejercicio2/features_de_zona.py) y córrela desde su carpeta. Completa los 4 `TODO`.

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

300 viajes tienen es_tarifa_fija en 1, y su tarifa mínima y máxima es la misma: 70,0. Un modelo lineal puede aprender un salto constante con esta columna (70 contra 21,64 de promedio en el resto), pero no que la tarifa deje de depender de la distancia en esos viajes; para eso haría falta una interacción.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea 8 columnas y la ordinal crea 1. Usaría one-hot: la ordinal inventa un orden y unas distancias que no existen (Upper West Side valdría 7 veces Brooklyn, y JFK quedaría entre Harlem y Lower Manhattan), y la regresión lineal las usaría como una escala.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

No existen duracion_min, metodo_pago, tarifa ni propina. Usarlas sería fuga de información: el modelo se vería excelente al entrenar y evaluar, pero en producción esas columnas no existirían.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

Mi modelo predice UNITS, las unidades vendidas de un producto en una tienda en un día, antes de que ocurra el día. Ya existen el calendario, los eventos, SNAP, el precio, la tienda y la categoría. No existen REVENUE (sale de UNITS) ni IS_SALES_SPIKE e IS_SUSPECTED_STOCKOUT_* (se marcan mirando las ventas ya ocurridas)

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

Revisar código. 

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

en_descuento: sale del conocimiento del dominio: las promociones suben las ventas. Evidencia: UNITS promedia 1,76 con descuento contra 1,06 sin descuento. Existe al predecir porque el precio se fija de antemano. Es binaria, no necesita codificación.

**Feature 2:**

snap_fin_de_semana: combina SNAP (beneficio que se cobra en días fijos del mes) con el fin de semana, cuando se compra más. Evidencia: UNITS promedia 1,45 en esos días contra 1,08 en el resto. El calendario se conoce de antemano. También es binaria; para CAT_ID o STORE_ID usaría one-hot si el modelo es lineal.

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Si uso todos los viajes, la prueba entra al cálculo de la media y hay fuga de información. Con muchos datos el número cambia poco, pero la evaluación deja de simular producción.
