# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tu notebook de exploración de la Semana 8, Sesión 2: las features de hoy salen de tus 3 hallazgos.

Los laboratorios de hoy siguen siendo notebooks: diseñar una feature es probar varias ideas y mirar cuál aporta. La próxima sesión, las features que se queden pasan a un script, que es lo que corre en producción.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio traduce algo que se puede explicar con una razón del mundo real. Una que solo mejora una métrica puede predecir bien, pero nadie sabe por qué y se puede romper si cambia el contexto.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

El de la proporción de transacciones en línea: dos clientes con 5 transacciones en línea pueden ser muy distintos si uno hizo 6 en total y el otro 100. El conocimiento es que importa qué tan en línea es el cliente, y se vuelve una columna que divide las transacciones en línea entre el total.

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Creando columnas que lo expresen: `es_tarifa_fija` (1 si el viaje es JFK-Manhattan) y, de la fecha y hora, la hora, si es hora pico y si es fin de semana. El modelo solo ve números, así que lo que sé del negocio tiene que ir en una columna.

## Ejercicio 1: features de fecha y hora

Notebook: [`starter/ejercicio1_features_tiempo.ipynb`](./starter/ejercicio1_features_tiempo.ipynb), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. Ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como número, la hora 23 queda a 23 de la hora 0, aunque en el reloj estén a una hora de distancia. Con seno y coseno la distancia es 0.26, igual que entre las 11 y las 12.

**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?

Más alta: en hora pico la mediana sube de 3.22 a 3.52 dólares por kilómetro. Aunque la tarifa dependa de la distancia y de la duración, en hora pico hay tráfico, el mismo trayecto dura más (mediana de 3.57 minutos por km contra 2.73) y cobra más.

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

Parte de lo que dice la duración: que a esas horas el tráfico hace los viajes más lentos y más caros por kilómetro. No da los minutos exactos de ese viaje, pero sí la tendencia, y es algo que sí se sabe al pedir el taxi.

**Lo que confirmó la salida, TODO por TODO:**

Tarifa por km (mediana): 3.22 fuera de hora pico (1842 viajes) y 3.52 en hora pico (812 viajes). Distancias del Paso 5: horas 23 y 0, como número 23 y como ciclo 0.26; horas 11 y 12, como número 1 y como ciclo 0.26.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Notebook: [`starter/ejercicio2_features_zona.ipynb`](./starter/ejercicio2_features_zona.ipynb), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. La última celda imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

300 viajes tienen `es_tarifa_fija` en 1, y todos tienen tarifa mínima y máxima de 70.00. Un modelo lineal con esa columna puede aprender un salto que lleve la tarifa a 70 sin depender de la distancia; sin ella, una recta de la distancia no puede ajustar un precio fijo.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea 8 columnas; la ordinal, una sola (de 0 a 7). Con regresión lineal usaría one-hot, porque la ordinal inventa un orden y una distancia entre zonas que no existen: el modelo creería que Queens "vale" más que Harlem.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

La duración (`duracion_min`) y la propina, porque se conocen cuando termina el viaje, no al pedir el taxi; la tarifa es lo que quiero predecir. Si las usara para entrenar, el modelo se vería muy bien en la prueba, pero en producción esas columnas no existirían: es una fuga.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de tu exploración de la Semana 8 y copia la plantilla [`plantilla-features/features_proyecto.ipynb`](./plantilla-features/features_proyecto.ipynb) a `psets/proyecto/modelado/`, al lado de tu notebook de exploración. Todo lo que pide este ejercicio queda en el notebook: el código de cada feature, la evidencia y la justificación en las celdas de texto.

### Paso 1: el momento de la predicción (5 minutos)

En la primera celda de texto, escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

### Paso 2: construye las 2 features (15 minutos)

Crea 2 columnas nuevas, derivadas de las existentes, cada una en su celda y con su evidencia (una correlación, una diferencia de promedios). Al menos una tiene que salir de uno de tus hallazgos de la exploración. Si pruebas una idea que no aporta, deja la celda y escribe por qué la descartas: también es evidencia.

### Paso 3: justifica cada feature (5 minutos)

Debajo de cada feature: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta, y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar. Antes de hacer commit, reinicia el kernel y corre todo.

**Ruta de tu notebook de features en tu fork:**

`psets/proyecto/modelado/features_proyecto.ipynb`

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Sí cambia: con todos, la media incluye viajes de prueba, que es una fuga. El número cambia poco (en la sesión siguiente 8.5576 contra 8.5034), pero la evaluación deja de ser honesta.
