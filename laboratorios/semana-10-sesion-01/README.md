# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el script `pipeline_proyecto.py` de tu proyecto, de la Semana 9, Sesión 2.

Los laboratorios de hoy son notebooks: entrenar y comparar modelos es experimentar, y en un notebook vuelves a correr solo el modelo que cambiaste, sin reentrenar los demás. La preparación no se copia en el notebook: se importa del script, para que todos los modelos usen exactamente la misma.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje es un modelo que combina las predicciones de varios modelos más simples para dar una sola. En bagging se entrenan muchos modelos en paralelo, cada uno con una muestra distinta de los datos (con reemplazo), y se promedian o votan sus resultados; los modelos no se influyen entre sí. En boosting los modelos se entrenan en secuencia: cada uno trata de corregir los errores que dejó el anterior, y la predicción final es la suma ponderada de todos.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Cada árbol profundo se equivoca de una forma distinta porque se ajusta a las particularidades de su muestra. Al promediar, esos errores no van todos en la misma dirección y se compensan en buena parte, mientras que lo que sí tienen en común (la señal real) se mantiene. Por eso la varianza del promedio es menor que la de un árbol solo, siempre que los árboles no estén demasiado correlacionados entre sí.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: memorizó el entrenamiento, ruido incluido, en lugar de aprender el patrón general. Por eso funciona casi perfecto con datos que ya vio y mal con datos nuevos. La brecha entre los dos errores es la señal.

## Ejercicio 1: línea base, un árbol y un bosque

Notebook: [`starter/ejercicio1_linea_base_y_ensamblaje.ipynb`](./starter/ejercicio1_linea_base_y_ensamblaje.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. La preparación de la Semana 9 está en [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py) y el notebook la importa. Ya trae una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos, cada uno en su celda. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Espero que se equivoque bastante, del orden de lo que varía la tarifa alrededor de su mediana: más de 10 dólares en promedio. Sirve como piso de comparación: si un modelo con columnas no le gana de forma clara, esas columnas no están aportando nada. También sirve para ponerle escala al error de los demás (un MAE de 1.3 se ve bien o mal según contra qué se compare).

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

En entrenamiento, casi cero, porque con una hoja por viaje el árbol puede devolver la tarifa exacta de cada uno que ya vio. En prueba espero que sea claramente peor que en entrenamiento, porque esas hojas aprendieron casos particulares, no reglas que sirvan para viajes nuevos.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad espero que la brecha se achique bastante: el error de entrenamiento sube un poco y el de prueba baja. Con 200 árboles promediados espero algo parecido, una brecha mucho menor que la del árbol libre, porque el promedio compensa los errores de cada árbol.

**Lo que confirmó la tabla, TODO por TODO:**

| Modelo | Error entrenamiento | Error prueba | Brecha |
|---|---|---|---|
| línea base | 13.1998 | 12.5308 | -0.6690 |
| árbol sin límite | 0.0029 | 1.7143 | 1.7114 |
| árbol max_depth=8 | 1.0098 | 1.2773 | 0.2675 |
| bosque aleatorio (200 árboles) | 0.4744 | 1.2993 | 0.8250 |

(Error absoluto medio, en las unidades de la tarifa.) Se cumplió lo esperado. La línea base se equivoca unos 12.5 en prueba. El árbol sin límite memoriza (0.0029 en entrenamiento) y pierde casi 1.7 en prueba. Con `max_depth=8` la brecha baja a 0.27 y además da el mejor error en prueba de los cuatro (1.2773). El bosque tiene una brecha de 0.83, menor que la del árbol libre pero mayor que la del árbol limitado; su error en prueba (1.2993) quedó muy cerca del árbol de profundidad 8, y tardó mucho más en entrenar (1.6 s contra 0.02 s).


## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Notebook: [`starter/ejercicio2_regularizacion.ipynb`](./starter/ejercicio2_regularizacion.ipynb), sobre el mismo `viajes_features.csv` y con la misma preparación importada. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Con interacciones de grado 3 la regresión recibe 3275 columnas, y solo hay 2380 filas de entrenamiento: más columnas que filas. Con tantos parámetros libres el modelo puede ajustar el entrenamiento casi a la perfección (0.8892 de error) sin aprender nada generalizable, y en prueba el error se dispara a 22.7273, peor incluso que la línea base (12.5308).

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En entrenamiento el error subió de 0.8892 a 1.0272, y en prueba bajó de 22.7273 a 1.3209; la brecha pasó de 21.84 a 0.29. Ridge penaliza los coeficientes grandes, así que ya no puede ajustar cada punto del entrenamiento tan de cerca. Esa pérdida de ajuste en entrenamiento es el precio de que generalice mucho mejor.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

El gradient boosting, con 1.2750 de error en prueba (el de Ridge grado 3 fue 1.3209). Frente a la línea base (12.5308) se equivoca 11.26 menos, es decir, reduce el error en cerca de un 90%.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

No hace falta que aprenda nada para que eso pase. La línea base solo usa la mediana del entrenamiento, así que su error en cada conjunto depende de qué tan dispersas estén las tarifas alrededor de esa mediana. Con 595 viajes de prueba, es normal que justo ese subconjunto tenga tarifas algo más cercanas a la mediana que las 2380 de entrenamiento. Es cuestión de la división y no de que el modelo sepa algo; no hay nada que sobreajustar, así que la brecha solo mide el azar de la muestra.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el notebook tú.

### Entregable 1: `modelos_proyecto.ipynb`

Un notebook, en la misma carpeta que tu `pipeline_proyecto.py` (`psets/proyecto/modelado/`), que importe de ahí la preparación y la división (`from pipeline_proyecto import cargar_y_dividir, construir_preparacion`) en vez de copiarlas, y que entrene:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento, escrita por el notebook. Antes de hacer commit, reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa, no de celdas corridas en momentos distintos.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

Mi línea base es un `DummyRegressor` que siempre predice la mediana de `UNITS` en entrenamiento, que es 0 (cerca del 70% de las filas vende 0 unidades). Elegí esa y no una regla del dominio como "lo mismo que la semana pasada" porque la muestra tiene unas 6 filas por producto y tienda repartidas en cinco años, así que casi nunca existe el dato de la semana anterior. Tampoco usé la media, porque con tantos ceros predice 1 unidad en todo y sale peor. La línea base da un RMSE de 3.8348 en prueba y es lo que tiene que superar cualquier modelo con columnas.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

Sí, el bosque sin límite. Su error en entrenamiento es 2.1645 y en prueba 3.0159, una brecha de 0.8514, contra 0.0649 del gradient boosting y 0.1099 del bosque regularizado. El bosque libre memoriza ventas de filas concretas y no mejora en prueba (3.0159 es casi lo mismo que el 3.0188 de la regresión lineal, que tiene mucho menos capacidad). Ridge, la línea base y el boosting no se sobreajustan: sus brechas son negativas o muy pequeñas (-0.0691, -0.0401 y 0.0649).

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

En el modelo lineal, Ridge con `alpha=10` no hizo falta: dio exactamente los mismos errores que la regresión sin regularizar (3.0879 en entrenamiento y 3.0188 en prueba), porque son 26 columnas contra casi 160 mil filas y no hay de dónde sobreajustar. Sí regularicé el bosque, exigiendo al menos 20 filas por hoja (`min_samples_leaf=20`): el error en entrenamiento subió de 2.1645 a 2.8233, el de prueba bajó de 3.0159 a 2.9332 y la brecha pasó de 0.8514 a 0.1099. Es el mejor modelo en prueba.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El mejor es el bosque con `min_samples_leaf=20`, con 2.9332 de RMSE en prueba contra 3.8348 de la línea base: 0.90 unidades menos, una mejora de alrededor de 23.5%. Es una mejora moderada, y se explica en parte porque no tengo el historial de cada producto (la muestra es muy dispersa) y la mayoría de las ventas son 0 o 1. El gradient boosting queda muy cerca (2.9595) y entrena en menos tiempo (20.9 s contra 26.8 s). Esta tabla es de una sola división por fecha, así que la diferencia entre los dos no es concluyente.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error, mediría: cuánto tarda en responder cada opción (el usuario ve la tarifa al pedir el viaje); cuánto cuesta cada predicción, sobre todo con un LLM que cobra por llamada; qué tan estables son las respuestas (el mismo viaje debería dar la misma tarifa, y un LLM puede cambiar de un intento a otro); si puedo explicar por qué sale cierto precio; cuánto trabajo es mantenerlo y reentrenarlo; y qué tan bien se comporta con viajes raros, como un destino poco frecuente. Para un número a partir de columnas tabulares, supongo que un ensamblaje va a ser más barato, rápido y estable, pero habría que medirlo.
