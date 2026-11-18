# Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la gobernanza de modelos, y qué componentes la forman?**

La gobernanza de modelos es el conjunto de procesos y controles con los que una organización define, aplica y mantiene el control sobre cómo se desarrollan, despliegan y usan sus modelos, durante todo su ciclo de vida. Busca transparencia, responsabilidad (quién respondió por qué) y repetibilidad, y nació en el sector financiero para controlar el riesgo de modelos; con la IA/ML se extendió a cualquier organización. Entre sus componentes: documentación del desarrollo del modelo, un inventario central de modelos, control de versiones, validación y pruebas (incluido back-testing), monitoreo y observabilidad en producción, y roles y políticas de cumplimiento y auditoría.


**El artículo pide un inventario central de modelos. ¿Qué datos tendría que tener ese inventario sobre cada modelo, y para qué los necesita alguien que no lo entrenó?**

Para cada modelo, el inventario debería registrar al menos: nombre e identificador único, versión y cuál está en producción; propósito y decisión de negocio que apoya; responsable (quién lo entrenó y quién responde por él); datos con que se entrenó (fuente, fecha, versión) y la corrida/código que lo produjo; hiperparámetros y métricas de validación; fecha de despliegue, estado (candidato, producción, retirado) y qué lo reemplazó; y limitaciones y riesgos conocidos.

Alguien que no lo entrenó los necesita para: saber qué modelo está tomando decisiones y quién responde por él, reproducirlo o auditarlo, decidir si es seguro reutilizarlo o cambiarlo, investigar un reclamo («¿qué modelo produjo esto?») y volver a una versión anterior sin depender de la memoria de quien lo entrenó.

## Pregunta de apertura

La sesión pasada, tu servicio cargaba `modelo.joblib`. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

**No sabría cuál lo produjo, y no podría volver atrás.** Si reentreno y re-exporto `modelo.joblib`, el archivo viejo desaparece: no queda ningún registro de qué modelo corría el día del reclamo, con qué datos ni con qué métrica. Y si el nuevo es peor, el anterior ya no existe; tendría que reentrenarlo a ciegas, esperando que salga igual.

Con un registro de modelos, cada reentrenamiento crea una **versión nueva e inmutable** (con su corrida, parámetros y métrica), y un **alias** (`produccion`) indica cuál se sirve. Si el servicio incluye la versión en cada respuesta, el reclamo se rastrea hasta una versión concreta; y volver atrás es mover el alias a la versión anterior (que sigue ahí) y reiniciar el servicio.

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

El error dice `Registered model alias produccion not found` (el texto exacto puede variar un poco según la versión de MLflow; en cualquier caso, es un error de «no existe»). Dice que **en el registro no hay nada**: `registrar_modelo.py` guardó el modelo como artefacto de la corrida, pero sin `registered_model_name` no creó ningún modelo registrado ni ninguna versión. Por eso imprime `Version None` (no hay `registered_model_version`), `versiones.py` no encuentra versiones que listar, y `promover.py` no puede poner un alias a una versión que no existe. Una corrida con un artefacto ≠ una versión registrada.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

Espero ver **2 versiones**: la 1 (64 neuronas, MAE 1.1695) y la 2 (8 neuronas, MAE 1.1877). Al registrar la segunda **no pasó nada con la primera**: no se reemplazó ni se borró. Cada `log_model(..., registered_model_name=...)` crea una versión nueva con número creciente y deja intacta a las anteriores; es justo la diferencia con sobrescribir `modelo.joblib`. (Confirmado al correr `versiones.py`: aparecen las dos.)

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

- Con `produccion` en la **versión 2**, `/salud` dice `"version_modelo": "2"`.
- Muevo el alias a la versión 1 **sin reiniciar** uvicorn: el servicio **sigue respondiendo la versión 2**. El modelo se cargó en memoria al arrancar y el alias solo se lee en ese momento; mover el alias no avisa al proceso.
- **Después de reiniciar**, `/salud` dice `"version_modelo": "1"`: ahora sí lee el alias y carga esa versión.

Consecuencia práctica: para volver atrás hay que mover el alias *y* reiniciar el servicio.

**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

Antes del TODO 4 falla `test_cada_prediccion_dice_que_version_la_calculo` (la de `/salud` ya pasaba, porque `/salud` ya devolvía la versión). Lo que comprueba y `/salud` no: que **cada predicción** lleve la versión que la calculó. `/salud` solo describe al servicio en general en ese instante; la respuesta de `/predecir` es lo que le llega al usuario y lo que se guarda o reclama después, así que es ahí donde tiene que estar la trazabilidad (y al ser una prueba con modelo de prueba y versión `"7"` inventada, se verifica sin servidor de MLflow, también en CI).


**Lo que respondió el servicio:**

Viaje de Midtown a JFK (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta):

| Versión que sirve | Tarifa estimada | `version_modelo` |
|---|---|---|
| 2 (8 neuronas) | 69.98 | `"2"` |
| 1 (64 neuronas) | 70.07 | `"1"` |

(Con el alias en la 1 pero sin reiniciar, el servicio siguió respondiendo 69.98 con `"2"`.)


## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

Quedó en producción la **versión 2** (gradient boosting: recall 0.661, precisión 0.429). No la última porque la regla no es «la más reciente» sino «la que detecta más fraudes que la actual»: la v1 (logística, recall 0.644) entró porque no había nada; la v2 la superó (0.661 > 0.644) y la reemplazó; la v3 (recall 0.102) no superó a la v2 y fue rechazada. Registrar una versión y promoverla son dos decisiones separadas.

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

La regla compara **solo recall**, y la regresión logística muy regularizada tiene un recall de apenas 0.102 contra 0.661 de la v2: con ese regularizador (`C=0.001`) casi ninguna probabilidad llega al umbral de 0.10, así que lanza pocas alertas; las que lanza son más acertadas (precisión 0.667), pero deja pasar casi 9 de cada 10 fraudes. Para la regla, eso es peor.

Si las **falsas alarmas cuestan más que los fraudes no detectados**, habría que cambiar el criterio de promoción: en vez de recall, comparar precisión (o una métrica que la pondere, como F0.5, o directamente un costo esperado `FP·costo_falsa_alarma + FN·costo_fraude_perdido` a minimizar), quizá con un recall mínimo aceptable; y reconsiderar el umbral de 0.10, que se eligió pensando en detectar mucho. Con esa regla, la v3 (precisión 0.667) sí pasaría a producción.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

En la interfaz de MLflow → **Models → `fraude-pagos` → versión 3**: ahí están las etiquetas `validacion = rechazada` y `comparada_con = 2`, y el enlace a la corrida con sus métricas (recall 0.102, precisión 0.667) frente a las de la versión 2. Es decir, el registro conserva tanto la decisión como la evidencia, sin depender de la memoria de nadie. (Lo que *no* queda guardado es la regla en sí; esa vive en el código de `ciclo_fraude.py`.)

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

Ahora hay **6 versiones** (la 4, 5 y 6 son copias de las anteriores, con las mismas métricas) y en producción sigue la **versión 2**. Sí es lo que esperaba: el entrenamiento es determinista (`random_state=42`), así que la v5 repite exactamente el recall de la v2 (0.661) y la regla exige *estrictamente mayor*, por lo que ninguna copia mejora a la de producción y las tres quedan `rechazada`. Efecto secundario: el registro se llena de versiones duplicadas; en la práctica convendría no registrar un candidato idéntico o limpiar esas versiones.

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

Una versión nueva de `ventas-walmart-m5` pasa a producción solo si su **`rmse_prueba` es menor que el de la versión actual**, medido sobre el mismo conjunto de prueba (división por fecha, corte 2015-04-30), y su **`costo_negocio`** (faltante 3, sobrante 1) **no empeora**. Hoy está en producción la versión 1 (bosque con `min_samples_leaf=20`, `rmse_prueba` ≈ 2.93); la versión 2 (`min_samples_leaf=5`) es la candidata de comparación.

La decisión la toma el autor del proyecto, comparando las versiones en la pestaña **Models** de MLflow, y **queda registrada** en el propio registro: el alias `produccion` apunta a la versión elegida, y `promover.py` deja en esa versión las etiquetas `promovida_el` y `motivo_promocion`. El criterio y los pasos para volver atrás están en `docs/despliegue.md` (sección «Versionado y registro»).

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

Lo sabe porque **la versión viaja con el modelo dentro de la imagen**: `servicio/exportar_modelo.py` consulta el registro *antes* de construir la imagen, descarga la versión con alias `produccion` y guarda su número en `servicio/modelo_version.txt`, junto a `modelo.joblib`. El `Dockerfile` copia los dos archivos, y `api.py` lee el número al arrancar y lo devuelve como `version_modelo` en `/salud` y en cada predicción. El contenedor nunca consulta el servidor de MLflow. (Contrapartida: si se mueve el alias, hay que re-exportar y reconstruir la imagen para que el cambio llegue al servicio.)

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Un modelo que responde rápido pero cada vez peor sufre **degradación silenciosa**: el servicio no falla (no hay errores, la latencia es buena, `/salud` dice ok), pero el mundo cambió (deriva de los datos de entrada, o cambia la relación entre las variables y el objetivo, p. ej. nuevas tarifas, nuevas zonas, otra estacionalidad) y el modelo ya no representa la realidad. Sin monitoreo de calidad, nadie se entera.

Para enterarme: registrar las entradas y predicciones (con `version_modelo`) y vigilar su distribución frente a la de entrenamiento (deriva de datos y de predicciones); y, cuando llegue la respuesta real (la tarifa efectivamente cobrada), calcular el error en producción y compararlo con el de validación, con alertas cuando pase un umbral. Un `/salud` técnico no basta: hay que monitorear la calidad del modelo, no solo que el servicio esté vivo.