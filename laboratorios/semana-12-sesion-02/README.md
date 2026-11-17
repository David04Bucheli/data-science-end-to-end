# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

Guardar el modelo entrenado en un archivo, escribir una API que lo cargue y reciba los datos de entrada, validar esas entradas, probar el servicio y empaquetarlo en un contenedor para desplegarlo en un servidor, donde otra aplicación lo llama.

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

Para que el servicio corra igual en cualquier máquina: el contenedor lleva el código, el modelo y las versiones exactas de Python y las librerías. Así se evita el "en mi máquina sí funciona".

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un servicio online (una API) al que la app le mande el viaje en ese momento: fecha y hora, zonas de origen y destino, distancia, pasajeros y método de pago. El servicio calcula el resto de las features y devuelve la tarifa estimada.

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que "todo ok" no demuestra que el servicio sirva: las pruebas pasan aunque no revisen que la tarifa sea correcta ni que se rechacen viajes mal armados (los `TODO` 3 y 4 no tienen `assert`), y `/salud` solo dice que el modelo está cargado.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

Sin `distancia_km` responde 200 con una tarifa, porque es opcional y la preparación la imputa. Sin `zona_origen` responde 422, porque es obligatoria.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El modelo recibiría una feature distinta a la que aprendió y daría una tarifa mal calculada sin ningún error a la vista (training-serving skew). Por eso el servicio reutiliza `features_taxis.py` para calcularla igual que en el entrenamiento.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce FastAPI, automáticamente, con el esquema `Viaje` (pydantic). La petición no llega al modelo: se rechaza antes de entrar a la función `predecir`.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Por la dirección donde escucha uvicorn. Dentro del contenedor, por defecto escucha en `127.0.0.1`, que solo se ve desde adentro; con `-p 8000:8000` la petición llega desde afuera y no encuentra nada. Hay que arrancarlo con `--host 0.0.0.0`. (Respondido de lo que sé: no pude correr Docker.)

**Lo que respondió el servicio:**

Tarifa del viaje de Midtown al aeropuerto: 70.39. Viaje sin zona de origen: código 422 con `{"detail": [{"type": "missing", "loc": ["body", "zona_origen"], "msg": "Field required", ...}]}`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

Con los pagos de `comprobar.py`: el de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo recibió probabilidad 0.9996 y `revisar: true`. El de la tarde, con tarjeta conocida y 22 viajes previos, recibió 0.0159 y `revisar: false`.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app recibe la decisión ya tomada y no necesita conocer el umbral ni repetir la lógica. Si el umbral pasa a 0.20 solo se cambia `UMBRAL` en el servicio y se vuelve a desplegar; la app no cambia.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Debería dar la misma, si es el mismo modelo (el mismo archivo `.joblib`), las features se calculan igual (la misma `agregar_features`) y las columnas entran en el mismo orden.

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde 422: FastAPI dice que el valor no es un número válido ("Input should be a valid number"). Lo produce la línea `monto: float` del esquema `Pago` en `servicio_fraude.py`.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

Recibe los datos como los conoce quien llama: producto, departamento, tienda, precio de lista (puede ir vacío si el producto aún no se vende), día de la semana, semana del año y los indicadores de SNAP, fin de semana, día de evento y pre-lanzamiento. El servicio calcula él mismo `PRECIO_LOG`, `SEMANA_SIN` y `SEMANA_COS`, y la codificación del producto, el departamento y la tienda la hace la preparación guardada dentro del modelo.

Sé que las calcula igual que en el entrenamiento por dos razones. La primera, que el servicio importa `agregar_features` de `modelado/pipeline_proyecto.py`, la misma función que usó el entrenamiento, sin copiarla. La segunda, que una prueba (`test_el_servicio_calcula_las_features_como_el_entrenamiento`) comprueba que la fila que el servicio le pasa al modelo es idéntica a la que produce `agregar_features` sobre los mismos datos.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

Petición (el servicio levantado con `uvicorn servicio.api:app --port 8000`):

```
curl -X POST localhost:8000/prediccion -H "Content-Type: application/json" \
  -d '{"item_id":"FOODS_3_090","dept_id":"FOODS_3","store_id":"CA_1","sell_price":1.5,"wday":3,"semana_del_anio":20,"is_snap":true,"is_weekend":false,"is_event_day":false,"is_pre_launch":false}'
```

Respuesta:

```
{"unidades_estimadas":10.276,"version_modelo":"2abf63a785674d0db32122197a396661"}
```

Con `"wday": 8` responde 422: `Input should be less than or equal to 7`.

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Guardaría cada modelo con un nombre o una huella de versión (como la del lote) y la devolvería en la respuesta o en un log, para ligar cada predicción a su modelo. No sobrescribiría el archivo: conservaría el anterior (o lo registraría en MLflow), y para volver atrás bastaría cargar el archivo anterior y reiniciar el servicio.
