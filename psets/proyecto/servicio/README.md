## Cómo correr el servicio

El servicio devuelve las unidades estimadas de un producto en una tienda y un día. Todo se corre desde `psets/proyecto/`.

**0. Registrar y promover el modelo.** El modelo vive en el registro de MLflow como `ventas-walmart-m5`. Cada corrida de `registrar_modelo.py` crea una versión nueva (el argumento es `min_samples_leaf` del bosque); `promover.py` pone el alias `produccion` en la que se va a servir, con el motivo opcional:

```
python -m servicio.registrar_modelo 20
python -m servicio.registrar_modelo 5
python -m servicio.promover 1 "menor rmse_prueba que la version 2"
```

**1. Exportar el modelo en producción desde MLflow.** Con el servidor de MLflow corriendo, sin argumentos: exporta la versión que tiene el alias `produccion`.

```
python3 servicio/exportar_modelo.py
```

Eso crea `servicio/modelo.joblib` y `servicio/modelo_version.txt` (el número de versión del registro). Ninguno se sube al repositorio, y la imagen los copia adentro: el contenedor no necesita llegar al servidor de MLflow.

**2. Levantarlo sin contenedor**, para probar:

```
uvicorn servicio.api:app --port 8000
```

**3. Construir y levantar el contenedor.** La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`:

```
docker build -f servicio/Dockerfile -t servicio-ventas .
docker run -p 8000:8000 servicio-ventas
```

**4. Probarlo.**

```
curl localhost:8000/salud
```

```
{"estado":"ok","version_modelo":"1"}
```

```
curl -X POST localhost:8000/prediccion -H "Content-Type: application/json" \
  -d '{"item_id":"FOODS_3_090","dept_id":"FOODS_3","store_id":"CA_1","sell_price":1.5,"wday":3,"semana_del_anio":20,"is_snap":true,"is_weekend":false,"is_event_day":false,"is_pre_launch":false,"fecha":"2015-05-20"}'
```

```
{"id_prediccion":"8c5b3a076d0f4712a823a9af114c21c1","unidades_estimadas":10.276,"version_modelo":"1"}
```

`/salud` y cada predicción dicen qué versión del registro las calculó (`version_modelo`). `fecha` (el día que se predice, `AAAA-MM-DD`) no la usa el modelo, pero queda en el registro para unir la predicción con las ventas reales. Los campos `sell_price`, `is_snap`, `is_weekend`, `is_event_day` e `is_pre_launch` son opcionales; `sell_price` va vacío cuando el producto todavía no se vende. Un `wday` fuera de 1 a 7, o un campo obligatorio que falta, devuelve 422.

Las pruebas del servicio no necesitan `modelo.joblib`: usan un modelo de juguete.

```
pytest psets/proyecto -v
```

## Monitoreo

Cada petición a `/prediccion` se mide (latencia y código de respuesta) y se resume en `GET /metricas`: `peticiones`, `errores_validacion` (422), `errores_servidor` (5xx), `latencia_p50_ms` y `latencia_p95_ms`. Son métricas en memoria: se reinician con el servicio.

Cada predicción válida se guarda además en `monitoreo/predicciones.jsonl` (una línea por predicción, con su `id_prediccion`, la versión, la entrada y la salida; la ruta se cambia con la variable `RUTA_REGISTRO`). En un contenedor, monta la carpeta para que el registro sobreviva:

```
docker run -p 8000:8000 -v "$(pwd)/monitoreo:/app/monitoreo" servicio-ventas
```

Para medir el error con las ventas reales: `python -m monitoreo.desempeno`. Las métricas, sus umbrales y qué dice cada una están en `docs/monitoreo.md`. `monitoreo/predicciones.jsonl` no se sube al repositorio.