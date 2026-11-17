## Cómo correr el servicio

El servicio devuelve las unidades estimadas de un producto en una tienda y un día. Todo se corre desde `psets/proyecto/`.

**1. Exportar el modelo final desde MLflow.** Con el servidor de MLflow corriendo, pasa el `run_id` de la corrida marcada como `candidato = final`:

```
python3 servicio/exportar_modelo.py 2abf63a785674d0db32122197a396661
```

Eso crea `servicio/modelo.joblib` (no se sube al repositorio) y `servicio/modelo_info.json`, que guarda de qué corrida salió el archivo.

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
{"estado":"ok"}
```

```
curl -X POST localhost:8000/prediccion -H "Content-Type: application/json" \
  -d '{"item_id":"FOODS_3_090","dept_id":"FOODS_3","store_id":"CA_1","sell_price":1.5,"wday":3,"semana_del_anio":20,"is_snap":true,"is_weekend":false,"is_event_day":false,"is_pre_launch":false}'
```

```
{"unidades_estimadas":10.276,"version_modelo":"2abf63a785674d0db32122197a396661"}
```

Los campos `sell_price`, `is_snap`, `is_weekend`, `is_event_day` e `is_pre_launch` son opcionales; `sell_price` va vacío cuando el producto todavía no se vende. Un `wday` fuera de 1 a 7, o un campo obligatorio que falta, devuelve 422.

Las pruebas del servicio no necesitan `modelo.joblib`: usan un modelo de juguete.

```
pytest psets/proyecto -v
```
