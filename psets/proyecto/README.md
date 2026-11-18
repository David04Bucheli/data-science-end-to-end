# Predicción de ventas por producto y tienda (Walmart M5)

> Este es el README de tu proyecto: reemplaza cada sección entre corchetes a medida que avanzas. Escríbelo para alguien que clona tu fork sin haber estado en clase y necesita entender qué hace el proyecto y correrlo, no como una nota para ti.

Predice `UNITS`, las unidades vendidas de un producto en una tienda y un día, para que el equipo de inventario de cada tienda decida cuánto pedir de cada producto. La predicción se sirve por lotes (cada noche, 7 días hacia adelante) y también por una API para casos puntuales; ver `docs/despliegue.md`.

**Métrica de éxito:** [la métrica de negocio y el umbral de decisión del PSet 1]

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| [nombre de la fuente] | [qué filas y columnas trae, desde qué fecha] | [la URL, la API o el procedimiento exacto] | `data/raw/[archivo]` |

Los datos no están en el repositorio. Para regenerarlos desde cero: [los comandos, en orden]. Se completa con el PSet 1.

## Cómo correr el pipeline de datos

[Los comandos, en orden y desde qué carpeta, para ir de la fuente al warehouse: levantar la infraestructura, construir la imagen, subir el flow a Kestra, limpiar, cargar y correr dbt. Cada paso dice qué archivo deja y dónde. Se completa con el PSet 2.]

## Cómo entrenar y evaluar el modelo

Los notebooks de `modelado/` importan la preparación de `modelado/pipeline_proyecto.py` y se corren en este orden: `explorar_proyecto`, `features_proyecto`, `modelos_proyecto` (línea base, lineales y ensamblajes), `experimentos_proyecto` (una corrida de MLflow por modelo; marca la final con `candidato = final`), `comparar_proyecto` (bosque, red neuronal y LLM) y `validacion_proyecto` (validación cruzada temporal y métrica de negocio). `experimentos_proyecto` necesita el servidor de MLflow levantado. Las pruebas de la preparación: `pytest modelado`.

## Modelo final

Bosque aleatorio (`RandomForestRegressor`, 100 árboles, `min_samples_leaf=20`) sobre la preparación de `pipeline_proyecto.py`. Corrida de MLflow `2abf63a785674d0db32122197a396661`, `rmse_prueba` = 2.9332 (división por fecha, corte 2015-04-30). Está en el registro como `ventas-walmart-m5`; la versión 1 es la que tiene el alias `produccion`. Criterio de promoción y cómo volver a una versión anterior: `docs/despliegue.md`, sección «Versionado y registro».

## Cómo correr el servicio

Desde `psets/proyecto/`, con MLflow levantado:

```
python -m servicio.registrar_modelo 20    # versión 1 (la final)
python -m servicio.registrar_modelo 5     # versión 2 (candidata)
python -m servicio.promover 1 "menor rmse_prueba"
python3 servicio/exportar_modelo.py       # exporta la versión con alias produccion
docker build -f servicio/Dockerfile -t servicio-ventas .
docker run -p 8000:8000 servicio-ventas
```

La imagen lleva adentro `modelo.joblib` y `modelo_version.txt`: el contenedor no necesita llegar a MLflow, y `/salud` y cada predicción responden `version_modelo`. Ejemplo:

```
curl -X POST localhost:8000/prediccion -H "Content-Type: application/json" \
  -d '{"item_id":"FOODS_3_090","dept_id":"FOODS_3","store_id":"CA_1","sell_price":1.5,"wday":3,"semana_del_anio":20,"is_snap":true}'
{"unidades_estimadas":10.276,"version_modelo":"1"}
```

Detalle en `servicio/README.md`.