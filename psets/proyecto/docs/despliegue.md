# Despliegue del modelo de ventas: por lotes

**Decisión: el modelo se sirve por lotes.** Una vez por noche, un proceso calcula las unidades esperadas de cada producto en cada tienda para los próximos 7 días y las guarda en una tabla. La app de reposición y el tablero leen esa tabla; no llaman al modelo.

## Qué decisión habilita y cuánto puede esperar

La predicción le sirve al equipo de inventario de cada tienda para decidir cuánto pedir de cada producto. Esa decisión se toma una vez al día, por la mañana, para pedidos que llegan de uno a tres días después. Por eso la predicción puede esperar **horas**: basta con que esté lista antes de las 6:00 a.m. del día en que se hace el pedido. No hay nadie esperando una respuesta frente a una pantalla.

## Cuántos casos hay y si se conocen de antemano

Se conocen todos de antemano: 3,049 productos × 10 tiendas × 7 días = 213,430 predicciones por noche. El catálogo y las tiendas cambian poco y los días futuros son simplemente los siguientes del calendario. Es un volumen pequeño: con el modelo actual, predecirlo todo toma segundos.

## Qué datos necesitan las features en el momento de predecir

Todas las features del modelo existen antes de que el día ocurra:

- `PRECIO_LOG`: el precio de lista, que se publica antes de la semana de venta.
- `SEMANA_SIN`, `SEMANA_COS`, día de la semana, fin de semana, día de evento: salen del calendario.
- `IS_SNAP`: el calendario de beneficios SNAP por estado se conoce con anticipación.
- `IS_PRE_LAUNCH`: se sabe si el producto ya salió a la venta en esa tienda.
- Producto, departamento y tienda: son datos del catálogo.

Las columnas que dependen de la venta del propio día (`REVENUE`, `IS_SALES_SPIKE` y los indicadores de desabastecimiento) no entran al modelo.

El único dato que podría faltar es el precio de un día futuro si todavía no se ha cargado. En ese caso la preparación imputa la mediana, y la predicción es menos confiable hasta que se cargue el precio real.

## Qué se pierde con el patrón que descarto

Con el servicio en línea se perdería simplicidad y se ganaría poco. Lo que se pierde al servir por lotes:

- **Cambios durante el día.** Si a mediodía alguien cambia un precio o se anuncia una promoción, la tabla no lo ve hasta la corrida de la noche siguiente.
- **Productos o tiendas nuevos.** Un producto que se agrega hoy no tiene predicción hasta mañana.

Ninguna de las dos es crítica porque la decisión de pedido es diaria. Por eso también se construyó el servicio (`servicio/`): cuando alguien necesita una predicción puntual, como un caso nuevo, se la pide a la API y obtiene la respuesta de inmediato, sin cambiar el patrón principal.

## Versionado y registro

- **Nombre en el registro:** `ventas-walmart-m5`, en el servidor de MLflow (`MLFLOW_TRACKING_URI`). Cada corrida de `servicio/registrar_modelo.py` crea una versión nueva, con su corrida, su `min_samples_leaf` y sus métricas (`rmse_prueba`, `rmse_entrenamiento`, `costo_negocio`); ninguna reemplaza a otra.
- **Qué versión está en producción:** la que tiene el alias `produccion`. La versión 1 es el bosque con `min_samples_leaf=20`, el modelo final de `modelado/` (`rmse_prueba` ≈ 2.93 sobre el conjunto de prueba por fecha). La 2 es el mismo bosque con `min_samples_leaf=5`, registrada como candidata para comparar.
- **Criterio para mover el alias:** una versión nueva pasa a `produccion` solo si su `rmse_prueba` es menor que el de la versión actual, medido sobre el mismo conjunto de prueba (corte 2015-04-30), y su `costo_negocio` (faltante 3, sobrante 1) no empeora. Decide el autor del proyecto, después de comparar las versiones en la pestaña **Models** de MLflow.
- **Dónde queda registrada la decisión:** en el alias `produccion` y en las etiquetas de la versión promovida (`promovida_el`, `motivo_promocion`), que escribe `python -m servicio.promover <versión> "<motivo>"`.
- **Cómo sabe el servicio qué versión sirve:** `servicio/exportar_modelo.py` guarda el número de la versión con alias `produccion` en `servicio/modelo_version.txt`, junto a `modelo.joblib`. La imagen copia los dos archivos; el contenedor no consulta MLflow. `/salud` y cada predicción devuelven `version_modelo`.

### Cómo volver a la versión anterior

Si la versión nueva resulta peor (por ejemplo, la 2 reemplazó a la 1):

1. `python -m servicio.promover 1 "vuelta a la version 1: la 2 empeoro"` mueve el alias a la versión anterior, que sigue en el registro.
2. `python3 servicio/exportar_modelo.py` regenera `modelo.joblib` y `modelo_version.txt` con esa versión.
3. `docker build -f servicio/Dockerfile -t servicio-ventas .` y volver a levantar el contenedor (mover el alias no cambia un contenedor que ya está corriendo: carga el modelo al arrancar).
4. `curl localhost:8000/salud` debe responder `"version_modelo":"1"`.