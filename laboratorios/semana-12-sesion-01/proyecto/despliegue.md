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
