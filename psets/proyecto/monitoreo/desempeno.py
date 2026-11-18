"""El desempeno del modelo de ventas en produccion: une cada prediccion
guardada con las unidades que se vendieron de verdad, y calcula el error
por semana y por version del modelo.

Uso, desde psets/proyecto/:

    python -m monitoreo.desempeno

La verdad son las ventas reales (UNITS) de cada producto, tienda y dia: se
conocen cuando cierra el dia, asi que la verdad de una prediccion llega
entre 1 y 8 dias despues (la prediccion nocturna es a 7 dias). Las
predicciones de dias que todavia no tienen ventas se cuentan aparte: no
tienen error todavia, y no se mezclan con las demas.

Tambien calcula lo que se puede saber SIN la verdad (el promedio de lo que
predice el modelo y la fraccion de peticiones sin precio): no dice cuanto se
equivoca el modelo, solo avisa que lo que recibe o lo que responde cambio.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import RUTA_DATOS

RUTA_REGISTRO = Path(os.environ.get("RUTA_REGISTRO", "monitoreo/predicciones.jsonl"))
RUTA_VENTAS = os.environ.get("RUTA_VENTAS", RUTA_DATOS)

# con lo que se aprobo la version 1: rmse en prueba (modelado/experimentos_proyecto.ipynb)
# y costo_negocio promedio de la validacion temporal (modelado/validacion_proyecto.csv)
RMSE_REFERENCIA = 2.9332
COSTO_REFERENCIA = 2.2891
FACTOR_ALERTA = 1.25  # alerta si el error semanal supera la referencia en mas de 25%

# los mismos costos de modelado/validacion_proyecto.ipynb: quedarse corto cuesta 3, pedir de mas cuesta 1
COSTO_FALTANTE = 3
COSTO_SOBRANTE = 1


def costo_negocio(real, estimado):
    faltante = np.maximum(real - estimado, 0)
    sobrante = np.maximum(estimado - real, 0)
    return float((COSTO_FALTANTE * faltante + COSTO_SOBRANTE * sobrante).mean())


def con_semana(predicciones):
    """Agrega "semana" (el lunes de la semana del dia predicho) y "fecha_key"
    (AAAAMMDD, como DATE_KEY en los datos). Las predicciones sin fecha no se
    pueden unir con las ventas y quedan con semana vacia."""
    predicciones = predicciones.copy()
    fecha = pd.to_datetime(predicciones["fecha"])
    predicciones["semana"] = fecha.dt.to_period("W").dt.start_time.dt.date
    predicciones["fecha_key"] = fecha.dt.strftime("%Y%m%d").astype("Int64")
    return predicciones


def unir_con_ventas(predicciones, ventas):
    """Cada prediccion con sus unidades reales ("unidades_reales", vacio si
    ese dia todavia no tiene ventas). Une por producto, tienda y dia."""
    ventas = ventas[["ITEM_ID", "STORE_ID", "DATE_KEY", "UNITS"]].rename(
        columns={"ITEM_ID": "item_id", "STORE_ID": "store_id", "DATE_KEY": "fecha_key", "UNITS": "unidades_reales"})
    ventas = ventas.drop_duplicates(["item_id", "store_id", "fecha_key"])
    return predicciones.merge(ventas, on=["item_id", "store_id", "fecha_key"], how="left")


def error_por_semana_y_version(unidas):
    """El error por semana y version, solo con las predicciones que ya tienen
    ventas reales: "predicciones", "rmse" y "costo_negocio"."""
    conocidas = unidas.dropna(subset=["unidades_reales", "semana"])
    filas = []
    for (semana, version), grupo in conocidas.groupby(["semana", "version_modelo"]):
        real = grupo["unidades_reales"].to_numpy(dtype=float)
        estimado = grupo["unidades_estimadas"].to_numpy(dtype=float)
        filas.append({
            "semana": semana, "version_modelo": version, "predicciones": len(grupo),
            "rmse": round(float(np.sqrt(np.mean((real - estimado) ** 2))), 4),
            "costo_negocio": round(costo_negocio(real, estimado), 4),
        })
    columnas = ["semana", "version_modelo", "predicciones", "rmse", "costo_negocio"]
    return pd.DataFrame(filas, columns=columnas).set_index(["semana", "version_modelo"])


def senales_sin_verdad(predicciones):
    """Lo que se sabe el mismo dia, sin las ventas reales: cuantas peticiones,
    cuanto predice el modelo en promedio y que fraccion llego sin precio."""
    con_fecha = predicciones.dropna(subset=["semana"])
    return con_fecha.groupby(["semana", "version_modelo"]).agg(
        predicciones=("id_prediccion", "size"),
        promedio_estimado=("unidades_estimadas", "mean"),
        fraccion_sin_precio=("sell_price", lambda precios: precios.isna().mean()),
    ).round(3)


def alertas(error):
    """Un mensaje por cada semana y version cuyo error supera la referencia."""
    mensajes = []
    for (semana, version), fila in error.iterrows():
        if fila["rmse"] > FACTOR_ALERTA * RMSE_REFERENCIA:
            mensajes.append(f"modelo: la semana del {semana} la version {version} tuvo rmse {fila['rmse']:.3f} "
                            f"(referencia {RMSE_REFERENCIA})")
        if fila["costo_negocio"] > FACTOR_ALERTA * COSTO_REFERENCIA:
            mensajes.append(f"modelo: la semana del {semana} la version {version} tuvo costo_negocio "
                            f"{fila['costo_negocio']:.3f} (referencia {COSTO_REFERENCIA})")
    return mensajes


if __name__ == "__main__":
    if not RUTA_REGISTRO.exists():
        raise SystemExit(f"No hay {RUTA_REGISTRO}: el servicio todavia no guardo ninguna prediccion.")
    predicciones = con_semana(pd.read_json(RUTA_REGISTRO, lines=True, dtype={"version_modelo": str}))
    ventas = pd.read_csv(RUTA_VENTAS, usecols=["ITEM_ID", "STORE_ID", "DATE_KEY", "UNITS"], low_memory=False)

    unidas = unir_con_ventas(predicciones, ventas)
    sin_fecha = int(predicciones["semana"].isna().sum())
    sin_verdad = int(unidas["unidades_reales"].isna().sum()) - sin_fecha
    print(f"{len(predicciones)} predicciones: {len(predicciones) - sin_fecha - sin_verdad} con ventas reales, "
          f"{sin_verdad} esperando las ventas del dia, {sin_fecha} sin fecha (no se pueden unir)")

    error = error_por_semana_y_version(unidas)
    print("\nError por semana y version (solo con ventas reales):")
    print(error.to_string())
    print("\nLo que se sabe sin la verdad:")
    print(senales_sin_verdad(predicciones).to_string())
    print("\nAlertas:")
    for mensaje in alertas(error):
        print("-", mensaje)