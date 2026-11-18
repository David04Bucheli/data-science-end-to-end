"""El desempeno del modelo en produccion: une cada prediccion guardada con
la tarifa que se cobro, y calcula el error por semana.

    python desempeno.py
"""

import os
import sys

import pandas as pd

MAE_REFERENCIA = 1.17  # el error en prueba con que se aprobo la version 1

if not os.path.exists("predicciones.jsonl"):
    sys.exit("No hay predicciones.jsonl: el servicio no esta guardando sus predicciones.")

predicciones = pd.read_json("predicciones.jsonl", lines=True)
cobros = pd.read_csv("cobros.csv")

datos = pd.merge(predicciones, cobros, on="id_prediccion")
datos["error"] = (datos["tarifa_estimada"] - datos["tarifa_cobrada"]).abs()
datos["semana"] = pd.to_datetime(datos["fecha_hora_inicio"]).dt.to_period("W").dt.start_time.dt.date
por_semana = datos.groupby("semana").agg(viajes=("error", "size"), mae=("error", "mean")).round(4)

por_semana["veces_la_referencia"] = (por_semana["mae"] / MAE_REFERENCIA).round(2)
print(por_semana.to_string())