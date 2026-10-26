import pandas as pd

# TODO: la misma ruta que usaste en la exploracion de la Semana 8.
datos = pd.read_csv("obt_sales.csv")

# TODO: la columna que tu modelo va a predecir.
OBJETIVO = "UNITS"

# TODO, feature 1: crea la columna a partir de una o mas columnas
# existentes.
# datos["TODO_feature_1"] = ...

precio_promedio_item = datos.groupby("ITEM_ID")["SELL_PRICE"].transform("mean")
datos["precio_relativo"] = datos["SELL_PRICE"] / precio_promedio_item
datos["en_descuento"] = (datos["precio_relativo"] < 0.95).astype(int)

# TODO, feature 2.
# datos["TODO_feature_2"] = ...

datos["snap_fin_de_semana"] = (datos["IS_SNAP"] & datos["IS_WEEKEND"]).astype(int)

# Evidencia: compara el objetivo segun cada feature nueva.
# Si es numerica, su correlacion con el objetivo:
# print(datos[["TODO_feature_1", OBJETIVO]].corr())
# Si es categorica o binaria, el objetivo promedio por grupo:
# print(datos.groupby("TODO_feature_2")[OBJETIVO].mean())

print(datos.groupby("en_descuento")[OBJETIVO].mean())
print(datos.groupby("snap_fin_de_semana")[OBJETIVO].mean())

print(datos.head())
