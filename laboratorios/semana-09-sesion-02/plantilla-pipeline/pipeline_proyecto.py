"""Preparacion de los datos de mi proyecto, lista para produccion.

Las features que decidiste en tu notebook de la sesion pasada viven aqui,
dentro de funciones. Desde la Semana 10, tus notebooks de experimentos
importan este archivo en vez de copiar el codigo:

    from pipeline_proyecto import cargar_y_dividir, construir_preparacion

Para comprobar que corre de principio a fin, sin notebook:

    python3 pipeline_proyecto.py
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

RUTA_DATOS = "/workspaces/data-science-end-to-end/laboratorios/semana-08-sesion-02/plantilla-eda/obt_sales.csv"

OBJETIVO = "UNITS"
NUMERICAS = ["PRECIO_LOG", "WDAY"]
CATEGORICAS = ["DEPT_ID", "STORE_ID"]
YA_LISTAS = ["SEMANA_SIN", "SEMANA_COS", "IS_SNAP", "IS_WEEKEND", "IS_EVENT_DAY", "IS_PRE_LAUNCH"]
PRODUCTO = ["ITEM_ID"]

# Las columnas que recibe el modelo, en el orden de la preparacion.
COLUMNAS_ENTRADA = NUMERICAS + CATEGORICAS + YA_LISTAS + PRODUCTO

# Todo lo que sea de la fecha de corte en adelante es prueba.
FECHA_CORTE = 20150430

# No entran, porque se calculan con UNITS o dependen de que ya se haya
# vendido: REVENUE, IS_SALES_SPIKE, IS_SUSPECTED_STOCKOUT_CONSERVATIVE,
# IS_SUSPECTED_STOCKOUT_NB. Si entraran, el modelo veria la respuesta.


def codificador_producto():
    # la semilla de la validacion cruzada interna se pasa distinto segun la
    # version de scikit-learn: desde la 1.9 va dentro de cv
    version = tuple(int(n) for n in sklearn.__version__.split(".")[:2])
    if version >= (1, 9):
        return TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=42))
    return TargetEncoder(target_type="continuous", random_state=42)


def agregar_features(datos):
    datos = datos.copy()
    # el precio tiene cola larga (de 0.05 a 61); en log se parece mas entre productos
    datos["PRECIO_LOG"] = np.log(datos["SELL_PRICE"])
    # semana del anio como circulo para que la 53 quede al lado de la 1
    angulo = 2 * np.pi * datos["WM_WEEK_OF_YEAR"] / 53
    datos["SEMANA_SIN"] = np.sin(angulo)
    datos["SEMANA_COS"] = np.cos(angulo)
    for col in ["IS_SNAP", "IS_WEEKEND", "IS_EVENT_DAY", "IS_PRE_LAUNCH"]:
        datos[col] = datos[col].astype(int)
    return datos


def cargar_y_dividir(ruta=RUTA_DATOS):
    datos = agregar_features(pd.read_csv(ruta, low_memory=False))
    X = datos[COLUMNAS_ENTRADA]
    y = datos[OBJETIVO]
    # El modelo va a predecir ventas futuras, asi que se entrena con lo
    # anterior a la fecha de corte y se prueba con lo posterior.
    es_prueba = datos["DATE_KEY"] >= FECHA_CORTE
    return X[~es_prueba], X[es_prueba], y[~es_prueba], y[es_prueba]


def construir_preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
        ("producto", codificador_producto(), PRODUCTO),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])

if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    preparacion = construir_preparacion()
    X_ent_listo = preparacion.fit_transform(X_ent, y_ent)
    X_prueba_listo = preparacion.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(preparacion.get_feature_names_out()))
