"""Preparacion de los datos de mi proyecto, lista para produccion.

Las features que decidiste en tu notebook de la sesion pasada viven aqui,
dentro de funciones. Desde la Semana 10, tus notebooks de experimentos
importan este archivo en vez de copiar el codigo:

    from pipeline_proyecto import cargar_y_dividir, construir_preparacion

Para comprobar que corre de principio a fin, sin notebook:

    python3 pipeline_proyecto.py
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

<<<<<<< HEAD
=======
# TODO: el mismo archivo de tu notebook de features de la sesion pasada.
RUTA_DATOS = "../data/processed/TODO_archivo.csv"
>>>>>>> 0459292cee18bfab206bcbb0deb753978024219a

datos = pd.read_csv("obt_sales.csv")
datos["DATE"] = pd.to_datetime(datos["DATE"])

<<<<<<< HEAD
OBJETIVO = "UNITS"

fecha_corte = datos["DATE"].quantile(0.8)
datos_ent = datos[datos["DATE"] <= fecha_corte]
datos_prueba = datos[datos["DATE"] > fecha_corte]
precio_promedio_item = datos_ent.groupby("ITEM_ID")["SELL_PRICE"].mean()
 
def agregar_features(df):
    df = df.copy()
    df["precio_relativo"] = df["SELL_PRICE"] / df["ITEM_ID"].map(precio_promedio_item)
    df["en_descuento"] = (df["precio_relativo"] < 0.95).astype(int)
    df["snap_fin_de_semana"] = (df["IS_SNAP"] & df["IS_WEEKEND"]).astype(int)
    return df
 
datos_ent = agregar_features(datos_ent)
datos_prueba = agregar_features(datos_prueba)
 
NUMERICAS = ["SELL_PRICE", "precio_relativo"]
CATEGORICAS = ["CAT_ID", "STORE_ID"]
YA_LISTAS = ["en_descuento", "snap_fin_de_semana"]
 
COLUMNAS = NUMERICAS + CATEGORICAS + YA_LISTAS
X_ent, y_ent = datos_ent[COLUMNAS], datos_ent[OBJETIVO]
X_prueba, y_prueba = datos_prueba[COLUMNAS], datos_prueba[OBJETIVO]
 
preparacion = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ]), NUMERICAS),
    ("categoricas", Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAS),
    ("ya_listas", "passthrough", YA_LISTAS),
])
 
X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
print("Fecha de corte:", fecha_corte.date())
print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
print(list(preparacion.get_feature_names_out()))
 
=======

def agregar_features(datos):
    # TODO: copia aqui, desde tu notebook de la sesion pasada, solo las
    # lineas que construyen tus 2 features. Sin graficos, sin prints y sin
    # las celdas que usaste para probar.
    return datos


def cargar_y_dividir(ruta=RUTA_DATOS):
    datos = agregar_features(pd.read_csv(ruta))
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
    y = datos[OBJETIVO]
    # TODO: decide como dividir. Si tus filas tienen fecha y vas a predecir
    # el futuro, divide por fecha (entrenamiento antes de una fecha de corte,
    # prueba despues) en vez de al azar.
    return train_test_split(X, y, test_size=0.2, random_state=42)


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
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


# Esto corre con "python3 pipeline_proyecto.py", pero no cuando un
# notebook hace "from pipeline_proyecto import ...".
if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    preparacion = construir_preparacion()
    X_ent_listo = preparacion.fit_transform(X_ent)
    X_prueba_listo = preparacion.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(preparacion.get_feature_names_out()))
>>>>>>> 0459292cee18bfab206bcbb0deb753978024219a
