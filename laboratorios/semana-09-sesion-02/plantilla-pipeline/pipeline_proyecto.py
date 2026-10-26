import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


datos = pd.read_csv("obt_sales.csv")
datos["DATE"] = pd.to_datetime(datos["DATE"])

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
 