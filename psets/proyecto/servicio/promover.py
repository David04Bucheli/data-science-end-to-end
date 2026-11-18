"""Entrena el modelo final de ventas, guarda su metrica principal en la
corrida y lo registra como una version NUEVA de "ventas-walmart-m5".

Uso, desde psets/proyecto/ y con el servidor de MLflow corriendo:

    python -m servicio.registrar_modelo 20

El numero es min_samples_leaf del bosque (el hiperparametro que se compara
entre versiones). Cada vez que lo corres el registro crea una version
nueva, con su corrida, su parametro y sus metricas: ninguna reemplaza a otra.
"""

import logging
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import RUTA_DATOS, cargar_y_dividir, construir_preparacion

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "ventas-walmart-m5"  # nombre del modelo en el registro (el del experimento)
METRICA = "rmse_prueba"       # la metrica principal: la que decide que version va a produccion

# mismos costos que modelado/validacion_proyecto.ipynb: quedarse corto cuesta 3, pedir de mas cuesta 1
COSTO_FALTANTE = 3
COSTO_SOBRANTE = 1


def costo_negocio(real, estimado):
    real = np.asarray(real)
    faltante = np.maximum(real - estimado, 0)
    sobrante = np.maximum(estimado - real, 0)
    return float((COSTO_FALTANTE * faltante + COSTO_SOBRANTE * sobrante).mean())


def entrenar_y_medir(min_samples_leaf, ruta=RUTA_DATOS):
    """Entrena el bosque con la misma preparacion y division que el resto del
    proyecto. Devuelve el pipeline entrenado y sus metricas."""
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir(ruta)
    pipeline = Pipeline([
        ("preparar", construir_preparacion()),
        ("modelo", RandomForestRegressor(
            n_estimators=100, min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X_ent, y_ent)
    estimado = pipeline.predict(X_prueba)
    metricas = {
        "rmse_entrenamiento": root_mean_squared_error(y_ent, pipeline.predict(X_ent)),
        METRICA: root_mean_squared_error(y_prueba, estimado),
        "costo_negocio": costo_negocio(y_prueba, estimado),
    }
    return pipeline, metricas


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python -m servicio.registrar_modelo <min_samples_leaf>")
    min_samples_leaf = int(sys.argv[1])
    # RUTA_DATOS_PROYECTO permite apuntar a otro CSV (p. ej. una muestra) sin tocar el codigo
    ruta = os.environ.get("RUTA_DATOS_PROYECTO", RUTA_DATOS)

    mlflow.set_experiment(f"{NOMBRE}-registro")
    with mlflow.start_run(run_name=f"bosque min_samples_leaf={min_samples_leaf}"):
        pipeline, metricas = entrenar_y_medir(min_samples_leaf, ruta)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metrics(metricas)
        # guarda el modelo en la corrida y lo registra a la vez: crea una version nueva
        info = mlflow.sklearn.log_model(pipeline, "modelo", registered_model_name=NOMBRE)
        version = info.registered_model_version

    print(f"Version {version} de {NOMBRE}: min_samples_leaf={min_samples_leaf}, "
          f"{METRICA} {metricas[METRICA]:.4f}, costo_negocio {metricas['costo_negocio']:.4f}")