"""Exporta a servicio/ la version de "ventas-walmart-m5" que tiene el alias
"produccion", y guarda su numero junto al modelo.

Uso, desde psets/proyecto/ y con el servidor de MLflow corriendo:

    python3 servicio/exportar_modelo.py

Deja dos archivos, que la imagen copia adentro (el contenedor no necesita
llegar al servidor de MLflow):

    servicio/modelo.joblib        el modelo
    servicio/modelo_version.txt   el numero de version del registro

Ninguno se sube al repositorio: se regeneran con este script. Para exportar
otra version, primero muevela a produccion con servicio/promover.py.
"""

import sys
from pathlib import Path

import joblib
import mlflow.sklearn
from mlflow import MlflowClient

CARPETA = Path(__file__).parent
NOMBRE = "ventas-walmart-m5"
ALIAS = "produccion"


def exportar():
    version = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS).version
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{version}")
    joblib.dump(modelo, CARPETA / "modelo.joblib")
    # el numero viaja con el modelo: asi cada prediccion puede decir quien la calculo
    (CARPETA / "modelo_version.txt").write_text(str(version), encoding="utf-8")
    print(f"Modelo exportado: version {version} de {NOMBRE} (alias {ALIAS})")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        sys.exit("Uso: python3 servicio/exportar_modelo.py  (sin argumentos: exporta la version con alias produccion)")
    exportar()