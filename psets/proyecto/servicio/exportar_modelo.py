"""Exporta el modelo final desde MLflow a servicio/modelo.joblib.

Uso, desde psets/proyecto/:

    python3 servicio/exportar_modelo.py <run_id>

El archivo no se sube al repositorio: se regenera con este script.
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow.sklearn

CARPETA = Path(__file__).parent


def exportar(run_id):
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")
    joblib.dump(modelo, CARPETA / "modelo.joblib")
    # guardo de que corrida salio el archivo, para poder rastrear una prediccion
    (CARPETA / "modelo_info.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    print("Modelo exportado de la corrida", run_id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python3 servicio/exportar_modelo.py <run_id>")
    exportar(sys.argv[1])
