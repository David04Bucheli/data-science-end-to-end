"""Servicio de prediccion de unidades vendidas.

Se levanta desde psets/proyecto/:

    uvicorn servicio.api:app --port 8000
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

# las features se calculan con las mismas funciones del entrenamiento
from modelado.pipeline_proyecto import COLUMNAS_ENTRADA, agregar_features

CARPETA = Path(__file__).parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))

app = FastAPI(title="Prediccion de ventas")


class Venta(BaseModel):
    """Los datos como los conoce quien llama: sin features calculadas."""
    item_id: str
    dept_id: str
    store_id: str
    sell_price: Optional[float] = Field(None, gt=0, description="vacio si el producto aun no se vende")
    wday: int = Field(ge=1, le=7, description="dia de la semana, 1 a 7")
    semana_del_anio: int = Field(ge=1, le=53)
    is_snap: bool = False
    is_weekend: bool = False
    is_event_day: bool = False
    is_pre_launch: bool = False


@lru_cache
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@lru_cache
def version_modelo():
    """La version del registro de MLflow que exportar_modelo.py guardo junto al modelo."""
    archivo = CARPETA / "modelo_version.txt"
    if archivo.exists():
        return archivo.read_text(encoding="utf-8").strip()
    return "desconocida"


@app.get("/salud")
def salud():
    return {"estado": "ok", "version_modelo": version_modelo()}


@app.post("/prediccion")
def prediccion(venta: Venta, modelo=Depends(cargar_modelo)):
    # mismos nombres de columna que el CSV con el que se entreno
    fila = pd.DataFrame([{
        "ITEM_ID": venta.item_id,
        "DEPT_ID": venta.dept_id,
        "STORE_ID": venta.store_id,
        "SELL_PRICE": float("nan") if venta.sell_price is None else venta.sell_price,
        "WDAY": venta.wday,
        "WM_WEEK_OF_YEAR": venta.semana_del_anio,
        "IS_SNAP": venta.is_snap,
        "IS_WEEKEND": venta.is_weekend,
        "IS_EVENT_DAY": venta.is_event_day,
        "IS_PRE_LAUNCH": venta.is_pre_launch,
    }])
    X = agregar_features(fila)[COLUMNAS_ENTRADA]
    unidades = max(float(modelo.predict(X)[0]), 0.0)  # no hay ventas negativas
    return {"unidades_estimadas": round(unidades, 3), "version_modelo": version_modelo()}