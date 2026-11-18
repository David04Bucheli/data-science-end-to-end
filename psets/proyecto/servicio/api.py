"""Servicio de prediccion de unidades vendidas, instrumentado.

Ademas de responder, mide cuanto tarda cada prediccion y con que codigo
responde (las metricas del sistema, en /metricas), y guarda cada
prediccion en monitoreo/predicciones.jsonl, para medir despues cuanto se
equivoco el modelo cuando se conozcan las ventas reales
(monitoreo/desempeno.py).

Se levanta desde psets/proyecto/:

    uvicorn servicio.api:app --port 8000
"""

import json
import os
import time
import uuid
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

# las features se calculan con las mismas funciones del entrenamiento
from modelado.pipeline_proyecto import COLUMNAS_ENTRADA, agregar_features

CARPETA = Path(__file__).parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
# un archivo JSONL: una linea por prediccion. En Docker, monta la carpeta con -v
# para que el registro sobreviva al contenedor.
RUTA_REGISTRO = Path(os.environ.get("RUTA_REGISTRO", "monitoreo/predicciones.jsonl"))

app = FastAPI(title="Prediccion de ventas")


class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por
    cada peticion a /prediccion. Se reinician si el servicio se reinicia."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):
        p50 = p95 = None
        if self.latencias_ms:
            p50 = round(float(np.percentile(self.latencias_ms, 50)), 2)
            p95 = round(float(np.percentile(self.latencias_ms, 95)), 2)
        return {
            "peticiones": len(self.codigos),
            "errores_validacion": sum(1 for c in self.codigos if c == 422),
            "errores_servidor": sum(1 for c in self.codigos if c >= 500),
            "latencia_p50_ms": p50,
            "latencia_p95_ms": p95,
        }


metricas = Metricas()


@app.middleware("http")
async def medir(request: Request, call_next):
    """Corre alrededor de cada peticion a /prediccion: mide la latencia y
    anota el codigo de la respuesta."""
    if request.url.path != "/prediccion":
        return await call_next(request)
    inicio = time.perf_counter()
    try:
        respuesta = await call_next(request)
    except Exception:
        # una excepcion sin atrapar es un error de servidor aunque no haya respuesta
        metricas.registrar((time.perf_counter() - inicio) * 1000, 500)
        raise
    metricas.registrar((time.perf_counter() - inicio) * 1000, respuesta.status_code)
    return respuesta


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
    # El modelo no la usa, pero es lo que permite unir despues la prediccion con
    # las ventas reales (producto + tienda + fecha): sin ella no hay forma de saber
    # a que dia se refiere cada prediccion.
    fecha: Optional[date] = Field(None, description="el dia que se predice (AAAA-MM-DD)")


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


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


def guardar_prediccion(registro):
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_REGISTRO, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro) + "\n")


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
    unidades = round(max(float(modelo.predict(X)[0]), 0.0), 3)  # no hay ventas negativas
    id_prediccion = uuid.uuid4().hex
    # con el id, el momento, la version, la entrada y la salida se puede unir despues
    # con las ventas reales y medir el error por periodo y por version
    guardar_prediccion({
        "id_prediccion": id_prediccion,
        "momento": datetime.now().isoformat(),
        "version_modelo": version_modelo(),
        **venta.model_dump(mode="json"),
        "unidades_estimadas": unidades,
    })
    return {"id_prediccion": id_prediccion, "unidades_estimadas": unidades, "version_modelo": version_modelo()}