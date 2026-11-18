import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import COLUMNAS_ENTRADA, OBJETIVO, agregar_features, construir_preparacion
from servicio import api
from servicio.api import app, cargar_modelo


@pytest.fixture
def registro(tmp_path, monkeypatch):
    """Un registro de predicciones temporal y unas metricas en blanco: las pruebas no escriben en monitoreo/."""
    ruta = tmp_path / "monitoreo" / "predicciones.jsonl"
    monkeypatch.setattr("servicio.api.RUTA_REGISTRO", ruta)
    monkeypatch.setattr("servicio.api.metricas", api.Metricas())
    return ruta


def leer_registro(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo]


@pytest.fixture
def cliente(monkeypatch, registro):
    """Un modelo de juguete entrenado con 20 filas escritas a mano: el modelo.joblib real no existe en GitHub.
    La version tambien es de juguete ("7", inventada): modelo_version.txt tampoco existe en GitHub."""
    monkeypatch.setattr("servicio.api.version_modelo", lambda: "7")
    n = 20
    datos = pd.DataFrame({
        "ITEM_ID": [f"FOODS_3_{i % 4:03d}" for i in range(n)],
        "DEPT_ID": ["FOODS_3"] * n,
        "STORE_ID": ["CA_1", "TX_2"] * (n // 2),
        "SELL_PRICE": [2.0 + i for i in range(n)],
        "WDAY": [1 + i % 7 for i in range(n)],
        "WM_WEEK_OF_YEAR": [1 + i for i in range(n)],
        "IS_SNAP": [True, False] * (n // 2),
        "IS_WEEKEND": [False] * n,
        "IS_EVENT_DAY": [False] * n,
        "IS_PRE_LAUNCH": [False] * n,
        "UNITS": [i % 3 for i in range(n)],
    })
    datos = agregar_features(datos)
    modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", DummyRegressor(strategy="mean"))])
    modelo.fit(datos[COLUMNAS_ENTRADA], datos[OBJETIVO])

    app.dependency_overrides[cargar_modelo] = lambda: modelo
    yield TestClient(app)
    app.dependency_overrides.clear()


VENTA_VALIDA = {
    "item_id": "FOODS_3_090", "dept_id": "FOODS_3", "store_id": "CA_1",
    "sell_price": 1.5, "wday": 3, "semana_del_anio": 20,
    "is_snap": True, "is_weekend": False, "is_event_day": False, "is_pre_launch": False,
}


def test_salud(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "version_modelo": "7"}


def test_prediccion_valida(cliente):
    respuesta = cliente.post("/prediccion", json=VENTA_VALIDA)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["unidades_estimadas"] >= 0
    assert "version_modelo" in cuerpo


def test_cada_prediccion_dice_que_version_la_calculo(cliente):
    respuesta = cliente.post("/prediccion", json=VENTA_VALIDA)
    assert respuesta.status_code == 200
    assert respuesta.json()["version_modelo"] == "7"


def test_producto_sin_precio_se_acepta(cliente):
    venta = {**VENTA_VALIDA, "sell_price": None, "is_pre_launch": True}
    assert cliente.post("/prediccion", json=venta).status_code == 200


def test_dia_de_la_semana_fuera_de_rango_da_422(cliente):
    respuesta = cliente.post("/prediccion", json={**VENTA_VALIDA, "wday": 9})
    assert respuesta.status_code == 422


def test_falta_un_campo_obligatorio_da_422(cliente):
    sin_tienda = {k: v for k, v in VENTA_VALIDA.items() if k != "store_id"}
    assert cliente.post("/prediccion", json=sin_tienda).status_code == 422


def test_el_servicio_calcula_las_features_como_el_entrenamiento(registro):
    """Un modelo falso guarda lo que recibe, y lo comparo con agregar_features sobre la misma fila."""
    class Espia:
        def predict(self, X):
            self.recibido = X
            return [1.0]

    espia = Espia()
    app.dependency_overrides[cargar_modelo] = lambda: espia
    try:
        TestClient(app).post("/prediccion", json=VENTA_VALIDA)
    finally:
        app.dependency_overrides.clear()

    cruda = pd.DataFrame([{
        "ITEM_ID": "FOODS_3_090", "DEPT_ID": "FOODS_3", "STORE_ID": "CA_1", "SELL_PRICE": 1.5,
        "WDAY": 3, "WM_WEEK_OF_YEAR": 20, "IS_SNAP": True, "IS_WEEKEND": False,
        "IS_EVENT_DAY": False, "IS_PRE_LAUNCH": False,
    }])
    esperado = agregar_features(cruda)[COLUMNAS_ENTRADA]
    pd.testing.assert_frame_equal(espia.recibido, esperado)


def test_cada_prediccion_queda_registrada(cliente, registro):
    venta = {**VENTA_VALIDA, "fecha": "2015-05-12"}
    respuesta = cliente.post("/prediccion", json=venta).json()
    lineas = leer_registro(registro)
    assert len(lineas) == 1
    linea = lineas[0]
    assert linea["id_prediccion"] == respuesta["id_prediccion"]
    assert linea["version_modelo"] == "7"
    assert linea["unidades_estimadas"] == respuesta["unidades_estimadas"]
    # lo que hace falta para unirla con las ventas reales: producto, tienda y fecha
    assert (linea["item_id"], linea["store_id"], linea["fecha"]) == ("FOODS_3_090", "CA_1", "2015-05-12")
    assert "momento" in linea


def test_una_peticion_invalida_no_queda_registrada(cliente, registro):
    cliente.post("/prediccion", json={**VENTA_VALIDA, "wday": 9})
    assert not registro.exists()


def test_metricas_cuentan_peticiones_y_separan_los_errores(cliente):
    cliente.post("/prediccion", json=VENTA_VALIDA)
    cliente.post("/prediccion", json={**VENTA_VALIDA, "wday": 9})  # 422: la peticion esta mal
    resumen = cliente.get("/metricas").json()
    assert resumen["peticiones"] == 2
    assert resumen["errores_validacion"] == 1
    assert resumen["errores_servidor"] == 0
    assert resumen["latencia_p95_ms"] > 0 and resumen["latencia_p50_ms"] <= resumen["latencia_p95_ms"]


def test_un_error_del_modelo_cuenta_como_error_de_servidor(cliente):
    class Roto:
        def predict(self, X):
            raise RuntimeError("el modelo fallo")

    app.dependency_overrides[cargar_modelo] = lambda: Roto()
    sin_excepcion = TestClient(app, raise_server_exceptions=False)
    assert sin_excepcion.post("/prediccion", json=VENTA_VALIDA).status_code == 500
    resumen = sin_excepcion.get("/metricas").json()
    assert resumen["errores_servidor"] == 1
    assert resumen["errores_validacion"] == 0