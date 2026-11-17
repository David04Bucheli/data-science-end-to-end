import numpy as np
import pandas as pd

from pipeline_proyecto import (
    COLUMNAS_ENTRADA,
    FECHA_CORTE,
    OBJETIVO,
    agregar_features,
    cargar_y_dividir,
    construir_preparacion,
)


def viajes_de_prueba(n=20):
    """Un DataFrame chico escrito a mano, con las columnas del CSV que usa el pipeline."""
    return pd.DataFrame({
        "DATE_KEY": [20150101 + i * 10 for i in range(n)],
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


def test_el_objetivo_y_las_columnas_con_fuga_no_entran_al_modelo():
    prohibidas = {OBJETIVO, "REVENUE", "IS_SALES_SPIKE",
                  "IS_SUSPECTED_STOCKOUT_CONSERVATIVE", "IS_SUSPECTED_STOCKOUT_NB"}
    assert not prohibidas & set(COLUMNAS_ENTRADA)


def test_precio_log_es_el_logaritmo_del_precio():
    datos = viajes_de_prueba()
    datos.loc[0, "SELL_PRICE"] = np.e
    resultado = agregar_features(datos)
    assert resultado.loc[0, "PRECIO_LOG"] == 1.0


def test_la_semana_53_queda_cerca_de_la_semana_1():
    datos = viajes_de_prueba(4).head(3)
    datos["WM_WEEK_OF_YEAR"] = [1, 27, 53]
    r = agregar_features(datos)
    distancia_1_53 = np.hypot(r.loc[0, "SEMANA_SIN"] - r.loc[2, "SEMANA_SIN"],
                              r.loc[0, "SEMANA_COS"] - r.loc[2, "SEMANA_COS"])
    distancia_1_27 = np.hypot(r.loc[0, "SEMANA_SIN"] - r.loc[1, "SEMANA_SIN"],
                              r.loc[0, "SEMANA_COS"] - r.loc[1, "SEMANA_COS"])
    assert distancia_1_53 < 0.2
    assert distancia_1_27 > 1.5


def test_la_division_pone_el_futuro_en_prueba(tmp_path):
    datos = viajes_de_prueba(40)
    datos["DATE_KEY"] = [FECHA_CORTE - 20 + i for i in range(40)]
    ruta = tmp_path / "ventas.csv"
    datos.to_csv(ruta, index=False)

    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir(str(ruta))

    assert len(X_ent) == 20 and len(X_prueba) == 20
    assert len(X_ent) == len(y_ent) and len(X_prueba) == len(y_prueba)
    fechas = pd.read_csv(ruta)["DATE_KEY"]
    assert fechas.loc[X_ent.index].max() < FECHA_CORTE <= fechas.loc[X_prueba.index].min()


def test_la_preparacion_acepta_categorias_que_no_vio_al_entrenar():
    datos = agregar_features(viajes_de_prueba())
    preparacion = construir_preparacion()
    preparacion.fit(datos[COLUMNAS_ENTRADA], datos[OBJETIVO])

    nuevo = datos[COLUMNAS_ENTRADA].head(2).copy()
    nuevo["STORE_ID"] = "WI_9"        # tienda nueva
    nuevo["ITEM_ID"] = "FOODS_3_999"  # producto nuevo
    nuevo["PRECIO_LOG"] = np.nan      # un producto sin precio todavia

    listo = preparacion.transform(nuevo)
    assert listo.shape[0] == 2
    assert not np.isnan(listo).any()
