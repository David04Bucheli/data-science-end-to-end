import pandas as pd

from monitoreo.desempeno import alertas, con_semana, error_por_semana_y_version, unir_con_ventas

VENTAS = pd.DataFrame({
    "ITEM_ID": ["A", "A", "B"], "STORE_ID": ["CA_1"] * 3,
    "DATE_KEY": [20150504, 20150505, 20150504], "UNITS": [2, 0, 4],
})


def predicciones():
    return con_semana(pd.DataFrame({
        "id_prediccion": ["1", "2", "3", "4"], "version_modelo": ["1"] * 4,
        "item_id": ["A", "A", "B", "B"], "store_id": ["CA_1"] * 4,
        "fecha": ["2015-05-04", "2015-05-05", "2015-05-04", "2015-05-20"],  # la ultima no tiene ventas aun
        "sell_price": [1.0, 1.0, None, 2.0], "unidades_estimadas": [2.0, 1.0, 1.0, 3.0],
    }))


def test_une_por_producto_tienda_y_dia_y_deja_vacio_lo_que_aun_no_se_vendio():
    unidas = unir_con_ventas(predicciones(), VENTAS)
    assert list(unidas["unidades_reales"].fillna(-1)) == [2, 0, 4, -1]


def test_el_error_ignora_las_predicciones_sin_ventas_y_se_calcula_por_semana():
    error = error_por_semana_y_version(unir_con_ventas(predicciones(), VENTAS))
    fila = error.loc[(pd.Timestamp("2015-05-04").date(), "1")]
    assert fila["predicciones"] == 3
    # errores: 0, 1 y 3 -> rmse = sqrt((0 + 1 + 9) / 3)
    assert abs(fila["rmse"] - 1.8257) < 0.001
    # costo: (0, sobrante 1 -> 1, faltante 3 -> 9) / 3
    assert abs(fila["costo_negocio"] - 3.3333) < 0.001


def test_alerta_solo_cuando_el_error_supera_la_referencia():
    error = error_por_semana_y_version(unir_con_ventas(predicciones(), VENTAS))
    # rmse 1.83 esta bajo la referencia; costo_negocio 3.33 supera 1.25 x 2.2891
    mensajes = alertas(error)
    assert len(mensajes) == 1 and "costo_negocio" in mensajes[0]