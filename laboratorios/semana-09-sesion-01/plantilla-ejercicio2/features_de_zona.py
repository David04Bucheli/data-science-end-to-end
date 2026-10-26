import pandas as pd

viajes = pd.read_csv("../starter/viajes_taxi.csv")
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()

ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

# TODO 1: crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
# va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
# JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
# y maxima de esos viajes.

jfk_a_manhattan = (viajes["zona_origen"] == "JFK Airport") & viajes["zona_destino"].isin(ZONAS_MANHATTAN)
manhattan_a_jfk = viajes["zona_origen"].isin(ZONAS_MANHATTAN) & (viajes["zona_destino"] == "JFK Airport")
viajes["es_tarifa_fija"] = (jfk_a_manhattan | manhattan_a_jfk).astype(int)
 
fijos = viajes[viajes["es_tarifa_fija"] == 1]
print("Viajes con es_tarifa_fija = 1:", len(fijos))
print("Tarifa minima:", fijos["tarifa"].min())
print("Tarifa maxima:", fijos["tarifa"].max())

# TODO 2: codifica zona_origen con one-hot (pd.get_dummies), con el
# prefijo "origen" y dtype=int. Imprime los nombres de las columnas
# nuevas.

origen_onehot = pd.get_dummies(viajes["zona_origen"], prefix="origen", dtype=int)
print("Columnas one-hot:", list(origen_onehot.columns))
print("Cantidad de columnas one-hot:", origen_onehot.shape[1])

# TODO 3: codifica zona_origen de forma ordinal: un numero entero por
# zona, en orden alfabetico, en una columna origen_ordinal.

zonas_ordenadas = sorted(viajes["zona_origen"].unique())
codigo_zona = {zona: i for i, zona in enumerate(zonas_ordenadas)}
viajes["origen_ordinal"] = viajes["zona_origen"].map(codigo_zona)
print("Codificacion ordinal:", codigo_zona)
print("Cantidad de columnas ordinales: 1 (origen_ordinal)")

# TODO 4: para cada columna original del dataset, decide si existe en el
# momento en que alguien pide el taxi (antes del viaje) e imprime la
# decision.

existe_al_pedir = {
    "id_viaje": True,           # se asigna al registrar la solicitud
    "fecha_hora_inicio": True,  # el momento del pedido
    "zona_origen": True,        # el pasajero indica donde esta
    "zona_destino": True,       # el pasajero indica a donde va
    "distancia_km": True,       # se estima con origen y destino
    "duracion_min": False,      # solo se conoce al terminar el viaje
    "pasajeros": True,          # se indica al pedir el taxi
    "metodo_pago": False,       # se elige al pagar, al final
    "tarifa": False,            # es el objetivo que se quiere predecir
    "propina": False,           # se da despues del viaje
}
for columna, existe in existe_al_pedir.items():
    print(f"{columna}: {'existe' if existe else 'NO existe'} al pedir el taxi")