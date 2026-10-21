import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# TODO 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
print("\nTarea 1")
limpios = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)]
print(f"Filas originales: {len(viajes)} | Filas tras quitar errores: {len(limpios)} "
      f"(se quitaron {len(viajes) - len(limpios)})")


# TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general y
# cuentalos con un filtro.

plt.figure(figsize=(10, 6))
plt.scatter(limpios["distancia_km"], limpios["tarifa"])
plt.title("Distancia vs tarifa")
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.savefig("figuras/distancia_vs_tarifa.png")
plt.close()


print("\nTarea 2")
# Puntos que no siguen la tendencia: viajes largos con tarifa plana de 70
jfk_plana = limpios[limpios["tarifa"] == 70.0]
print("\nViajes con tarifa exactamente 70.0:", len(jfk_plana))
print(jfk_plana["zona_destino"].value_counts().to_string())
print(jfk_plana["zona_origen"].value_counts().head(3).to_string())
print("Rango de distancia de esos viajes (km):", jfk_plana["distancia_km"].min(), "-", jfk_plana["distancia_km"].max())



# TODO 3: propina promedio y cantidad de viajes con propina 0, por
# metodo de pago (groupby + agg).

print("\nTarea 3")

por_pago = (
    limpios.groupby("metodo_pago", dropna=False)
    .agg(
        viajes=("propina", "size"),
        propina_promedio=("propina", "mean"),
        propina_cero=("propina", lambda s: (s == 0).sum()),
    )
    .round(2)
)
por_pago["pct_propina_cero"] = (por_pago["propina_cero"] / por_pago["viajes"] * 100).round(1)
print("\nPropina por metodo de pago:")
print(por_pago)


# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
# con la tarifa.

print("\nTarea 4")

cols = ["distancia_km", "duracion_min", "pasajeros", "propina"]
corr = limpios[cols + ["tarifa"]].corr()["tarifa"].drop("tarifa").round(3)
print("\nCorrelacion (Pearson) con la tarifa:")
print(corr.to_string())


# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.

print("\nTarea 5")

print("\nPasajeros vacio (NaN):", limpios["pasajeros"].isna().sum())
print("Pasajeros igual a 0:", (limpios["pasajeros"] == 0).sum())
