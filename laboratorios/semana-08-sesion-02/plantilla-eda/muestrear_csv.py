"""Saca una muestra aleatoria de un CSV gigante sin cargarlo en memoria.

Lee el archivo por pedazos (chunks), conserva cada fila con probabilidad
p = filas_deseadas / filas_totales y va escribiendo la muestra a disco.
Solo hay un pedazo en memoria a la vez, asi que no se cuelga VS Code.

Uso (desde la carpeta plantilla-eda):

    python3 muestrear_csv.py RUTA_AL_CSV_COMPLETO
    python3 muestrear_csv.py RUTA_AL_CSV_COMPLETO --filas 150000 --salida data/obt_sales.csv
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np


def contar_filas(ruta):
    """Cuenta filas (sin la cabecera) leyendo bytes; es rapido y usa poca memoria.
    Supone que ninguna celda tiene saltos de linea dentro de comillas."""
    n = 0
    ultimo = b"\n"
    with open(ruta, "rb") as f:
        while bloque := f.read(1 << 24):  # bloques de 16 MB
            n += bloque.count(b"\n")
            ultimo = bloque[-1:]
    if ultimo != b"\n":  # el archivo no termina en salto de linea
        n += 1
    return max(n - 1, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", help="CSV completo")
    ap.add_argument("--salida", default="data/obt_sales_muestra.csv", help="donde guardar la muestra")
    ap.add_argument("--filas", type=int, default=200_000, help="filas aproximadas de la muestra (default 200000)")
    ap.add_argument("--total", type=int, default=None, help="filas totales del CSV; si no lo das, se cuentan")
    ap.add_argument("--semilla", type=int, default=42, help="semilla para que la muestra sea reproducible")
    ap.add_argument("--chunk", type=int, default=500_000, help="filas leidas por pedazo")
    args = ap.parse_args()

    if os.path.abspath(args.entrada) == os.path.abspath(args.salida):
        sys.exit("La salida no puede ser el mismo archivo que la entrada.")

    total = args.total if args.total else contar_filas(args.entrada)
    if total == 0:
        sys.exit("El archivo no tiene filas.")
    p = min(1.0, args.filas / total)
    print(f"Filas totales: {total:,} | probabilidad de muestreo: {p:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.salida)), exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    escritas = 0
    primera = True
    # dtype=str + keep_default_na=False: se copia el texto tal cual, sin que pandas
    # reinterprete numeros, fechas o vacios. Menos memoria y cero cambios en los datos.
    lector = pd.read_csv(args.entrada, chunksize=args.chunk, dtype=str, keep_default_na=False)
    for i, pedazo in enumerate(lector, 1):
        muestra = pedazo[rng.random(len(pedazo)) < p]
        muestra.to_csv(args.salida, mode="w" if primera else "a", header=primera, index=False)
        primera = False
        escritas += len(muestra)
        print(f"  pedazo {i}: {min(i * args.chunk, total):,} filas leidas | {escritas:,} en la muestra", end="\r")
    print()

    mb = os.path.getsize(args.salida) / 1e6
    print(f"Listo: {escritas:,} filas en {args.salida} ({mb:.1f} MB)")
    if mb > 50:
        print("AVISO: GitHub avisa desde 50 MB y bloquea desde 100 MB. Baja --filas.")


if __name__ == "__main__":
    main()