"""
PREDIMIN - Modulo 0: importacion y depuracion del Excel de la mina.

Lee "data/Pm 10 nas voladura.xlsx", corrige los problemas detectados y guarda
"data/registros_voladura.csv", que es lo que usan los demas modulos.

Problemas que corrige (y que deben mencionarse en la tesis, seccion de
depuracion de datos):
  1. PM10 = 0  -> se trata como dato FALTANTE (falla o mantenimiento del
     monitor). Un PM10 horario de exactamente 0 ug/m3 no es fisicamente
     plausible y los ceros se concentran en jun, sep y oct de 2023.
  2. "S/D" (sin dato) en variables meteorologicas -> faltante.
  3. "TOTAL DE EXPLOSIVOS" es el doble del Heavy ANFO (suma ANFO + emulsion +
     Heavy ANFO, que ya es la mezcla de ambos). Se usa Heavy ANFO como total.
  4. Tiempos de retardo escritos como texto ("7 y 10", "13, 17 y 42") -> se
     extrae el menor valor y el numero de retardos distintos.
  5. Valores imposibles (tonelaje 0, retardo 46 246 ms, P80 = 197) -> faltante.
  6. Columnas constantes (diametro 6 1/8", taco de grava, limite 45) -> se
     descartan porque no aportan informacion al modelo.

Uso:
    python src/importar_excel.py
"""

import re

import numpy as np
import pandas as pd

from config import ARCHIVO_DATOS, ARCHIVO_EXCEL


def _numeros(v):
    """Extrae todos los numeros de un texto como '13, 17 y 42 ms'."""
    if pd.isna(v):
        return []
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(v))]


def _hora(v):
    if pd.isna(v):
        return np.nan
    if hasattr(v, "hour"):
        return v.hour
    m = re.match(r"(\d{1,2})", str(v))
    return float(m.group(1)) if m else np.nan


def importar(ruta=ARCHIVO_EXCEL, verbose=True) -> pd.DataFrame:
    raw = pd.read_excel(ruta)
    raw.columns = [str(c).strip() for c in raw.columns]

    def col(nombre):
        # busca la columna ignorando mayusculas y tildes parciales
        for c in raw.columns:
            if c.lower().startswith(nombre.lower()):
                return raw[c]
        raise KeyError(f"No se encontro la columna que empieza por '{nombre}'")

    num = lambda s: pd.to_numeric(s, errors="coerce")

    df = pd.DataFrame({
        "fecha": pd.to_datetime(col("Fecha"), errors="coerce"),
        "hora": col("Hora").map(_hora),
        "pm10_ugm3": num(col("PM10")),
        "humedad_relativa_pct": num(col("Humedad")),
        "direccion_viento_grados": num(col("Direcci")),
        "velocidad_viento_ms": num(col("Velocidad")),
        "precipitacion_mm": num(col("Precipitaci")),
        "condicion_lluvia": col("Rango de clasificaci"),
        "estacion_anio": col("Estaciones"),
        "n_eventos": num(col("N° de eventos")),
        "numero_taladros": num(col("Numero de taladros")),
        "tonelaje_tm": num(col("Tonelaje")),
        "anfo_kg": num(col("Kilogramos de ANFO")),
        "emulsion_kg": num(col("Kilogramos de Emulsion")),
        "explosivo_total_kg": num(col("Kilogramos de Heavy ANFO 1")),
        "p80": num(col("P80")),
    })
    df["mes"] = df["fecha"].dt.month

    t_tal = col("Tiempo entre taladro").map(_numeros)
    t_fil = col("Tiempo entre fila").map(_numeros)
    df["tiempo_taladro_ms"] = t_tal.map(lambda l: min(l) if l else np.nan)
    df["tiempo_fila_ms"] = t_fil.map(lambda l: min(l) if l else np.nan)
    df["n_retardos_taladro"] = t_tal.map(len)
    df["n_retardos_fila"] = t_fil.map(len)

    n = len(df)
    log = {}

    # 1. PM10 = 0 -> faltante
    log["PM10 = 0 (falla del monitor) -> faltante"] = int((df["pm10_ugm3"] == 0).sum())
    df.loc[df["pm10_ugm3"] <= 0, "pm10_ugm3"] = np.nan

    # 5. valores imposibles
    reglas = {
        "tonelaje_tm": (1, None),
        "tiempo_taladro_ms": (0, 1000),
        "tiempo_fila_ms": (0, 2000),
        "p80": (0, 50),
        "humedad_relativa_pct": (0, 100),
        "velocidad_viento_ms": (0, 40),
        "direccion_viento_grados": (0, 360),
    }
    for c, (lo, hi) in reglas.items():
        fuera = pd.Series(False, index=df.index)
        if lo is not None:
            fuera |= df[c] < lo
        if hi is not None:
            fuera |= df[c] > hi
        if fuera.any():
            log[f"{c} fuera de rango -> faltante"] = int(fuera.sum())
        df.loc[fuera, c] = np.nan

    df = df.sort_values(["fecha", "hora"]).reset_index(drop=True)
    log["Registros leidos"] = n
    log["Registros con PM10 valido"] = int(df["pm10_ugm3"].notna().sum())
    df.attrs["log_depuracion"] = log

    if verbose:
        print(f"[Importacion] {n} registros leidos de {ruta.name}")
        for k, v in log.items():
            print(f"  - {k}: {v}")
        print(f"  Registros con PM10 valido: {df['pm10_ugm3'].notna().sum()}")
        print(f"  Periodo: {df['fecha'].min():%d/%m/%Y} - {df['fecha'].max():%d/%m/%Y}")
    return df


if __name__ == "__main__":
    d = importar()
    d.to_csv(ARCHIVO_DATOS, index=False, date_format="%Y-%m-%d")
    print(f"[OK] Datos depurados -> {ARCHIVO_DATOS}")
