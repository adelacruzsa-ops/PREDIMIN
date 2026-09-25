"""
PREDIMIN - Modulo 1: carga, control de calidad y preparacion de variables.

Flujo (Zhang et al., 2026): organizacion -> control de calidad ->
tratamiento de variables -> division 80:20.
"""

import numpy as np
import pandas as pd

from config import ARCHIVO_DATOS, OBJETIVOS, PROPORCION_PRUEBA, SEMILLA, VARIABLES_ENTRADA


def cargar(ruta=ARCHIVO_DATOS) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute primero: python src/importar_excel.py")
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    faltantes = [c for c in VARIABLES_ENTRADA + OBJETIVOS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el archivo de datos: {faltantes}")
    return df


def control_de_calidad(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Solo se conservan registros con variable objetivo y meteorologia completas."""
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=OBJETIVOS + ["humedad_relativa_pct", "velocidad_viento_ms",
                                       "direccion_viento_grados"])
    if verbose:
        print(f"[QC] Registros: {n0} -> {len(df)} "
              f"({n0 - len(df)} sin PM10 o sin meteorologia)")
    return df.reset_index(drop=True)


def ingenieria_de_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas con sentido fisico, calculadas a partir de campos existentes."""
    df = df.copy()

    # Direccion del viento: variable circular (359 y 1 grado son vecinos)
    rad = np.deg2rad(df["direccion_viento_grados"])
    df["viento_sin"] = np.sin(rad)
    df["viento_cos"] = np.cos(rad)

    # Mes: tambien circular (diciembre y enero son vecinos) -> estacionalidad
    ang = 2 * np.pi * df["mes"] / 12
    df["mes_sin"] = np.sin(ang)
    df["mes_cos"] = np.cos(ang)

    # Factor de carga en kg/t (no hay volumen, pero si tonelaje)
    df["factor_carga_kg_t"] = df["explosivo_total_kg"] / df["tonelaje_tm"]

    # Explosivo por taladro
    df["explosivo_por_taladro"] = df["explosivo_total_kg"] / df["numero_taladros"]

    # Proporcion de emulsion en la mezcla
    df["fraccion_emulsion"] = df["emulsion_kg"] / df["explosivo_total_kg"]

    # Indice seco: baja humedad + viento (condicion mas desfavorable)
    df["indice_seco"] = (100 - df["humedad_relativa_pct"]) / 100 * df["velocidad_viento_ms"]

    return df.replace([np.inf, -np.inf], np.nan)


VARIABLES_DERIVADAS = [
    "viento_sin", "viento_cos", "mes_sin", "mes_cos", "factor_carga_kg_t",
    "explosivo_por_taladro", "fraccion_emulsion", "indice_seco",
]

EXCLUIR_DEL_MODELO = {"direccion_viento_grados", "mes"}   # se usan sus versiones seno/coseno


def columnas_modelo():
    return [c for c in VARIABLES_ENTRADA if c not in EXCLUIR_DEL_MODELO] + VARIABLES_DERIVADAS


def matriz_de_variables(df: pd.DataFrame, medianas: dict | None = None):
    """X numerica. Los faltantes se imputan con la mediana del ENTRENAMIENTO."""
    columnas = columnas_modelo()
    X = df[columnas].astype(float).copy()
    if medianas is not None:
        X = X.fillna(medianas)
    return X, columnas


def dividir(df: pd.DataFrame, objetivo: str):
    """
    Division 80:20 estratificada por cuartiles del PM10.
    Las medianas para imputar se calculan solo con el entrenamiento (sin fuga).
    """
    from sklearn.model_selection import train_test_split

    X, columnas = matriz_de_variables(df)
    y = df[objetivo]
    estratos = pd.qcut(y, q=4, labels=False, duplicates="drop")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, random_state=SEMILLA, stratify=estratos)

    medianas = X_tr.median().to_dict()
    return X_tr.fillna(medianas), X_te.fillna(medianas), y_tr, y_te, columnas, medianas


def preparar(ruta=ARCHIVO_DATOS, verbose: bool = True) -> pd.DataFrame:
    df = cargar(ruta)
    df = control_de_calidad(df, verbose=verbose)
    df = ingenieria_de_variables(df)
    return df


if __name__ == "__main__":
    d = preparar()
    print(d.shape)
    print(d[columnas_modelo() + OBJETIVOS].describe().T.round(2))
