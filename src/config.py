"""
PREDIMIN - Configuracion central del sistema (version 3: datos reales)
Tesis: Desarrollo del sistema inteligente de prediccion PREDIMIN para la
mitigacion de material particulado generado por voladuras en mineria superficial.
UNSA - Escuela Profesional de Ingenieria de Minas

Esta version esta adaptada a los registros reales de la Unidad Minera La Arena
(archivo "Pm 10 nas voladura.xlsx": ene-2023 a may-2024, 505 registros).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_SALIDAS = RAIZ / "outputs"

for _d in (DIR_DATOS, DIR_MODELOS, DIR_SALIDAS):
    _d.mkdir(parents=True, exist_ok=True)

ARCHIVO_EXCEL = DIR_DATOS / "Pm 10 nas voladura.xlsx"      # datos originales de la mina
ARCHIVO_DATOS = DIR_DATOS / "registros_voladura.csv"       # datos depurados (lo genera importar_excel.py)

# ---------------------------------------------------------------------------
# Variables predictoras (X) - solo las que se conocen ANTES del disparo
# ---------------------------------------------------------------------------
VARS_OPERACIONALES = [
    "numero_taladros",        # taladros disparados
    "tonelaje_tm",            # tonelaje fracturado (t)
    "anfo_kg",                # kg de ANFO
    "emulsion_kg",            # kg de emulsion
    "explosivo_total_kg",     # kg de Heavy ANFO (= ANFO + emulsion)
    "n_eventos",              # numero de disparos en el dia
    "tiempo_taladro_ms",      # retardo entre taladros (ms, el menor si hay varios)
    "tiempo_fila_ms",         # retardo entre filas (ms, el menor si hay varios)
]

VARS_METEOROLOGICAS = [
    "humedad_relativa_pct",     # %
    "velocidad_viento_ms",      # m/s
    "direccion_viento_grados",  # 0-360 (se transforma a seno/coseno)
    "precipitacion_mm",         # mm
]

VARS_TEMPORALES = [
    "hora",   # hora del registro / disparo (0-23)
    "mes",    # 1-12 (se transforma a seno/coseno: captura la estacionalidad)
]

VARIABLES_ENTRADA = VARS_OPERACIONALES + VARS_METEOROLOGICAS + VARS_TEMPORALES

# El P80 (fragmentacion) se registra DESPUES del disparo: no puede usarse para
# predecir antes de disparar. Se conserva en el CSV solo para analisis.

# ---------------------------------------------------------------------------
# Variable objetivo (Y)
# ---------------------------------------------------------------------------
# Los datos reales solo tienen PM10. Cuando la mina entregue PM2.5, basta con
# agregar "pm25_ugm3" a esta lista.
OBJETIVOS = ["pm10_ugm3"]

# ---------------------------------------------------------------------------
# Umbrales normativos  (ug/m3, media de 24 horas)
# ---------------------------------------------------------------------------
ECA_PERU = {"pm10_ugm3": 100.0, "pm25_ugm3": 50.0}   # D.S. N 003-2017-MINAM
GUIA_OMS = {"pm10_ugm3": 45.0, "pm25_ugm3": 15.0}    # WHO 2021 (el Excel lo llama "Limite WMO")

# Umbral para el modelo de clasificacion (riesgo de superacion)
UMBRAL_RIESGO = {"pm10_ugm3": 45.0}

# Niveles de alerta: se expresan respecto a la guia OMS y al ECA
#   VERDE    < 60 % de la guia OMS
#   AMARILLO 60 % OMS  - guia OMS
#   NARANJA  guia OMS  - ECA Peru
#   ROJO     > ECA Peru
COLORES_ALERTA = {
    "VERDE": "#1B7F4C",
    "AMARILLO": "#C9A227",
    "NARANJA": "#D9741E",
    "ROJO": "#B3261E",
}

# ---------------------------------------------------------------------------
# Parametros de modelamiento
# ---------------------------------------------------------------------------
SEMILLA = 42
PROPORCION_PRUEBA = 0.20   # division 80:20 (coherente con Zhang et al., 2026)
K_FOLDS = 5
N_TRIALS_OPTUNA = 40       # ensayos de optimizacion bayesiana por algoritmo

# Entrenar sobre log(1+PM10). Con estos datos empeora el R2 en ug/m3 porque
# subestima los picos, por eso queda desactivado. Se deja como opcion.
USAR_LOG = False
