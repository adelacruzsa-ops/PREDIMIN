"""
PREDIMIN - Modulo 3: motor de prediccion y recomendacion de mitigacion.

Convierte la salida de los modelos en una decision operativa:
  1. PM10 esperado (regresion) y probabilidad de superar 45 ug/m3 (clasificacion).
  2. Nivel de alerta frente a la guia OMS 2021 y al ECA de Aire (D.S. 003-2017-MINAM).
  3. Medidas de mitigacion segun el nivel y las variables criticas del evento.
  4. Escenarios: cuanto cambia la prediccion si se modifica una variable controlable.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import DIR_MODELOS, ECA_PERU, GUIA_OMS, OBJETIVOS
from preprocesamiento import ingenieria_de_variables, matriz_de_variables

_CACHE = {}


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------
def _cargar(nombre: str, obligatorio: bool = True):
    if nombre not in _CACHE:
        ruta = DIR_MODELOS / f"{nombre}.joblib"
        if not Path(ruta).exists():
            if not obligatorio:
                return None
            raise FileNotFoundError(
                f"No existe {ruta}. Ejecute primero: python src/entrenamiento.py")
        try:
            _CACHE[nombre] = joblib.load(ruta)
        except (ModuleNotFoundError, AttributeError, ImportError) as e:
            raise RuntimeError(
                f"El modelo {ruta.name} fue creado con otra version de las librerias. "
                f"Vuelva a entrenar: python src/entrenamiento.py  ({e})") from e
    return _CACHE[nombre]


def _X(evento: dict, paquete: dict):
    df = ingenieria_de_variables(pd.DataFrame([evento]))
    X, _ = matriz_de_variables(df, paquete.get("medianas"))
    return X[paquete["columnas"]]


# ---------------------------------------------------------------------------
# Clasificacion frente a umbrales
# ---------------------------------------------------------------------------
ORDEN = {"VERDE": 0, "AMARILLO": 1, "NARANJA": 2, "ROJO": 3}


def nivel_por_valor(valor: float, objetivo: str) -> str:
    oms, eca = GUIA_OMS[objetivo], ECA_PERU[objetivo]
    if valor < 0.6 * oms:
        return "VERDE"
    if valor < oms:
        return "AMARILLO"
    if valor < eca:
        return "NARANJA"
    return "ROJO"


def nivel_por_probabilidad(p: float) -> str:
    if p >= 0.70:
        return "NARANJA"
    if p >= 0.40:
        return "AMARILLO"
    return "VERDE"


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------
def predecir(evento: dict) -> dict:
    """PM10 esperado, probabilidad de superar 45 ug/m3 y nivel de alerta."""
    salida = {}
    for objetivo in OBJETIVOS:
        paq = _cargar(f"predimin_{objetivo}")
        valor = max(float(paq["modelo"].predict(_X(evento, paq))[0]), 0.0)

        paq_r = _cargar(f"predimin_riesgo_{objetivo}", obligatorio=False)
        prob = (float(paq_r["modelo"].predict_proba(_X(evento, paq_r))[0, 1])
                if paq_r else None)

        n_val = nivel_por_valor(valor, objetivo)
        n_prob = nivel_por_probabilidad(prob) if prob is not None else "VERDE"
        nivel = max(n_val, n_prob, key=lambda n: ORDEN[n])

        salida[objetivo] = {
            "valor": round(valor, 1),
            "probabilidad_superar_oms": None if prob is None else round(prob, 3),
            "guia_oms": GUIA_OMS[objetivo],
            "eca_peru": ECA_PERU[objetivo],
            "porcentaje_oms": round(100 * valor / GUIA_OMS[objetivo], 1),
            "excede_oms": bool(valor > GUIA_OMS[objetivo]),
            "excede_eca": bool(valor > ECA_PERU[objetivo]),
            "nivel": nivel,
        }
    return salida


def prediccion_red_neuronal(evento: dict, nivel: float = 0.90) -> dict | None:
    """
    Segunda opinion de la red neuronal (ensamble): PM10 esperado con intervalo
    de incertidumbre y probabilidad de superar el umbral. None si no se ha
    ejecutado src/red_neuronal.py.
    """
    paq = _cargar("predimin_red_neuronal", obligatorio=False)
    if paq is None:
        return None
    X = _X(evento, paq)
    media, lo, hi = paq["regresion"].predict_intervalo(X, nivel)
    prob = float(paq["clasificacion"].predict_proba(X)[0, 1])
    return {"valor": round(float(media[0]), 1), "limite_inferior": round(float(lo[0]), 1),
            "limite_superior": round(float(hi[0]), 1), "nivel_intervalo": nivel,
            "probabilidad_superar_oms": round(prob, 3), "umbral": paq["umbral"]}


def contribuciones(evento: dict, objetivo: str = "pm10_ugm3", top: int = 8):
    """Contribucion SHAP (ug/m3) de cada variable a ESTE evento."""
    try:
        import shap
    except ImportError:
        return None
    paq = _cargar(f"predimin_{objetivo}")
    X = _X(evento, paq)
    modelo = getattr(paq["modelo"], "regressor_", paq["modelo"])
    try:
        valores = shap.TreeExplainer(modelo).shap_values(X)[0]
    except Exception:
        return None   # modelo no basado en arboles (red neuronal)
    tabla = pd.DataFrame({"variable": paq["columnas"], "contribucion": np.round(valores, 2)})
    tabla["abs"] = tabla["contribucion"].abs()
    return tabla.sort_values("abs", ascending=False).head(top).drop(columns="abs")


# ---------------------------------------------------------------------------
# Catalogo de medidas
# ---------------------------------------------------------------------------
# IMPORTANTE (tesis): los rangos de "reduccion_esperada" son REFERENCIALES y
# deben sustentarse con literatura (Cecala et al., 2019; Kissell, 2003) o con
# registros de la unidad minera. El modelo NO estima el efecto del riego ni de
# la nebulizacion porque esas medidas no estan en los datos historicos.
CATALOGO_MEDIDAS = {
    "riego_previo": {
        "nombre": "Riego previo de plataforma y accesos",
        "detalle": "Humedecer con cisterna la plataforma del disparo y los accesos "
                   "en las 2 horas previas.",
        "reduccion_esperada": "Referencial (citar fuente)",
    },
    "nebulizacion": {
        "nombre": "Nebulizacion en el area del disparo",
        "detalle": "Operar nebulizadores o cañones de niebla antes y despues de la "
                   "detonacion.",
        "reduccion_esperada": "Referencial (citar fuente)",
    },
    "reprogramar": {
        "nombre": "Reprogramar la hora del disparo",
        "detalle": "Buscar una ventana con mayor humedad y menor viento "
                   "(ver escenarios estimados por el modelo).",
        "reduccion_esperada": "Ver tabla de escenarios",
    },
    "fraccionar": {
        "nombre": "Fraccionar el disparo o reducir disparos por dia",
        "detalle": "Dividir el proyecto en disparos menores o evitar varios "
                   "eventos el mismo dia.",
        "reduccion_esperada": "Ver tabla de escenarios",
    },
    "rediseno_carga": {
        "nombre": "Revisar el diseño de carga",
        "detalle": "Evaluar si el factor de carga puede reducirse sin afectar la "
                   "fragmentacion (P80).",
        "reduccion_esperada": "Ver tabla de escenarios",
    },
    "aviso": {
        "nombre": "Aviso y monitoreo reforzado",
        "detalle": "Comunicar a Medio Ambiente y Relaciones Comunitarias e "
                   "intensificar el monitoreo durante el evento.",
        "reduccion_esperada": "No reduce la emision; reduce el riesgo social",
    },
}


def recomendar(evento: dict, prediccion: dict | None = None) -> dict:
    pred = prediccion or predecir(evento)
    nivel = max((pred[o]["nivel"] for o in OBJETIVOS), key=lambda n: ORDEN[n])
    p = pred["pm10_ugm3"]

    medidas, motivos = [], []

    def agregar(clave, motivo):
        if clave not in medidas:
            medidas.append(clave)
            motivos.append(motivo)

    txt_prob = ("" if p["probabilidad_superar_oms"] is None
                else f", probabilidad de superar 45 µg/m³: {p['probabilidad_superar_oms'] * 100:.0f} %")
    if ORDEN[nivel] >= 1:
        agregar("riego_previo", f"Nivel {nivel} (PM10 esperado {p['valor']:.0f} µg/m³{txt_prob}).")
    if ORDEN[nivel] >= 2:
        agregar("nebulizacion", "Escenario con riesgo de superar la guia OMS.")
        agregar("reprogramar", "Evaluar una ventana horaria mas favorable.")
    if ORDEN[nivel] >= 3:
        agregar("fraccionar", "La prediccion supera el ECA nacional de 24 h.")
        agregar("aviso", "Posible incumplimiento del ECA.")

    humedad = float(evento.get("humedad_relativa_pct", 100))
    viento = float(evento.get("velocidad_viento_ms", 0))
    eventos = float(evento.get("n_eventos", 1))
    ton = float(evento.get("tonelaje_tm") or 0)
    fc = float(evento.get("explosivo_total_kg", 0)) / ton if ton > 0 else 0

    if humedad < 50:
        agregar("riego_previo", f"Humedad relativa de {humedad:.0f} %: condicion seca "
                                f"(la variable de mayor influencia en los datos).")
    if viento > 3.0:
        agregar("reprogramar", f"Viento de {viento:.1f} m/s (percentil 90 historico: ~3 m/s).")
    if eventos > 1:
        agregar("fraccionar", f"{eventos:.0f} disparos el mismo dia.")
    if fc > 0.30:
        agregar("rediseno_carga", f"Factor de carga de {fc:.2f} kg/t (mediana historica: 0.23).")

    if not medidas:
        recs = [{"nombre": "Operacion normal",
                 "detalle": "No se requieren medidas adicionales. Mantener el riego y "
                            "monitoreo de rutina.",
                 "reduccion_esperada": "-",
                 "motivo": "Escenario dentro del rango habitual."}]
    else:
        recs = [{**CATALOGO_MEDIDAS[k], "motivo": m} for k, m in zip(medidas, motivos)]

    return {"nivel_alerta": nivel, "prediccion": pred, "recomendaciones": recs}


# ---------------------------------------------------------------------------
# Escenarios (objetivo especifico 5)
# ---------------------------------------------------------------------------
def escenarios_estandar(evento: dict) -> dict:
    e = dict(evento)
    esc = {"Base (disparo planificado)": e}
    esc["Explosivo -10 %"] = {**e,
                              "explosivo_total_kg": e["explosivo_total_kg"] * 0.9,
                              "anfo_kg": e["anfo_kg"] * 0.9,
                              "emulsion_kg": e["emulsion_kg"] * 0.9}
    esc["Fraccionar (50 % del disparo)"] = {**e,
        **{k: e[k] * 0.5 for k in ("explosivo_total_kg", "anfo_kg", "emulsion_kg",
                                   "numero_taladros", "tonelaje_tm") if e.get(k)}}
    esc["Un solo disparo en el dia"] = {**e, "n_eventos": 1}
    esc["Disparar a las 07:00"] = {**e, "hora": 7}
    esc["Condicion mas humeda (+15 % HR)"] = {
        **e, "humedad_relativa_pct": min(100.0, e["humedad_relativa_pct"] + 15)}
    esc["Viento calmo (0.5 m/s)"] = {**e, "velocidad_viento_ms": 0.5}
    return esc


def evaluar_escenarios(evento: dict, escenarios: dict | None = None) -> pd.DataFrame:
    escenarios = escenarios or escenarios_estandar(evento)
    filas, base = [], None
    for nombre, ev in escenarios.items():
        p = predecir(ev)["pm10_ugm3"]
        fila = {"escenario": nombre, "pm10_ugm3": p["valor"],
                "prob_superar_45": p["probabilidad_superar_oms"], "nivel": p["nivel"]}
        base = base or fila
        fila["reduccion_pm10_%"] = (round(100 * (base["pm10_ugm3"] - fila["pm10_ugm3"])
                                          / base["pm10_ugm3"], 1)
                                    if base["pm10_ugm3"] > 0 else 0.0)
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Evento de ejemplo: valores tipicos de los registros, con condicion seca
EVENTO_EJEMPLO = {
    "numero_taladros": 130,
    "tonelaje_tm": 64729.0,
    "anfo_kg": 2760.0,
    "emulsion_kg": 11277.0,
    "explosivo_total_kg": 14037.0,
    "n_eventos": 2,
    "tiempo_taladro_ms": 17.0,
    "tiempo_fila_ms": 182.0,
    "humedad_relativa_pct": 35.0,
    "velocidad_viento_ms": 2.5,
    "direccion_viento_grados": 306.0,
    "precipitacion_mm": 0.0,
    "hora": 12,
    "mes": 8,
}

if __name__ == "__main__":
    salida = recomendar(EVENTO_EJEMPLO)
    p = salida["prediccion"]["pm10_ugm3"]
    print(f"\nNIVEL DE ALERTA: {salida['nivel_alerta']}")
    print(f"  PM10 esperado: {p['valor']:.1f} ug/m3  |  "
          f"Prob. de superar 45: {p['probabilidad_superar_oms']}")
    print("\nMEDIDAS RECOMENDADAS:")
    for i, r in enumerate(salida["recomendaciones"], 1):
        print(f"  {i}. {r['nombre']}  -- {r['motivo']}")
    print("\nESCENARIOS:")
    print(evaluar_escenarios(EVENTO_EJEMPLO).to_string(index=False))
    rn = prediccion_red_neuronal(EVENTO_EJEMPLO)
    if rn:
        print(f"\nRED NEURONAL: {rn['valor']} ug/m3 (intervalo 90 %: "
              f"{rn['limite_inferior']} - {rn['limite_superior']}), "
              f"prob. de superar 45: {rn['probabilidad_superar_oms']}")
    print("\nCONTRIBUCION SHAP (ug/m3):")
    c = contribuciones(EVENTO_EJEMPLO)
    if c is not None:
        print(c.to_string(index=False))
