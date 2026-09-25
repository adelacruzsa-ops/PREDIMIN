"""
PREDIMIN - Modulo 4: interfaz web del sistema inteligente.

Ejecutar con:
    streamlit run src/app.py

En Streamlit Community Cloud: archivo principal "src/app.py". Si no existen los
modelos (no se suben a GitHub), la app los reconstruye al abrirse.
"""

import pandas as pd
import streamlit as st

from config import COLORES_ALERTA, DIR_MODELOS, DIR_SALIDAS, ECA_PERU, GUIA_OMS
from recomendador import (contribuciones, evaluar_escenarios, prediccion_red_neuronal,
                          recomendar)

st.set_page_config(page_title="PREDIMIN", page_icon="⛏", layout="wide")



@st.cache_resource(show_spinner="Preparando los modelos por primera vez (menos de 1 minuto)...")
def asegurar_modelos():
    """En la nube no se suben los modelos: se reconstruyen con los resultados de outputs/."""
    faltan = [n for n in ("predimin_pm10_ugm3", "predimin_riesgo_pm10_ugm3",
                          "predimin_red_neuronal")
              if not (DIR_MODELOS / f"{n}.joblib").exists()]
    if faltan and (DIR_SALIDAS / "resumen_entrenamiento.json").exists():
        from entrenamiento import reconstruir_modelos
        reconstruir_modelos()


asegurar_modelos()

st.title("PREDIMIN")
st.caption(
    "Sistema inteligente de prediccion y mitigacion de material particulado "
    "generado por voladuras en mineria superficial — Unidad Minera La Arena"
)

# ---------------------------------------------------------------------------
# Entradas (valores por defecto = medianas de los registros 2023-2024)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Parametros del disparo")

    st.subheader("Diseño de voladura")
    numero_taladros = st.number_input("Numero de taladros", 1, 1000, 130)
    tonelaje = st.number_input("Tonelaje fracturado (t)", 100.0, 300000.0, 64729.0, step=1000.0)
    anfo = st.number_input("ANFO (kg)", 0.0, 20000.0, 2760.0, step=100.0)
    emulsion = st.number_input("Emulsion (kg)", 0.0, 60000.0, 11277.0, step=100.0)
    explosivo = anfo + emulsion
    st.info(f"Heavy ANFO total: **{explosivo:,.0f} kg**  \n"
            f"Factor de carga: **{explosivo / tonelaje:.3f} kg/t**")
    n_eventos = st.number_input("Disparos en el dia", 1, 6, 1)
    t_taladro = st.number_input("Retardo entre taladros (ms)", 0.0, 500.0, 17.0)
    t_fila = st.number_input("Retardo entre filas (ms)", 0.0, 2000.0, 182.0)

    st.subheader("Programacion")
    hora = st.slider("Hora del disparo", 6, 18, 12)
    mes = st.selectbox("Mes", list(range(1, 13)), index=7,
                       format_func=lambda m: ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul",
                                              "Ago", "Sep", "Oct", "Nov", "Dic"][m - 1])

    st.subheader("Condiciones meteorologicas")
    humedad = st.slider("Humedad relativa (%)", 0, 100, 35)
    viento = st.slider("Velocidad del viento (m/s)", 0.0, 8.0, 2.5, 0.1)
    direccion = st.slider("Direccion del viento (grados)", 0, 360, 306)
    precip = st.number_input("Precipitacion (mm)", 0.0, 50.0, 0.0, step=0.1)

    evaluar = st.button("Evaluar voladura", type="primary", width="stretch")

evento = {
    "numero_taladros": numero_taladros,
    "tonelaje_tm": tonelaje,
    "anfo_kg": anfo,
    "emulsion_kg": emulsion,
    "explosivo_total_kg": explosivo,
    "n_eventos": n_eventos,
    "tiempo_taladro_ms": t_taladro,
    "tiempo_fila_ms": t_fila,
    "humedad_relativa_pct": float(humedad),
    "velocidad_viento_ms": viento,
    "direccion_viento_grados": float(direccion),
    "precipitacion_mm": precip,
    "hora": hora,
    "mes": mes,
}

def pestana_evaluacion():
    if evaluar or st.session_state.get("ya_evaluado"):
        st.session_state["ya_evaluado"] = True

        try:
            resultado = recomendar(evento)
        except (FileNotFoundError, RuntimeError) as e:
            st.error(str(e))
            return

        nivel = resultado["nivel_alerta"]
        p = resultado["prediccion"]["pm10_ugm3"]

        st.markdown(
            f"<div style='background:{COLORES_ALERTA[nivel]};color:#fff;padding:14px 18px;"
            f"border-radius:10px;font-size:1.25rem;font-weight:700;'>"
            f"NIVEL DE ALERTA: {nivel}</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("PM10 esperado", f"{p['valor']:.1f} µg/m³",
                      f"{p['porcentaje_oms']:.0f} % de la guia OMS", delta_color="inverse")
            st.progress(min(p["valor"] / GUIA_OMS["pm10_ugm3"], 1.0))
            st.caption(f"Guia OMS 2021 (24 h): {GUIA_OMS['pm10_ugm3']:.0f} µg/m³ · "
                       f"ECA Peru (24 h): {ECA_PERU['pm10_ugm3']:.0f} µg/m³")
        with c2:
            if p["probabilidad_superar_oms"] is not None:
                st.metric("Probabilidad de superar 45 µg/m³",
                          f"{p['probabilidad_superar_oms'] * 100:.0f} %")
                st.progress(p["probabilidad_superar_oms"])
                st.caption("Modelo de clasificacion entrenado con los registros historicos.")

        rn = prediccion_red_neuronal(evento)
        if rn is not None:
            st.markdown("**Segunda opinion: red neuronal (ensamble de 10 redes)**")
            r1, r2 = st.columns(2)
            r1.metric("PM10 esperado (red neuronal)", f"{rn['valor']:.1f} µg/m³",
                      f"incertidumbre del modelo: {rn['limite_inferior']:.0f} – "
                      f"{rn['limite_superior']:.0f} µg/m³", delta_color="off")
            r2.metric("Probabilidad de superar 45 µg/m³ (red neuronal)",
                      f"{rn['probabilidad_superar_oms'] * 100:.0f} %")
            st.caption("El rango refleja el desacuerdo entre las redes del ensamble "
                       "(percentiles 5-95): cuanto mas ancho, menos segura es la prediccion. "
                       "No incluye el ruido del monitor, asi que el PM10 real puede caer fuera.")

        st.info(
            "Con los datos actuales el modelo explica una parte limitada de la "
            "variacion del PM10 (ver README). Use el nivel de alerta y la "
            "probabilidad como apoyo a la decision, no como valor exacto."
        )

        st.divider()
        st.subheader("Medidas de mitigacion recomendadas")
        for i, r in enumerate(resultado["recomendaciones"], 1):
            with st.expander(f"{i}. {r['nombre']}", expanded=(i <= 2)):
                st.markdown(f"**Por que se recomienda:** {r['motivo']}")
                st.markdown(f"**Accion concreta:** {r['detalle']}")
                st.caption(f"Efecto: {r['reduccion_esperada']}")

        st.divider()
        st.subheader("Escenarios estimados por el modelo")
        st.caption("Se cambia una variable a la vez y se vuelve a predecir.")
        esc = evaluar_escenarios(evento)
        esc["prob_superar_45"] = (esc["prob_superar_45"] * 100).round(0)
        st.dataframe(
            esc.rename(columns={"escenario": "Escenario", "pm10_ugm3": "PM10 (µg/m³)",
                                "prob_superar_45": "Prob. > 45 (%)", "nivel": "Nivel",
                                "reduccion_pm10_%": "Reduccion PM10 (%)"}),
            width="stretch", hide_index=True,
        )

        st.divider()
        st.subheader("Por que el modelo predice este valor")
        st.caption("Contribucion SHAP de cada variable (µg/m³). Positivo = sube el PM10.")
        tabla = contribuciones(evento)
        if tabla is None:
            st.info("Instale la libreria `shap` para ver esta seccion.")
        else:
            st.bar_chart(tabla.set_index("variable")["contribucion"])
            st.dataframe(tabla, width="stretch", hide_index=True)
    else:
        st.info("Complete los parametros del disparo en el panel izquierdo y pulse "
                "**Evaluar voladura**.")


def pestana_resultados():
    st.subheader("Desempeño de los modelos (datos 2023-2024)")
    def leer(nombre):
        ruta = DIR_SALIDAS / nombre
        return pd.read_csv(ruta) if ruta.exists() else None

    t = leer("comparacion_modelos_pm10_ugm3.csv")
    if t is None:
        st.info("Ejecute primero el entrenamiento (2_EJECUTAR_TODO.bat).")
        return
    st.markdown("**Regresion: prediccion del valor de PM10** "
                "(el modelo se elige por R² de validacion cruzada)")
    st.dataframe(t, width="stretch", hide_index=True)

    c = leer("comparacion_clasificadores_pm10_ugm3.csv")
    if c is not None:
        st.markdown("**Clasificacion: riesgo de superar 45 µg/m³** (AUC: 0,5 = azar, 1 = perfecto)")
        st.dataframe(c, width="stretch", hide_index=True)

    corr = leer("correlaciones_pm10.csv")
    if corr is not None:
        st.markdown("**Correlacion de Spearman de cada variable con el PM10**")
        st.bar_chart(corr.set_index("variable")["rho_spearman"])

    figuras = [
        ("pm10_vs_variables.png", "PM10 frente a variables meteorologicas y de voladura"),
        ("pm10_por_mes.png", "Estacionalidad del PM10"),
        ("observado_vs_predicho_pm10_ugm3.png", "Modelo seleccionado: observado vs. predicho"),
        ("shap_beeswarm_pm10_ugm3.png", "Importancia de variables (SHAP)"),
        ("red_neuronal_arquitectura.png", "Arquitectura de la red neuronal"),
        ("red_neuronal_roc.png", "Red neuronal: curva ROC del riesgo de superar 45 µg/m³"),
        ("red_neuronal_observado_vs_predicho.png", "Red neuronal: observado vs. predicho con intervalo 90 %"),
        ("red_neuronal_curva_aprendizaje.png", "Curva de aprendizaje de la red neuronal"),
        ("red_neuronal_regularizacion.png", "Red neuronal: efecto de la regularizacion"),
        ("red_neuronal_importancia.png", "Red neuronal: importancia de variables (permutacion)"),
    ]
    for archivo, titulo in figuras:
        ruta = DIR_SALIDAS / archivo
        if ruta.exists():
            st.markdown(f"**{titulo}**")
            st.image(str(ruta))


tab1, tab2 = st.tabs(["Evaluar voladura", "Resultados del modelo"])
with tab1:
    pestana_evaluacion()
with tab2:
    pestana_resultados()
