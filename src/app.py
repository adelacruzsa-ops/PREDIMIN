"""
PREDIMIN - Modulo 4: interfaz web del sistema inteligente.

Ejecutar con:
    streamlit run src/app.py

En Streamlit Community Cloud: archivo principal "src/app.py". Si no existen los
modelos (no se suben a GitHub), la app los reconstruye al abrirse.
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ARCHIVO_DATOS, DIR_MODELOS, DIR_SALIDAS, ECA_PERU, ETIQUETAS, GUIA_OMS
from recomendador import (contribuciones, evaluar_escenarios, prediccion_red_neuronal,
                          recomendar)

st.set_page_config(page_title="PREDIMIN · Predicción de PM10 en voladuras",
                   page_icon="⛰️", layout="wide", initial_sidebar_state="auto")

# ---------------------------------------------------------------------------
# Identidad visual
# ---------------------------------------------------------------------------
AZUL = "#1F3864"          # institucional (tambien en el Excel de resultados)
AZUL_SERIE = "#2a78d6"    # serie de datos
ROJO_SERIE = "#e34948"    # contribucion que SUBE el PM10
TINTA, TINTA_2, REJILLA = "#1B1F24", "#5B6470", "#E6E9EE"

# Niveles de alerta: color de estado + icono + etiqueta (nunca solo color)
NIVELES = {
    "VERDE":    {"color": "#0ca30c", "fondo": "#E8F6E8", "icono": "✔", "texto": "Riesgo bajo",
                 "detalle": "PM10 esperado por debajo del 60 % de la guía OMS."},
    "AMARILLO": {"color": "#C98A00", "fondo": "#FEF6E3", "icono": "!", "texto": "Riesgo moderado",
                 "detalle": "PM10 esperado cercano a la guía OMS (45 µg/m³)."},
    "NARANJA":  {"color": "#D9632F", "fondo": "#FDEEE7", "icono": "▲", "texto": "Riesgo alto",
                 "detalle": "Se espera superar la guía OMS; aplicar medidas preventivas."},
    "ROJO":     {"color": "#d03b3b", "fondo": "#FBE9E9", "icono": "✖", "texto": "Riesgo crítico",
                 "detalle": "Se espera superar el ECA nacional (100 µg/m³)."},
}

st.markdown(f"""
<style>
  #MainMenu, footer {{visibility: hidden;}}
  .block-container {{padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}}
  h1, h2, h3, h4 {{color: {TINTA}; letter-spacing: -0.01em;}}
  .pm-header {{
      background: linear-gradient(120deg, {AZUL} 0%, #2E5597 100%);
      border-radius: 14px; padding: 22px 28px; color: #fff; margin-bottom: 1.2rem;
  }}
  .pm-header h1 {{color: #fff; margin: 0; font-size: 2rem; font-weight: 800;}}
  .pm-header p {{margin: .25rem 0 .7rem; opacity: .92; font-size: 1rem;}}
  .pm-chip {{display: inline-block; background: rgba(255,255,255,.16); border-radius: 999px;
             padding: 3px 12px; margin: 0 6px 4px 0; font-size: .78rem;}}
  .pm-card {{background: #fff; border: 1px solid {REJILLA}; border-radius: 12px;
             padding: 16px 18px; height: 100%;}}
  .pm-kpi-label {{color: {TINTA_2}; font-size: .82rem; font-weight: 600;}}
  .pm-kpi-value {{color: {TINTA}; font-size: 1.9rem; font-weight: 800; line-height: 1.2;}}
  .pm-kpi-sub {{color: {TINTA_2}; font-size: .85rem;}}
  .pm-alert {{border-radius: 12px; padding: 16px 20px; display: flex; gap: 16px;
              align-items: center; border-left: 8px solid;}}
  .pm-alert-icon {{width: 44px; height: 44px; border-radius: 50%; color: #fff; display: flex;
                   align-items: center; justify-content: center; font-size: 1.3rem;
                   font-weight: 800; flex-shrink: 0;}}
  .pm-alert-title {{font-size: 1.25rem; font-weight: 800; color: {TINTA};}}
  .pm-alert-text {{color: {TINTA_2}; font-size: .92rem;}}
  .pm-rec {{background: #fff; border: 1px solid {REJILLA}; border-radius: 12px;
            padding: 14px 16px; margin-bottom: 10px;}}
  .pm-rec-num {{display: inline-flex; width: 26px; height: 26px; border-radius: 50%;
                background: {AZUL}; color: #fff; align-items: center; justify-content: center;
                font-weight: 700; font-size: .85rem; margin-right: 8px;}}
  .pm-rec-title {{font-weight: 700; color: {TINTA}; font-size: 1rem;}}
  .pm-rec-body {{color: {TINTA_2}; font-size: .9rem; margin-top: 6px;}}
  .pm-tag {{display: inline-block; background: #EEF2F8; color: {AZUL}; border-radius: 6px;
            padding: 1px 8px; font-size: .75rem; font-weight: 600; margin-top: 8px;}}
  .pm-note {{color: {TINTA_2}; font-size: .82rem;}}
  .pm-step {{background: #fff; border: 1px solid {REJILLA}; border-top: 4px solid {AZUL};
             border-radius: 10px; padding: 12px 14px; min-height: 170px;}}
  .pm-step b {{color: {AZUL};}}
  .pm-footer {{color: {TINTA_2}; font-size: .78rem; text-align: center; margin-top: 2.5rem;
               border-top: 1px solid {REJILLA}; padding-top: 1rem;}}
  section[data-testid="stSidebar"] h2 {{font-size: 1.1rem;}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Modelos (en la nube se reconstruyen a partir de outputs/)
# ---------------------------------------------------------------------------
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


def leer_csv(nombre):
    ruta = DIR_SALIDAS / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


def leer_json(nombre):
    ruta = DIR_SALIDAS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def etiqueta(v):
    return ETIQUETAS.get(v, v)


def estilo_figura(fig, alto):
    fig.update_layout(
        height=alto, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA, size=13),
        hoverlabel=dict(bgcolor="white", font_size=13), showlegend=False,
    )
    fig.update_xaxes(gridcolor=REJILLA, zeroline=False, linecolor=REJILLA)
    fig.update_yaxes(gridcolor=REJILLA, zeroline=False, linecolor=REJILLA)
    return fig


def tarjeta_kpi(col, titulo, valor, sub=""):
    col.markdown(f"<div class='pm-card'><div class='pm-kpi-label'>{titulo}</div>"
                 f"<div class='pm-kpi-value'>{valor}</div>"
                 f"<div class='pm-kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown("""
<div class="pm-header">
  <h1>PREDIMIN</h1>
  <p>Sistema inteligente de predicción y mitigación de material particulado (PM10)
     generado por voladuras en minería superficial</p>
  <span class="pm-chip">Unidad Minera La Arena · La Libertad</span>
  <span class="pm-chip">401 voladuras · ene 2023 – may 2024</span>
  <span class="pm-chip">7 modelos de aprendizaje automático + red neuronal</span>
  <span class="pm-chip">UNSA · Ingeniería de Minas</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Entradas (valores por defecto = registros tipicos 2023-2024)
# ---------------------------------------------------------------------------
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
         "Septiembre", "Octubre", "Noviembre", "Diciembre"]

with st.sidebar:
    st.markdown("## Parámetros de la voladura")
    st.caption("Los resultados se actualizan al cambiar cualquier valor.")

    with st.expander("Diseño de carga", expanded=True):
        numero_taladros = st.number_input("Número de taladros", 1, 1000, 130)
        tonelaje = st.number_input("Tonelaje fracturado (t)", 100.0, 300000.0, 64729.0,
                                   step=1000.0, format="%.0f")
        anfo = st.number_input("ANFO (kg)", 0.0, 20000.0, 2760.0, step=100.0, format="%.0f")
        emulsion = st.number_input("Emulsión (kg)", 0.0, 60000.0, 11277.0, step=100.0,
                                   format="%.0f")
        explosivo = anfo + emulsion
        st.markdown(f"<div class='pm-card' style='padding:10px 14px'>"
                    f"<span class='pm-note'>Heavy ANFO total</span><br><b>{explosivo:,.0f} kg</b>"
                    f"<br><span class='pm-note'>Factor de carga</span><br>"
                    f"<b>{explosivo / tonelaje:.3f} kg/t</b></div>", unsafe_allow_html=True)

    with st.expander("Secuencia y programación", expanded=True):
        n_eventos = st.number_input("Disparos en el día", 1, 6, 2)
        t_taladro = st.number_input("Retardo entre taladros (ms)", 0.0, 500.0, 17.0)
        t_fila = st.number_input("Retardo entre filas (ms)", 0.0, 2000.0, 182.0)
        hora = st.slider("Hora del disparo", 6, 18, 12, format="%d:00")
        mes = st.selectbox("Mes", list(range(1, 13)), index=7,
                           format_func=lambda m: MESES[m - 1])

    with st.expander("Condiciones meteorológicas", expanded=True):
        humedad = st.slider("Humedad relativa (%)", 0, 100, 35)
        viento = st.slider("Velocidad del viento (m/s)", 0.0, 8.0, 2.5, 0.1)
        direccion = st.slider("Dirección del viento (°)", 0, 360, 306)
        precip = st.number_input("Precipitación (mm)", 0.0, 50.0, 0.0, step=0.1)

evento = {
    "numero_taladros": numero_taladros, "tonelaje_tm": tonelaje, "anfo_kg": anfo,
    "emulsion_kg": emulsion, "explosivo_total_kg": explosivo, "n_eventos": n_eventos,
    "tiempo_taladro_ms": t_taladro, "tiempo_fila_ms": t_fila,
    "humedad_relativa_pct": float(humedad), "velocidad_viento_ms": viento,
    "direccion_viento_grados": float(direccion), "precipitacion_mm": precip,
    "hora": hora, "mes": mes,
}


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def grafico_pm10(valor, rango=None):
    """Barra de bala: PM10 esperado frente a la guia OMS y al ECA."""
    oms, eca = GUIA_OMS["pm10_ugm3"], ECA_PERU["pm10_ugm3"]
    tope = max(130.0, valor * 1.15, (rango[1] if rango else 0) * 1.05)
    fig = go.Figure()
    for x0, x1, n in ((0, 0.6 * oms, "VERDE"), (0.6 * oms, oms, "AMARILLO"),
                      (oms, eca, "NARANJA"), (eca, tope, "ROJO")):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.5, y1=0.5, line_width=0,
                      fillcolor=NIVELES[n]["fondo"], layer="below")
    fig.add_trace(go.Bar(x=[valor], y=[""], orientation="h", width=0.34,
                         marker=dict(color=AZUL, cornerradius=4),
                         hovertemplate="PM10 esperado: %{x:.1f} µg/m³<extra></extra>"))
    if rango:
        fig.add_trace(go.Scatter(x=list(rango), y=["", ""], mode="lines+markers",
                                 line=dict(color=TINTA_2, width=2),
                                 marker=dict(size=9, symbol="line-ns-open", color=TINTA_2),
                                 hovertemplate="Rango de la red neuronal: %{x:.0f} µg/m³"
                                               "<extra></extra>"))
    for x, txt in ((oms, "Guía OMS 45"), (eca, "ECA Perú 100")):
        fig.add_vline(x=x, line=dict(color=TINTA, width=1.5, dash="dot"))
        fig.add_annotation(x=x, y=0.5, yshift=12, xshift=4, xanchor="left", text=txt,
                           showarrow=False, font=dict(size=11, color=TINTA_2))
    fig.update_xaxes(range=[0, tope], title="PM10 (µg/m³, media de 24 h)")
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return estilo_figura(fig, 150)


def grafico_escenarios(esc):
    d = esc.iloc[::-1]
    colores = [AZUL if e.startswith("Base") else AZUL_SERIE for e in d["escenario"]]
    fig = go.Figure(go.Bar(
        x=d["pm10_ugm3"], y=d["escenario"], orientation="h", marker=dict(
            color=colores, cornerradius=4), text=[f"{v:.1f}" for v in d["pm10_ugm3"]],
        textposition="outside", cliponaxis=False,
        customdata=(d["prob_superar_45"] * 100).round(0),
        hovertemplate="%{y}<br>PM10: %{x:.1f} µg/m³<br>Prob. > 45: %{customdata:.0f} %"
                      "<extra></extra>"))
    fig.add_vline(x=GUIA_OMS["pm10_ugm3"], line=dict(color=NIVELES["ROJO"]["color"],
                                                     width=1.5, dash="dot"))
    fig.add_annotation(x=GUIA_OMS["pm10_ugm3"], y=1, yref="paper", yshift=8,
                       xshift=4, xanchor="left", text="Guía OMS", showarrow=False,
                       font=dict(size=11, color=TINTA_2))
    fig.update_xaxes(title="PM10 estimado (µg/m³)",
                     range=[0, max(60, d["pm10_ugm3"].max() * 1.2)])
    return estilo_figura(fig, 330)


def grafico_contribuciones(tabla):
    d = tabla.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["contribucion"], y=[etiqueta(v) for v in d["variable"]], orientation="h",
        marker=dict(color=[ROJO_SERIE if v > 0 else AZUL_SERIE for v in d["contribucion"]],
                    cornerradius=4),
        hovertemplate="%{y}: %{x:+.1f} µg/m³<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=TINTA_2, width=1))
    fig.update_xaxes(title="Contribución al PM10 esperado (µg/m³)")
    return estilo_figura(fig, 330)


# ---------------------------------------------------------------------------
# Pestaña 1: evaluacion del disparo
# ---------------------------------------------------------------------------
def pestana_evaluacion():
    try:
        resultado = recomendar(evento)
    except (FileNotFoundError, RuntimeError) as e:
        st.error(str(e))
        return

    nivel = resultado["nivel_alerta"]
    info = NIVELES[nivel]
    p = resultado["prediccion"]["pm10_ugm3"]
    rn = prediccion_red_neuronal(evento)

    st.markdown(f"""
    <div class="pm-alert" style="background:{info['fondo']};border-color:{info['color']};">
      <div class="pm-alert-icon" style="background:{info['color']};">{info['icono']}</div>
      <div><div class="pm-alert-title">Nivel de alerta {nivel.lower()} · {info['texto']}</div>
           <div class="pm-alert-text">{info['detalle']}</div></div>
    </div>""", unsafe_allow_html=True)
    st.write("")

    k1, k2, k3, k4 = st.columns(4)
    tarjeta_kpi(k1, "PM10 esperado", f"{p['valor']:.1f} <small>µg/m³</small>",
                f"{p['porcentaje_oms']:.0f} % de la guía OMS")
    prob = p["probabilidad_superar_oms"]
    tarjeta_kpi(k2, "Probabilidad de PM10 > 45 µg/m³",
                "–" if prob is None else f"{prob * 100:.0f} %", "Modelo de clasificación")
    if rn:
        tarjeta_kpi(k3, "Red neuronal · PM10", f"{rn['valor']:.1f} <small>µg/m³</small>",
                    f"rango del ensamble: {rn['limite_inferior']:.0f}–"
                    f"{rn['limite_superior']:.0f}")
        tarjeta_kpi(k4, "Red neuronal · Prob. > 45 µg/m³",
                    f"{rn['probabilidad_superar_oms'] * 100:.0f} %", "Ensamble de 10 redes")

    st.markdown("#### PM10 esperado frente a la normativa")
    st.plotly_chart(grafico_pm10(p["valor"], (rn["limite_inferior"], rn["limite_superior"])
                                 if rn else None),
                    width="stretch", config={"displayModeBar": False})
    st.markdown(
        "<div class='pm-note'>Barra azul: modelo principal · línea gris: rango de las 10 redes "
        "neuronales (percentiles 5–95; refleja la incertidumbre del modelo, no el ruido del "
        "monitor) · franjas: niveles de alerta.</div>", unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([1, 1], gap="large")
    with izq:
        st.markdown("#### Medidas de mitigación recomendadas")
        for i, r in enumerate(resultado["recomendaciones"], 1):
            st.markdown(f"""
            <div class="pm-rec"><span class="pm-rec-num">{i}</span>
              <span class="pm-rec-title">{r['nombre']}</span>
              <div class="pm-rec-body"><b>Por qué:</b> {r['motivo']}<br>
                   <b>Acción:</b> {r['detalle']}</div>
              <span class="pm-tag">Efecto: {r['reduccion_esperada']}</span>
            </div>""", unsafe_allow_html=True)
    with der:
        st.markdown("#### ¿Por qué el modelo predice este valor?")
        tabla = contribuciones(evento)
        if tabla is None:
            st.info("La explicación SHAP está disponible cuando el modelo principal es de árboles.")
        else:
            st.plotly_chart(grafico_contribuciones(tabla), width="stretch",
                            config={"displayModeBar": False})
            st.markdown("<div class='pm-note'>Valores SHAP: en rojo lo que <b>sube</b> el PM10 "
                        "esperado y en azul lo que lo <b>baja</b>, respecto al promedio "
                        "histórico.</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Escenarios de mitigación")
    st.caption("Se modifica una variable a la vez y se vuelve a predecir con el modelo principal.")
    esc = evaluar_escenarios(evento)
    st.plotly_chart(grafico_escenarios(esc), width="stretch",
                    config={"displayModeBar": False})
    vista = pd.DataFrame({
        "Escenario": esc["escenario"],
        "PM10 (µg/m³)": esc["pm10_ugm3"],
        "Prob. > 45": esc["prob_superar_45"] * 100,
        "Cambio vs. base (%)": -esc["reduccion_pm10_%"] + 0.0,
        "Nivel": [f"{NIVELES[n]['icono']} {NIVELES[n]['texto']}" for n in esc["nivel"]],
    })
    st.dataframe(vista, hide_index=True, width="stretch", column_config={
        "PM10 (µg/m³)": st.column_config.NumberColumn(format="%.1f"),
        "Prob. > 45": st.column_config.ProgressColumn(format="%.0f %%", min_value=0,
                                                      max_value=100),
        "Cambio vs. base (%)": st.column_config.NumberColumn(format="%+.1f"),
    })
    st.markdown(
        "<div class='pm-note'>Con los datos actuales solo los escenarios meteorológicos "
        "(humedad) tienen respaldo estadístico; las variables de voladura no muestran relación "
        "significativa con el PM10 registrado. Use la predicción como apoyo a la decisión, no "
        "como un valor exacto.</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Ficha de evaluación pre-voladura (PDF)")
    st.caption("Documento imprimible con esta evaluación: parámetros, nivel de alerta, gráficos, "
               "medidas, escenarios y firmas de conformidad.")
    c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    codigo = c1.text_input("Código del disparo (opcional)", placeholder="p. ej. LA-0101")
    responsable = c2.text_input("Elaborado por (opcional)", placeholder="Nombre y cargo")
    if c3.button("Preparar ficha PDF", type="primary", width="stretch"):
        from ficha_pdf import generar_ficha
        with st.spinner("Generando la ficha..."):
            st.session_state["ficha_pdf"] = generar_ficha(evento, codigo.strip(),
                                                          responsable.strip())
            st.session_state["ficha_nombre"] = (
                f"ficha_prevoladura_{codigo.strip() or 'PREDIMIN'}.pdf".replace(" ", "_"))
    if "ficha_pdf" in st.session_state:
        st.download_button("⬇ Descargar ficha PDF", st.session_state["ficha_pdf"],
                           file_name=st.session_state["ficha_nombre"], mime="application/pdf",
                           width="stretch")
        st.caption("Si cambia los parámetros, vuelva a pulsar «Preparar ficha PDF».")


# ---------------------------------------------------------------------------
# Pestaña: programa de voladuras
# ---------------------------------------------------------------------------
def pestana_programa():
    from programa import a_excel, evaluar_programa, leer, plantilla

    st.markdown("#### Evaluación del programa de voladuras")
    st.markdown(
        "<div class='pm-note'>Evalúe todos los disparos programados de la semana a la vez. "
        "El sistema los ordena de mayor a menor riesgo para priorizar el riego, la "
        "nebulización o la reprogramación.</div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown("**1. Descargue la plantilla**")
        st.download_button("⬇ Plantilla Excel", plantilla(), "plantilla_programa_PREDIMIN.xlsx",
                           width="stretch")
        st.markdown("**2. Llénela con los disparos programados**")
        st.caption("Una fila por disparo. Use el pronóstico meteorológico del día previsto.")
    with c2:
        st.markdown("**3. Suba el archivo**")
        archivo = st.file_uploader("Programa de voladuras (.xlsx)", type=["xlsx"],
                                   label_visibility="collapsed")
        usar_ejemplo = st.checkbox("Probar con el programa de ejemplo de la plantilla")

    if archivo is None and not usar_ejemplo:
        return
    from io import BytesIO
    try:
        df = leer(archivo if archivo is not None else BytesIO(plantilla()))
        with st.spinner(f"Evaluando {len(df)} disparos..."):
            res = evaluar_programa(df)
    except Exception as e:  # archivo mal llenado
        st.error(f"No se pudo evaluar el archivo: {e}")
        return

    st.divider()
    evaluados = res.dropna(subset=["Nivel"]) if "Nivel" in res else res.iloc[0:0]
    k = st.columns(4)
    tarjeta_kpi(k[0], "Disparos evaluados", f"{len(evaluados)}",
                f"de {len(res)} filas del archivo")
    for col, (n, t) in zip(k[1:], (("ROJO", "Riesgo crítico"), ("NARANJA", "Riesgo alto"),
                                   ("AMARILLO", "Riesgo moderado"))):
        cant = int((res.get("Nivel") == n).sum()) if "Nivel" in res else 0
        tarjeta_kpi(col, f"{NIVELES[n]['icono']} {t}", f"{cant}", "disparos")
    st.write("")

    if "Nivel" in res and len(evaluados):
        d = evaluados.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=d["PM10 esperado (µg/m³)"], y=d["Código"].astype(str), orientation="h",
            marker=dict(color=[NIVELES[n]["color"] for n in d["Nivel"]], cornerradius=4),
            text=[f"{v:.1f} · {NIVELES[n]['texto']}" for v, n in
                  zip(d["PM10 esperado (µg/m³)"], d["Nivel"])],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f} µg/m³<extra></extra>"))
        fig.add_vline(x=GUIA_OMS["pm10_ugm3"], line=dict(color=TINTA, width=1.2, dash="dot"))
        fig.update_xaxes(title="PM10 esperado (µg/m³)",
                         range=[0, max(70, d["PM10 esperado (µg/m³)"].max() * 1.35)])
        st.plotly_chart(estilo_figura(fig, 80 + 34 * len(d)), width="stretch",
                        config={"displayModeBar": False})

    vista = res.copy()
    if "Nivel" in vista:
        vista["Riesgo"] = [f"{NIVELES[n]['icono']} {NIVELES[n]['texto']}" if n in NIVELES
                           else "" for n in vista["Nivel"]]
        vista = vista.drop(columns="Nivel")
    st.dataframe(vista, hide_index=True, width="stretch", column_config={
        "Prob. > 45 µg/m³ (%)": st.column_config.ProgressColumn(
            format="%.0f %%", min_value=0, max_value=100)})
    st.download_button("⬇ Descargar resultado en Excel", a_excel(res),
                       "evaluacion_programa_PREDIMIN.xlsx", type="primary")


# ---------------------------------------------------------------------------
# Pestaña: datos historicos
# ---------------------------------------------------------------------------
@st.cache_data
def datos_historicos():
    df = pd.read_csv(ARCHIVO_DATOS, parse_dates=["fecha"])
    return df.dropna(subset=["pm10_ugm3"])


def pestana_historico():
    df = datos_historicos()
    ini, fin = df["fecha"].min().date(), df["fecha"].max().date()
    rango = st.slider("Periodo", min_value=ini, max_value=fin, value=(ini, fin),
                      format="MM/YYYY")
    d = df[(df["fecha"].dt.date >= rango[0]) & (df["fecha"].dt.date <= rango[1])]
    if d.empty:
        st.info("No hay registros en el periodo elegido.")
        return
    oms, eca = GUIA_OMS["pm10_ugm3"], ECA_PERU["pm10_ugm3"]

    k = st.columns(4)
    tarjeta_kpi(k[0], "Voladuras con PM10 válido", f"{len(d)}",
                f"{d['fecha'].dt.date.nunique()} días con registro")
    tarjeta_kpi(k[1], "PM10 mediano", f"{d['pm10_ugm3'].median():.1f} <small>µg/m³</small>",
                f"máximo {d['pm10_ugm3'].max():.0f} µg/m³")
    tarjeta_kpi(k[2], "Superan la guía OMS", f"{(d['pm10_ugm3'] > oms).mean() * 100:.0f} %",
                f"{int((d['pm10_ugm3'] > oms).sum())} voladuras &gt; 45 µg/m³")
    tarjeta_kpi(k[3], "Superan el ECA", f"{(d['pm10_ugm3'] > eca).mean() * 100:.1f} %",
                f"{int((d['pm10_ugm3'] > eca).sum())} voladuras &gt; 100 µg/m³")
    st.write("")

    st.markdown("#### PM10 registrado en cada voladura")
    fig = go.Figure(go.Scatter(
        x=d["fecha"], y=d["pm10_ugm3"], mode="markers",
        marker=dict(size=8, color=AZUL_SERIE, opacity=0.75, line=dict(width=1, color="white")),
        customdata=d[["humedad_relativa_pct", "velocidad_viento_ms", "explosivo_total_kg"]],
        hovertemplate="%{x|%d/%m/%Y}<br>PM10: %{y:.1f} µg/m³<br>Humedad: %{customdata[0]:.0f} %"
                      "<br>Viento: %{customdata[1]:.1f} m/s<br>Explosivo: %{customdata[2]:,.0f} kg"
                      "<extra></extra>"))
    mensual = d.set_index("fecha")["pm10_ugm3"].resample("MS").mean()
    fig.add_trace(go.Scatter(x=mensual.index + pd.Timedelta(days=14), y=mensual.values,
                             mode="lines", line=dict(color=AZUL, width=2), connectgaps=False,
                             hovertemplate="Promedio de %{x|%m/%Y}: %{y:.1f} µg/m³<extra></extra>"))
    for y, t in ((oms, "Guía OMS 45"), (eca, "ECA Perú 100")):
        fig.add_hline(y=y, line=dict(color=NIVELES["ROJO"]["color"] if y == eca else TINTA,
                                     width=1.2, dash="dot"),
                      annotation_text=t, annotation_position="top left",
                      annotation_font=dict(size=11, color=TINTA_2))
    fig.update_yaxes(title="PM10 (µg/m³)")
    fig.update_xaxes(tickformat="%m/%Y", dtick="M2")
    st.plotly_chart(estilo_figura(fig, 380), width="stretch", config={"displayModeBar": False})
    st.markdown("<div class='pm-note'>Puntos: cada voladura · línea: promedio mensual. "
                "Octubre de 2023 no tiene registros válidos (PM10 = 0, falla del monitor).</div>",
                unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns(2, gap="large")
    with izq:
        st.markdown("#### % de voladuras sobre la guía OMS, por mes")
        m = (d.assign(mes=d["fecha"].dt.to_period("M").dt.to_timestamp())
             .groupby("mes")["pm10_ugm3"].agg(lambda s: (s > oms).mean() * 100))
        fig = go.Figure(go.Bar(x=m.index, y=m.values, marker=dict(color=AZUL_SERIE,
                                                                  cornerradius=4),
                               hovertemplate="%{x|%m/%Y}: %{y:.0f} %<extra></extra>"))
        fig.update_yaxes(title="% de voladuras > 45 µg/m³", range=[0, 100])
        fig.update_xaxes(dtick="M2", tickformat="%m/%Y")
        st.plotly_chart(estilo_figura(fig, 320), width="stretch",
                        config={"displayModeBar": False})
    with der:
        st.markdown("#### Humedad relativa y PM10")
        from scipy.stats import spearmanr
        dd = d.dropna(subset=["humedad_relativa_pct"])
        rho = spearmanr(dd["humedad_relativa_pct"], dd["pm10_ugm3"])[0]
        fig = go.Figure(go.Scatter(
            x=dd["humedad_relativa_pct"], y=dd["pm10_ugm3"], mode="markers",
            marker=dict(size=8, color=AZUL_SERIE, opacity=0.7, line=dict(width=1, color="white")),
            hovertemplate="Humedad %{x:.0f} % · PM10 %{y:.1f} µg/m³<extra></extra>"))
        fig.add_hline(y=oms, line=dict(color=TINTA, width=1.2, dash="dot"))
        fig.update_xaxes(title="Humedad relativa (%)")
        fig.update_yaxes(title="PM10 (µg/m³)")
        st.plotly_chart(estilo_figura(fig, 320), width="stretch",
                        config={"displayModeBar": False})
        st.markdown(f"<div class='pm-note'>ρ de Spearman = {rho:.2f}: a menor humedad, mayor "
                    "PM10. Es la variable de mayor influencia en los datos.</div>",
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pestaña 2: desempeño de los modelos
# ---------------------------------------------------------------------------
NOMBRE_MODELO = {"RedNeuronal_MLP": "Red neuronal (ensamble)", "ExtraTrees": "Extra Trees",
                 "RandomForest": "Random Forest", "GradientBoosting": "Gradient Boosting"}


def pestana_resultados():
    reg = leer_csv("comparacion_modelos_pm10_ugm3.csv")
    clf = leer_csv("comparacion_clasificadores_pm10_ugm3.csv")
    resumen = leer_json("resumen_entrenamiento.json")
    if reg is None or resumen is None:
        st.info("Ejecute primero el entrenamiento (2_EJECUTAR_TODO.bat).")
        return
    r = resumen["pm10_ugm3"]
    shap = leer_csv("shap_importancia_pm10_ugm3.csv")

    k1, k2, k3, k4 = st.columns(4)
    tarjeta_kpi(k1, "Registros válidos", f"{r['n_registros']}",
                "80 % entrenamiento · 20 % prueba")
    tarjeta_kpi(k2, "Mejor modelo de regresión", NOMBRE_MODELO.get(
        r["modelo_seleccionado"], r["modelo_seleccionado"]),
        f"R² prueba = {r['metricas']['R2_test']:.2f} · MAE = {r['metricas']['MAE_test']:.1f}")
    c = r["clasificador_riesgo"]
    tarjeta_kpi(k3, "Clasificación del riesgo", f"AUC {c['AUC_CV']:.2f}",
                f"{NOMBRE_MODELO.get(c['Modelo'], c['Modelo'])} · validación cruzada")
    if shap is not None:
        tarjeta_kpi(k4, "Variable más influyente",
                    f"<span style='font-size:1.35rem'>{etiqueta(shap.iloc[0]['variable'])}</span>",
                    "según el análisis SHAP")
    st.write("")

    izq, der = st.columns(2, gap="large")
    with izq:
        st.markdown("#### Regresión: R² en validación cruzada")
        d = reg.sort_values("R2_CV_medio")
        fig = go.Figure(go.Bar(
            x=d["R2_CV_medio"], y=[NOMBRE_MODELO.get(m, m) for m in d["Modelo"]],
            orientation="h", error_x=dict(array=d["R2_CV_desv"], color=TINTA_2, thickness=1),
            marker=dict(color=[AZUL if m == r["modelo_seleccionado"] else AZUL_SERIE
                               for m in d["Modelo"]], cornerradius=4),
            hovertemplate="%{y}: R² = %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="R² medio (± desviación, 5 particiones agrupadas por fecha)")
        st.plotly_chart(estilo_figura(fig, 320), width="stretch",
                        config={"displayModeBar": False})
    with der:
        st.markdown("#### Clasificación del riesgo (PM10 > 45): AUC")
        if clf is not None:
            d = clf.sort_values("AUC_CV")
            fig = go.Figure(go.Bar(
                x=d["AUC_CV"], y=[NOMBRE_MODELO.get(m, m) for m in d["Modelo"]],
                orientation="h", marker=dict(color=[AZUL if m == c["Modelo"] else AZUL_SERIE
                                                    for m in d["Modelo"]], cornerradius=4),
                text=[f"{v:.3f}" for v in d["AUC_CV"]], textposition="outside",
                cliponaxis=False, hovertemplate="%{y}: AUC = %{x:.3f}<extra></extra>"))
            fig.add_vline(x=0.5, line=dict(color=TINTA_2, dash="dot"))
            fig.update_xaxes(range=[0.5, 0.9], title="AUC (0,5 = azar · 1 = perfecto)")
            st.plotly_chart(estilo_figura(fig, 320), width="stretch",
                            config={"displayModeBar": False})

    with st.expander("Tablas completas de métricas"):
        st.markdown("**Regresión**")
        st.dataframe(reg.rename(columns={
            "R2_CV_medio": "R² CV", "R2_CV_desv": "Desv. R² CV", "RMSE_CV_medio": "RMSE CV",
            "R2_test": "R² prueba", "RMSE_test": "RMSE prueba", "MAE_test": "MAE prueba",
            "MAPE_%_test": "MAPE prueba (%)", "R2_temporal_ganador": "R² temporal"}),
            hide_index=True, width="stretch")
        if clf is not None:
            st.markdown("**Clasificación**")
            st.dataframe(clf.rename(columns={
                "AUC_CV": "AUC CV", "AUC_test": "AUC prueba",
                "Sensibilidad_test": "Sensibilidad", "Precision_test": "Precisión",
                "F1_test": "F1"}), hide_index=True, width="stretch")

    st.divider()
    st.markdown("#### Figuras del análisis")
    grupos = {
        "Datos": [("pm10_vs_variables.png", "PM10 frente a variables meteorológicas y de voladura"),
                  ("pm10_por_mes.png", "Estacionalidad del PM10")],
        "Modelo principal": [
            ("observado_vs_predicho_pm10_ugm3.png", "Observado vs. predicho (prueba)"),
            ("shap_beeswarm_pm10_ugm3.png", "Importancia de variables (SHAP)")],
        "Red neuronal": [
            ("red_neuronal_arquitectura.png", "Arquitectura de la red"),
            ("red_neuronal_roc.png", "Curva ROC del riesgo"),
            ("red_neuronal_observado_vs_predicho.png", "Observado vs. predicho con incertidumbre"),
            ("red_neuronal_curva_aprendizaje.png", "Curva de aprendizaje"),
            ("red_neuronal_regularizacion.png", "Efecto de la regularización"),
            ("red_neuronal_importancia.png", "Importancia de variables (permutación)")],
    }
    for pest, (nombre, figs) in zip(st.tabs(list(grupos)), grupos.items()):
        with pest:
            cols = st.columns(2, gap="large")
            for i, (archivo, titulo) in enumerate(f for f in figs if (DIR_SALIDAS / f[0]).exists()):
                with cols[i % 2]:
                    st.markdown(f"**{titulo}**")
                    st.image(str(DIR_SALIDAS / archivo), width="stretch")


# ---------------------------------------------------------------------------
# Pestaña 3: acerca del sistema
# ---------------------------------------------------------------------------
def pestana_acerca():
    st.markdown("#### Cómo funciona PREDIMIN")
    pasos = [
        ("1 · Datos", "Registros de voladura, meteorología y PM10 de la U.M. La Arena, depurados "
                      "(505 → 401 registros válidos)."),
        ("2 · Modelos", "Seis ensambles de árboles y una red neuronal, optimizados con búsqueda "
                        "bayesiana y validados por fecha."),
        ("3 · Predicción", "PM10 esperado, probabilidad de superar la guía OMS y rango de "
                           "incertidumbre."),
        ("4 · Explicación", "Valores SHAP: qué variables empujan la predicción hacia arriba o "
                            "hacia abajo."),
        ("5 · Recomendación", "Nivel de alerta, medidas de mitigación, escenarios, ficha PDF y "
                              "evaluación del programa semanal."),
    ]
    for col, (t, d) in zip(st.columns(5), pasos):
        col.markdown(f"<div class='pm-step'><b>{t}</b><br><span class='pm-note'>{d}</span></div>",
                     unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns(2, gap="large")
    with izq:
        st.markdown("#### Niveles de alerta")
        st.dataframe(pd.DataFrame({
            "Nivel": [f"{v['icono']} {k.capitalize()}" for k, v in NIVELES.items()],
            "PM10 esperado (µg/m³)": ["< 27", "27 – 45", "45 – 100", "> 100"],
            "Referencia": ["< 60 % guía OMS", "Hasta la guía OMS 2021", "Sobre la guía OMS",
                           "Sobre el ECA (D.S. 003-2017-MINAM)"],
        }), hide_index=True, width="stretch")
        st.markdown("<div class='pm-note'>El nivel también sube si la probabilidad de superar "
                    "45 µg/m³ es ≥ 40 % (moderado) o ≥ 70 % (alto).</div>",
                    unsafe_allow_html=True)
    with der:
        st.markdown("#### Limitaciones")
        st.markdown(
            "- El PM10 de la estación incluye polvo de todas las fuentes, no solo de la voladura.\n"
            "- Los datos no incluyen la distancia disparo–estación ni el PM10 previo al disparo.\n"
            "- El modelo explica una parte limitada del PM10 (R² ≈ 0,2); la clasificación del "
            "riesgo es más confiable (AUC ≈ 0,8).\n"
            "- Los porcentajes de reducción del riego y la nebulización son referenciales "
            "(Cecala et al., 2019; Kissell, 2003).\n"
            "- Calibrado para La Arena: otra unidad minera requiere reentrenar el sistema.")

    st.markdown("#### Equipo de investigación")
    st.markdown(
        "Borda Callañaupa, Héctor Elías · De la Cruz Sallo, Alexa Yoselin · "
        "Onton Olivares, José Joao  \n"
        "Asesor: Mg. Canahua Loza, Reynaldo Sabino  \n"
        "Escuela Profesional de Ingeniería de Minas · Universidad Nacional de San Agustín de "
        "Arequipa")


pestanas = st.tabs(["Evaluar voladura", "Programa de voladuras", "Datos históricos",
                    "Desempeño de los modelos", "Acerca del sistema"])
for pestana, funcion in zip(pestanas, (pestana_evaluacion, pestana_programa, pestana_historico,
                                       pestana_resultados, pestana_acerca)):
    with pestana:
        funcion()

st.markdown(
    "<div class='pm-footer'>PREDIMIN · Universidad Nacional de San Agustín de Arequipa · "
    "Normativa: Guía OMS 2021 (45 µg/m³) · ECA para Aire, D.S. N.° 003-2017-MINAM "
    "(100 µg/m³)</div>", unsafe_allow_html=True)
