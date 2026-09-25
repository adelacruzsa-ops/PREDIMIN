"""
PREDIMIN - Ficha de evaluacion pre-voladura (PDF).

Resume en un documento imprimible la evaluacion de un disparo: parametros,
nivel de alerta, PM10 esperado frente a la normativa, explicacion del modelo,
medidas de mitigacion, escenarios y firmas de conformidad.

Uso desde la aplicacion (boton "Descargar ficha PDF") o desde Python:
    from ficha_pdf import generar_ficha
    pdf_bytes = generar_ficha(evento)
"""

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from config import ECA_PERU, ETIQUETAS, GUIA_OMS
from estilo_pdf import (ANCHO, AZUL, COLOR_NIVEL, E, FONDO_NIVEL, ICONO_NIVEL, MARGEN,
                        REJILLA, TEXTO_NIVEL, TINTA_2, P, fecha_hoy, pie_y_cabecera, tabla)
from recomendador import contribuciones, evaluar_escenarios, prediccion_red_neuronal, recomendar

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def _png(fig, ancho_mm):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=ancho_mm * mm, height=ancho_mm * mm * h / w)


def _grafico_pm10(valor, rango):
    oms, eca = GUIA_OMS["pm10_ugm3"], ECA_PERU["pm10_ugm3"]
    tope = max(130.0, valor * 1.15, (rango[1] if rango else 0) * 1.05)
    fig, ax = plt.subplots(figsize=(7.2, 1.25))
    for x0, x1, n in ((0, 0.6 * oms, "VERDE"), (0.6 * oms, oms, "AMARILLO"),
                      (oms, eca, "NARANJA"), (eca, tope, "ROJO")):
        ax.axvspan(x0, x1, color=FONDO_NIVEL[n], zorder=0)
    ax.barh([0], [valor], height=0.36, color="#1F3864", zorder=2)
    if rango:
        ax.plot(rango, [0, 0], color="#5B6470", lw=1.5, zorder=3)
        ax.plot(rango, [0, 0], "|", color="#5B6470", ms=12, zorder=3)
    for x, t in ((oms, "Guía OMS 45"), (eca, "ECA Perú 100")):
        ax.axvline(x, color="#1B1F24", lw=1, ls=":", zorder=1)
        ax.text(x + 1, 0.42, t, fontsize=7, color="#5B6470", va="bottom")
    ax.text(valor, 0.24, f"{valor:.1f}", fontsize=8.5, fontweight="bold", ha="center",
            va="bottom", color="#1B1F24", zorder=4)
    ax.set_xlim(0, tope)
    ax.set_ylim(-0.5, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("PM10 (µg/m³, media de 24 h)", fontsize=7.5, color="#5B6470")
    ax.tick_params(labelsize=7, colors="#5B6470")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D9DEE5")
    return fig


def _grafico_contribuciones(t):
    d = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.barh([ETIQUETAS.get(v, v) for v in d["variable"]], d["contribucion"],
            color=["#e34948" if v > 0 else "#2a78d6" for v in d["contribucion"]], height=0.65)
    ax.axvline(0, color="#5B6470", lw=0.8)
    ax.set_xlabel("Contribución al PM10 (µg/m³)", fontsize=7, color="#5B6470")
    ax.tick_params(labelsize=6.5, colors="#1B1F24")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#E6E9EE", lw=0.6)
    ax.set_axisbelow(True)
    return fig


def _kpis(celdas):
    datos = [[P(v, "kpi_valor") for _, v in celdas], [P(t, "kpi_titulo") for t, _ in celdas]]
    ancho = (ANCHO - 2 * MARGEN) / len(celdas)
    t = Table(datos, colWidths=[ancho] * len(celdas))
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, REJILLA), ("INNERGRID", (0, 0), (-1, -1), 0.5, REJILLA),
        ("TOPPADDING", (0, 0), (-1, 0), 7), ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
    ]))
    return t


def generar_ficha(evento: dict, codigo: str = "", responsable: str = "") -> bytes:
    """Devuelve el PDF (bytes) con la evaluacion del disparo."""
    resultado = recomendar(evento)
    nivel = resultado["nivel_alerta"]
    p = resultado["prediccion"]["pm10_ugm3"]
    rn = prediccion_red_neuronal(evento)
    esc = evaluar_escenarios(evento)
    contrib = contribuciones(evento)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, leftMargin=MARGEN, rightMargin=MARGEN, topMargin=15 * mm,
                            bottomMargin=16 * mm, title="Ficha de evaluación pre-voladura",
                            author="PREDIMIN")
    util = ANCHO - 2 * MARGEN
    h = []

    h.append(P("Ficha de evaluación pre-voladura", "titulo"))
    ident = f"Código del disparo: <b>{codigo}</b> · " if codigo else ""
    h.append(P(f"{ident}Unidad Minera La Arena · Generada el {fecha_hoy()}", "subtitulo"))

    # Nivel de alerta
    detalle = {
        "VERDE": "PM10 esperado por debajo del 60 % de la guía OMS. Operación normal.",
        "AMARILLO": "PM10 esperado cercano a la guía OMS (45 µg/m³). Aplicar medidas preventivas.",
        "NARANJA": "Se espera superar la guía OMS. Aplicar medidas y evaluar reprogramar.",
        "ROJO": "Se espera superar el ECA nacional (100 µg/m³). Reprogramar o fraccionar.",
    }[nivel]
    icono = Table([[P(f"<font color='white' size='16'><b>{ICONO_NIVEL[nivel]}</b></font>")]],
                  colWidths=[12 * mm], rowHeights=[12 * mm])
    icono.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(COLOR_NIVEL[nivel])),
                               ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    alerta = Table([[icono, P(f"<font size='12.5'><b>Nivel de alerta {nivel.lower()} · "
                              f"{TEXTO_NIVEL[nivel]}</b></font><br/>{detalle}")]],
                   colWidths=[16 * mm, util - 16 * mm])
    alerta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(FONDO_NIVEL[nivel])),
        ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor(COLOR_NIVEL[nivel])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    h += [alerta, Spacer(1, 6)]

    prob = p["probabilidad_superar_oms"]
    celdas = [("PM10 esperado (µg/m³)", f"{p['valor']:.1f}"),
              ("Prob. de PM10 &gt; 45 µg/m³", "–" if prob is None else f"{prob * 100:.0f} %")]
    if rn:
        celdas += [("Red neuronal: PM10 (µg/m³)",
                    f"{rn['valor']:.1f} <font size='8'>({rn['limite_inferior']:.0f}–"
                    f"{rn['limite_superior']:.0f})</font>"),
                   ("Red neuronal: prob. &gt; 45", f"{rn['probabilidad_superar_oms'] * 100:.0f} %")]
    h += [_kpis(celdas), Spacer(1, 4)]
    h.append(_png(_grafico_pm10(p["valor"], (rn["limite_inferior"], rn["limite_superior"])
                                if rn else None), util / mm))
    h.append(P("Barra azul: modelo principal. Línea gris: rango de las 10 redes neuronales "
               "(percentiles 5–95, incertidumbre del modelo). Franjas: niveles de alerta.", "chico"))

    # Parametros + explicacion
    ton = evento["tonelaje_tm"]
    filas = [["Parámetro", "Valor"],
             ["Número de taladros", f"{evento['numero_taladros']:,.0f}"],
             ["Tonelaje fracturado", f"{ton:,.0f} t"],
             ["ANFO / Emulsión", f"{evento['anfo_kg']:,.0f} / {evento['emulsion_kg']:,.0f} kg"],
             ["Heavy ANFO total", f"{evento['explosivo_total_kg']:,.0f} kg"],
             ["Factor de carga", f"{evento['explosivo_total_kg'] / ton:.3f} kg/t" if ton else "–"],
             ["Disparos en el día", f"{evento['n_eventos']:.0f}"],
             ["Retardos taladro / fila", f"{evento['tiempo_taladro_ms']:.0f} / "
                                         f"{evento['tiempo_fila_ms']:.0f} ms"],
             ["Hora · mes", f"{evento['hora']:02.0f}:00 · {MESES[int(evento['mes']) - 1]}"],
             ["Humedad relativa", f"{evento['humedad_relativa_pct']:.0f} %"],
             ["Viento", f"{evento['velocidad_viento_ms']:.1f} m/s · "
                        f"{evento['direccion_viento_grados']:.0f}°"],
             ["Precipitación", f"{evento['precipitacion_mm']:.1f} mm"]]
    izq = [P("Parámetros del disparo", "h2"), tabla(filas, [36 * mm, 42 * mm])]
    der = [P("¿Por qué el modelo predice este valor?", "h2")]
    if contrib is not None:
        der += [_png(_grafico_contribuciones(contrib), 92),
                P("Rojo: sube el PM10 esperado · azul: lo baja (valores SHAP).", "chico")]
    else:
        der.append(P("Explicación no disponible para este modelo.", "chico"))
    bloque = Table([[izq, der]], colWidths=[82 * mm, util - 82 * mm])
    bloque.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    h += [Spacer(1, 4), bloque]

    # Recomendaciones
    filas = [["N.°", "Medida", "Motivo", "Acción"]]
    for i, r in enumerate(resultado["recomendaciones"], 1):
        filas.append([str(i), f"<b>{r['nombre']}</b>", r["motivo"], r["detalle"]])
    h.append(KeepTogether([P("Medidas de mitigación recomendadas", "h2"),
                           tabla(filas, [9 * mm, 42 * mm, 58 * mm, util - 109 * mm])]))

    # Escenarios
    filas = [["Escenario", "PM10 (µg/m³)", "Prob. > 45", "Cambio vs. base", "Nivel"]]
    for _, f in esc.iterrows():
        filas.append([f["escenario"], f"{f['pm10_ugm3']:.1f}",
                      "–" if f["prob_superar_45"] is None else f"{f['prob_superar_45'] * 100:.0f} %",
                      f"{-f['reduccion_pm10_%'] + 0.0:+.1f} %",
                      f"{ICONO_NIVEL[f['nivel']]} {TEXTO_NIVEL[f['nivel']]}"])
    h.append(KeepTogether([
        P("Escenarios evaluados por el modelo", "h2"),
        tabla(filas, [62 * mm, 25 * mm, 22 * mm, 27 * mm, util - 136 * mm]),
        Spacer(1, 3),
        P("Se modifica una variable a la vez. Con los datos actuales solo los escenarios "
          "meteorológicos (humedad) tienen respaldo estadístico.", "chico")]))

    # Firmas
    sep = 8 * mm
    col = (util - 2 * sep) / 3
    firmas = Table([["", "", "", "", ""],
                    [P("Elaborado por" + (f"<br/><b>{responsable}</b>" if responsable else ""),
                       "kpi_titulo"), "",
                     P("Revisado · Medio Ambiente", "kpi_titulo"), "",
                     P("V.° B.° · Jefe de Perforación y Voladura", "kpi_titulo")]],
                   colWidths=[col, sep, col, sep, col], rowHeights=[16 * mm, None])
    firmas.setStyle(TableStyle([("LINEBELOW", (0, 0), (0, 0), 0.6, TINTA_2),
                                ("LINEBELOW", (2, 0), (2, 0), 0.6, TINTA_2),
                                ("LINEBELOW", (4, 0), (4, 0), 0.6, TINTA_2),
                                ("VALIGN", (0, 1), (-1, 1), "TOP")]))
    h.append(KeepTogether([Spacer(1, 8), firmas, Spacer(1, 8), P(
        "Documento de apoyo a la decisión generado por PREDIMIN. Las predicciones tienen "
        "incertidumbre (R² ≈ 0,2 en regresión; AUC ≈ 0,8 en la clasificación del riesgo) y no "
        "reemplazan el monitoreo de calidad del aire. Normativa de referencia: Guía OMS 2021 "
        "(45 µg/m³, 24 h) y ECA para Aire, D.S. N.° 003-2017-MINAM (100 µg/m³, 24 h).",
        "chico")]))

    doc.build(h, onFirstPage=pie_y_cabecera("Ficha de evaluación pre-voladura"),
              onLaterPages=pie_y_cabecera("Ficha de evaluación pre-voladura"))
    return buf.getvalue()


if __name__ == "__main__":
    from config import DIR_SALIDAS
    from recomendador import EVENTO_EJEMPLO

    ruta = DIR_SALIDAS / "ficha_prevoladura_ejemplo.pdf"
    ruta.write_bytes(generar_ficha(EVENTO_EJEMPLO, codigo="EJEMPLO-001"))
    print(f"[OK] {ruta}")
