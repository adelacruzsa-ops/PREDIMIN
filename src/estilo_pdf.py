"""
PREDIMIN - Estilo comun de los documentos PDF (ficha pre-voladura y manual).

Usa la fuente DejaVu Sans que viene incluida con matplotlib, de modo que las
tildes, la "ñ" y los simbolos (µg/m³, ≥, ✔) se ven igual en cualquier PC o
servidor sin instalar nada adicional.
"""

import os
from datetime import datetime

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle

# ---------------------------------------------------------------------------
# Fuentes
# ---------------------------------------------------------------------------
_DIR_FUENTES = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for _nombre, _archivo in (("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                          ("Sans-Oblique", "DejaVuSans-Oblique.ttf")):
    if _nombre not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(_nombre, os.path.join(_DIR_FUENTES, _archivo)))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique")

# ---------------------------------------------------------------------------
# Colores (los mismos de la aplicacion)
# ---------------------------------------------------------------------------
AZUL = colors.HexColor("#1F3864")
AZUL_CLARO = colors.HexColor("#EEF2F8")
TINTA = colors.HexColor("#1B1F24")
TINTA_2 = colors.HexColor("#5B6470")
REJILLA = colors.HexColor("#D9DEE5")

COLOR_NIVEL = {"VERDE": "#0ca30c", "AMARILLO": "#C98A00", "NARANJA": "#D9632F", "ROJO": "#d03b3b"}
FONDO_NIVEL = {"VERDE": "#E8F6E8", "AMARILLO": "#FEF6E3", "NARANJA": "#FDEEE7", "ROJO": "#FBE9E9"}
TEXTO_NIVEL = {"VERDE": "Riesgo bajo", "AMARILLO": "Riesgo moderado", "NARANJA": "Riesgo alto",
               "ROJO": "Riesgo crítico"}
ICONO_NIVEL = {"VERDE": "✔", "AMARILLO": "!", "NARANJA": "▲", "ROJO": "✖"}

ANCHO, ALTO = A4
MARGEN = 16 * mm

# ---------------------------------------------------------------------------
# Estilos de texto
# ---------------------------------------------------------------------------
E = {
    "titulo": ParagraphStyle("titulo", fontName="Sans-Bold", fontSize=17, leading=21,
                             textColor=AZUL, spaceAfter=2),
    "subtitulo": ParagraphStyle("subtitulo", fontName="Sans", fontSize=9.5, leading=13,
                                textColor=TINTA_2, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14, leading=18, textColor=AZUL,
                         spaceBefore=10, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11, leading=14, textColor=AZUL,
                         spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "normal": ParagraphStyle("normal", fontName="Sans", fontSize=9.5, leading=13.5,
                             textColor=TINTA, spaceAfter=4),
    "chico": ParagraphStyle("chico", fontName="Sans", fontSize=8, leading=10.5,
                            textColor=TINTA_2),
    "celda": ParagraphStyle("celda", fontName="Sans", fontSize=8.5, leading=11,
                            textColor=TINTA),
    "celda_b": ParagraphStyle("celda_b", fontName="Sans-Bold", fontSize=8.5, leading=11,
                              textColor=TINTA),
    "cabecera": ParagraphStyle("cabecera", fontName="Sans-Bold", fontSize=8.5, leading=11,
                               textColor=colors.white),
    "kpi_valor": ParagraphStyle("kpi_valor", fontName="Sans-Bold", fontSize=15, leading=18,
                                textColor=TINTA, alignment=TA_CENTER),
    "kpi_titulo": ParagraphStyle("kpi_titulo", fontName="Sans", fontSize=7.5, leading=9.5,
                                 textColor=TINTA_2, alignment=TA_CENTER),
}


def P(texto, estilo="normal"):
    return Paragraph(texto, E[estilo])


def tabla(filas, anchos, cabecera=True, zebra=True):
    """Tabla con cabecera azul y filas alternadas."""
    datos = [[c if not isinstance(c, str) else P(c, "cabecera" if (cabecera and i == 0)
                                                  else "celda") for c in fila]
             for i, fila in enumerate(filas)]
    t = Table(datos, colWidths=anchos, repeatRows=1 if cabecera else 0)
    estilo = [
        ("GRID", (0, 0), (-1, -1), 0.4, REJILLA),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if cabecera:
        estilo.append(("BACKGROUND", (0, 0), (-1, 0), AZUL))
    if zebra:
        for i in range(1 if cabecera else 0, len(filas)):
            if i % 2 == 0:
                estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F6F7F9")))
    t.setStyle(TableStyle(estilo))
    return t


def pie_y_cabecera(titulo_doc):
    """Funcion para onPage: franja superior y pie con numero de pagina."""
    def dibujar(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(AZUL)
        canvas.rect(0, ALTO - 9 * mm, ANCHO, 9 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Sans-Bold", 9)
        canvas.drawString(MARGEN, ALTO - 6 * mm, "PREDIMIN")
        canvas.setFont("Sans", 8)
        canvas.drawRightString(ANCHO - MARGEN, ALTO - 6 * mm, titulo_doc)
        canvas.setStrokeColor(REJILLA)
        canvas.line(MARGEN, 12 * mm, ANCHO - MARGEN, 12 * mm)
        canvas.setFillColor(TINTA_2)
        canvas.setFont("Sans", 7.5)
        canvas.drawString(MARGEN, 8 * mm,
                          "Universidad Nacional de San Agustín de Arequipa · Escuela Profesional "
                          "de Ingeniería de Minas")
        canvas.drawRightString(ANCHO - MARGEN, 8 * mm, f"Página {doc.page}")
        canvas.restoreState()
    return dibujar


def fecha_hoy() -> str:
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre"]
    h = datetime.now()
    return f"{h.day} de {meses[h.month - 1]} de {h.year}, {h:%H:%M}"
