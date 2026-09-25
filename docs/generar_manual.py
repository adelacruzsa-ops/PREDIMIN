"""
Genera docs/MANUAL_USUARIO_PREDIMIN.pdf (anexo de la tesis).

Uso (desde la carpeta del proyecto):
    python docs/generar_manual.py

Las capturas de pantalla estan en docs/img/. Si cambia la aplicacion, reemplace
las capturas con los mismos nombres y vuelva a ejecutar este archivo.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from estilo_pdf import (ALTO, ANCHO, AZUL, COLOR_NIVEL, FONDO_NIVEL, ICONO_NIVEL,  # noqa: E402
                        MARGEN, REJILLA, TEXTO_NIVEL, TINTA_2, P, pie_y_cabecera, tabla)

IMG = RAIZ / "docs" / "img"
SALIDA = RAIZ / "docs" / "MANUAL_USUARIO_PREDIMIN.pdf"
UTIL = ANCHO - 2 * MARGEN
URL = "https://predimin.streamlit.app"


def figura(nombre, pie, ancho=UTIL, alto_max=150 * mm):
    ruta = IMG / nombre
    w, h = PILImage.open(ruta).size
    ancho_final = min(ancho, alto_max * w / h)
    img = Image(str(ruta), width=ancho_final, height=ancho_final * h / w)
    marco = Table([[img]], colWidths=[ancho_final + 2])
    marco.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, REJILLA),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return KeepTogether([Spacer(1, 4), marco, Spacer(1, 3), P(pie, "chico"), Spacer(1, 6)])


def nota(texto, color="#EEF2F8", borde=AZUL):
    t = Table([[P(texto)]], colWidths=[UTIL])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, borde),
                           ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9)]))
    return KeepTogether([Spacer(1, 3), t, Spacer(1, 6)])


def pasos(lista):
    return [P(f"<b>{i}.</b> {t}") for i, t in enumerate(lista, 1)]


def portada(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(AZUL)
    canvas.rect(0, ALTO * 0.52, ANCHO, ALTO * 0.48, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Sans", 11)
    canvas.drawString(MARGEN + 4 * mm, ALTO - 40 * mm,
                      "UNIVERSIDAD NACIONAL DE SAN AGUSTÍN DE AREQUIPA")
    canvas.setFont("Sans", 9.5)
    canvas.drawString(MARGEN + 4 * mm, ALTO - 46 * mm,
                      "Facultad de Geología, Geofísica y Minas · Escuela Profesional de "
                      "Ingeniería de Minas")
    canvas.setFont("Sans-Bold", 44)
    canvas.drawString(MARGEN + 4 * mm, ALTO - 95 * mm, "PREDIMIN")
    canvas.setFont("Sans", 13)
    for i, linea in enumerate(("Sistema inteligente de soporte a la decisión para la",
                               "predicción y mitigación de material particulado (PM10)",
                               "generado por voladuras en minería superficial")):
        canvas.drawString(MARGEN + 4 * mm, ALTO - 108 * mm - i * 7 * mm, linea)
    canvas.setFillColor(AZUL)
    canvas.setFont("Sans-Bold", 26)
    canvas.drawString(MARGEN + 4 * mm, ALTO * 0.52 - 25 * mm, "Manual de usuario")
    canvas.setFillColor(TINTA_2)
    canvas.setFont("Sans", 11)
    canvas.drawString(MARGEN + 4 * mm, ALTO * 0.52 - 34 * mm,
                      "Versión 1.0 · Caso de estudio: Unidad Minera La Arena")
    y = 70 * mm
    canvas.setFont("Sans-Bold", 10)
    canvas.setFillColor(AZUL)
    canvas.drawString(MARGEN + 4 * mm, y, "Autores")
    canvas.setFont("Sans", 10)
    canvas.setFillColor(colors.HexColor("#1B1F24"))
    for i, a in enumerate(("Borda Callañaupa, Héctor Elías", "De la Cruz Sallo, Alexa Yoselin",
                           "Onton Olivares, José Joao")):
        canvas.drawString(MARGEN + 4 * mm, y - 6 * mm - i * 5.5 * mm, a)
    canvas.setFont("Sans-Bold", 10)
    canvas.setFillColor(AZUL)
    canvas.drawString(MARGEN + 4 * mm, y - 30 * mm, "Asesor")
    canvas.setFont("Sans", 10)
    canvas.setFillColor(colors.HexColor("#1B1F24"))
    canvas.drawString(MARGEN + 4 * mm, y - 36 * mm, "Mg. Canahua Loza, Reynaldo Sabino")
    canvas.setFillColor(TINTA_2)
    canvas.setFont("Sans", 9)
    canvas.drawString(MARGEN + 4 * mm, 18 * mm, "Arequipa – Perú · 2026")
    canvas.drawRightString(ANCHO - MARGEN, 18 * mm, URL.replace("https://", ""))
    canvas.restoreState()


def nivel_tabla():
    filas = [["Nivel", "PM10 esperado", "Significado", "Acción sugerida"]]
    info = {
        "VERDE": ("< 27 µg/m³", "Menos del 60 % de la guía OMS.", "Operación normal; riego de rutina."),
        "AMARILLO": ("27 – 45 µg/m³", "Cercano a la guía OMS 2021.",
                     "Riego previo de la plataforma y accesos."),
        "NARANJA": ("45 – 100 µg/m³", "Supera la guía OMS.",
                    "Riego, nebulización y evaluar reprogramar la hora."),
        "ROJO": ("> 100 µg/m³", "Supera el ECA (D.S. 003-2017-MINAM).",
                 "Reprogramar o fraccionar; aviso a Medio Ambiente y Relaciones Comunitarias."),
    }
    for n, (rango, sig, acc) in info.items():
        filas.append([f"<font color='{COLOR_NIVEL[n]}'><b>{ICONO_NIVEL[n]}</b></font> "
                      f"<b>{TEXTO_NIVEL[n]}</b>", rango, sig, acc])
    t = tabla(filas, [34 * mm, 28 * mm, 50 * mm, UTIL - 112 * mm], zebra=False)
    t.setStyle(TableStyle([("BACKGROUND", (0, i), (-1, i), colors.HexColor(FONDO_NIVEL[n]))
                           for i, n in enumerate(info, 1)]))
    return t


def contenido():
    h = [PageBreak()]

    h += [P("Contenido", "h1")]
    for s in ("1. Introducción", "2. Acceso al sistema", "3. Evaluar una voladura",
              "4. Interpretación de los niveles de alerta", "5. Ficha de evaluación pre-voladura",
              "6. Evaluar el programa de voladuras", "7. Datos históricos",
              "8. Desempeño de los modelos", "9. Actualizar el sistema con nuevos datos",
              "10. Buenas prácticas y limitaciones", "11. Solución de problemas"):
        h.append(P(s))
    h.append(Spacer(1, 10))

    # 1
    h += [P("1. Introducción", "h1"), P(
        "PREDIMIN es un sistema inteligente de <b>soporte a la decisión</b> que estima, "
        "<b>antes de ejecutar una voladura</b>, la concentración de material particulado PM10 "
        "que se registrará y la probabilidad de superar la guía de la Organización Mundial de la "
        "Salud (45 µg/m³). Con esa información asigna un nivel de alerta y recomienda medidas de "
        "mitigación como el riego previo, la nebulización o la reprogramación del disparo."),
        P("El sistema fue entrenado con 401 registros de voladuras de la Unidad Minera La Arena "
          "(enero de 2023 a mayo de 2024), que combinan parámetros de voladura, condiciones "
          "meteorológicas y el PM10 medido por la estación de monitoreo. Compara siete modelos de "
          "aprendizaje automático (seis ensambles de árboles y una red neuronal) y explica cada "
          "predicción mediante valores SHAP."),
        P("<b>Usuarios previstos:</b> ingenieros de perforación y voladura, supervisores de "
          "medio ambiente y responsables de relaciones comunitarias.")]
    h.append(tabla([
        ["Módulo", "Función"],
        ["Procesamiento de datos", "Depura el Excel de la mina y calcula variables derivadas "
                                   "(factor de carga, índice seco, codificación del viento)."],
        ["Modelos predictivos", "Estiman el PM10 esperado y la probabilidad de superar 45 µg/m³."],
        ["Explicación", "Indica qué variables suben o bajan la predicción (SHAP)."],
        ["Recomendación", "Asigna el nivel de alerta, propone medidas y evalúa escenarios."],
        ["Interfaz web", "Aplicación accesible desde el navegador, con reportes en PDF y Excel."],
    ], [45 * mm, UTIL - 45 * mm]))

    # 2
    h += [P("2. Acceso al sistema", "h1"), P("2.1 Versión web", "h2"),
          P(f"Abra <b>{URL}</b> en cualquier navegador (Chrome, Edge o Firefox), en computadora o "
            "celular. No requiere instalación. Si la aplicación estuvo sin uso varios días, "
            "Streamlit la pone en reposo: pulse <i>«Yes, get this app back up!»</i> y espere "
            "uno o dos minutos."),
          P("2.2 Versión local (Windows)", "h2"),
          P("Requiere Python 3.12 instalado con la opción <i>«Add Python to PATH»</i>. En la "
            "carpeta del proyecto, ejecute con doble clic:"),
          tabla([["Paso", "Archivo", "Función", "Tiempo"],
                 ["1", "1_INSTALAR.bat", "Crea el entorno e instala las librerías "
                                         "(solo la primera vez).", "5–10 min"],
                 ["2", "2_EJECUTAR_TODO.bat", "Depura los datos, entrena los modelos y genera "
                                              "tablas y figuras.", "20–45 min"],
                 ["3", "3_ABRIR_APP.bat", "Abre PREDIMIN en el navegador.", "Inmediato"]],
                [14 * mm, 42 * mm, UTIL - 86 * mm, 30 * mm]),
          P("2.3 Estructura de la pantalla", "h2"),
          P("A la izquierda está el <b>panel de parámetros</b> del disparo. En la parte central, "
            "cinco pestañas: <b>Evaluar voladura</b>, <b>Programa de voladuras</b>, <b>Datos "
            "históricos</b>, <b>Desempeño de los modelos</b> y <b>Acerca del sistema</b>. En "
            "celular, el panel de parámetros se abre con el botón «&gt;» de la esquina superior "
            "izquierda.")]
    h.append(figura("01_evaluar_general.png",
                    "Figura 1. Pantalla principal: panel de parámetros (izquierda), nivel de "
                    "alerta, indicadores y PM10 esperado frente a la normativa."))

    # 3
    h += [P("3. Evaluar una voladura", "h1"),
          P("Los resultados se actualizan automáticamente al modificar cualquier valor del panel "
            "izquierdo. Complete los parámetros del disparo planificado:"),
          tabla([["Grupo", "Parámetro", "Unidad", "Fuente del dato"],
                 ["Diseño de carga", "Número de taladros", "u", "Diseño del disparo"],
                 ["", "Tonelaje fracturado", "t", "Diseño del disparo"],
                 ["", "ANFO y emulsión", "kg", "Diseño de carga (el Heavy ANFO total y el "
                                                "factor de carga se calculan solos)"],
                 ["Secuencia", "Disparos en el día", "u", "Programa diario"],
                 ["", "Retardo entre taladros y entre filas", "ms", "Diseño de amarre"],
                 ["Programación", "Hora del disparo y mes", "h, mes", "Programa diario"],
                 ["Meteorología", "Humedad relativa", "%", "Estación meteorológica o pronóstico"],
                 ["", "Velocidad y dirección del viento", "m/s, °", "Estación o pronóstico"],
                 ["", "Precipitación", "mm", "Estación o pronóstico"]],
                [30 * mm, 55 * mm, 18 * mm, UTIL - 103 * mm]),
          Spacer(1, 6)]
    lista = pasos([
        "<b>Nivel de alerta</b> con color, icono y texto (ver sección 4).",
        "<b>Indicadores:</b> PM10 esperado del modelo principal, probabilidad de superar "
        "45 µg/m³ y la «segunda opinión» de la red neuronal con su rango de incertidumbre.",
        "<b>Gráfico frente a la normativa:</b> la barra azul es el PM10 esperado; la línea gris, "
        "el rango de las 10 redes neuronales; las franjas de color, los niveles de alerta.",
        "<b>Medidas de mitigación</b> numeradas, con el motivo y la acción concreta.",
        "<b>¿Por qué el modelo predice este valor?:</b> en rojo, las variables que suben el PM10 "
        "esperado; en azul, las que lo bajan.",
        "<b>Escenarios:</b> PM10 estimado al modificar una variable a la vez (menos explosivo, "
        "fraccionar, disparar temprano, mayor humedad, viento calmo)."])
    h += [KeepTogether([P("<b>Resultados que muestra la pestaña:</b>"), lista[0]])] + lista[1:]
    h.append(figura("02_evaluar_medidas.png",
                    "Figura 2. Medidas de mitigación recomendadas y explicación de la predicción "
                    "(valores SHAP)."))
    h.append(figura("03_escenarios_ficha.png",
                    "Figura 3. Escenarios de mitigación y generación de la ficha PDF.",
                    alto_max=125 * mm))
    h.append(nota("<b>Importante:</b> con los datos actuales, solo los escenarios meteorológicos "
                  "(humedad) tienen respaldo estadístico. Las variables de voladura no muestran "
                  "relación significativa con el PM10 registrado, por lo que sus escenarios deben "
                  "interpretarse con cautela."))

    # 4
    h += [P("4. Interpretación de los niveles de alerta", "h1"),
          P("El nivel se determina con el PM10 esperado y con la probabilidad de superar "
            "45 µg/m³; se toma el más alto de ambos criterios. Una probabilidad de 40 % o más "
            "eleva el nivel a moderado, y una de 70 % o más, a alto."), nivel_tabla(),
          Spacer(1, 4),
          P("Los umbrales se basan en la Guía de calidad del aire de la OMS (2021) y en los "
            "Estándares de Calidad Ambiental para Aire del Perú (D.S. N.° 003-2017-MINAM), "
            "ambos para una media de 24 horas.", "chico")]

    # 5
    h += [P("5. Ficha de evaluación pre-voladura", "h1"),
          P("Al final de la pestaña <b>Evaluar voladura</b> se genera un documento PDF "
            "imprimible con toda la evaluación del disparo, útil como registro y para la "
            "aprobación previa del disparo.")]
    h += pasos(["Escriba, si lo desea, el código del disparo y el nombre de quien elabora la ficha.",
                "Pulse <b>Preparar ficha PDF</b>.",
                "Pulse <b>Descargar ficha PDF</b>. Si cambia los parámetros, vuelva a prepararla."])
    h.append(P("La ficha contiene: nivel de alerta, indicadores, gráfico frente a la normativa, "
               "parámetros del disparo, explicación del modelo, medidas recomendadas, escenarios y "
               "espacios de firma para quien la elabora, el área de Medio Ambiente y el Jefe de "
               "Perforación y Voladura."))

    # 6
    h += [P("6. Evaluar el programa de voladuras", "h1"),
          P("Permite evaluar todos los disparos programados de la semana a la vez y "
            "priorizar las medidas de control en los de mayor riesgo.")]
    h += pasos(["En la pestaña <b>Programa de voladuras</b>, descargue la <b>Plantilla Excel</b>.",
                "Complete una fila por disparo. La hoja «Instrucciones» describe cada columna. Para "
                "la meteorología use el pronóstico del día previsto.",
                "Suba el archivo. Para probar, marque «Probar con el programa de ejemplo».",
                "Revise el resumen, el gráfico y la tabla, ordenados de mayor a menor riesgo.",
                "Descargue el resultado en Excel: cada fila aparece coloreada según su nivel."])
    h.append(figura("04_programa.png", "Figura 4. Evaluación del programa de voladuras.",
                    alto_max=140 * mm))
    h.append(nota("Las filas con celdas vacías o con texto en columnas numéricas no se evalúan; "
                  "aparecen con la observación «No evaluado» y el motivo."))

    # 7
    h += [P("7. Datos históricos", "h1"),
          P("Muestra los registros con los que se entrenó el sistema. Con el deslizador "
            "<b>Periodo</b> se filtra el rango de fechas. Incluye indicadores (PM10 mediano, "
            "porcentaje de voladuras sobre la guía OMS y el ECA), la serie temporal con el "
            "promedio mensual, el porcentaje mensual de superaciones y la relación entre la "
            "humedad relativa y el PM10. Al pasar el cursor sobre un punto se ven la fecha, la "
            "humedad, el viento y el explosivo de esa voladura.")]
    h.append(figura("05_historico.png", "Figura 5. Explorador de datos históricos.",
                    alto_max=140 * mm))

    # 8
    h += [P("8. Desempeño de los modelos", "h1"),
          P("Resume la validación de los siete modelos y reúne las figuras del análisis "
            "(datos, modelo principal y red neuronal)."),
          tabla([["Métrica", "Qué mide", "Cómo leerla"],
                 ["R²", "Proporción de la variación del PM10 explicada por el modelo.",
                  "1 = perfecto; 0 = igual que predecir el promedio."],
                 ["MAE / RMSE", "Error medio de la predicción, en µg/m³.", "Cuanto menor, mejor."],
                 ["AUC", "Capacidad de distinguir las voladuras que superan 45 µg/m³ de las "
                         "que no.", "0,5 = azar; 1 = perfecto; 0,8 = buena."],
                 ["Sensibilidad", "Porcentaje de superaciones reales que el sistema detecta.",
                  "Cuanto mayor, menos superaciones pasan inadvertidas."]],
                [28 * mm, 80 * mm, UTIL - 108 * mm])]
    h.append(figura("06_desempeno.png", "Figura 6. Desempeño de los modelos.", alto_max=140 * mm))

    # 9
    h += [P("9. Actualizar el sistema con nuevos datos", "h1")]
    h += pasos(["Reemplace <b>data/Pm 10 nas voladura.xlsx</b> por el archivo actualizado, con "
                "las mismas columnas.",
                "Ejecute <b>2_EJECUTAR_TODO.bat</b>. Los modelos se vuelven a entrenar y se "
                "actualizan todas las tablas y figuras de la carpeta <b>outputs</b>.",
                "Para actualizar la versión web, suba los cambios de las carpetas <b>data</b> y "
                "<b>outputs</b> al repositorio de GitHub; Streamlit se actualiza solo en pocos "
                "minutos."])
    h.append(nota("Datos que mejorarían el sistema (ver <b>data/datos_adicionales_a_solicitar.csv"
                  "</b>): PM10 antes y después del disparo, ubicación del disparo y de la estación, "
                  "PM2.5, temperatura, geometría de perforación y registro de las medidas de control "
                  "aplicadas. La curva de aprendizaje de la red neuronal muestra que el desempeño "
                  "sigue mejorando con más registros."))

    # 10
    h += [P("10. Buenas prácticas y limitaciones", "h1")]
    for t in (
        "Use PREDIMIN como <b>apoyo a la decisión</b>: no reemplaza el monitoreo de calidad del aire "
        "ni el criterio del ingeniero.",
        "Priorice el <b>nivel de alerta y la probabilidad</b> antes que el valor exacto de PM10: la "
        "clasificación del riesgo es más confiable (AUC ≈ 0,8) que la regresión (R² ≈ 0,2).",
        "El PM10 de la estación incluye polvo de todas las fuentes (acarreo, chancado, viento), no "
        "solo de la voladura.",
        "Los porcentajes de reducción del riego y de la nebulización son referenciales (Cecala et "
        "al., 2019; Kissell, 2003); el sistema no los estima con datos propios.",
        "El sistema está calibrado para la U.M. La Arena. Para otra operación debe reentrenarse "
        "con sus propios registros.",
        "No introduzca valores fuera del rango histórico (por ejemplo, disparos mucho mayores que "
        "los registrados): la predicción pierde confiabilidad."):
        h.append(P(f"• {t}"))

    # 11
    h += [P("11. Solución de problemas", "h1"),
          tabla([["Problema", "Causa probable", "Solución"],
                 ["La página web tarda en abrir", "La aplicación estaba en reposo.",
                  "Pulse «Yes, get this app back up!» y espere 1–2 minutos."],
                 ["«ModuleNotFoundError» o error al cargar el modelo",
                  "Los modelos se crearon con otra versión de las librerías.",
                  "Ejecute de nuevo 2_EJECUTAR_TODO.bat."],
                 ["«No se encontró Python» al instalar",
                  "Python no está instalado o no está en el PATH.",
                  "Instale Python 3.12 marcando «Add Python to PATH»."],
                 ["«Faltan columnas en el archivo»", "El Excel no sigue la plantilla.",
                  "Descargue la plantilla y copie los datos sin cambiar los encabezados."],
                 ["La ficha PDF no refleja los cambios", "La ficha se generó antes del cambio.",
                  "Pulse de nuevo «Preparar ficha PDF»."]],
                [48 * mm, 55 * mm, UTIL - 103 * mm])]
    return h


def main():
    doc = SimpleDocTemplate(str(SALIDA), leftMargin=MARGEN, rightMargin=MARGEN,
                            topMargin=16 * mm, bottomMargin=17 * mm,
                            title="PREDIMIN - Manual de usuario", author="PREDIMIN")
    pie = pie_y_cabecera("Manual de usuario · v1.0")
    doc.build([Spacer(1, 1)] + contenido(), onFirstPage=portada, onLaterPages=pie)
    print(f"[OK] {SALIDA}")


if __name__ == "__main__":
    main()
