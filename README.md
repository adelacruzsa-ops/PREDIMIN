# PREDIMIN — versión 5 (datos reales de La Arena)

Sistema inteligente de predicción y mitigación de material particulado (PM10)
generado por voladuras en minería superficial.
Tesis — Ingeniería de Minas, UNSA. Caso de estudio: U.M. La Arena.

---

## Uso fácil (doble clic, en Windows)

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `1_INSTALAR.bat` | Crea el entorno e instala las librerías (solo la primera vez) |
| 2 | `2_EJECUTAR_TODO.bat` | Limpia el Excel, analiza, entrena los 7 modelos, analiza la red neuronal y genera el reporte (20-45 min) |
| 2b | `2b_EJECUTAR_RAPIDO.bat` | Lo mismo pero sin optimización bayesiana (1-3 min), para probar |
| 3 | `3_ABRIR_APP.bat` | Abre la aplicación PREDIMIN en el navegador |

Resultados: carpeta `outputs`, en especial **`RESULTADOS_TESIS.xlsx`** (todas las tablas).

## Publicar la app en internet (Streamlit Community Cloud, gratis)

1. Entrar a https://share.streamlit.io y pulsar **Continue with GitHub**.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository: `adelacruzsa-ops/PREDIMIN` · Branch: `main` · Main file path: `src/app.py`.
4. En **Advanced settings**, elegir Python **3.12**. Pulsar **Deploy**.

La primera vez tarda unos minutos (instala las librerías y reconstruye los modelos
a partir de `outputs/`). Antes de publicar, confirmar con la mina que autoriza
mostrar sus datos.

## Uso desde la terminal de VS Code

```
.venv\Scripts\python.exe src/importar_excel.py         # 1. limpia el Excel
.venv\Scripts\python.exe src/analisis_exploratorio.py  # 2. correlaciones y gráficos
.venv\Scripts\python.exe src/entrenamiento.py          # 3. entrena y compara modelos
.venv\Scripts\python.exe src/red_neuronal.py           # 4. análisis completo de la red neuronal
.venv\Scripts\python.exe src/reporte_tesis.py          # 5. RESULTADOS_TESIS.xlsx
.venv\Scripts\python.exe -m streamlit run src/app.py   # 6. aplicación web
```

Si la mina envía más datos, reemplace `data/Pm 10 nas voladura.xlsx` (mismas
columnas) y vuelva a ejecutar `2_EJECUTAR_TODO.bat`.

## Estructura

```
data/   Pm 10 nas voladura.xlsx         datos originales de la mina
        registros_voladura.csv          datos depurados (generado)
        datos_adicionales_a_solicitar.csv
src/    config.py                       variables, umbrales OMS/ECA, parámetros
        importar_excel.py               Módulo 0: depuración del Excel
        analisis_exploratorio.py        Objetivo 1: influencia de las variables
        preprocesamiento.py             Módulo 1: variables derivadas y división 80:20
        entrenamiento.py                Módulo 2: 6 algoritmos + red neuronal + optimización bayesiana + SHAP + clasificador
        red_neuronal.py                 Red neuronal: ensamble de 10 MLP con incertidumbre; ejecutada sola genera su análisis completo
        reporte_tesis.py                Reúne todas las tablas en outputs/RESULTADOS_TESIS.xlsx
        recomendador.py                 Módulo 3: alerta, medidas y escenarios
        app.py                          Módulo 4: interfaz web (Streamlit)
models/ outputs/                        modelos, tablas y figuras (generados)
docs/   REVISION_PLAN_TESIS.md          revisión del plan de tesis, textos propuestos y resultados v5
```

## Novedades de la versión 5

- **Red neuronal rediseñada**: ensamble de 10 perceptrones multicapa con remuestreo bootstrap
  (Monte Carlo), objetivo en log(1+PM10), optimizador L-BFGS y arquitectura elegida por
  optimización bayesiana. R² de prueba: −0,04 → 0,20. Compite también en la clasificación
  del riesgo (AUC ≈ 0,79) y la app la muestra como "segunda opinión" con su rango de incertidumbre.
- **Análisis de la red** (`src/red_neuronal.py`): arquitectura, regularización, curva de
  aprendizaje, ROC, observado vs. predicho e importancia por permutación.
- **Validación agrupada por fecha**: los registros de un mismo día (comparten polvo de fondo)
  ya no se reparten entre entrenamiento y prueba, así se evita la fuga de información.

---

## Depuración aplicada a los datos (para la sección de metodología)

| Problema encontrado en el Excel | Tratamiento |
|---|---|
| 101 registros con PM10 = 0 (concentrados en jun, sep y oct 2023) | Dato faltante: falla o mantenimiento del monitor. **Confirmar con la mina** |
| "S/D" en humedad, viento y precipitación | Dato faltante |
| `TOTAL DE EXPLOSIVOS` es el doble del Heavy ANFO (suma ANFO + emulsión + Heavy ANFO) | Se usa Heavy ANFO (= ANFO + emulsión) como total |
| Retardos escritos como texto ("7 y 10", "13, 17 y 42") | Se toma el menor valor |
| Tonelaje = 0 (14 casos), retardo de 46 246 ms, P80 = 197 | Dato faltante |
| Diámetro (6 1/8"), taco (grava) y límite (45) son constantes | Se descartan: no aportan información |
| El P80 se mide **después** del disparo | No se usa para predecir (solo para análisis) |

Quedan **401 registros válidos** (ene-2023 a may-2024).

## Resultados con los datos actuales (leer antes de escribir la tesis)

1. **La humedad relativa es la variable que más influye** en el PM10
   (ρ de Spearman = −0,44), seguida de la estacionalidad (mes) y la precipitación.
   SHAP (Extra Trees) y la importancia por permutación (red neuronal) coinciden.
2. **Las variables de voladura (explosivo, taladros, tonelaje, factor de carga) no
   muestran relación significativa con el PM10 registrado** (|ρ| < 0,1, p > 0,05).
   El factor de carga casi no varía (≈ 0,23 kg/t), así que el modelo no puede
   aprender su efecto.
3. El modelo de regresión explica una parte limitada del PM10 (Extra Trees:
   R² = 0,22 en validación cruzada y 0,25 en prueba; red neuronal: 0,13 y 0,20).
   La **clasificación** (¿superará 45 µg/m³?) funciona mejor (AUC ≈ 0,8 en todos
   los modelos, incluida la red neuronal), por eso la app muestra ambas.
4. En la validación temporal (entrenar con meses pasados y probar con meses
   futuros) el R² es cercano a 0: el modelo todavía no generaliza bien a
   periodos que no vio.
5. La curva de aprendizaje de la red muestra que el desempeño **sigue mejorando
   con más registros**.

Detalle completo y cómo defenderlo ante el jurado: `docs/REVISION_PLAN_TESIS.md`.

**Posible explicación:** el PM10 de la estación incluye polvo de todas las
fuentes (vías de acarreo, chancado, viento), y no sabemos a qué distancia está
la estación del disparo ni cuánto PM10 había antes. La señal de la voladura
queda enmascarada por el polvo de fondo.

**Qué pedir a la mina** (`data/datos_adicionales_a_solicitar.csv`): el PM10 de
la hora anterior y de las horas posteriores al disparo, la ubicación de la
estación y del disparo, y la confirmación de los registros con PM10 = 0.
Con eso se puede modelar el **aporte neto de la voladura** (PM10 después −
PM10 antes), que es lo que plantea la tesis.

## Notas

- Los % de reducción del catálogo de medidas (riego, nebulización) son
  referenciales: deben citarse de la literatura. El modelo solo estima los
  escenarios que cambian variables presentes en los datos.
- Si aparece `ModuleNotFoundError` al abrir la app, vuelva a entrenar
  (`python src/entrenamiento.py`): los modelos se deben crear en su propia PC.
