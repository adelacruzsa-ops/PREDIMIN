# Revisión del plan de tesis y del sistema PREDIMIN

**Documento revisado:** `PLAN_TESIS_FINAL.docx` (23 páginas) y el código PREDIMIN v4.
**Objetivo de esta revisión:** dejar el plan y el software listos para presentarlos ante el jurado.

---

## 1. Veredicto general

| Parte | Estado | Comentario |
|---|---|---|
| Software PREDIMIN (código) | **Bien hecho** | Ordenado, reproducible y metodológicamente honesto: elige el modelo por validación cruzada (no por la prueba), valida en el tiempo, usa SHAP y no inventa efectos de mitigación. En la versión 5 se mejoraron la red neuronal y la validación (ver sección 3). |
| Planteamiento, justificación y antecedentes | **Bien** | Bien redactados. Faltan corregir detalles de formato y citas (sección 2.3). |
| Hipótesis y variables | **Debe corregirse** | Tal como está, la hipótesis no se puede comprobar con los datos (sección 2.1). |
| Metodología (2.4) y aspectos administrativos (2.5, 2.6) | **Incompleto** | Están vacíos (`[ ]`). En la sección 4 hay textos listos para completar. |
| PM2.5 | **Inconsistente** | El plan promete PM10 **y** PM2.5, pero los datos de la mina solo tienen PM10. |
| Resultados esperados vs. antecedentes | **Riesgo ante el jurado** | Los antecedentes reportan R² de 0,95–0,999; con los datos reales se obtiene R² ≈ 0,2 en regresión. Hay que explicarlo (sección 5) y apoyarse en la clasificación del riesgo (AUC ≈ 0,8), que sí funciona. |

---

## 2. Observaciones al documento del plan

### 2.1 Observaciones críticas (corregir sí o sí)

1. **Hipótesis no contrastable.** "Que con el desarrollo del sistema inteligente de predicción PREDIMIN podrá mitigar material particulado…" no se puede verificar: las medidas de mitigación **no se implementan en campo** (lo dice el propio plan en 2.2.2.4), así que nunca se mide una reducción real. La hipótesis debe referirse a algo que los datos sí permiten medir: la capacidad predictiva del modelo y la influencia de las variables. Ver la propuesta en la sección 4.1.

2. **Variables mal definidas.** "Variable independiente: Sistema PREDIMIN / Variable dependiente: Mitigación" no se puede operacionalizar. Lo correcto es:
   - Variables independientes (X): parámetros operacionales de voladura y variables meteorológicas.
   - Variable dependiente (Y): concentración de PM10 (µg/m³) y su superación de la guía OMS (45 µg/m³).
   Ver la tabla de operacionalización en la sección 4.2.

3. **PM2.5.** El título no lo menciona, pero el resumen, el problema, los objetivos y los indicadores sí. El Excel de La Arena solo tiene PM10. Hay dos caminos: (a) pedir el PM2.5 a la mina (ya figura en `data/datos_adicionales_a_solicitar.csv`), o (b) **recomendado**: limitar los objetivos a PM10 y mencionar el PM2.5 como limitación y trabajo futuro. El código ya está preparado: basta con agregar `"pm25_ugm3"` en `config.py` si llegan los datos.

4. **Metodología vacía.** Las secciones 2.4.1 a 2.4.4, 2.5 y 2.6 están en blanco o con `[ ]`. En la sección 4 están los textos propuestos.

5. **Cronograma incompleto.** "Fase II Determinación de" está cortado.

### 2.2 Observaciones de coherencia

6. **Problemas, objetivos e hipótesis no están alineados.** Hay 5 objetivos específicos y solo 3 problemas específicos. Se recomienda incluir una **matriz de consistencia** (problema → objetivo → hipótesis → variables → indicadores) con una fila por objetivo.

7. **Los objetivos 1 y 3 se superponen** ("analizar la influencia" e "identificar las variables de mayor importancia"). Para diferenciarlos:
   - Objetivo 1: análisis **estadístico** (correlación de Spearman con prueba de significancia).
   - Objetivo 3: importancia **dentro del modelo** (SHAP en los árboles y permutación en la red neuronal).

8. **Factor de carga.** El plan lo define en kg/m³ (FC = Qe / Vr), pero la mina registra **tonelaje**, así que el sistema lo calcula en **kg/t** (FC = Qe / T). Cambiar la fórmula o indicar la densidad usada para convertir.

9. **Falta teoría de la red neuronal.** El sistema usa una red neuronal artificial y el plan cita redes neuronales en los antecedentes, pero el marco teórico no la explica (neurona, capas, activación, entrenamiento, regularización). Tampoco se definen las métricas de clasificación (AUC, sensibilidad, precisión), que son las que mejor resultado dan. Ver el texto propuesto en la sección 4.5.

### 2.3 Forma y citas

10. **Hosseini y Pourmirzaee**: dice 2024 en el texto y 2023 en la Tabla 1 y en las referencias. El artículo es del volumen 240 de *Expert Systems with Applications*, de **2024**. Usar 2024 en todo el documento.
11. **Tabla 1**: dice "R² = 0,999 (PM10); R² = 0,999 (TSP)", pero el texto dice 0,999 y 0,991 para PM10, y 0,999 y 0,998 para TSP. Unificar (indicar entrenamiento/prueba).
12. **Numeración duplicada**: "2.1.2" aparece dos veces (Formulación y Justificación) y "2.2.2.4" también (Estrategias de mitigación e Inteligencia artificial).
13. **Citas sin referencia**: *Lundberg & Lee (2017)* y *Russell & Norvig (2021)* se citan en el texto pero **no están** en la lista de referencias.
14. Erratas: "voladura.," (p. 4), "Tran sformer" (Tabla 1), "Ch ange" (referencia de Seinfeld & Pandis), "Resumen (problema , metodología…)".
15. Referencias que conviene agregar por el software utilizado (ver la sección 4.7).

---

## 3. Cambios hechos al software (versión 5)

### 3.1 Red neuronal: de un solo perceptrón a un ensamble probabilístico

En la versión 4 la red neuronal era **el peor de los 7 modelos** (R² de prueba = −0,04): peor que predecir siempre el promedio. En la versión 5 se rediseñó así:

| Aspecto | v4 | v5 | Por qué |
|---|---|---|---|
| Variable que aprende | PM10 en µg/m³ | log(1 + PM10) | El PM10 es muy asimétrico (asimetría ≈ 2,5): los pocos picos dominaban el error. |
| Optimizador | Adam + early stopping | L-BFGS (cuasi-Newton) | Con ~320 registros converge de forma más estable y no desperdicia el 10 % de los datos en early stopping. |
| Activación | ReLU | tanh o ReLU (lo decide la optimización bayesiana) | |
| Número de redes | 1 | **Ensamble de 10 redes con remuestreo bootstrap (Monte Carlo)** | Reduce la varianza y entrega un **intervalo de incertidumbre**, la misma idea de Hosseini y Pourmirzaee (2024). |
| Clasificación del riesgo | No participaba | **Compite contra los árboles** | Es la salida más útil del sistema. |
| Salida en la app | No se mostraba | "Segunda opinión" con intervalo del 90 % | |

Además, `python src/red_neuronal.py` genera un análisis completo de la red para el capítulo de resultados:

| Archivo en `outputs/` | Contenido | Uso en la tesis |
|---|---|---|
| `red_neuronal_arquitectura.png` | Diagrama de la red | Metodología |
| `red_neuronal_regularizacion.png` | R² de entrenamiento vs. validación según alpha | Justifica la regularización (control del sobreajuste) |
| `red_neuronal_curva_aprendizaje.png` | Desempeño según la cantidad de registros | Muestra si más datos mejorarían el modelo |
| `red_neuronal_observado_vs_predicho.png` | Predicciones con intervalo del 90 % | Resultados, objetivo 2 |
| `red_neuronal_roc.png` | Curva ROC del riesgo de superar 45 µg/m³ | Resultados, objetivo 2 |
| `red_neuronal_importancia.png` / `.csv` | Importancia por permutación | Objetivo 3: se contrasta con SHAP |
| `red_neuronal_resumen.json` | Arquitectura, parámetros y métricas | Tablas |

### 3.2 Validación más rigurosa: agrupada por fecha

En 118 días hay 2 o 3 registros el mismo día, y comparten el polvo de fondo. Antes podía quedar un registro de un día en entrenamiento y otro del **mismo día** en prueba, y el modelo "adivinaba" usando información del mismo día (fuga de información). Ahora la división 80:20 y la validación cruzada están **agrupadas por fecha**. Las métricas bajan un poco, pero son las correctas y resisten la pregunta del jurado "¿cómo evitaron la fuga de datos?".

### 3.3 Resultados actualizados

Resultados con la validación agrupada por fecha y 40 ensayos de optimización bayesiana por algoritmo (401 registros: 321 para entrenamiento y 80 para prueba):

**Regresión (valor de PM10)**

| Modelo | R² validación cruzada | R² prueba | RMSE prueba (µg/m³) | MAE prueba (µg/m³) |
|---|---|---|---|---|
| **Extra Trees (seleccionado)** | **0,216 ± 0,110** | **0,254** | **25,9** | **17,5** |
| LightGBM | 0,195 ± 0,069 | 0,212 | 26,6 | – |
| Random Forest | 0,192 ± 0,078 | 0,202 | 26,8 | – |
| CatBoost | 0,183 ± 0,113 | 0,160 | 27,5 | – |
| Gradient Boosting | 0,180 ± 0,078 | 0,216 | 26,6 | – |
| XGBoost | 0,167 ± 0,077 | 0,237 | 26,2 | – |
| Red neuronal (ensamble) | 0,129 ± 0,124 | 0,204 | 26,8 | **16,7** |

- La red neuronal pasó de un R² de prueba de **−0,04 (v4) a 0,20 (v5)**, y tiene el **menor MAE** de todos los modelos (16,7 µg/m³).
- La validación temporal del modelo seleccionado sigue cerca de 0 (R² = 0,03): el modelo aún no generaliza a meses que no vio. Es una limitación que debe declararse.

**Clasificación (riesgo de superar 45 µg/m³; lo supera el 22 % de los registros)**

| Modelo | AUC validación cruzada | AUC prueba | Sensibilidad | Precisión |
|---|---|---|---|---|
| **Extra Trees (seleccionado)** | **0,801** | 0,760 | 0,69 | 0,41 |
| Random Forest | 0,795 | 0,778 | 0,69 | 0,45 |
| CatBoost | 0,791 | 0,791 | 0,54 | 0,50 |
| Gradient Boosting | 0,790 | 0,782 | 0,38 | 0,42 |
| Red neuronal (ensamble) | 0,789 | 0,790 | 0,46 | **0,55** |
| LightGBM | 0,775 | 0,796 | 0,46 | 0,35 |

Todos los clasificadores quedan entre 0,77 y 0,80 de AUC: **la red neuronal es competitiva con los árboles** y tiene la mayor precisión (menos falsas alarmas).

**Red neuronal seleccionada por la optimización bayesiana:** 20 entradas → 1 capa oculta de 4 neuronas (tanh) → 1 salida; α = 2,17; 89 parámetros por red; ensamble de 10 redes. Que la optimización prefiera una red **pequeña** es coherente con datos ruidosos y pocos registros: una red grande memorizaría el ruido (ver `red_neuronal_regularizacion.png`).

**Lo que muestran las figuras de la red:**
- *Curva de aprendizaje*: el R² y el AUC de validación **siguen subiendo** al aumentar los registros → **con más datos el modelo mejoraría**. Es un buen argumento para la sección de recomendaciones.
- *Importancia por permutación*: la **humedad relativa** es, por lejos, la variable más importante también para la red, al igual que en SHAP con Extra Trees. Que dos métodos distintos coincidan hace el hallazgo más sólido (objetivo 3).
- *Incertidumbre*: el rango del ensamble (percentiles 5–95 de las 10 redes) contiene al 60 % de los valores observados, no al 90 %. Esto es esperable, porque mide la incertidumbre **del modelo** y no el ruido del monitor. Hay que decirlo así en la tesis: es un indicador de confianza relativa, no un intervalo de predicción calibrado.

**Precaución con los escenarios (objetivo 5):** con estos datos, los escenarios que cambian variables de voladura (−10 % de explosivo, fraccionar, un solo disparo) dan cambios pequeños y a veces **contraintuitivos** (el PM10 sube), porque esas variables no tienen relación significativa con el PM10 registrado. Solo el escenario de **mayor humedad** (+15 % HR → −19 % de PM10 en el evento de ejemplo) es consistente con el análisis estadístico. En la tesis conviene presentar los escenarios meteorológicos (reprogramar a una ventana más húmeda, riego) como los respaldados por los datos y explicar por qué los de voladura no lo están.

---

## 4. Textos propuestos para completar el plan

> Son borradores: adáptenlos al formato de la escuela y revísenlos con el asesor.

### 4.1 Hipótesis (2.3.1)

**Hipótesis general.** Un sistema inteligente basado en modelos de aprendizaje automático, entrenado con los registros operacionales y meteorológicos de las voladuras de la U.M. La Arena, permite anticipar los eventos de voladura con riesgo de superar la guía de la OMS para PM10 (45 µg/m³) con una capacidad de discriminación significativamente superior al azar (AUC > 0,5), lo que sirve de base para recomendar medidas de mitigación preventivas.

**Hipótesis específicas.**
- H1: Las variables meteorológicas (humedad relativa, velocidad y dirección del viento, precipitación) tienen una influencia estadísticamente significativa (p < 0,05) sobre la concentración de PM10 registrada durante las voladuras.
- H2: Al menos uno de los modelos de aprendizaje automático evaluados (ensambles de árboles y red neuronal artificial) predice la concentración de PM10 mejor que un modelo de referencia que predice siempre el promedio (R² > 0 en validación cruzada).
- H3: La humedad relativa es la variable de mayor importancia en el modelo predictivo según el análisis SHAP y de importancia por permutación.
- H4: Los escenarios que modifican variables controlables (hora del disparo, fraccionamiento, cantidad de explosivo) producen cambios cuantificables en el PM10 estimado por el modelo.

### 4.2 Variables y operacionalización (2.3.2)

| Tipo | Variable | Dimensión | Indicador | Unidad | Fuente / instrumento |
|---|---|---|---|---|---|
| Independiente | Parámetros operacionales de voladura | Diseño de carga | Explosivo total (Heavy ANFO), ANFO, emulsión | kg | Reporte de voladura |
| | | | Factor de carga | kg/t | Calculado |
| | | Geometría del disparo | Número de taladros, tonelaje fracturado | u, t | Reporte de voladura |
| | | Secuencia | Retardo entre taladros y entre filas; disparos por día | ms, u | Reporte de voladura |
| Independiente | Condiciones meteorológicas | Humedad | Humedad relativa | % | Estación meteorológica |
| | | Viento | Velocidad; dirección (seno y coseno) | m/s, ° | Estación meteorológica |
| | | Precipitación | Precipitación | mm | Estación meteorológica |
| | | Temporalidad | Hora del disparo; mes (seno y coseno) | h, mes | Registro |
| Dependiente | Material particulado | Concentración | PM10 | µg/m³ | Monitor de calidad de aire |
| | | Riesgo normativo | Superación de la guía OMS (45 µg/m³) y del ECA (100 µg/m³) | sí/no | Calculado |
| Interviniente | Sistema PREDIMIN | Desempeño predictivo | R², RMSE, MAE, AUC, sensibilidad | – | Validación cruzada y conjunto de prueba |

### 4.3 Diseño metodológico (2.4.1)

- **Enfoque: cuantitativo**, ya que se analizan datos numéricos de voladura, meteorología y concentración de PM10 mediante técnicas estadísticas y algoritmos de aprendizaje automático, y los resultados se expresan en métricas objetivas (R², RMSE, MAE, AUC).
- **Tipo: aplicada**, porque busca resolver un problema concreto de gestión ambiental en una unidad minera mediante el desarrollo de una herramienta tecnológica.
- **Alcance: correlacional–predictivo (explicativo)**, dado que se determina la relación entre las variables operacionales y meteorológicas y la concentración de PM10, y se construye un modelo que estima dicha concentración para nuevos eventos de voladura.
- **Diseño: no experimental, longitudinal, retrospectivo**, pues las variables no se manipulan: se analizan registros históricos de voladuras realizadas entre enero de 2023 y mayo de 2024, tal como ocurrieron.

Fases: (1) recolección de los registros de la unidad minera; (2) depuración y control de calidad de los datos; (3) análisis exploratorio y correlacional (objetivo 1); (4) entrenamiento, optimización y validación de los modelos (objetivo 2); (5) interpretación con SHAP e importancia por permutación (objetivo 3); (6) diseño del módulo de recomendación y de la interfaz web (objetivo 4); (7) evaluación de escenarios (objetivo 5).

### 4.4 Diseño muestral (2.4.2) y técnicas (2.4.3)

**Población:** registros diarios de voladura de la U.M. La Arena con monitoreo de PM10 y meteorología, entre enero de 2023 y mayo de 2024 (505 registros).
**Muestra:** muestreo **no probabilístico por conveniencia (censal)**: se usan todos los registros que superan el control de calidad, es decir, **401 registros válidos**. Se excluyen los registros con PM10 = 0 (falla o mantenimiento del monitor), sin datos meteorológicos o con valores físicamente imposibles.
**División:** 80 % para entrenamiento (321 registros) y 20 % para prueba independiente (80 registros), estratificada por cuartiles de PM10 y agrupada por fecha.

**Técnicas de procesamiento de datos:**
1. Depuración: tratamiento de faltantes ("S/D", PM10 = 0), corrección de campos de texto (retardos) y valores fuera de rango; imputación por la mediana del conjunto de entrenamiento.
2. Ingeniería de variables: codificación circular (seno/coseno) de la dirección del viento y del mes, factor de carga, explosivo por taladro, fracción de emulsión e índice seco.
3. Análisis estadístico: correlación de Spearman con prueba de significancia al 5 %.
4. Modelamiento: Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost y red neuronal artificial (ensamble de perceptrones multicapa); optimización bayesiana de hiperparámetros (Optuna, 40 ensayos por algoritmo); validación cruzada de 5 particiones agrupada por fecha; validación temporal (entrenar con meses pasados y probar con meses futuros).
5. Clasificación del riesgo de superar 45 µg/m³ (AUC, sensibilidad, precisión, F1).
6. Interpretabilidad: valores SHAP e importancia por permutación.
7. Herramientas: Python 3.12, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP y Streamlit.

### 4.5 Texto para el marco teórico: red neuronal artificial

> **2.2.2.x Redes neuronales artificiales.** Una red neuronal artificial (RNA) es un modelo compuesto por unidades de cálculo llamadas neuronas, organizadas en capas: una capa de entrada que recibe las variables predictoras, una o más capas ocultas y una capa de salida (Goodfellow et al., 2016). Cada neurona calcula una suma ponderada de sus entradas y le aplica una función de activación no lineal:
>
> *a = g(w₁x₁ + w₂x₂ + … + wₙxₙ + b)*
>
> donde *wᵢ* son los pesos, *b* es el sesgo y *g* es la función de activación (por ejemplo, la tangente hiperbólica o ReLU). Al combinar varias neuronas y capas, la red puede aproximar relaciones no lineales complejas entre las variables de voladura, la meteorología y la concentración de PM10 (Bishop, 2006).
>
> El perceptrón multicapa (MLP) se entrena ajustando los pesos para minimizar el error cuadrático medio entre los valores observados y los predichos. En este trabajo se usa el algoritmo L-BFGS (Liu & Nocedal, 1989), un método cuasi-Newton adecuado para conjuntos de datos pequeños. Para evitar el sobreajuste (que la red memorice los datos de entrenamiento), se añade una penalización L2 sobre la magnitud de los pesos, controlada por el parámetro α.
>
> Como una sola red es sensible a sus pesos iniciales y a los datos con que se entrena, se emplea un **ensamble de redes**: se entrenan *N* redes con muestras bootstrap (remuestreo aleatorio con reemplazo; Efron & Tibshirani, 1993) y se promedian sus predicciones. La dispersión entre las redes permite construir un intervalo de incertidumbre para cada predicción (Lakshminarayanan et al., 2017), en la línea del enfoque probabilístico con simulación de Monte Carlo de Hosseini y Pourmirzaee (2024).

> **Métricas de clasificación.** Para evaluar la predicción del riesgo de superar la guía de la OMS se usan la sensibilidad (proporción de superaciones reales detectadas), la precisión (proporción de alertas que resultaron correctas) y el área bajo la curva ROC (AUC), que mide la capacidad del modelo para distinguir entre eventos que superan y no superan el umbral: AUC = 0,5 equivale al azar y AUC = 1 a una discriminación perfecta (Fawcett, 2006).

### 4.6 Aspectos éticos (2.4.4)

- Los datos fueron proporcionados por la U.M. La Arena y se usan solo con fines académicos, con la autorización de la empresa y respetando su confidencialidad.
- No se trabaja con personas ni con datos personales.
- Se respetan la autoría y las fuentes consultadas (normas APA 7.ª edición).
- Los resultados se reportan tal como se obtuvieron, incluidas sus limitaciones; no se manipulan datos para mejorar las métricas.
- Marco normativo: D.S. N.° 003-2017-MINAM (ECA para Aire) y guías de calidad del aire de la OMS (2021).

### 4.7 Referencias que conviene agregar

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
- Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning, 63*(1), 3–42. https://doi.org/10.1007/s10994-006-6226-1
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems, 30*.
- Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming, 45*, 503–528. https://doi.org/10.1007/BF01589116
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. **(ya se cita; falta en la lista)**
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems, 31*.
- Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern approach* (4th ed.). Pearson. **(ya se cita; falta en la lista)**

> Verifiquen cada referencia (DOI, páginas) antes de entregar.

---

## 5. Cómo defender los resultados ante el jurado

**Pregunta probable: "Los antecedentes obtienen R² de 0,95 o más y ustedes 0,2. ¿Por qué?"**

Respuesta sugerida (con datos):
1. **El PM10 registrado no es solo de la voladura.** La estación mide el polvo de **todas** las fuentes (vías de acarreo, chancado, viento). Zhang et al. (2026) midieron el polvo **del disparo**, e incluyeron la **distancia** entre el disparo y el monitor, que fue una de sus variables más importantes. En La Arena no se tiene la distancia ni el PM10 de fondo antes del disparo.
2. **Hosseini y Pourmirzaee (2024) trabajaron con datos generados por simulación de Monte Carlo**, que son mucho más "limpios" que los registros reales.
3. **Las variables de voladura casi no varían** en La Arena (factor de carga ≈ 0,23 kg/t en casi todos los disparos), así que ningún modelo puede aprender su efecto. Esto es un hallazgo, no un error: con este diseño de voladura estandarizado, **lo que determina el PM10 es la meteorología** (humedad relativa, estacionalidad).
4. **Por eso el sistema se apoya en la clasificación del riesgo** (¿superará 45 µg/m³?), que tiene un AUC ≈ 0,8: identifica bien los días de riesgo aunque no acierte el valor exacto. Para decidir si se riega o se reprograma un disparo, eso es lo que se necesita.
5. **Validación honesta**: el modelo se eligió por validación cruzada agrupada por fecha, se evaluó con datos que nunca vio y se hizo una validación temporal. Muchos trabajos con R² altos no hacen esto.
6. **Trabajo futuro concreto**: con el PM10 antes y después del disparo y la ubicación del disparo y de la estación (`data/datos_adicionales_a_solicitar.csv`), se puede modelar el aporte neto de la voladura.

**Otras preguntas probables:**
- *¿Por qué una red neuronal y no solo árboles?* → Porque se compararon ambas familias con el mismo protocolo, tal como lo hacen los antecedentes; la red aporta además un intervalo de incertidumbre.
- *¿Cómo evitaron el sobreajuste?* → Regularización L2 (figura de regularización), validación cruzada agrupada por fecha, conjunto de prueba independiente y ensamble de redes.
- *¿El sistema reduce el polvo?* → No directamente: **anticipa** el riesgo y **recomienda** medidas. Los porcentajes de reducción del riego y la nebulización deben citarse de la literatura (Cecala et al., 2019; Kissell, 2003); el modelo solo estima escenarios que cambian variables presentes en los datos.
- *¿Se puede usar en otra mina?* → Requiere reentrenarse con los datos de esa mina (ya está en las limitaciones).
