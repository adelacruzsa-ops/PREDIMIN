"""
PREDIMIN - Analisis exploratorio (objetivo especifico 1).

"Analizar la influencia de las variables meteorologicas y parametros
operacionales de voladura sobre la generacion de PM10."

Genera en outputs/:
  - correlaciones_pm10.csv       correlacion de Spearman de cada variable con PM10
  - pm10_vs_variables.png        diagramas de dispersion de las variables clave
  - pm10_por_mes.png             estacionalidad del PM10
  - datos_faltantes_por_mes.csv  PM10 = 0 (falla del monitor) por mes

Uso:
    python src/analisis_exploratorio.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from config import ARCHIVO_DATOS, DIR_SALIDAS, GUIA_OMS
from preprocesamiento import columnas_modelo, ingenieria_de_variables

ETIQUETAS = {
    "humedad_relativa_pct": "Humedad relativa (%)",
    "velocidad_viento_ms": "Velocidad del viento (m/s)",
    "explosivo_total_kg": "Explosivo total (kg)",
    "numero_taladros": "Numero de taladros",
    "tonelaje_tm": "Tonelaje fracturado (t)",
    "factor_carga_kg_t": "Factor de carga (kg/t)",
}


def main():
    raw = pd.read_csv(ARCHIVO_DATOS, parse_dates=["fecha"])
    faltantes = (raw.assign(mes=raw["fecha"].dt.to_period("M"))
                 .groupby("mes")["pm10_ugm3"].apply(lambda s: s.isna().sum()))
    faltantes.to_csv(DIR_SALIDAS / "datos_faltantes_por_mes.csv", header=["pm10_faltantes"])

    df = ingenieria_de_variables(raw.dropna(subset=["pm10_ugm3"]))
    filas = []
    for c in columnas_modelo():
        d = df[[c, "pm10_ugm3"]].dropna()
        rho, pval = spearmanr(d[c], d["pm10_ugm3"])
        filas.append({"variable": c, "rho_spearman": round(rho, 3),
                      "p_valor": round(pval, 4), "significativa_5%": pval < 0.05,
                      "n": len(d)})
    corr = pd.DataFrame(filas).sort_values("rho_spearman", key=abs, ascending=False)
    corr.to_csv(DIR_SALIDAS / "correlaciones_pm10.csv", index=False)
    print("Correlacion de Spearman con PM10:")
    print(corr.to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, (c, et) in zip(axes.flat, ETIQUETAS.items()):
        d = df[[c, "pm10_ugm3"]].dropna()
        rho = corr.set_index("variable").loc[c, "rho_spearman"]
        ax.scatter(d[c], d["pm10_ugm3"], s=12, alpha=0.5, edgecolor="none")
        ax.axhline(GUIA_OMS["pm10_ugm3"], color="#B3261E", linestyle="--", linewidth=1)
        ax.set_xlabel(et)
        ax.set_title(f"ρ Spearman = {rho:.2f}", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("PM10 (µg/m³)")
    fig.suptitle("PM10 frente a variables meteorologicas y de voladura "
                 "(linea roja: guia OMS 45 µg/m³)")
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "pm10_vs_variables.png", dpi=200)
    plt.close(fig)

    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    fig, ax = plt.subplots(figsize=(8, 4))
    datos = [df.loc[df["mes"] == m, "pm10_ugm3"].values for m in range(1, 13)]
    ax.boxplot(datos, tick_labels=meses, showfliers=True)
    ax.axhline(GUIA_OMS["pm10_ugm3"], color="#B3261E", linestyle="--", linewidth=1)
    ax.set_ylabel("PM10 (µg/m³)")
    ax.set_title("PM10 registrado por mes (2023-2024)")
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "pm10_por_mes.png", dpi=200)
    plt.close(fig)
    print(f"\n[OK] Tablas y figuras en {DIR_SALIDAS}")


if __name__ == "__main__":
    main()
