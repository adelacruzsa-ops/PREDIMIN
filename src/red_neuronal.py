"""
PREDIMIN - Red neuronal artificial (Perceptron multicapa, MLP).

Aqui se DEFINE la red neuronal. Se usa de dos formas:
  1. entrenamiento.py la incluye como un candidato mas en la comparacion de
     modelos (junto a Random Forest, XGBoost, etc.).
  2. Este archivo se puede ejecutar solo para ver la red en detalle:
         python src/red_neuronal.py
     Genera outputs/red_neuronal_curva_aprendizaje.png y
            outputs/red_neuronal_observado_vs_predicho.png

Estructura de la red:

    ENTRADA (22 variables)  ->  CAPA OCULTA 1  ->  CAPA OCULTA 2  ->  SALIDA (PM10)
    humedad, viento,           32 neuronas        16 neuronas        1 neurona
    explosivo, taladros...     activacion ReLU    activacion ReLU

  - Cada neurona calcula: salida = ReLU(w1*x1 + w2*x2 + ... + b)
  - Los pesos (w) y sesgos (b) se ajustan con el algoritmo Adam,
    minimizando el error cuadratico medio (MSE) entre PM10 real y predicho.
  - Las variables se ESTANDARIZAN (media 0, desviacion 1) antes de entrar,
    porque la red es sensible a la escala (kg de explosivo ~ 15 000 vs.
    velocidad del viento ~ 1.5).
  - "alpha" es la regularizacion L2: penaliza pesos grandes para evitar que
    la red memorice los datos (sobreajuste), un riesgo alto con 401 registros.
  - "early_stopping" separa el 10 % del entrenamiento y detiene el
    aprendizaje cuando el error en ese 10 % deja de bajar.
"""

import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEMILLA

warnings.filterwarnings("ignore")

PARAMETROS_POR_DEFECTO = {
    "n_capas": 2,               # numero de capas ocultas
    "neuronas": 32,             # neuronas en la primera capa (cada capa siguiente tiene la mitad)
    "alpha": 1.0,               # regularizacion L2
    "learning_rate_init": 0.001,
}


def arquitectura(n_capas: int, neuronas: int) -> tuple:
    """(32, 16) para 2 capas y 32 neuronas; (64, 32, 16) para 3 capas y 64."""
    return tuple(max(4, neuronas // (2 ** i)) for i in range(n_capas))


def crear_red(p: dict | None = None) -> Pipeline:
    """Construye la red: estandarizacion + perceptron multicapa."""
    p = {**PARAMETROS_POR_DEFECTO, **(p or {})}
    return Pipeline([
        ("estandarizar", StandardScaler()),
        ("red", MLPRegressor(
            hidden_layer_sizes=arquitectura(p["n_capas"], p["neuronas"]),
            activation="relu",
            solver="adam",
            alpha=p["alpha"],
            learning_rate_init=p["learning_rate_init"],
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=SEMILLA,
        )),
    ])


def espacio_busqueda(trial) -> dict:
    """Hiperparametros que explora la optimizacion bayesiana (Optuna)."""
    return {
        "n_capas": trial.suggest_int("n_capas", 1, 3),
        "neuronas": trial.suggest_int("neuronas", 8, 64, step=8),
        "alpha": trial.suggest_float("alpha", 1e-3, 30, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold, cross_val_score

    from config import DIR_SALIDAS
    from entrenamiento import metricas
    from preprocesamiento import dividir, preparar

    df = preparar()
    X_tr, X_te, y_tr, y_te, columnas, _ = dividir(df, "pm10_ugm3")

    red = crear_red()
    capas = arquitectura(PARAMETROS_POR_DEFECTO["n_capas"], PARAMETROS_POR_DEFECTO["neuronas"])
    print(f"\nRed neuronal: {len(columnas)} entradas -> capas ocultas {capas} -> 1 salida")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    cv = cross_val_score(red, X_tr, y_tr, cv=kf, scoring="r2")
    red.fit(X_tr, y_tr)
    m = metricas(y_te, red.predict(X_te))
    mlp = red.named_steps["red"]
    n_pesos = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)

    print(f"  Parametros (pesos + sesgos) aprendidos: {n_pesos}")
    print(f"  Iteraciones (epocas) de entrenamiento:  {mlp.n_iter_}")
    print(f"  R2 validacion cruzada: {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"  Prueba: R2={m['R2']}  RMSE={m['RMSE']}  MAE={m['MAE']}")

    # Curva de aprendizaje: como baja el error mientras la red aprende
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mlp.loss_curve_, label="Error en entrenamiento")
    ax2 = ax.twinx()
    ax2.plot(mlp.validation_scores_, color="#D9741E", label="R² en validacion interna")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Error de entrenamiento (MSE/2, (µg/m³)²)")
    ax2.set_ylabel("R² validacion")
    ax.set_title(f"Curva de aprendizaje de la red neuronal {capas}", fontsize=10)
    fig.legend(loc="center right", bbox_to_anchor=(0.85, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "red_neuronal_curva_aprendizaje.png", dpi=200)
    plt.close(fig)

    pred = red.predict(X_te)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, pred, s=25, alpha=0.8)
    lim = [0, float(max(y_te.max(), pred.max())) * 1.05]
    ax.plot(lim, lim, "--", color="grey")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("PM10 observado (µg/m³)")
    ax.set_ylabel("PM10 predicho (µg/m³)")
    ax.set_title(f"Red neuronal — prueba: R² = {m['R2']:.3f}")
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "red_neuronal_observado_vs_predicho.png", dpi=200)
    plt.close(fig)
    print(f"\n[OK] Figuras en {DIR_SALIDAS}")
