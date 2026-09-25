"""
PREDIMIN - Red neuronal artificial: ensamble de perceptrones multicapa (MLP).

Aqui se DEFINE la red neuronal. Se usa de dos formas:
  1. entrenamiento.py la incluye como un candidato mas en la comparacion de
     modelos de regresion (junto a Random Forest, XGBoost, etc.) y de
     clasificacion del riesgo de superar 45 ug/m3.
  2. Este archivo se puede ejecutar solo para el analisis detallado de la red:
         python src/red_neuronal.py
     Genera en outputs/ las figuras y tablas del capitulo de la red neuronal
     (ver la funcion main) y guarda models/predimin_red_neuronal.joblib, que la
     aplicacion usa como "segunda opinion" con intervalo de incertidumbre.

Estructura de cada red (valores por defecto; la optimizacion bayesiana los ajusta):

    ENTRADA (20 variables)  ->  CAPA OCULTA 1  ->  CAPA OCULTA 2  ->  SALIDA
    humedad, viento,            16 neuronas        8 neuronas         log(1+PM10)
    explosivo, taladros...      tangente hiperb.   tangente hiperb.   (regresion)
                                                                      o P(PM10 > 45)
                                                                      (clasificacion)

Decisiones de diseño (justificarlas en la tesis):
  - ESTANDARIZACION: las variables entran con media 0 y desviacion 1, porque
    la red es sensible a la escala (kg de explosivo ~ 15 000 vs. viento ~ 1.5).
  - OBJETIVO EN ESCALA LOGARITMICA: el PM10 es muy asimetrico (asimetria ~ 2.5,
    pocos picos de hasta 235 ug/m3). La red aprende log(1+PM10) y la salida se
    devuelve en ug/m3. Sin esta transformacion los picos dominan el error y la
    red se vuelve inestable.
  - L-BFGS: con ~320 registros de entrenamiento, un optimizador de segundo
    orden (cuasi-Newton) converge de forma mas estable que Adam y no necesita
    separar datos para "early stopping".
  - REGULARIZACION L2 (alpha): penaliza pesos grandes para que la red no
    memorice los datos (sobreajuste), un riesgo alto con pocos registros.
    Ver outputs/red_neuronal_regularizacion.png.
  - ENSAMBLE CON REMUESTREO BOOTSTRAP (Monte Carlo): se entrenan N redes, cada
    una con una muestra aleatoria con reemplazo de los datos y pesos iniciales
    distintos. La prediccion es el promedio de las N redes y la dispersion
    entre ellas da un INTERVALO DE INCERTIDUMBRE. Es la misma idea de
    Hosseini y Pourmirzaee (2024), que combinan redes neuronales con
    simulacion de Monte Carlo para obtener predicciones probabilisticas.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEMILLA

warnings.filterwarnings("ignore")

N_REDES = 10   # redes en el ensamble

PARAMETROS_POR_DEFECTO = {
    "n_capas": 2,           # numero de capas ocultas
    "neuronas": 16,         # neuronas en la primera capa (cada capa siguiente tiene la mitad)
    "activacion": "tanh",   # funcion de activacion de las capas ocultas
    "alpha": 10.0,          # regularizacion L2
}


def arquitectura(n_capas: int, neuronas: int) -> tuple:
    """(16, 8) para 2 capas y 16 neuronas; (32, 16, 8) para 3 capas y 32."""
    return tuple(max(4, neuronas // (2 ** i)) for i in range(n_capas))


def _red_base(p: dict, tarea: str) -> Pipeline:
    """Una sola red: estandarizacion + perceptron multicapa."""
    Red = MLPRegressor if tarea == "regresion" else MLPClassifier
    return Pipeline([
        ("estandarizar", StandardScaler()),
        ("red", Red(
            hidden_layer_sizes=arquitectura(p["n_capas"], p["neuronas"]),
            activation=p["activacion"],
            solver="lbfgs",
            alpha=p["alpha"],
            max_iter=2000,
            random_state=SEMILLA,
        )),
    ])


class _EnsambleBase(BaseEstimator):
    """Entrena N redes, cada una sobre una muestra bootstrap y con otra semilla."""

    _tarea = None

    def __init__(self, n_capas=2, neuronas=16, activacion="tanh", alpha=10.0,
                 n_redes=N_REDES, semilla=SEMILLA):
        self.n_capas = n_capas
        self.neuronas = neuronas
        self.activacion = activacion
        self.alpha = alpha
        self.n_redes = n_redes
        self.semilla = semilla

    def _parametros(self):
        return {"n_capas": self.n_capas, "neuronas": self.neuronas,
                "activacion": self.activacion, "alpha": self.alpha}

    def _objetivo(self, y):
        return y

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._objetivo(np.asarray(y))
        rng = np.random.RandomState(self.semilla)
        base = _red_base(self._parametros(), self._tarea)
        self.redes_ = []
        for _ in range(self.n_redes):
            idx = rng.randint(0, len(X), len(X))           # muestra bootstrap
            if self._tarea == "clasificacion":
                while len(np.unique(y[idx])) < 2:          # ambas clases presentes
                    idx = rng.randint(0, len(X), len(X))
            red = clone(base).set_params(red__random_state=int(rng.randint(1_000_000)))
            self.redes_.append(red.fit(X[idx], y[idx]))
        self.n_features_in_ = X.shape[1]
        return self

    def n_parametros(self) -> int:
        """Pesos + sesgos de UNA red (todas tienen la misma arquitectura)."""
        mlp = self.redes_[0].named_steps["red"]
        return int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))


class RedNeuronalRegresion(RegressorMixin, _EnsambleBase):
    """Ensamble de redes que predice el PM10 (ug/m3) con intervalo de incertidumbre."""

    _tarea = "regresion"

    def _objetivo(self, y):
        return np.log1p(y.astype(float))

    def predicciones_individuales(self, X) -> np.ndarray:
        """Matriz (n_redes, n_registros) con la prediccion de cada red en ug/m3."""
        X = np.asarray(X, dtype=float)
        return np.clip(np.expm1([r.predict(X) for r in self.redes_]), 0, None)

    def predict(self, X):
        return self.predicciones_individuales(X).mean(axis=0)

    def predict_intervalo(self, X, nivel: float = 0.90):
        """
        Promedio y limites del intervalo (percentiles de las N redes).
        OJO: mide la incertidumbre DEL MODELO (cuanto discrepan las redes), no el
        ruido propio del monitor; por eso cubre menos del 90 % de los valores
        observados (ver "cobertura_intervalo_90_%" en red_neuronal_resumen.json).
        """
        P = self.predicciones_individuales(X)
        a = (1 - nivel) / 2 * 100
        return P.mean(axis=0), np.percentile(P, a, axis=0), np.percentile(P, 100 - a, axis=0)


class RedNeuronalClasificacion(ClassifierMixin, _EnsambleBase):
    """Ensamble de redes que estima la probabilidad de superar el umbral."""

    _tarea = "clasificacion"

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return super().fit(X, y)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([r.predict_proba(X) for r in self.redes_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def crear_red(p: dict | None = None) -> RedNeuronalRegresion:
    """Construye el ensamble de redes para regresion (usado por entrenamiento.py)."""
    return RedNeuronalRegresion(**{**PARAMETROS_POR_DEFECTO, **(p or {})})


def crear_red_clasificacion(p: dict | None = None) -> RedNeuronalClasificacion:
    """Construye el ensamble de redes para clasificar el riesgo."""
    return RedNeuronalClasificacion(**{**PARAMETROS_POR_DEFECTO, **(p or {})})


def espacio_busqueda(trial) -> dict:
    """Hiperparametros que explora la optimizacion bayesiana (Optuna)."""
    return {
        "n_capas": trial.suggest_int("n_capas", 1, 2),
        "neuronas": trial.suggest_int("neuronas", 4, 32, step=4),
        "activacion": trial.suggest_categorical("activacion", ["tanh", "relu"]),
        "alpha": trial.suggest_float("alpha", 1.0, 100.0, log=True),
    }


# ---------------------------------------------------------------------------
# Analisis detallado (python src/red_neuronal.py)
# ---------------------------------------------------------------------------
def _hiperparametros_optimizados() -> dict:
    """Usa los hiperparametros que encontro entrenamiento.py, si existen."""
    import json

    from config import DIR_SALIDAS
    ruta = DIR_SALIDAS / "hiperparametros_pm10_ugm3.json"
    if ruta.exists():
        hp = json.loads(ruta.read_text(encoding="utf-8")).get("RedNeuronal_MLP")
        if hp:
            return {k: v for k, v in hp.items() if k in PARAMETROS_POR_DEFECTO}
    return dict(PARAMETROS_POR_DEFECTO)


def figura_arquitectura(columnas, capas, ruta):
    """Diagrama de la red: entradas, capas ocultas y salida."""
    import matplotlib.pyplot as plt

    niveles = [len(columnas), *capas, 1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axis("off")
    max_dibujo = 12
    posiciones = []
    for i, n in enumerate(niveles):
        m = min(n, max_dibujo)
        ys = np.linspace(0.08, 0.92, m) if m > 1 else np.array([0.5])
        posiciones.append((i, ys, n > max_dibujo))
    for (i, ys, _), (j, ys2, _) in zip(posiciones, posiciones[1:]):
        for y1 in ys:
            for y2 in ys2:
                ax.plot([i, j], [y1, y2], color="#9AA5B1", linewidth=0.3, alpha=0.6, zorder=1)
    colores = ["#2F6FAE"] + ["#D9741E"] * len(capas) + ["#1B7F4C"]
    for (i, ys, recortado), color in zip(posiciones, colores):
        ax.scatter([i] * len(ys), ys, s=170, color=color, edgecolor="white", zorder=2)
        if recortado:
            ax.text(i, 0.5, "⋮", ha="center", va="center", fontsize=18, zorder=3,
                    bbox=dict(facecolor="white", edgecolor="none"))
    for k, c in enumerate(columnas[:max_dibujo]):
        ax.text(-0.12, posiciones[0][1][k], c, ha="right", va="center", fontsize=7)
    titulos = ([f"Entrada\n{len(columnas)} variables"]
               + [f"Capa oculta {k + 1}\n{n} neuronas" for k, n in enumerate(capas)]
               + ["Salida\nPM10 / P(PM10 > 45)"])
    for i, t in enumerate(titulos):
        ax.text(i, 1.0, t, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlim(-1.3, len(niveles) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_title("Arquitectura de cada red del ensamble PREDIMIN", fontsize=11, pad=18)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def main():
    import json

    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, roc_curve)
    from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                         cross_val_score, learning_curve,
                                         validation_curve)

    from config import DIR_MODELOS, DIR_SALIDAS, K_FOLDS, UMBRAL_RIESGO
    from entrenamiento import metricas
    from preprocesamiento import dividir, grupos_de, preparar

    objetivo = "pm10_ugm3"
    umbral = UMBRAL_RIESGO[objetivo]
    df = preparar()
    X_tr, X_te, y_tr, y_te, columnas, medianas = dividir(df, objetivo)
    g_tr = grupos_de(df, X_tr.index)
    c_tr, c_te = (y_tr > umbral).astype(int), (y_te > umbral).astype(int)
    cv_reg = GroupKFold(n_splits=K_FOLDS)
    cv_clf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMILLA)

    p = _hiperparametros_optimizados()
    capas = arquitectura(p["n_capas"], p["neuronas"])
    print(f"\nRed neuronal: {len(columnas)} entradas -> capas ocultas {capas} "
          f"({p['activacion']}, alpha={p['alpha']:.3g}) -> 1 salida")
    print(f"Ensamble de {N_REDES} redes con remuestreo bootstrap (Monte Carlo)")

    # --- 1. Regresion ------------------------------------------------------
    reg = crear_red(p)
    r2_cv = cross_val_score(reg, X_tr, y_tr, cv=cv_reg, groups=g_tr, scoring="r2", n_jobs=-1)
    reg.fit(X_tr, y_tr)
    media, lo, hi = reg.predict_intervalo(X_te, 0.90)
    m_reg = metricas(y_te, media)
    cobertura = float(np.mean((y_te.values >= lo) & (y_te.values <= hi)))
    print(f"\n  REGRESION  R2 validacion cruzada: {r2_cv.mean():.3f} ± {r2_cv.std():.3f}")
    print(f"             Prueba: R2={m_reg['R2']}  RMSE={m_reg['RMSE']}  MAE={m_reg['MAE']}")
    print(f"             Intervalo 90 %: cobertura real {cobertura * 100:.0f} %, "
          f"ancho medio {np.mean(hi - lo):.1f} ug/m3")

    # --- 2. Clasificacion del riesgo --------------------------------------
    clf = crear_red_clasificacion(p)
    auc_cv = cross_val_score(clf, X_tr, c_tr, cv=cv_clf, groups=g_tr, scoring="roc_auc", n_jobs=-1)
    clf.fit(X_tr, c_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    m_clf = {
        "AUC_CV": round(float(auc_cv.mean()), 4), "AUC_CV_desv": round(float(auc_cv.std()), 4),
        "AUC_test": round(float(roc_auc_score(c_te, prob)), 4),
        "Sensibilidad_test": round(float(recall_score(c_te, pred)), 4),
        "Precision_test": round(float(precision_score(c_te, pred, zero_division=0)), 4),
        "F1_test": round(float(f1_score(c_te, pred)), 4),
    }
    print(f"\n  CLASIFICACION (PM10 > {umbral:.0f})  AUC validacion cruzada: "
          f"{auc_cv.mean():.3f} ± {auc_cv.std():.3f}   AUC prueba: {m_clf['AUC_test']:.3f}")
    print(f"             Sensibilidad={m_clf['Sensibilidad_test']}  "
          f"Precision={m_clf['Precision_test']}  F1={m_clf['F1_test']}")

    # --- 3. Figuras --------------------------------------------------------
    figura_arquitectura(columnas, capas, DIR_SALIDAS / "red_neuronal_arquitectura.png")

    # 3a. Regularizacion: R2 de entrenamiento y de validacion segun alpha
    alphas = np.logspace(-1, 2.5, 8)
    tr_s, va_s = validation_curve(crear_red({**p, "alpha": 1.0}), X_tr, y_tr,
                                  param_name="alpha", param_range=alphas, cv=cv_reg,
                                  groups=g_tr, scoring="r2", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(alphas, tr_s.mean(1), "o-", label="Entrenamiento")
    ax.semilogx(alphas, va_s.mean(1), "o-", label="Validacion cruzada")
    ax.fill_between(alphas, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=0.15)
    ax.axvline(p["alpha"], color="grey", linestyle="--", linewidth=1, label="alpha elegido")
    ax.set_ylim(max(-0.5, float(va_s.mean(1).min()) - 0.1), 1)
    ax.set_xlabel("Regularizacion L2 (alpha)")
    ax.set_ylabel("R²")
    ax.set_title("Sobreajuste vs. regularizacion de la red neuronal", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "red_neuronal_regularizacion.png", dpi=200)
    plt.close(fig)

    # 3b. Curva de aprendizaje: desempeño segun la cantidad de registros
    tam = np.linspace(0.3, 1.0, 5)
    n_r, tr_r, va_r = learning_curve(crear_red(p), X_tr, y_tr, train_sizes=tam, cv=cv_reg,
                                     groups=g_tr, scoring="r2", n_jobs=-1)
    n_c, tr_c, va_c = learning_curve(crear_red_clasificacion(p), X_tr, c_tr, train_sizes=tam,
                                     cv=cv_clf, groups=g_tr, scoring="roc_auc", n_jobs=-1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, n, tr_, va_, et in ((axes[0], n_r, tr_r, va_r, "R² (regresion)"),
                                (axes[1], n_c, tr_c, va_c, "AUC (riesgo > 45 µg/m³)")):
        ax.plot(n, tr_.mean(1), "o-", label="Entrenamiento")
        ax.plot(n, va_.mean(1), "o-", label="Validacion cruzada")
        ax.fill_between(n, va_.mean(1) - va_.std(1), va_.mean(1) + va_.std(1), alpha=0.15)
        ax.set_xlabel("Registros de entrenamiento")
        ax.set_ylabel(et)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Curva de aprendizaje de la red neuronal: ¿mejoraria con mas datos?",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "red_neuronal_curva_aprendizaje.png", dpi=200)
    plt.close(fig)

    # 3c. Observado vs. predicho con intervalo del 90 %
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.errorbar(y_te, media, yerr=[media - lo, hi - media], fmt="o", ms=4, alpha=0.7,
                elinewidth=0.8, capsize=0, label="Prediccion ± incertidumbre del ensamble (90 %)")
    lim = [0, float(max(y_te.max(), hi.max())) * 1.05]
    ax.plot(lim, lim, "--", color="grey", linewidth=1, label="Ajuste perfecto (1:1)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("PM10 observado (µg/m³)")
    ax.set_ylabel("PM10 predicho (µg/m³)")
    ax.set_title(f"Red neuronal (ensamble) — prueba\nR² = {m_reg['R2']:.3f} · "
                 f"RMSE = {m_reg['RMSE']:.1f} · cobertura del intervalo = {cobertura * 100:.0f} %",
                 fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "red_neuronal_observado_vs_predicho.png", dpi=200)
    plt.close(fig)

    # 3d. Curva ROC del clasificador
    fpr, tpr, _ = roc_curve(c_te, prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"Red neuronal (AUC = {m_clf['AUC_test']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Azar (AUC = 0,5)")
    ax.set_xlabel("Tasa de falsas alarmas (1 − especificidad)")
    ax.set_ylabel("Sensibilidad (superaciones detectadas)")
    ax.set_title(f"Curva ROC — riesgo de PM10 > {umbral:.0f} µg/m³ (prueba)", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "red_neuronal_roc.png", dpi=200)
    plt.close(fig)

    # 3e. Importancia por permutacion (independiente del tipo de modelo)
    imp_r = permutation_importance(reg, X_te, y_te, scoring="r2", n_repeats=30,
                                   random_state=SEMILLA, n_jobs=-1)
    imp_c = permutation_importance(clf, X_te, c_te, scoring="roc_auc", n_repeats=30,
                                   random_state=SEMILLA, n_jobs=-1)
    imp = pd.DataFrame({
        "variable": columnas,
        "perdida_R2_regresion": imp_r.importances_mean.round(4),
        "perdida_AUC_clasificacion": imp_c.importances_mean.round(4),
    }).sort_values("perdida_AUC_clasificacion", ascending=False).reset_index(drop=True)
    imp.to_csv(DIR_SALIDAS / "red_neuronal_importancia_permutacion.csv", index=False)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["variable"], top["perdida_AUC_clasificacion"], color="#2F6FAE")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Disminucion del AUC al desordenar la variable")
    ax.set_title("Importancia de variables para la red neuronal (permutacion)", fontsize=10)
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / "red_neuronal_importancia.png", dpi=200)
    plt.close(fig)

    # --- 4. Guardar ---------------------------------------------------------
    resumen = {
        "arquitectura": {"entradas": len(columnas), "capas_ocultas": list(capas),
                         "activacion": p["activacion"], "salida": 1,
                         "optimizador": "L-BFGS", "regularizacion_L2_alpha": p["alpha"],
                         "redes_en_ensamble": N_REDES,
                         "parametros_por_red": reg.n_parametros(),
                         "objetivo_regresion": "log(1 + PM10)"},
        "regresion": {"R2_CV_medio": round(float(r2_cv.mean()), 4),
                      "R2_CV_desv": round(float(r2_cv.std()), 4),
                      **{f"{k}_test": v for k, v in m_reg.items()},
                      "cobertura_intervalo_90_%": round(cobertura * 100, 1),
                      "ancho_medio_intervalo_ugm3": round(float(np.mean(hi - lo)), 2)},
        "clasificacion": m_clf,
        "n_entrenamiento": int(len(X_tr)), "n_prueba": int(len(X_te)),
    }
    with open(DIR_SALIDAS / "red_neuronal_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    joblib.dump({"regresion": reg, "clasificacion": clf, "columnas": columnas,
                 "medianas": medianas, "umbral": umbral, "hiperparametros": p},
                DIR_MODELOS / "predimin_red_neuronal.joblib")
    print(f"\n  Parametros (pesos + sesgos) por red: {reg.n_parametros()}")
    print(f"[OK] Figuras y tablas en {DIR_SALIDAS}")
    print(f"[OK] Modelo -> {DIR_MODELOS / 'predimin_red_neuronal.joblib'}")


if __name__ == "__main__":
    # Se importa el propio modulo para que los modelos guardados queden
    # asociados a "red_neuronal" (y no a "__main__") y la app pueda cargarlos.
    import red_neuronal
    red_neuronal.main()
