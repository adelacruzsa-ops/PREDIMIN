"""
PREDIMIN - Modulo 2: entrenamiento, optimizacion y seleccion del modelo predictivo.

Flujo (coherente con Zhang et al., 2026 - Figura 3 del plan de tesis):

  1. Division 80:20 (entrenamiento / prueba independiente).
  2. Para cada uno de los 6 algoritmos de ensamble (Random Forest, Extra Trees,
     Gradient Boosting, XGBoost, LightGBM y CatBoost):
       - optimizacion bayesiana de hiperparametros (Optuna, muestreador TPE)
         usando SOLO el conjunto de entrenamiento, con validacion cruzada de
         5 particiones y funcion objetivo MSE.
  3. Seleccion del modelo ganador por su desempeno en VALIDACION CRUZADA.
     El conjunto de prueba NO se usa para elegir: solo para reportar el
     desempeno final. (Elegir con el conjunto de prueba lo convierte en parte
     del entrenamiento y las metricas reportadas quedan infladas.)
  4. Metricas finales sobre el conjunto de prueba: R2, RMSE, MAE y MAPE.
  5. Interpretabilidad con valores SHAP sobre el modelo ganador.
  6. Validacion TEMPORAL (entrenar con el pasado, probar con el futuro): mide
     si el modelo sirve para meses que no vio. Es la prueba mas exigente.
  7. Modelo de CLASIFICACION del riesgo de superar la guia OMS (45 ug/m3):
     con datos ruidosos suele ser mas util "va a superar o no" que el valor exacto.

Todas las metricas se calculan en ug/m3. (Opcion USAR_LOG en config.py para
entrenar sobre log(1+PM10); con estos datos no mejora el resultado.)

Uso:
    python src/entrenamiento.py                # 40 ensayos por algoritmo
    python src/entrenamiento.py --trials 100   # busqueda mas exhaustiva
    python src/entrenamiento.py --trials 0     # sin optimizar (rapido, para pruebas)
"""

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_predict, cross_val_score

from config import (
    DIR_MODELOS,
    DIR_SALIDAS,
    K_FOLDS,
    N_TRIALS_OPTUNA,
    OBJETIVOS,
    SEMILLA,
    UMBRAL_RIESGO,
    USAR_LOG,
)
from preprocesamiento import dividir, matriz_de_variables, preparar


def con_log(modelo):
    """Si USAR_LOG, entrena sobre log(1+PM) y devuelve predicciones en ug/m3."""
    if not USAR_LOG:
        return modelo
    return TransformedTargetRegressor(regressor=modelo, func=np.log1p,
                                      inverse_func=np.expm1, check_inverse=False)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Espacios de busqueda de hiperparametros
# ---------------------------------------------------------------------------
# Cada entrada: (constructor(params) -> modelo, espacio(trial) -> params, params_por_defecto)

def _rf(p):
    return RandomForestRegressor(random_state=SEMILLA, n_jobs=1, **p)


def _et(p):
    return ExtraTreesRegressor(random_state=SEMILLA, n_jobs=1, **p)


def _gb(p):
    return GradientBoostingRegressor(random_state=SEMILLA, **p)


def _espacio_bosque(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 150, 700, step=50),
        "max_depth": t.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 8),
        "max_features": t.suggest_float("max_features", 0.3, 1.0),
    }


def _espacio_gb(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
    }


def catalogo_modelos() -> dict:
    cat = {
        "RandomForest": (_rf, _espacio_bosque,
                         {"n_estimators": 500, "min_samples_leaf": 2}),
        "ExtraTrees": (_et, _espacio_bosque,
                       {"n_estimators": 500, "min_samples_leaf": 2}),
        "GradientBoosting": (_gb, _espacio_gb,
                             {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3}),
    }

    try:
        from xgboost import XGBRegressor

        def _xgb(p):
            return XGBRegressor(random_state=SEMILLA, n_jobs=1, **p)

        def _esp_xgb(t):
            return {
                "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": t.suggest_int("max_depth", 2, 8),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 10),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            }

        cat["XGBoost"] = (_xgb, _esp_xgb,
                          {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 5})
    except ImportError:
        print("[aviso] xgboost no instalado; se omite ese candidato.")

    try:
        from lightgbm import LGBMRegressor

        def _lgb(p):
            return LGBMRegressor(random_state=SEMILLA, n_jobs=1, verbose=-1, **p)

        def _esp_lgb(t):
            return {
                "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": t.suggest_int("num_leaves", 4, 40),
                "min_child_samples": t.suggest_int("min_child_samples", 3, 30),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            }

        cat["LightGBM"] = (_lgb, _esp_lgb,
                           {"n_estimators": 400, "learning_rate": 0.05,
                            "num_leaves": 15, "min_child_samples": 5})
    except ImportError:
        print("[aviso] lightgbm no instalado; se omite ese candidato.")

    try:
        from catboost import CatBoostRegressor

        def _cb(p):
            return CatBoostRegressor(random_seed=SEMILLA, verbose=0,
                                     allow_writing_files=False, thread_count=1, **p)

        def _esp_cb(t):
            return {
                "iterations": t.suggest_int("iterations", 200, 1000, step=100),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": t.suggest_int("depth", 3, 8),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10, log=True),
            }

        cat["CatBoost"] = (_cb, _esp_cb,
                           {"iterations": 800, "learning_rate": 0.05, "depth": 6})
    except ImportError:
        print("[aviso] catboost no instalado; se omite ese candidato.")

    # Red neuronal artificial (perceptron multicapa) - definida en red_neuronal.py
    from red_neuronal import PARAMETROS_POR_DEFECTO, crear_red, espacio_busqueda
    cat["RedNeuronal_MLP"] = (crear_red, espacio_busqueda, dict(PARAMETROS_POR_DEFECTO))

    return cat


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def metricas(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = float(np.mean(np.abs((y_real - y_pred) / np.clip(y_real, 1e-6, None))) * 100)
    return {
        "R2": round(float(r2_score(y_real, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_real, y_pred))), 4),
        "MAE": round(float(mean_absolute_error(y_real, y_pred)), 4),
        "MAPE_%": round(mape, 2),
    }


def _particiones():
    return KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMILLA), None


def _cv(modelo, X, y, cv, grupos, scoring):
    return cross_val_score(modelo, X, y, cv=cv, groups=grupos,
                           scoring=scoring, n_jobs=-1)


# ---------------------------------------------------------------------------
# Optimizacion bayesiana
# ---------------------------------------------------------------------------
def optimizar(nombre, constructor, espacio, por_defecto, X, y, cv, grupos, n_trials):
    if n_trials <= 0:
        return por_defecto
    try:
        import optuna
    except ImportError:
        print("[aviso] optuna no instalado; se usan hiperparametros por defecto.")
        return por_defecto

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objetivo(trial):
        mse = -_cv(constructor(espacio(trial)), X, y, cv, grupos,
                   "neg_mean_squared_error").mean()
        return mse

    estudio = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA),
    )
    # El primer ensayo evalua los hiperparametros por defecto, de modo que la
    # optimizacion nunca termina peor que la configuracion de partida.
    claves = _valores_minimos(espacio)
    estudio.enqueue_trial({k: v for k, v in por_defecto.items() if k in claves})
    estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    # espacio(FixedTrial) reconstruye el diccionario completo (incluye constantes)
    return espacio(optuna.trial.FixedTrial(estudio.best_params))


def _valores_minimos(espacio):
    """Nombres de los hiperparametros que define un espacio de busqueda."""

    class _Sonda:
        def __init__(self):
            self.v = {}

        def suggest_int(self, n, lo, hi, step=1, log=False):
            self.v[n] = lo
            return lo

        def suggest_float(self, n, lo, hi, step=None, log=False):
            self.v[n] = lo
            return lo

    s = _Sonda()
    espacio(s)
    return s.v


# ---------------------------------------------------------------------------
# Entrenamiento para una variable objetivo
# ---------------------------------------------------------------------------
def entrenar_objetivo(df: pd.DataFrame, objetivo: str, n_trials: int):
    print(f"\n{'=' * 72}\n  VARIABLE OBJETIVO: {objetivo}\n{'=' * 72}")
    X_tr, X_te, y_tr, y_te, columnas, medianas = dividir(df, objetivo)
    print(f"  Entrenamiento: {len(X_tr)} registros | Prueba: {len(X_te)} registros")
    print(f"  Optimizacion bayesiana: {n_trials} ensayos por algoritmo\n")

    cv, grupos_cv = _particiones()
    resultados, ajustados, hiperparametros = [], {}, {}

    for nombre, (constructor_base, espacio, por_defecto) in catalogo_modelos().items():
        constructor = (lambda cb: (lambda p: con_log(cb(p))))(constructor_base)
        params = optimizar(nombre, constructor, espacio, por_defecto,
                           X_tr, y_tr, cv, grupos_cv, n_trials)
        modelo = constructor(params)
        cv_r2 = _cv(modelo, X_tr, y_tr, cv, grupos_cv, "r2")
        cv_rmse = np.sqrt(-_cv(modelo, X_tr, y_tr, cv, grupos_cv,
                               "neg_mean_squared_error"))
        modelo.fit(X_tr, y_tr)
        m_test = metricas(y_te, modelo.predict(X_te))

        ajustados[nombre] = modelo
        hiperparametros[nombre] = params
        resultados.append({
            "Modelo": nombre,
            "R2_CV_medio": round(float(cv_r2.mean()), 4),
            "R2_CV_desv": round(float(cv_r2.std()), 4),
            "RMSE_CV_medio": round(float(cv_rmse.mean()), 4),
            **{f"{k}_test": v for k, v in m_test.items()},
        })
        print(f"  {nombre:<17} R2cv={cv_r2.mean():.4f} ± {cv_r2.std():.3f}   "
              f"R2test={m_test['R2']:.4f}  RMSEtest={m_test['RMSE']:.2f}")

    # Seleccion por VALIDACION CRUZADA (no por el conjunto de prueba)
    tabla = pd.DataFrame(resultados).sort_values("R2_CV_medio", ascending=False)
    ganador = tabla.iloc[0]["Modelo"]
    fila = tabla.iloc[0]
    print(f"\n  >> Modelo seleccionado (mejor R2 en validacion cruzada): {ganador}")
    print(f"     Desempeno en prueba independiente: R2={fila['R2_test']}  "
          f"RMSE={fila['RMSE_test']}  MAE={fila['MAE_test']}  MAPE={fila['MAPE_%_test']} %")

    modelo_final = ajustados[ganador]

    # Validacion temporal del modelo ganador (pasado -> futuro)
    X_all, _ = matriz_de_variables(df)
    X_all = X_all.fillna(medianas)
    r2_t = cross_val_score(con_log(catalogo_modelos()[ganador][0](
        {k: v for k, v in hiperparametros[ganador].items()})),
        X_all, df[objetivo], cv=TimeSeriesSplit(n_splits=K_FOLDS), scoring="r2")
    print(f"     Validacion temporal (pasado -> futuro): R2 medio = {r2_t.mean():.3f} "
          f"(por bloque: {', '.join(f'{v:.2f}' for v in r2_t)})")
    tabla["R2_temporal_ganador"] = np.where(tabla["Modelo"] == ganador, round(float(r2_t.mean()), 4), np.nan)

    joblib.dump(
        {"modelo": modelo_final, "columnas": columnas, "objetivo": objetivo,
         "algoritmo": ganador, "hiperparametros": hiperparametros[ganador],
         "medianas": medianas},
        DIR_MODELOS / f"predimin_{objetivo}.joblib",
    )
    tabla.to_csv(DIR_SALIDAS / f"comparacion_modelos_{objetivo}.csv", index=False)
    with open(DIR_SALIDAS / f"hiperparametros_{objetivo}.json", "w", encoding="utf-8") as f:
        json.dump(hiperparametros, f, indent=2, default=float)

    grafico_observado_vs_predicho(y_te, modelo_final.predict(X_te), objetivo, ganador,
                                  metricas(y_te, modelo_final.predict(X_te)))

    return modelo_final, X_tr, X_te, columnas, tabla, ganador, float(r2_t.mean())


# ---------------------------------------------------------------------------
# Clasificacion: riesgo de superar el umbral
# ---------------------------------------------------------------------------
def entrenar_clasificador(df: pd.DataFrame, objetivo: str):
    from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)

    umbral = UMBRAL_RIESGO[objetivo]
    X_tr, X_te, y_tr, y_te, columnas, medianas = dividir(df, objetivo)
    c_tr, c_te = (y_tr > umbral).astype(int), (y_te > umbral).astype(int)
    print(f"\n  CLASIFICACION: riesgo de {objetivo} > {umbral:.0f} ug/m3 "
          f"({c_tr.mean() * 100:.0f} % de los registros superan)")

    cands = {
        "RandomForest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                               class_weight="balanced", random_state=SEMILLA),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5,
                                           class_weight="balanced", random_state=SEMILLA),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.03,
                                                       max_depth=3, subsample=0.8,
                                                       random_state=SEMILLA),
    }
    try:
        from lightgbm import LGBMClassifier
        cands["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8,
                                           min_child_samples=10, class_weight="balanced",
                                           random_state=SEMILLA, verbose=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        cands["CatBoost"] = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03,
                                               auto_class_weights="Balanced", verbose=0,
                                               random_seed=SEMILLA, allow_writing_files=False)
    except ImportError:
        pass

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMILLA)
    filas, ajustados = [], {}
    for nombre, m in cands.items():
        auc_cv = cross_val_score(m, X_tr, c_tr, cv=skf, scoring="roc_auc").mean()
        m.fit(X_tr, c_tr)
        p = m.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        filas.append({
            "Modelo": nombre, "AUC_CV": round(float(auc_cv), 4),
            "AUC_test": round(float(roc_auc_score(c_te, p)), 4),
            "Sensibilidad_test": round(float(recall_score(c_te, pred)), 4),
            "Precision_test": round(float(precision_score(c_te, pred, zero_division=0)), 4),
            "F1_test": round(float(f1_score(c_te, pred)), 4),
        })
        ajustados[nombre] = m
        print(f"  {nombre:<17} AUCcv={auc_cv:.3f}  AUCtest={filas[-1]['AUC_test']:.3f}  "
              f"Sensibilidad={filas[-1]['Sensibilidad_test']:.2f}")

    tabla = pd.DataFrame(filas).sort_values("AUC_CV", ascending=False)
    ganador = tabla.iloc[0]["Modelo"]
    print(f"  >> Clasificador seleccionado: {ganador}")
    tabla.to_csv(DIR_SALIDAS / f"comparacion_clasificadores_{objetivo}.csv", index=False)
    joblib.dump({"modelo": ajustados[ganador], "columnas": columnas, "umbral": umbral,
                 "algoritmo": ganador, "medianas": medianas},
                DIR_MODELOS / f"predimin_riesgo_{objetivo}.joblib")
    return tabla.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def grafico_observado_vs_predicho(y_real, y_pred, objetivo, nombre, m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    etiqueta = "PM10" if "pm10" in objetivo else "PM2.5"
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y_real, y_pred, s=28, alpha=0.8, edgecolor="white", linewidth=0.5)
    lim = [0, max(float(np.max(y_real)), float(np.max(y_pred))) * 1.05]
    ax.plot(lim, lim, "--", color="grey", linewidth=1, label="Ajuste perfecto (1:1)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"{etiqueta} observado (µg/m³)")
    ax.set_ylabel(f"{etiqueta} predicho (µg/m³)")
    ax.set_title(f"{nombre} — conjunto de prueba\n"
                 f"R² = {m['R2']:.3f} · RMSE = {m['RMSE']:.2f} · MAE = {m['MAE']:.2f}",
                 fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_SALIDAS / f"observado_vs_predicho_{objetivo}.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interpretabilidad SHAP
# ---------------------------------------------------------------------------
def analisis_shap(modelo, X_tr, X_te, columnas, objetivo: str):
    try:
        import shap
    except ImportError:
        print("[aviso] shap no instalado; se omite el analisis de importancia.")
        return None

    # Si se uso log, SHAP se calcula sobre el modelo interno (escala log).
    interno = getattr(modelo, "regressor_", modelo)
    try:
        valores = shap.TreeExplainer(interno).shap_values(X_te)
    except Exception:
        # modelo no basado en arboles (p. ej. la red neuronal): explicador general
        fondo = shap.sample(X_tr, 50, random_state=SEMILLA)
        valores = shap.KernelExplainer(interno.predict, fondo).shap_values(X_te, nsamples=200)
    importancia = (
        pd.DataFrame({
            "variable": columnas,
            "shap_medio_abs": np.abs(valores).mean(axis=0).round(4),
        })
        .sort_values("shap_medio_abs", ascending=False)
        .reset_index(drop=True)
    )
    importancia.to_csv(DIR_SALIDAS / f"shap_importancia_{objetivo}.csv", index=False)

    print(f"\n  Variables mas influyentes sobre {objetivo} (SHAP):")
    for _, f in importancia.head(8).iterrows():
        print(f"    {f['variable']:<26} {f['shap_medio_abs']:>9.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(valores, X_te, feature_names=columnas, show=False,
                          max_display=14)
        plt.tight_layout()
        plt.savefig(DIR_SALIDAS / f"shap_beeswarm_{objetivo}.png", dpi=200)
        plt.close()
    except Exception as e:  # pragma: no cover
        print(f"    [aviso] no se pudo generar la figura SHAP: {e}")

    return importancia


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Entrenamiento de PREDIMIN")
    ap.add_argument("--trials", type=int, default=N_TRIALS_OPTUNA,
                    help="ensayos de optimizacion bayesiana por algoritmo (0 = sin optimizar)")
    args = ap.parse_args()

    df = preparar()
    print(f"\nRegistros utilizables: {len(df)}")
    if len(df) < 100:  # noqa
        print("[aviso] Menos de 100 registros: las metricas seran inestables. "
              "Reporte la desviacion de la validacion cruzada junto al R2.")

    resumen = {}
    for objetivo in OBJETIVOS:
        modelo, X_tr, X_te, columnas, tabla, ganador, r2_temp = entrenar_objetivo(
            df, objetivo, args.trials)
        analisis_shap(modelo, X_tr, X_te, columnas, objetivo)
        clas = entrenar_clasificador(df, objetivo) if objetivo in UMBRAL_RIESGO else None
        resumen[objetivo] = {
            "modelo_seleccionado": ganador,
            "criterio_seleccion": "mayor R2 medio en validacion cruzada (entrenamiento)",
            "metricas": tabla.iloc[0].to_dict(),
            "r2_validacion_temporal": round(r2_temp, 4),
            "clasificador_riesgo": clas,
            "n_registros": int(len(df)),
            "n_trials_optuna": args.trials,
        }

    with open(DIR_SALIDAS / "resumen_entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[OK] Modelos guardados en {DIR_MODELOS}")
    print(f"[OK] Tablas y figuras guardadas en {DIR_SALIDAS}")


if __name__ == "__main__":
    main()
