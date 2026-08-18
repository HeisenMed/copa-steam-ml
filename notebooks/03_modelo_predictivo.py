# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 03: Modelo Predictivo del Puntaje  (Fase 2 — Modelos Predictivos)
================================================================================

Objetivo
--------
Predecir el `puntaje_obtenido` (0–100) de un estudiante a partir de sus
características demográficas, socioeconómicas y de experiencia previa, y a
partir del modelo construir un **Índice de Potencial STEM** exportable a
Supabase para mostrarlo en la web.

Secciones
---------
    A) Preparación de datos  (features, imputación, split 80/20 estratificado)
    B) Entrenamiento de 4 modelos (Lineal, Random Forest, XGBoost, LightGBM)
       con validación cruzada 5-fold, evaluación en test e importancias.
    C) Comparación de modelos y selección del mejor.
    D) Exportación para producción:
         - models/mejor_modelo_puntaje.joblib
         - models/modelo_coeficientes.json  (coeficientes/importancias + fórmula/reglas)
         - models/predictor.py  (función Python PURA, sin sklearn ni libs ML)
    E) Análisis de residuos (real vs predicho, distribución del error, sesgos por grupo).
    F) Índice de Potencial STEM por estudiante → models/scores_potencial_stem.csv
    G) Informe reports/03_modelo_predictivo.md

Principios de diseño
--------------------
- Autocontenido y reproducible: `random_state=42`.
- Paleta de marca Copa STEM aplicada explícitamente en TODOS los gráficos.
- El predictor puro replica el modelo ganador (lineal o de árboles) usando solo
  la librería estándar (json + aritmética); se valida numéricamente contra el
  modelo real antes de exportarlo.

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
import json
import inspect
import warnings
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from textwrap import dedent

RANDOM_STATE = 42

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    import joblib

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, KFold, cross_validate
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
except ImportError as exc:  # pragma: no cover
    print("ERROR: falta una dependencia del entorno.")
    print(f"       Detalle: {exc}")
    print("       Instale: pandas numpy scikit-learn matplotlib seaborn joblib "
          "xgboost lightgbm")
    sys.exit(1)

# XGBoost y LightGBM son opcionales: si faltan, el script continúa con los demás.
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0. CONFIGURACIÓN GLOBAL (idéntica a los scripts 01/02 para consistencia visual)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
for _d in (OUTPUTS_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Preferimos el dataset LIMPIO (sin exámenes anulados; ver 05b). Si no existe,
# se usa el original. Así 03 se entrena sobre datos limpios por defecto.
DATASET_CANDIDATOS = ["copa_stem_dataset_limpio.csv", "copa_stem_dataset.csv"]
DATASET_NAME = DATASET_CANDIDATOS[0]

COLORS = {
    "cyan":   "#00d4ff",
    "violet": "#8b5cf6",
    "amber":  "#f59e0b",
    "dark":   "#050816",
    "green":  "#10b981",
    "red":    "#ef4444",
    "blue":   "#0f77ee",
}
PALETTE = ["#00d4ff", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444", "#0f77ee"]

sns.set_theme(style="whitegrid")
sns.set_palette(PALETTE)
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":    "#333333",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "font.size":         10,
    "figure.autolayout": True,
    "axes.prop_cycle":   plt.cycler(color=PALETTE),
})
DPI = 150

STEM_GRAD = LinearSegmentedColormap.from_list("stem_grad", PALETTE)


def gradient_colors(n: int) -> list:
    """n colores muestreados del gradiente de marca Copa STEM."""
    if n <= 1:
        return [COLORS["cyan"]]
    return [STEM_GRAD(i / (n - 1)) for i in range(n)]


# Colores fijos por modelo (para que cada modelo tenga su color en todos los gráficos).
MODEL_COLORS = {
    "Regresión Lineal": COLORS["cyan"],
    "Random Forest":    COLORS["violet"],
    "XGBoost":          COLORS["amber"],
    "LightGBM":         COLORS["green"],
}

FIGURES: dict[str, str] = {}
REPORT: list[str] = []

# Umbrales de categoría del Índice de Potencial STEM (sobre el percentil 0–100).
CAT_ALTO, CAT_MEDIO, CAT_DESARROLLO = 75, 50, 25


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def savefig(fig, filename: str, key: str | None = None) -> str:
    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES[key or filename] = filename
    log(f"    figura guardada → outputs/{filename}")
    return filename


def img(key: str, alt: str) -> str:
    f = FIGURES.get(key)
    return f"![{alt}](../outputs/{f})" if f else f"_(figura '{alt}' no disponible)_"


def tabla_md(df: pd.DataFrame) -> str:
    enc = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    filas = ["| " + " | ".join(str(v) for v in row) + " |"
             for row in df.itertuples(index=False)]
    return "\n".join([enc, sep] + filas)


# =============================================================================
# CONFIGURACIÓN DE FEATURES
# =============================================================================
# Numéricos "directos" (se imputan con la MEDIANA del train).
NUMERIC = ["grado_escolar", "estrato", "interes_prog_robotica",
           "n_herramientas", "n_areas_interes"]
# Ordinales 0–3 derivados de nivel_programacion / nivel_robotica.
ORDINAL = ["nivel_programacion_ord", "nivel_robotica_ord"]
# Binarias Sí/No (se imputan con la MODA del train).
BINARY = ["computador_bin", "internet_bin", "olimpiadas_bin"]
BINARY_SRC = {
    "computador_bin": "computador_en_casa",
    "internet_bin":   "internet_en_casa",
    "olimpiadas_bin": "participacion_olimpiadas",
}
# Categóricas one-hot (se imputan con la MODA del train).
ONEHOT = ["genero", "municipio", "tipo_institucion"]

TARGET = "puntaje_obtenido"


# =============================================================================
# FUNCIONES DE TRANSFORMACIÓN "PURAS"
# -----------------------------------------------------------------------------
# Estas funciones se usan aquí para construir la matriz de entrenamiento Y se
# EMBEBEN literalmente (via inspect.getsource) en models/predictor.py, de modo
# que la predicción en producción sea idéntica bit a bit a la del entrenamiento.
# No dependen de numpy/pandas: solo builtins + json (stdlib).
# =============================================================================

def _isnan(v):
    return isinstance(v, float) and v != v


def _to_float(v):
    """Convierte a float; devuelve None si no es un número real."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _parse_count(v):
    """Cuenta elementos de una lista JSON (o CSV) ignorando 'Ninguna/Ninguno'."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "[]"):
        return 0
    items = None
    try:
        import json as _json
        parsed = _json.loads(s)
        if isinstance(parsed, list):
            items = parsed
    except Exception:
        items = None
    if items is None:
        items = [x for x in s.strip("[]").replace('"', "").split(",") if x.strip()]
    cnt = 0
    for it in items:
        t = str(it).strip().lower()
        if t and t not in ("ninguna", "ninguno", "ninguna.", "ninguno."):
            cnt += 1
    return cnt


def _ord_level(v):
    """Mapea Ninguna/Básica/Intermedia/Avanzada → 0/1/2/3 (None si desconocido)."""
    if v is None:
        return None
    s = str(v).strip().lower()
    m = {"ninguna": 0, "ninguno": 0, "básica": 1, "basica": 1,
         "intermedia": 2, "avanzada": 3}
    return m.get(s)


def _bin_si(v):
    """Sí* → 1, No* → 0 (None si desconocido)."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("nan", "none", ""):
        return None
    if s.startswith("s"):
        return 1
    if s.startswith("n"):
        return 0
    return None


def features_from_raw(raw, PRE):
    """Convierte un dict de features CRUDOS en el vector numérico del modelo.

    El orden de las columnas es:
        NUMERIC + ORDINAL + BINARY + one-hot(por cada categoría de ONEHOT).
    Los NaN se imputan con la mediana (numéricos/ordinales) o la moda
    (binarias/categóricas) calculadas sobre el train.
    """
    feats = []
    # --- Numéricos directos ---
    for name in PRE["numeric"]:
        v = _to_float(raw.get(name))
        if v is None and name == "n_herramientas":
            v = _parse_count(raw.get("herramientas_conocidas"))
        if v is None and name == "n_areas_interes":
            v = _parse_count(raw.get("areas_interes"))
        if v is None or _isnan(v):
            v = PRE["medians"][name]
        feats.append(float(v))
    # --- Ordinales (0–3) ---
    for name in PRE["ordinal"]:
        rawcol = name[:-4]  # quita el sufijo "_ord"
        lv = _ord_level(raw.get(rawcol))
        if lv is None:
            lv = PRE["medians"][name]
        feats.append(float(lv))
    # --- Binarias Sí/No ---
    for name in PRE["binary"]:
        src = PRE["binary_src"][name]
        b = _bin_si(raw.get(src))
        if b is None:
            b = PRE["modes"][name]
        feats.append(float(b))
    # --- One-hot ---
    for col in PRE["onehot_order"]:
        val = raw.get(col)
        if val is None or _isnan(val) or str(val).strip().lower() in ("nan", "none", ""):
            val = PRE["onehot_mode"][col]
        val = str(val).strip()
        for cat in PRE["onehot_cats"][col]:
            feats.append(1.0 if val == cat else 0.0)
    return feats


def predict_from_features(feats, MODEL):
    """Predice el puntaje (0–100) a partir del vector de features.

    Soporta dos tipos de modelo:
      - 'linear'   : intercepto + Σ coef_i · feat_i.
      - 'ensemble' : suma/promedio de hojas de árboles + término de sesgo.
    """
    if MODEL["type"] == "linear":
        y = MODEL["intercept"]
        coef = MODEL["coef"]
        for i in range(len(coef)):
            y += coef[i] * feats[i]
    else:
        op = MODEL["op"]           # 'lt' (x < umbral) o 'le' (x <= umbral)
        total = 0.0
        for tree in MODEL["trees"]:
            i = 0
            while tree[i][0] != -1:            # -1 marca una hoja
                fi, thr = tree[i][0], tree[i][1]
                if op == "lt":
                    go_left = feats[fi] < thr
                else:
                    go_left = feats[fi] <= thr
                i = tree[i][2] if go_left else tree[i][3]
            total += tree[i][4]
        if MODEL["combine"] == "mean":
            total = total / len(MODEL["trees"])
        y = total + MODEL["bias"]
    if y < 0.0:
        y = 0.0
    if y > 100.0:
        y = 100.0
    return y


# =============================================================================
# A. PREPARACIÓN DE DATOS
# =============================================================================

def cargar_y_limpiar() -> pd.DataFrame:
    log("SECCIÓN A — Carga y limpieza")
    # Resolución del dataset: primero limpio, luego original, luego autodetectar.
    ruta = next((DATA_DIR / n for n in DATASET_CANDIDATOS
                 if (DATA_DIR / n).exists()), None)
    if ruta is None:
        csvs = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
        if not csvs:
            print(f"\n  ⚠  No se encontró '{DATASET_NAME}' en {DATA_DIR}\n")
            sys.exit(0)
        ruta = csvs[0]
    log(f"    dataset: {ruta.name}")

    df = pd.read_csv(ruta, encoding="utf-8")
    log(f"    registros crudos: {len(df):,} | columnas: {df.shape[1]}")

    # Eliminar documentos de prueba.
    docs_prueba = ["1234", "123456", "123456789", "1234567899", "0", "00000000"]
    if "numero_documento" in df.columns:
        df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
        df = df[~df["numero_documento"].isin(docs_prueba)]
        df = df[df["numero_documento"].str.len() >= 5]

    # Normalizar strings vacíos a NaN.
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Tipos numéricos.
    for c in ["puntaje_obtenido", "grado_escolar", "estrato",
              "interes_prog_robotica", "edad_calculada"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reset_index(drop=True)
    log(f"    registros tras limpieza: {len(df):,}")
    return df


def fit_preprocessor(train_records: list[dict], full_df: pd.DataFrame) -> dict:
    """Calcula medianas/modas/categorías (imputación) SOLO con el train."""
    medians, modes, onehot_mode, onehot_cats = {}, {}, {}, {}

    def _num_vals(name):
        out = []
        for r in train_records:
            v = _to_float(r.get(name))
            if v is None and name == "n_herramientas":
                v = _parse_count(r.get("herramientas_conocidas"))
            if v is None and name == "n_areas_interes":
                v = _parse_count(r.get("areas_interes"))
            if v is not None and not _isnan(v):
                out.append(float(v))
        return out

    for name in NUMERIC:
        vals = _num_vals(name)
        medians[name] = float(statistics.median(vals)) if vals else 0.0
    for name in ORDINAL:
        rawcol = name[:-4]
        vals = [_ord_level(r.get(rawcol)) for r in train_records]
        vals = [v for v in vals if v is not None]
        medians[name] = float(statistics.median(vals)) if vals else 0.0
    for name in BINARY:
        src = BINARY_SRC[name]
        vals = [_bin_si(r.get(src)) for r in train_records]
        vals = [v for v in vals if v is not None]
        modes[name] = float(round(sum(vals) / len(vals))) if vals else 0.0
    for col in ONEHOT:
        cnt = Counter(
            str(r.get(col)).strip() for r in train_records
            if r.get(col) is not None
            and str(r.get(col)).strip().lower() not in ("nan", "none", ""))
        onehot_mode[col] = cnt.most_common(1)[0][0] if cnt else ""
        cats = sorted({
            str(v).strip() for v in full_df[col].dropna().unique()
            if str(v).strip().lower() not in ("nan", "none", "")})
        onehot_cats[col] = cats

    return {"numeric": NUMERIC, "ordinal": ORDINAL, "binary": BINARY,
            "binary_src": BINARY_SRC, "onehot_order": ONEHOT,
            "medians": medians, "modes": modes,
            "onehot_mode": onehot_mode, "onehot_cats": onehot_cats}


def feature_names_from_pre(PRE: dict) -> list[str]:
    names = list(PRE["numeric"]) + list(PRE["ordinal"]) + list(PRE["binary"])
    for col in PRE["onehot_order"]:
        for cat in PRE["onehot_cats"][col]:
            names.append(f"{col}={cat}")
    return names


def build_matrix(records: list[dict], PRE: dict) -> np.ndarray:
    return np.array([features_from_raw(r, PRE) for r in records], dtype=float)


def preparar_datos(df: pd.DataFrame) -> dict:
    """Construye X/y de train y test con split 80/20 estratificado por grado."""
    log("SECCIÓN A — Preparación de features y split 80/20 (estratificado por grado)")

    # Todos los registros del CSV tienen puntaje; usamos los que presentaron.
    modelo_df = df[df[TARGET].notna()].copy().reset_index(drop=True)
    log(f"    estudiantes con puntaje (modelables): {len(modelo_df):,}")

    # Estrato de estratificación: grado imputado con la moda (para no perder filas).
    grado_moda = modelo_df["grado_escolar"].mode(dropna=True)
    grado_moda = float(grado_moda.iloc[0]) if len(grado_moda) else 10.0
    strata = modelo_df["grado_escolar"].fillna(grado_moda)

    idx = np.arange(len(modelo_df))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_STATE, stratify=strata)

    df_tr = modelo_df.iloc[tr_idx].reset_index(drop=True)
    df_te = modelo_df.iloc[te_idx].reset_index(drop=True)

    PRE = fit_preprocessor(df_tr.to_dict("records"), modelo_df)
    feat_names = feature_names_from_pre(PRE)

    Xtr = build_matrix(df_tr.to_dict("records"), PRE)
    Xte = build_matrix(df_te.to_dict("records"), PRE)
    ytr = df_tr[TARGET].to_numpy(dtype=float)
    yte = df_te[TARGET].to_numpy(dtype=float)

    log(f"    train={len(df_tr):,}  test={len(df_te):,}  features={len(feat_names)}")
    return {"PRE": PRE, "feat_names": feat_names, "modelo_df": modelo_df,
            "df_tr": df_tr, "df_te": df_te,
            "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte}


# =============================================================================
# B. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# =============================================================================

def construir_modelos() -> dict:
    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=8,
            random_state=RANDOM_STATE, n_jobs=-1),
    }
    if _HAS_XGB:
        modelos["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    else:
        log("    ⚠ XGBoost no disponible: se omite ese modelo.")
    if _HAS_LGBM:
        modelos["LightGBM"] = LGBMRegressor(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    else:
        log("    ⚠ LightGBM no disponible: se omite ese modelo.")
    return modelos


def importancias_modelo(nombre: str, modelo, feat_names: list[str]) -> np.ndarray:
    if hasattr(modelo, "feature_importances_"):
        return np.asarray(modelo.feature_importances_, dtype=float)
    if hasattr(modelo, "coef_"):
        return np.abs(np.asarray(modelo.coef_, dtype=float))
    return np.zeros(len(feat_names))


def grafico_importancias(nombre: str, imp: np.ndarray, feat_names: list[str],
                         filename: str, key: str) -> None:
    orden = np.argsort(imp)[::-1][:15][::-1]  # top 15, ascendente para barh
    vals = imp[orden]
    labels = [feat_names[i] for i in orden]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(orden))))
    colores = gradient_colors(len(orden))
    ax.barh(labels, vals, color=colores, edgecolor="white")
    ax.set_title(f"Importancia de variables — {nombre}")
    ax.set_xlabel("Importancia (|coef| o ganancia)" if nombre == "Regresión Lineal"
                  else "Importancia relativa")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.3g}", va="center", fontsize=8)
    savefig(fig, filename, key)


def entrenar_y_evaluar(ctx: dict) -> dict:
    log("SECCIÓN B — Entrenamiento con CV 5-fold + evaluación en test")
    Xtr, Xte, ytr, yte = ctx["Xtr"], ctx["Xte"], ctx["ytr"], ctx["yte"]
    feat_names = ctx["feat_names"]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2",
               "rmse": "neg_root_mean_squared_error",
               "mae": "neg_mean_absolute_error"}

    resultados = {}
    for i, (nombre, modelo) in enumerate(construir_modelos().items()):
        log(f"    · {nombre}")
        cvres = cross_validate(modelo, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1)
        cv_r2 = cvres["test_r2"]
        cv_rmse = -cvres["test_rmse"]
        cv_mae = -cvres["test_mae"]

        modelo.fit(Xtr, ytr)  # modelo final con todo el train
        pred_te = modelo.predict(Xte)
        test_r2 = float(r2_score(yte, pred_te))
        test_rmse = float(np.sqrt(mean_squared_error(yte, pred_te)))
        test_mae = float(mean_absolute_error(yte, pred_te))

        imp = importancias_modelo(nombre, modelo, feat_names)
        key = f"imp_{i}"
        grafico_importancias(nombre, imp, feat_names,
                             f"F03_importancia_{i}_{nombre.split()[0].lower()}.png", key)

        resultados[nombre] = {
            "modelo": modelo,
            "cv_r2": (float(cv_r2.mean()), float(cv_r2.std())),
            "cv_rmse": (float(cv_rmse.mean()), float(cv_rmse.std())),
            "cv_mae": (float(cv_mae.mean()), float(cv_mae.std())),
            "test_r2": test_r2, "test_rmse": test_rmse, "test_mae": test_mae,
            "importancias": imp, "imp_key": key, "pred_te": pred_te,
        }
        print(f"        CV  R²={cv_r2.mean():.3f}±{cv_r2.std():.3f} | "
              f"RMSE={cv_rmse.mean():.2f}±{cv_rmse.std():.2f} | "
              f"MAE={cv_mae.mean():.2f}±{cv_mae.std():.2f}")
        print(f"        TEST R²={test_r2:.3f} | RMSE={test_rmse:.2f} | "
              f"MAE={test_mae:.2f}")
    return resultados


# =============================================================================
# C. COMPARACIÓN Y SELECCIÓN DEL MEJOR MODELO
# =============================================================================

def comparar_modelos(resultados: dict) -> str:
    log("SECCIÓN C — Comparación y selección del mejor modelo")
    nombres = list(resultados.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colores = [MODEL_COLORS.get(n, PALETTE[i % len(PALETTE)])
               for i, n in enumerate(nombres)]

    r2s = [resultados[n]["test_r2"] for n in nombres]
    ax1.bar(nombres, r2s, color=colores, edgecolor="white")
    ax1.set_title("R² en test (mayor es mejor)")
    ax1.set_ylabel("R²")
    ax1.tick_params(axis="x", rotation=20)
    for i, v in enumerate(r2s):
        ax1.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    rmses = [resultados[n]["test_rmse"] for n in nombres]
    ax2.bar(nombres, rmses, color=colores, edgecolor="white")
    ax2.set_title("RMSE en test (menor es mejor)")
    ax2.set_ylabel("RMSE")
    ax2.tick_params(axis="x", rotation=20)
    for i, v in enumerate(rmses):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("C. Comparación de modelos — puntaje predicho",
                 fontsize=15, fontweight="bold")
    savefig(fig, "F03_comparacion_modelos.png", "comparacion")

    # Mejor modelo: mayor R² en test (desempate por menor RMSE).
    mejor = max(nombres, key=lambda n: (resultados[n]["test_r2"],
                                        -resultados[n]["test_rmse"]))
    log(f"    → mejor modelo: {mejor} "
        f"(R²={resultados[mejor]['test_r2']:.3f}, "
        f"RMSE={resultados[mejor]['test_rmse']:.2f})")
    return mejor


# =============================================================================
# D. EXPORTACIÓN PARA PRODUCCIÓN
# =============================================================================

def _extract_sklearn_tree(t) -> list:
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:  # hoja
            nodes.append([-1, 0.0, -1, -1, float(t.value[i].ravel()[0])])
        else:
            nodes.append([int(t.feature[i]), float(t.threshold[i]),
                          int(t.children_left[i]), int(t.children_right[i]), 0.0])
    return nodes


def _extract_rf(modelo) -> tuple[list, str, str]:
    trees = [_extract_sklearn_tree(est.tree_) for est in modelo.estimators_]
    return trees, "mean", "le"   # RF: promedio de árboles, split x <= umbral


def _extract_xgb(modelo) -> tuple[list, str, str]:
    dumps = modelo.get_booster().get_dump(dump_format="json")
    trees = []
    for d in dumps:
        root = json.loads(d)
        nodes = []

        def add(n):
            idx = len(nodes)
            nodes.append(None)
            if "leaf" in n:
                nodes[idx] = [-1, 0.0, -1, -1, float(n["leaf"])]
            else:
                split = n["split"]
                f = int(split[1:]) if isinstance(split, str) and split.startswith("f") \
                    else int(split)
                thr = float(n["split_condition"])
                hijos = {c["nodeid"]: c for c in n["children"]}
                left = add(hijos[n["yes"]])
                right = add(hijos[n["no"]])
                nodes[idx] = [f, thr, left, right, 0.0]
            return idx

        add(root)
        trees.append(nodes)
    return trees, "sum", "lt"    # XGB: suma de árboles, split x < umbral


def _extract_lgbm(modelo) -> tuple[list, str, str]:
    m = modelo.booster_.dump_model()
    trees = []
    for ti in m["tree_info"]:
        nodes = []

        def add(n):
            idx = len(nodes)
            nodes.append(None)
            if "leaf_value" in n and "split_feature" not in n:
                nodes[idx] = [-1, 0.0, -1, -1, float(n["leaf_value"])]
            else:
                f = int(n["split_feature"])
                thr = float(n["threshold"])
                left = add(n["left_child"])
                right = add(n["right_child"])
                nodes[idx] = [f, thr, left, right, 0.0]
            return idx

        add(ti["tree_structure"])
        trees.append(nodes)
    return trees, "sum", "le"    # LGBM: suma de árboles, split x <= umbral


def _ensemble_raw(feats, trees, op, combine):
    total = 0.0
    for tree in trees:
        i = 0
        while tree[i][0] != -1:
            go_left = (feats[tree[i][0]] < tree[i][1]) if op == "lt" \
                else (feats[tree[i][0]] <= tree[i][1])
            i = tree[i][2] if go_left else tree[i][3]
        total += tree[i][4]
    if combine == "mean":
        total /= len(trees)
    return total


def construir_model_spec(nombre: str, modelo, Xtr: np.ndarray) -> dict:
    """Representación portátil del modelo ganador para el predictor puro."""
    if nombre == "Regresión Lineal":
        return {"type": "linear",
                "intercept": float(modelo.intercept_),
                "coef": [float(c) for c in modelo.coef_]}

    if nombre == "Random Forest":
        trees, combine, op = _extract_rf(modelo)
    elif nombre == "XGBoost":
        trees, combine, op = _extract_xgb(modelo)
    elif nombre == "LightGBM":
        trees, combine, op = _extract_lgbm(modelo)
    else:  # pragma: no cover
        raise ValueError(nombre)

    # Sesgo empírico: absorbe cualquier constante aditiva del modelo
    # (base_score de XGBoost, init score de LightGBM, etc.) para que la
    # réplica pura coincida con el modelo real.
    real = modelo.predict(Xtr)
    crudo = np.array([_ensemble_raw(list(x), trees, op, combine) for x in Xtr])
    bias = float(np.mean(real - crudo))
    return {"type": "ensemble", "combine": combine, "op": op,
            "bias": bias, "trees": trees}


def exportar_coeficientes_json(nombre: str, modelo, model_spec: dict,
                               feat_names: list[str]) -> dict:
    """models/modelo_coeficientes.json — coeficientes/importancias + fórmula/reglas."""
    doc = {"modelo": nombre, "generado": datetime.now().isoformat(timespec="seconds"),
           "n_features": len(feat_names), "features": feat_names}

    if model_spec["type"] == "linear":
        coefs = {f: float(c) for f, c in zip(feat_names, modelo.coef_)}
        doc["intercepto"] = float(modelo.intercept_)
        doc["coeficientes"] = coefs
        # Fórmula exacta legible.
        partes = [f"{modelo.intercept_:.4f}"]
        for f, c in zip(feat_names, modelo.coef_):
            signo = "+" if c >= 0 else "-"
            partes.append(f"{signo} {abs(c):.4f}*[{f}]")
        doc["formula"] = "puntaje = " + " ".join(partes)
        doc["tipo_export"] = "formula_lineal"
    else:
        imp = importancias_modelo(nombre, modelo, feat_names)
        doc["importancias"] = {f: float(v) for f, v in zip(feat_names, imp)}
        # Pseudo-código de reglas: primer árbol del ensemble como ejemplo.
        doc["reglas_ejemplo"] = _reglas_pseudocodigo(model_spec, feat_names,
                                                     max_lineas=60)
        doc["tipo_export"] = "reglas_arbol"
        doc["n_arboles"] = len(model_spec["trees"])
        doc["combina"] = model_spec["combine"]
        doc["sesgo"] = model_spec["bias"]

    destino = MODELS_DIR / "modelo_coeficientes.json"
    destino.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    coeficientes/importancias → models/{destino.name}")
    return doc


def _reglas_pseudocodigo(model_spec: dict, feat_names: list[str],
                         max_lineas: int = 60) -> str:
    """Convierte el primer árbol del ensemble en pseudo-código legible."""
    tree = model_spec["trees"][0]
    op = "<" if model_spec["op"] == "lt" else "<="
    lineas = []

    def recorrer(i, prof):
        if len(lineas) >= max_lineas:
            return
        sangria = "  " * prof
        nodo = tree[i]
        if nodo[0] == -1:
            lineas.append(f"{sangria}=> hoja: {nodo[4]:.4f}")
            return
        fname = feat_names[nodo[0]] if nodo[0] < len(feat_names) else f"f{nodo[0]}"
        lineas.append(f"{sangria}si [{fname}] {op} {nodo[1]:.4f}:")
        recorrer(nodo[2], prof + 1)
        lineas.append(f"{sangria}si no:")
        recorrer(nodo[3], prof + 1)

    recorrer(0, 0)
    combina = "PROMEDIO" if model_spec["combine"] == "mean" else "SUMA"
    cab = (f"# Ejemplo: árbol 1 de {len(model_spec['trees'])}. "
           f"Predicción = {combina} de las hojas de todos los árboles "
           f"+ sesgo({model_spec['bias']:.4f}).\n")
    return cab + "\n".join(lineas)


def generar_predictor_py(model_spec: dict, PRE: dict, feat_names: list[str],
                         nombre: str, ejemplo: dict) -> None:
    """Escribe models/predictor.py: función pura, solo stdlib (json + aritmética)."""
    fns = [_isnan, _to_float, _parse_count, _ord_level, _bin_si,
           features_from_raw, predict_from_features]
    fuente = "\n\n".join(inspect.getsource(f) for f in fns)

    spec = {"preprocess": PRE, "model": model_spec,
            "feature_names": feat_names, "modelo": nombre}
    # ensure_ascii=True + raw string ⇒ el archivo no depende de la codificación
    # y json.loads reconstruye los acentos de forma segura.
    js = json.dumps(spec, ensure_ascii=True)

    contenido = f'''# -*- coding: utf-8 -*-
"""
Predictor puro del puntaje Copa STEM — modelo: {nombre}.
GENERADO AUTOMÁTICAMENTE por notebooks/03_modelo_predictivo.py — no editar a mano.

Uso:
    from predictor import predecir_puntaje
    puntaje = predecir_puntaje({{"grado_escolar": 10, "genero": "Femenino", ...}})

No requiere sklearn, xgboost, lightgbm, numpy ni pandas.
Solo usa la librería estándar de Python (json, math).
"""
import json
import math  # noqa: F401  (disponible para extensiones futuras)

_SPEC = json.loads(r"""{js}""")
PRE = _SPEC["preprocess"]
MODEL = _SPEC["model"]
FEATURE_NAMES = _SPEC["feature_names"]


{fuente}


def predecir_puntaje(estudiante):
    """Recibe un dict con los features crudos y devuelve el puntaje estimado (0–100)."""
    feats = features_from_raw(estudiante, PRE)
    return predict_from_features(feats, MODEL)


if __name__ == "__main__":
    ejemplo = {json.dumps(ejemplo, ensure_ascii=False)}
    print("Modelo:", _SPEC["modelo"])
    print("Puntaje estimado del ejemplo:", round(predecir_puntaje(ejemplo), 2))
'''
    destino = MODELS_DIR / "predictor.py"
    destino.write_text(contenido, encoding="utf-8")
    log(f"    predictor puro → models/{destino.name}")


def exportar_produccion(ctx: dict, resultados: dict, mejor: str) -> dict:
    log("SECCIÓN D — Exportación para producción")
    modelo = resultados[mejor]["modelo"]
    PRE, feat_names = ctx["PRE"], ctx["feat_names"]

    # 1) Modelo real con joblib (para uso con sklearn en el backend si se desea).
    bundle = {"modelo": modelo, "preprocessor": PRE, "feature_names": feat_names,
              "nombre": mejor, "random_state": RANDOM_STATE}
    joblib.dump(bundle, MODELS_DIR / "mejor_modelo_puntaje.joblib")
    log("    modelo serializado → models/mejor_modelo_puntaje.joblib")

    # 2) Representación portátil + JSON de coeficientes/importancias.
    model_spec = construir_model_spec(mejor, modelo, ctx["Xtr"])
    doc = exportar_coeficientes_json(mejor, modelo, model_spec, feat_names)

    # 3) Ejemplo representativo (medianas/modas) para la demo del predictor.
    ejemplo = _estudiante_ejemplo(ctx["df_tr"])

    # 4) Predictor puro.
    generar_predictor_py(model_spec, PRE, feat_names, mejor, ejemplo)

    # 5) Auto-verificación: el predictor puro debe coincidir con el modelo real.
    max_diff = _verificar_predictor(ctx, modelo)
    log(f"    verificación predictor puro vs modelo real: máx|Δ| = {max_diff:.4g}")

    return {"model_spec": model_spec, "doc": doc, "max_diff": max_diff,
            "ejemplo": ejemplo}


def _estudiante_ejemplo(df_tr: pd.DataFrame) -> dict:
    def moda(col):
        m = df_tr[col].mode(dropna=True)
        return (m.iloc[0] if len(m) else None)
    ej = {}
    for c in ["grado_escolar", "genero", "municipio", "tipo_institucion",
              "estrato", "computador_en_casa", "internet_en_casa",
              "participacion_olimpiadas", "nivel_programacion", "nivel_robotica",
              "interes_prog_robotica", "herramientas_conocidas", "areas_interes"]:
        if c in df_tr.columns:
            v = moda(c)
            if isinstance(v, float) and v == v and c in ("grado_escolar", "estrato"):
                v = int(v)
            ej[c] = None if (isinstance(v, float) and v != v) else v
    return ej


def _verificar_predictor(ctx: dict, modelo) -> float:
    """Compara la predicción pura (recargando predictor.py) con el modelo real."""
    import importlib.util
    ruta = MODELS_DIR / "predictor.py"
    spec = importlib.util.spec_from_file_location("predictor_check", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df_te = ctx["df_te"]
    real = np.clip(modelo.predict(ctx["Xte"]), 0.0, 100.0)
    puro = np.array([mod.predecir_puntaje(r) for r in df_te.to_dict("records")])
    return float(np.max(np.abs(real - puro)))


# =============================================================================
# E. ANÁLISIS DE RESIDUOS
# =============================================================================

def analisis_residuos(ctx: dict, resultados: dict, mejor: str) -> dict:
    log("SECCIÓN E — Análisis de residuos")
    df_te = ctx["df_te"].copy()
    yte = ctx["yte"]
    pred = resultados[mejor]["pred_te"]
    resid = yte - pred
    df_te["_pred"] = pred
    df_te["_resid"] = resid

    # --- Real vs predicho + distribución del error ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.scatter(yte, pred, s=20, alpha=0.35, color=COLORS["cyan"],
                edgecolor="white", linewidth=0.3)
    lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    ax1.plot(lim, lim, "--", color=COLORS["red"], linewidth=1.5,
             label="Predicción perfecta")
    ax1.set_title(f"Real vs. predicho — {mejor}\nR² = {resultados[mejor]['test_r2']:.3f}")
    ax1.set_xlabel("Puntaje real")
    ax1.set_ylabel("Puntaje predicho")
    ax1.legend()

    sns.histplot(resid, bins=30, color=COLORS["violet"], edgecolor="white",
                 stat="density", ax=ax2)
    ax2.axvline(0, color=COLORS["red"], linestyle="--", linewidth=1.5)
    ax2.axvline(float(np.mean(resid)), color=COLORS["amber"], linestyle="--",
                linewidth=1.5, label=f"Media = {np.mean(resid):.2f}")
    ax2.set_title("Distribución de los errores (real − predicho)")
    ax2.set_xlabel("Error")
    ax2.set_ylabel("Densidad")
    ax2.legend()
    fig.suptitle("E. Análisis de residuos", fontsize=15, fontweight="bold")
    savefig(fig, "F03_residuos.png", "residuos")

    # --- Residuos por grupo: ¿el modelo es peor para ciertos grupos? ---
    grupos = [("genero", "Género"), ("municipio", "Municipio"),
              ("estrato", "Estrato")]
    grupos = [(c, t) for c, t in grupos if c in df_te.columns]
    resumen_grupos = {}
    if grupos:
        fig2, axes = plt.subplots(1, len(grupos), figsize=(6 * len(grupos), 5))
        axes = np.atleast_1d(axes)
        for ax, (col, titulo) in zip(axes, grupos):
            sub = df_te[[col, "_resid"]].dropna()
            orden = sorted(sub[col].unique(), key=str)
            sns.boxplot(data=sub, x=col, y="_resid", order=orden,
                        palette=PALETTE, ax=ax)
            ax.axhline(0, color=COLORS["red"], linestyle="--", linewidth=1.2)
            ax.set_title(f"Error por {titulo}")
            ax.set_xlabel("")
            ax.set_ylabel("Error (real − predicho)")
            ax.tick_params(axis="x", rotation=20)
            resumen_grupos[col] = (
                sub.assign(abs_err=sub["_resid"].abs())
                .groupby(col)["abs_err"].mean().round(2).to_dict())
        fig2.suptitle("E. ¿El modelo es peor para ciertos grupos? "
                      "(error absoluto medio por grupo)",
                      fontsize=14, fontweight="bold")
        savefig(fig2, "F03_residuos_por_grupo.png", "residuos_grupo")

    return {"resid": resid, "mean_resid": float(np.mean(resid)),
            "std_resid": float(np.std(resid)), "por_grupo": resumen_grupos}


# =============================================================================
# F. ÍNDICE DE POTENCIAL STEM
# =============================================================================

def indice_potencial(ctx: dict, resultados: dict, mejor: str) -> dict:
    log("SECCIÓN F — Índice de Potencial STEM (todos los inscritos)")
    modelo = resultados[mejor]["modelo"]
    PRE = ctx["PRE"]
    df_full = ctx["modelo_df"]  # todos los inscritos con datos (presentaron)

    X_all = build_matrix(df_full.to_dict("records"), PRE)
    est = np.clip(modelo.predict(X_all), 0.0, 100.0)

    out = pd.DataFrame({
        "numero_documento": df_full["numero_documento"].astype(str).values,
        "puntaje_estimado": np.round(est, 2),
    })
    # Índice de potencial = percentil (0–100) del puntaje estimado en la cohorte.
    out["indice_potencial"] = (out["puntaje_estimado"].rank(pct=True) * 100).round(2)

    def categorizar(p):
        if p >= CAT_ALTO:
            return "Alto potencial"
        if p >= CAT_MEDIO:
            return "Medio"
        if p >= CAT_DESARROLLO:
            return "En desarrollo"
        return "Requiere apoyo"

    out["categoria"] = out["indice_potencial"].apply(categorizar)

    destino = MODELS_DIR / "scores_potencial_stem.csv"
    out.to_csv(destino, index=False, encoding="utf-8-sig")
    log(f"    scores exportados ({len(out):,} estudiantes) → models/{destino.name}")

    # --- Gráfico: distribución del índice + conteo por categoría ---
    orden_cat = ["Requiere apoyo", "En desarrollo", "Medio", "Alto potencial"]
    conteo = out["categoria"].value_counts().reindex(orden_cat).fillna(0).astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.histplot(out["puntaje_estimado"], bins=30, color=COLORS["cyan"],
                 edgecolor="white", ax=ax1)
    ax1.set_title("Distribución del puntaje estimado")
    ax1.set_xlabel("Puntaje estimado (0–100)")
    ax1.set_ylabel("N estudiantes")

    cat_colors = [COLORS["red"], COLORS["amber"], COLORS["violet"], COLORS["green"]]
    ax2.bar(conteo.index, conteo.values, color=cat_colors, edgecolor="white")
    ax2.set_title("Estudiantes por categoría de potencial")
    ax2.set_ylabel("N estudiantes")
    ax2.tick_params(axis="x", rotation=15)
    for i, v in enumerate(conteo.values):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("F. Índice de Potencial STEM", fontsize=15, fontweight="bold")
    savefig(fig, "F03_indice_potencial.png", "potencial")

    return {"tabla": out, "conteo": conteo.to_dict(), "n": len(out)}


# =============================================================================
# G. INFORME MARKDOWN
# =============================================================================

def construir_informe(ctx: dict, resultados: dict, mejor: str,
                      exp: dict, resid: dict, pot: dict) -> None:
    log("SECCIÓN G — Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    nombres = list(resultados.keys())

    R = REPORT.append
    R("# Modelo Predictivo del Puntaje — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Fase 2 · Informe generado: {fecha}\n")
    R("---\n")

    # Resumen ejecutivo
    r = resultados[mejor]
    R("## Resumen ejecutivo\n")
    R(dedent(f"""\
        Se entrenaron **{len(nombres)} modelos** de regresión para predecir el
        `puntaje_obtenido` (0–100) a partir de {len(ctx['feat_names'])} variables
        demográficas, socioeconómicas y de experiencia previa, sobre
        **{len(ctx['modelo_df']):,} estudiantes** (split 80/20 estratificado por
        grado). El mejor modelo fue **{mejor}** (R² test = {r['test_r2']:.3f},
        RMSE = {r['test_rmse']:.2f}, MAE = {r['test_mae']:.2f}). A partir de él se
        construyó un **Índice de Potencial STEM** (0–100) para cada estudiante,
        exportado a `models/scores_potencial_stem.csv` para su carga en Supabase.
        La réplica pura del modelo (`models/predictor.py`) coincide con el modelo
        real dentro de un margen de **{exp['max_diff']:.2g}** puntos.\n"""))

    # Metodología
    R("## Metodología\n")
    R(dedent(f"""\
        - **Features:** grado, género (one-hot), municipio (one-hot), tipo de
          institución (one-hot), estrato, computador e internet en casa (binarias),
          participación previa en olimpiadas (binaria), nivel de programación y de
          robótica (ordinales 0–3), interés en prog/robótica, nº de herramientas y
          nº de áreas de interés (conteos).
        - **Imputación:** mediana (numéricos/ordinales) o moda (binarias/categóricas),
          calculada **solo con el train** para evitar fuga de información.
        - **Validación:** 5-fold CV sobre el train + evaluación final en el test 20%.
        - **Métricas:** R², RMSE y MAE. Reproducible con `random_state=42`.\n"""))

    # Comparación de modelos
    R("## Comparación de modelos\n")
    filas = []
    for n in nombres:
        rr = resultados[n]
        filas.append({
            "Modelo": n + (" ⭐" if n == mejor else ""),
            "R² (CV)": f"{rr['cv_r2'][0]:.3f} ± {rr['cv_r2'][1]:.3f}",
            "RMSE (CV)": f"{rr['cv_rmse'][0]:.2f} ± {rr['cv_rmse'][1]:.2f}",
            "MAE (CV)": f"{rr['cv_mae'][0]:.2f} ± {rr['cv_mae'][1]:.2f}",
            "R² (test)": f"{rr['test_r2']:.3f}",
            "RMSE (test)": f"{rr['test_rmse']:.2f}",
            "MAE (test)": f"{rr['test_mae']:.2f}",
        })
    R(tabla_md(pd.DataFrame(filas)) + "\n")
    R(f"\n{img('comparacion', 'Comparación de modelos')}\n")
    R(f"\n**Modelo seleccionado: {mejor}** (mayor R² en test, desempate por RMSE).\n")

    # Importancias del mejor modelo
    R("## Importancia de variables (mejor modelo)\n")
    imp = resultados[mejor]["importancias"]
    orden = np.argsort(imp)[::-1][:10]
    fi = [{"Variable": ctx["feat_names"][i], "Importancia": round(float(imp[i]), 4)}
          for i in orden]
    R(tabla_md(pd.DataFrame(fi)) + "\n")
    R(f"\n{img(resultados[mejor]['imp_key'], 'Importancia de variables')}\n")

    # Exportación
    R("## Exportación para producción\n")
    doc = exp["doc"]
    R(dedent(f"""\
        - `models/mejor_modelo_puntaje.joblib` — modelo + preprocesador (uso con sklearn).
        - `models/modelo_coeficientes.json` — {'fórmula lineal exacta' if doc['tipo_export']=='formula_lineal' else 'importancias + reglas de decisión (pseudo-código)'}.
        - `models/predictor.py` — **función Python pura** `predecir_puntaje(dict)`
          que NO requiere sklearn ni librerías de ML (solo `json` + aritmética).
        - `models/scores_potencial_stem.csv` — índice de potencial por estudiante.\n"""))
    if doc["tipo_export"] == "formula_lineal":
        formula = doc["formula"]
        if len(formula) > 900:
            formula = formula[:900] + " …"
        R("\n**Fórmula del modelo (extracto):**\n")
        R("```\n" + formula + "\n```\n")
    else:
        R("\n**Reglas de decisión (ejemplo, primer árbol):**\n")
        R("```\n" + doc["reglas_ejemplo"] + "\n```\n")

    # Residuos
    R("## Análisis de residuos\n")
    R(dedent(f"""\
        El error medio (real − predicho) es **{resid['mean_resid']:.2f}** con
        desviación **{resid['std_resid']:.2f}**. Un error medio cercano a 0 indica
        ausencia de sesgo sistemático.\n"""))
    R(f"\n{img('residuos', 'Residuos')}\n")
    if resid["por_grupo"]:
        R("\n**Error absoluto medio por grupo** (¿el modelo es peor para alguien?):\n")
        for col, d in resid["por_grupo"].items():
            detalle = "; ".join(f"{k}: {v}" for k, v in d.items())
            R(f"- **{col}** → {detalle}\n")
        R(f"\n{img('residuos_grupo', 'Residuos por grupo')}\n")

    # Índice de potencial
    R("## Índice de Potencial STEM\n")
    R(dedent(f"""\
        Para cada estudiante se estima el puntaje con el mejor modelo, se normaliza
        a percentil (0–100) dentro de la cohorte (`indice_potencial`) y se clasifica:
        **Alto potencial** (≥{CAT_ALTO}), **Medio** ({CAT_MEDIO}–{CAT_ALTO-1}),
        **En desarrollo** ({CAT_DESARROLLO}–{CAT_MEDIO-1}) y
        **Requiere apoyo** (<{CAT_DESARROLLO}).\n"""))
    conteo = pot["conteo"]
    filas_cat = [{"Categoría": k, "N estudiantes": v} for k, v in conteo.items()]
    R(tabla_md(pd.DataFrame(filas_cat)) + "\n")
    R(f"\n{img('potencial', 'Índice de potencial')}\n")
    R(dedent("""\
        El CSV `models/scores_potencial_stem.csv` (columnas: `numero_documento`,
        `puntaje_estimado`, `indice_potencial`, `categoria`) se sube a Supabase
        para mostrar el potencial de cada estudiante en la web.\n"""))

    # Limitaciones
    R("## Limitaciones\n")
    R(dedent(f"""\
        - **Poder predictivo moderado:** el puntaje depende de factores no medidos
          (preparación, motivación puntual); un R² de {r['test_r2']:.3f} implica que
          buena parte de la varianza no es explicable con estas variables.
        - **Datos observacionales y autorreportados** (estrato, acceso, niveles).
        - **Categorías con poca muestra** (p. ej. género no binario) tienen
          estimaciones menos fiables.
        - El **índice de potencial es relativo** a esta cohorte (percentil), no una
          medida absoluta de habilidad.\n"""))

    R("## Referencias técnicas\n")
    R(dedent("""\
        - Breiman, L. (2001). *Random Forests*. · Chen & Guestrin (2016). *XGBoost*.
        - Ke et al. (2017). *LightGBM*. · Pedregosa et al. (2011). *scikit-learn*.\n"""))
    R("\n---\n_Generado por `notebooks/03_modelo_predictivo.py` — Copa STEM 2026._\n")

    destino = REPORTS_DIR / "03_modelo_predictivo.md"
    destino.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"    informe escrito → reports/{destino.name}")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Modelo Predictivo del Puntaje (Fase 2)")
    print(" Fundación SapienceLab")
    print("=" * 70)

    df = cargar_y_limpiar()
    ctx = preparar_datos(df)
    resultados = entrenar_y_evaluar(ctx)
    mejor = comparar_modelos(resultados)
    exp = exportar_produccion(ctx, resultados, mejor)
    resid = analisis_residuos(ctx, resultados, mejor)
    pot = indice_potencial(ctx, resultados, mejor)
    construir_informe(ctx, resultados, mejor, exp, resid, pot)

    print("\n" + "=" * 70)
    print(" ✔ MODELO PREDICTIVO COMPLETADO")
    print(f"   · Mejor modelo:      {mejor} "
          f"(R²={resultados[mejor]['test_r2']:.3f})")
    print(f"   · Predictor puro:    máx|Δ| = {exp['max_diff']:.4g} (vs modelo real)")
    print(f"   · Figuras:           {len(FIGURES)} → outputs/")
    print(f"   · Modelo:            models/mejor_modelo_puntaje.joblib")
    print(f"   · Scores potencial:  models/scores_potencial_stem.csv ({pot['n']:,})")
    print(f"   · Informe:           reports/03_modelo_predictivo.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
