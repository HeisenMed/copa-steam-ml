# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 06: Detección de Talento Oculto  — Fase 2/3
================================================================================

Definición
----------
**Talento oculto** = estudiante con **alto rendimiento** PERO **condiciones
socioeconómicas adversas**. Es el foco prioritario de intervención de la
Fundación: alto potencial STEM que el contexto podría estar frenando.

    Alto rendimiento (≥1 de):
        - puntaje_obtenido ≥ percentil 75 de los datos limpios
        - indice_potencial ≥ 75  (de models/deploy/scores_potencial_stem.csv)

    Condiciones adversas (≥2 de):
        - estrato 1 o 2
        - sin computador en casa
        - sin internet en casa
        - no vive con ambos padres
        - no ha participado antes en olimpiadas
        - nivel de programación "Ninguna"

    talento_oculto = alto_rendimiento AND (≥2 condiciones adversas)

Nota metodológica sobre el clasificador (sección B)
---------------------------------------------------
El target `talento_oculto` es una **regla determinista** sobre las mismas
variables que se usan como features. Por eso los clasificadores alcanzan
métricas muy altas: en buena medida *reconstruyen la regla* (fuga de etiqueta
esperada, no generalización). Su valor real es doble: (1) producir una
**probabilidad continua** para priorizar casos límite, y (2) confirmar vía
importancia de variables **qué condiciones pesan más** en la definición.

Entregables
-----------
    models/deploy/talento_oculto_scores.csv
    models/deploy/talento_oculto_predictor.py   (función pura, sin sklearn)
    models/deploy/talento_oculto_predictor.js    (misma función en JS ES6)
    reports/06_talento_oculto.md
    outputs/F06_*.png  (paleta Copa STEM)

Reproducible (`random_state=42`).
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
import json
import inspect
import warnings
import importlib.util
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from collections import Counter

RANDOM_STATE = 42

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
    from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0. CONFIGURACIÓN GLOBAL
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
for _d in (OUTPUTS_DIR, REPORTS_DIR, DEPLOY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_CANDIDATOS = ["copa_stem_dataset_limpio.csv", "copa_stem_dataset.csv"]
SCORES_CSV = DEPLOY_DIR / "scores_potencial_stem.csv"

COLORS = {"cyan": "#00d4ff", "violet": "#8b5cf6", "amber": "#f59e0b",
          "dark": "#050816", "green": "#10b981", "red": "#ef4444", "blue": "#0f77ee"}
PALETTE = ["#00d4ff", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444", "#0f77ee"]

sns.set_theme(style="whitegrid")
sns.set_palette(PALETTE)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "#333333",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "font.size": 10, "figure.autolayout": True,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})
DPI = 150
STEM_GRAD = LinearSegmentedColormap.from_list("stem_grad", PALETTE)
STEM_SEQ = LinearSegmentedColormap.from_list(
    "stem_seq", ["#eafcff", "#8ee9ff", COLORS["cyan"], COLORS["blue"]])

INDICE_ALTO = 75          # umbral de indice_potencial para "alto rendimiento"
ADVERSAS_MIN = 2          # nº mínimo de condiciones adversas
MODEL_COLORS = {"Regresión Logística": COLORS["cyan"],
                "Random Forest": COLORS["violet"],
                "XGBoost": COLORS["amber"]}

FIGURES: dict[str, str] = {}
REPORT: list[str] = []


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def gradient_colors(n: int) -> list:
    if n <= 1:
        return [COLORS["cyan"]]
    return [STEM_GRAD(i / (n - 1)) for i in range(n)]


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
# CONFIGURACIÓN DE FEATURES DEL CLASIFICADOR
# =============================================================================
NUMERIC = ["grado_escolar", "estrato", "interes_prog_robotica",
           "n_herramientas", "n_areas_interes", "puntaje_obtenido",
           "indice_potencial"]
ORDINAL = ["nivel_programacion_ord", "nivel_robotica_ord"]
BINARY = ["computador_bin", "internet_bin", "olimpiadas_bin"]
BINARY_SRC = {"computador_bin": "computador_en_casa",
              "internet_bin": "internet_en_casa",
              "olimpiadas_bin": "participacion_olimpiadas"}
ONEHOT = ["genero", "municipio", "tipo_institucion", "con_quien_vive"]

# Nombres legibles de las condiciones adversas (para el detalle exportado).
COND_LABELS = {
    "estrato_bajo": "Estrato 1-2",
    "sin_computador": "Sin computador",
    "sin_internet": "Sin internet",
    "no_ambos_padres": "No vive con ambos padres",
    "sin_olimpiadas": "Sin olimpiadas previas",
    "prog_ninguna": "Programación: ninguna",
}


# =============================================================================
# FUNCIONES PURAS
# -----------------------------------------------------------------------------
# Se usan aquí y se EMBEBEN (inspect.getsource) en el predictor .py de deploy.
# La versión .js las replica a mano. Solo stdlib (json + math).
# =============================================================================

def _isnan(v):
    return isinstance(v, float) and v != v


def _to_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _parse_count(v):
    if v is None:
        return 0
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
    if v is None:
        return None
    m = {"ninguna": 0, "ninguno": 0, "básica": 1, "basica": 1,
         "intermedia": 2, "avanzada": 3}
    return m.get(str(v).strip().lower())


def _bin_si(v):
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


def _sigmoid(z):
    import math
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def condiciones_adversas(raw):
    """Devuelve (lista de condiciones adversas activas, detalle_legible). Puro."""
    activas = []
    est = _to_float(raw.get("estrato"))
    if est is not None and est <= 2:
        activas.append("estrato_bajo")
    if _bin_si(raw.get("computador_en_casa")) == 0:
        activas.append("sin_computador")
    if _bin_si(raw.get("internet_en_casa")) == 0:
        activas.append("sin_internet")
    cqv = raw.get("con_quien_vive")
    if cqv is not None and str(cqv).strip().lower() not in ("nan", "none", "") \
            and str(cqv).strip() != "Ambos padres":
        activas.append("no_ambos_padres")
    if _bin_si(raw.get("participacion_olimpiadas")) == 0:
        activas.append("sin_olimpiadas")
    prog = raw.get("nivel_programacion")
    if prog is not None and str(prog).strip().lower().startswith("ningun"):
        activas.append("prog_ninguna")
    return activas


def _features_clf(raw, PRE):
    """Vector de features para el clasificador (mismo orden que en el entrenamiento)."""
    feats = []
    for name in PRE["numeric"]:
        v = _to_float(raw.get(name))
        if v is None and name == "n_herramientas":
            v = _parse_count(raw.get("herramientas_conocidas"))
        if v is None and name == "n_areas_interes":
            v = _parse_count(raw.get("areas_interes"))
        if v is None:
            v = PRE["medians"][name]
        feats.append(float(v))
    for name in PRE["ordinal"]:
        lv = _ord_level(raw.get(name[:-4]))
        if lv is None:
            lv = PRE["medians"][name]
        feats.append(float(lv))
    for name in PRE["binary"]:
        b = _bin_si(raw.get(PRE["binary_src"][name]))
        if b is None:
            b = PRE["modes"][name]
        feats.append(float(b))
    for col in PRE["onehot_order"]:
        val = raw.get(col)
        if val is None or _isnan(val) or str(val).strip().lower() in ("nan", "none", ""):
            val = PRE["onehot_mode"][col]
        val = str(val).strip()
        for cat in PRE["onehot_cats"][col]:
            feats.append(1.0 if val == cat else 0.0)
    return feats


def _predict_proba(feats, MODEL):
    """Probabilidad de talento oculto (0–1) según el modelo embebido."""
    if MODEL["type"] == "logistic":
        z = MODEL["intercept"]
        coef = MODEL["coef"]
        for i in range(len(coef)):
            z += coef[i] * feats[i]
        return _sigmoid(z)
    # ensemble de árboles
    op = MODEL["op"]
    total = 0.0
    for tree in MODEL["trees"]:
        i = 0
        while tree[i][0] != -1:
            fi, thr = tree[i][0], tree[i][1]
            go_left = feats[fi] < thr if op == "lt" else feats[fi] <= thr
            i = tree[i][2] if go_left else tree[i][3]
        total += tree[i][4]
    if MODEL["combine"] == "mean":
        total /= len(MODEL["trees"])
    if MODEL.get("proba_transform") == "sigmoid":
        return _sigmoid(total + MODEL["bias"])
    return total


def evaluar_talento(raw, SPEC):
    """Evalúa a un estudiante: condiciones, regla determinista y probabilidad."""
    activas = condiciones_adversas(raw)
    n_adv = len(activas)

    punt = _to_float(raw.get("puntaje_obtenido"))
    indice = _to_float(raw.get("indice_potencial"))
    alto = False
    if punt is not None and punt >= SPEC["p75_puntaje"]:
        alto = True
    if indice is not None and indice >= SPEC["indice_alto"]:
        alto = True

    es_talento = bool(alto and n_adv >= SPEC["adversas_min"])
    proba = _predict_proba(_features_clf(raw, SPEC["preprocess"]), SPEC["model"])

    return {
        "probabilidad_talento": round(float(proba), 4),
        "es_talento_oculto": es_talento,
        "n_condiciones_adversas": n_adv,
        "condiciones_detalle": "|".join(activas),
        "_alto_rendimiento": bool(alto),
    }


# =============================================================================
# CARGA Y ETIQUETADO
# =============================================================================

def cargar_datos() -> pd.DataFrame:
    log("Carga de datos limpios + índice de potencial")
    ruta = next((DATA_DIR / n for n in DATASET_CANDIDATOS
                 if (DATA_DIR / n).exists()), None)
    if ruta is None:
        print("\n  ⚠  No se encontró el dataset. Ejecute 05b primero.\n")
        sys.exit(1)
    log(f"    dataset: {ruta.name}")
    df = pd.read_csv(ruta, encoding="utf-8", dtype={"numero_documento": str})

    docs_prueba = ["1234", "123456", "123456789", "1234567899", "0", "00000000"]
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df = df[~df["numero_documento"].isin(docs_prueba)]
    df = df[df["numero_documento"].str.len() >= 5]
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    for c in ["puntaje_obtenido", "grado_escolar", "estrato",
              "interes_prog_robotica", "edad_calculada"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["puntaje_obtenido"].notna()].reset_index(drop=True)

    # Índice de potencial (de script 04); mapa por documento (primer valor).
    if not SCORES_CSV.exists():
        print(f"\n  ⚠  Falta {SCORES_CSV}. Ejecute antes 04_indice_potencial_stem.py.\n")
        sys.exit(1)
    sc = pd.read_csv(SCORES_CSV, dtype={"numero_documento": str})
    idx_map = (sc.drop_duplicates("numero_documento")
               .set_index("numero_documento")["indice_potencial"].to_dict())
    df["indice_potencial"] = df["numero_documento"].map(idx_map)
    log(f"    estudiantes: {len(df):,} | con índice: {df['indice_potencial'].notna().sum():,}")

    # Ordinales derivados.
    for c in ["nivel_programacion", "nivel_robotica"]:
        if c in df.columns:
            df[c + "_ord"] = df[c].map(lambda v: _ord_level(v))
    # Conteos.
    df["n_herramientas"] = df.get("herramientas_conocidas").map(_parse_count) \
        if "herramientas_conocidas" in df.columns else 0
    df["n_areas_interes"] = df.get("areas_interes").map(_parse_count) \
        if "areas_interes" in df.columns else 0
    return df


def etiquetar_talento(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log("SECCIÓN A — Etiquetado de talento oculto (regla determinista)")
    p75 = float(df["puntaje_obtenido"].quantile(0.75))
    log(f"    percentil 75 del puntaje (datos limpios): {p75:.1f}")

    cond = df.apply(lambda r: condiciones_adversas(r.to_dict()), axis=1)
    df["condiciones_activas"] = cond
    df["n_condiciones_adversas"] = cond.map(len)
    df["condiciones_detalle"] = cond.map(lambda xs: "|".join(xs))

    alto = (df["puntaje_obtenido"] >= p75) | (df["indice_potencial"] >= INDICE_ALTO)
    df["alto_rendimiento"] = alto.fillna(False)
    df["talento_oculto"] = (df["alto_rendimiento"]
                            & (df["n_condiciones_adversas"] >= ADVERSAS_MIN))

    n_tal = int(df["talento_oculto"].sum())
    log(f"    alto rendimiento: {int(df['alto_rendimiento'].sum()):,} | "
        f"talento oculto: {n_tal:,} ({100*n_tal/len(df):.1f}%)")
    meta = {"p75_puntaje": p75, "indice_alto": INDICE_ALTO,
            "adversas_min": ADVERSAS_MIN, "n_talento": n_tal,
            "n_total": len(df), "pct": 100 * n_tal / len(df)}
    return df, meta


def fit_preprocessor(train: list[dict], full_df: pd.DataFrame) -> dict:
    import statistics
    medians, modes, onehot_mode, onehot_cats = {}, {}, {}, {}

    def num_vals(name):
        out = []
        for r in train:
            v = _to_float(r.get(name))
            if v is None and name == "n_herramientas":
                v = _parse_count(r.get("herramientas_conocidas"))
            if v is None and name == "n_areas_interes":
                v = _parse_count(r.get("areas_interes"))
            if v is not None:
                out.append(float(v))
        return out

    for name in NUMERIC:
        vals = num_vals(name)
        medians[name] = float(statistics.median(vals)) if vals else 0.0
    for name in ORDINAL:
        vals = [_ord_level(r.get(name[:-4])) for r in train]
        vals = [v for v in vals if v is not None]
        medians[name] = float(statistics.median(vals)) if vals else 0.0
    for name in BINARY:
        vals = [_bin_si(r.get(BINARY_SRC[name])) for r in train]
        vals = [v for v in vals if v is not None]
        modes[name] = float(round(sum(vals) / len(vals))) if vals else 0.0
    for col in ONEHOT:
        cnt = Counter(str(r.get(col)).strip() for r in train
                      if r.get(col) is not None
                      and str(r.get(col)).strip().lower() not in ("nan", "none", ""))
        onehot_mode[col] = cnt.most_common(1)[0][0] if cnt else ""
        cats = sorted({str(v).strip() for v in full_df[col].dropna().unique()
                       if str(v).strip().lower() not in ("nan", "none", "")})
        onehot_cats[col] = cats

    return {"numeric": NUMERIC, "ordinal": ORDINAL, "binary": BINARY,
            "binary_src": BINARY_SRC, "onehot_order": ONEHOT,
            "medians": medians, "modes": modes,
            "onehot_mode": onehot_mode, "onehot_cats": onehot_cats}


def feature_names(PRE: dict) -> list[str]:
    names = list(PRE["numeric"]) + list(PRE["ordinal"]) + list(PRE["binary"])
    for col in PRE["onehot_order"]:
        for cat in PRE["onehot_cats"][col]:
            names.append(f"{col}={cat}")
    return names


def build_X(records: list[dict], PRE: dict) -> np.ndarray:
    return np.array([_features_clf(r, PRE) for r in records], dtype=float)


# =============================================================================
# B. MODELOS DE CLASIFICACIÓN
# =============================================================================

def construir_modelos():
    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=5000,
                                                  random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    if _HAS_XGB:
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    else:
        log("    ⚠ XGBoost no disponible: se omite.")
    return modelos


def entrenar(df: pd.DataFrame):
    log("SECCIÓN B — Entrenamiento de clasificadores (CV 5-fold estratificada)")
    y = df["talento_oculto"].astype(int).to_numpy()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE,
                              stratify=y)
    df_tr, df_te = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)

    PRE = fit_preprocessor(df_tr.to_dict("records"), df)
    feats = feature_names(PRE)
    Xtr, Xte = build_X(df_tr.to_dict("records"), PRE), build_X(df_te.to_dict("records"), PRE)
    ytr, yte = df_tr["talento_oculto"].astype(int).to_numpy(), \
        df_te["talento_oculto"].astype(int).to_numpy()
    log(f"    train={len(df_tr):,} (pos={ytr.sum()}) | "
        f"test={len(df_te):,} (pos={yte.sum()}) | features={len(feats)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision",
               "recall": "recall", "f1": "f1", "auc": "roc_auc"}

    resultados = {}
    for nombre, modelo in construir_modelos().items():
        log(f"    · {nombre}")
        cvres = cross_validate(modelo, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1)
        modelo.fit(Xtr, ytr)
        proba_te = modelo.predict_proba(Xte)[:, 1]
        pred_te = (proba_te >= 0.5).astype(int)
        resultados[nombre] = {
            "modelo": modelo,
            "cv": {m: (float(cvres[f"test_{m}"].mean()),
                       float(cvres[f"test_{m}"].std())) for m in scoring},
            "test": {"accuracy": accuracy_score(yte, pred_te),
                     "precision": precision_score(yte, pred_te, zero_division=0),
                     "recall": recall_score(yte, pred_te, zero_division=0),
                     "f1": f1_score(yte, pred_te, zero_division=0),
                     "auc": roc_auc_score(yte, proba_te) if yte.sum() else float("nan")},
            "proba_te": proba_te, "yte": yte,
        }
        c = resultados[nombre]["cv"]
        t = resultados[nombre]["test"]
        print(f"        CV  F1={c['f1'][0]:.3f}±{c['f1'][1]:.3f} | "
              f"AUC={c['auc'][0]:.3f} | TEST F1={t['f1']:.3f} AUC={t['auc']:.3f}")

    # --- Curva ROC comparativa ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    for nombre, r in resultados.items():
        fpr, tpr, _ = roc_curve(r["yte"], r["proba_te"])
        ax.plot(fpr, tpr, linewidth=2, color=MODEL_COLORS.get(nombre, COLORS["cyan"]),
                label=f"{nombre} (AUC={r['test']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    ax.set_title("Curva ROC comparativa — detección de talento oculto")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(loc="lower right")
    savefig(fig, "F06_roc.png", "roc")

    mejor = max(resultados, key=lambda n: (resultados[n]["test"]["auc"],
                                           resultados[n]["test"]["f1"]))
    log(f"    → mejor modelo: {mejor} (AUC={resultados[mejor]['test']['auc']:.3f})")

    # --- Importancia de variables del mejor modelo -------------------------
    modelo = resultados[mejor]["modelo"]
    if hasattr(modelo, "feature_importances_"):
        imp = np.asarray(modelo.feature_importances_, float)
    else:
        imp = np.abs(np.asarray(modelo.coef_, float).ravel())
    orden = np.argsort(imp)[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([feats[i] for i in orden], imp[orden],
            color=gradient_colors(len(orden)), edgecolor="white")
    ax.set_title(f"Importancia de variables — {mejor}")
    ax.set_xlabel("Importancia")
    savefig(fig, "F06_importancia.png", "imp")

    return {"resultados": resultados, "mejor": mejor, "PRE": PRE,
            "feat_names": feats, "df_tr": df_tr, "df_te": df_te,
            "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "imp": imp}


# =============================================================================
# EXTRACCIÓN DEL MODELO PARA EL PREDICTOR PURO
# =============================================================================

def _tree_rf(t, cls1: int) -> list:
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            v = t.value[i][0]
            p1 = float(v[cls1] / v.sum()) if v.sum() > 0 else 0.0
            nodes.append([-1, 0.0, -1, -1, p1])
        else:
            nodes.append([int(t.feature[i]), float(t.threshold[i]),
                          int(t.children_left[i]), int(t.children_right[i]), 0.0])
    return nodes


def _extract_xgb_clf(modelo) -> list:
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
                hijos = {c["nodeid"]: c for c in n["children"]}
                left = add(hijos[n["yes"]])
                right = add(hijos[n["no"]])
                nodes[idx] = [f, float(n["split_condition"]), left, right, 0.0]
            return idx
        add(root)
        trees.append(nodes)
    return trees


def construir_model_spec(nombre: str, modelo, Xtr: np.ndarray) -> dict:
    if nombre == "Regresión Logística":
        return {"type": "logistic", "intercept": float(modelo.intercept_[0]),
                "coef": [float(c) for c in modelo.coef_.ravel()]}
    if nombre == "Random Forest":
        cls1 = list(modelo.classes_).index(1)
        trees = [_tree_rf(est.tree_, cls1) for est in modelo.estimators_]
        return {"type": "ensemble", "combine": "mean", "op": "le",
                "proba_transform": "none", "bias": 0.0, "trees": trees}
    if nombre == "XGBoost":
        trees = _extract_xgb_clf(modelo)
        dm = xgb.DMatrix(Xtr)
        margin = modelo.get_booster().predict(dm, output_margin=True)
        crudo = np.array([_ensemble_sum(list(x), trees) for x in Xtr])
        bias = float(np.mean(margin - crudo))
        return {"type": "ensemble", "combine": "sum", "op": "lt",
                "proba_transform": "sigmoid", "bias": bias, "trees": trees}
    raise ValueError(nombre)


def _ensemble_sum(feats, trees):
    total = 0.0
    for tree in trees:
        i = 0
        while tree[i][0] != -1:
            go_left = feats[tree[i][0]] < tree[i][1]
            i = tree[i][2] if go_left else tree[i][3]
        total += tree[i][4]
    return total


# =============================================================================
# C. ANÁLISIS
# =============================================================================

def analisis(df: pd.DataFrame, meta: dict) -> dict:
    log("SECCIÓN C — Análisis descriptivo")
    res = {}
    tal = df[df["talento_oculto"]]

    # --- C.1 Por colegio (barplot) -----------------------------------------
    if "institucion_educativa" in df.columns:
        g = (df.groupby("institucion_educativa")
             .agg(n=("talento_oculto", "size"),
                  n_tal=("talento_oculto", "sum")).reset_index())
        g = g[g["n"] >= 10]
        g["tasa"] = (g["n_tal"] / g["n"] * 100).round(1)
        res["colegios"] = g.sort_values("n_tal", ascending=False)
        top = g.sort_values("n_tal", ascending=True).tail(15)
        if not top.empty and top["n_tal"].sum() > 0:
            fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(top))))
            ax.barh(top["institucion_educativa"], top["n_tal"],
                    color=gradient_colors(len(top)), edgecolor="white")
            ax.set_title("Talentos ocultos por institución (N ≥ 10)")
            ax.set_xlabel("Nº de talentos ocultos")
            for i, (_, f) in enumerate(top.iterrows()):
                ax.text(f["n_tal"], i, f" {int(f['n_tal'])} de {int(f['n'])} "
                        f"({f['tasa']:.0f}%)", va="center", fontsize=8)
            savefig(fig, "F06_colegios.png", "colegios")

    # --- C.2 Por grado y género --------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, titulo in [(ax1, "grado_escolar", "Grado"),
                            (ax2, "genero", "Género")]:
        if col in tal.columns and not tal.empty:
            vc = tal[col].value_counts().sort_index() if col == "grado_escolar" \
                else tal[col].value_counts()
            ax.bar([str(k) for k in vc.index], vc.values,
                   color=PALETTE[:len(vc)], edgecolor="white")
            ax.set_title(f"Talento oculto por {titulo}")
            ax.set_ylabel("N")
            ax.tick_params(axis="x", rotation=15)
            for i, v in enumerate(vc.values):
                ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    fig.suptitle("C. Talento oculto por grado y género",
                 fontsize=15, fontweight="bold")
    savefig(fig, "F06_grado_genero.png", "grado_genero")
    res["por_grado"] = tal["grado_escolar"].value_counts().sort_index().to_dict()
    res["por_genero"] = tal["genero"].value_counts().to_dict()

    # --- C.3 Perfil comparativo talento oculto vs resto --------------------
    resto = df[~df["talento_oculto"]]
    filas = []
    num_cols = [("puntaje_obtenido", "Puntaje medio"),
                ("indice_potencial", "Índice potencial medio"),
                ("estrato", "Estrato medio"),
                ("n_condiciones_adversas", "Condiciones adversas (media)"),
                ("n_herramientas", "Herramientas (media)"),
                ("n_areas_interes", "Áreas interés (media)")]
    for c, etq in num_cols:
        if c in df.columns:
            filas.append({"Variable": etq,
                          "Talento oculto": round(float(tal[c].mean()), 2),
                          "Resto": round(float(resto[c].mean()), 2)})
    pct_cols = [("computador_en_casa", "% sin computador", lambda s: (s.map(_bin_si) == 0).mean() * 100),
                ("internet_en_casa", "% sin internet", lambda s: (s.map(_bin_si) == 0).mean() * 100),
                ("participacion_olimpiadas", "% sin olimpiadas", lambda s: (s.map(_bin_si) == 0).mean() * 100)]
    for c, etq, fn in pct_cols:
        if c in df.columns:
            filas.append({"Variable": etq,
                          "Talento oculto": round(float(fn(tal[c])), 1),
                          "Resto": round(float(fn(resto[c])), 1)})
    res["perfil"] = pd.DataFrame(filas)

    # --- C.4 Heatmap colegio × grado ---------------------------------------
    if not tal.empty and "institucion_educativa" in tal.columns:
        piv = (tal.pivot_table(index="institucion_educativa", columns="grado_escolar",
                               values="talento_oculto", aggfunc="size")
               .fillna(0).astype(int))
        piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
        if piv.shape[0] > 0:
            fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(piv))))
            sns.heatmap(piv, annot=True, fmt="d", cmap=STEM_SEQ, linewidths=0.5,
                        linecolor="white", cbar_kws={"shrink": 0.7}, ax=ax)
            ax.set_title("Talentos ocultos por institución × grado")
            ax.set_xlabel("Grado")
            ax.set_ylabel("")
            savefig(fig, "F06_heatmap_colegio_grado.png", "heatmap")

    # --- C.5 Top 10 casos destacados ---------------------------------------
    if not tal.empty:
        cols = [c for c in ["numero_documento", "nombres", "apellidos",
                            "institucion_educativa", "grado_escolar", "genero",
                            "puntaje_obtenido", "indice_potencial",
                            "n_condiciones_adversas", "condiciones_detalle"]
                if c in tal.columns]
        res["top10"] = (tal[cols].sort_values("puntaje_obtenido", ascending=False)
                        .head(10))
    return res


# =============================================================================
# D. EXPORTACIÓN
# =============================================================================

def exportar(df: pd.DataFrame, ctx: dict, meta: dict) -> dict:
    log("SECCIÓN D — Exportación (CSV + predictores puros)")
    mejor = ctx["mejor"]
    modelo = ctx["resultados"][mejor]["modelo"]
    PRE, feats = ctx["PRE"], ctx["feat_names"]
    model_spec = construir_model_spec(mejor, modelo, ctx["Xtr"])

    SPEC = {"meta": {"generado": datetime.now().isoformat(timespec="seconds"),
                     "modelo": mejor, "n_total": meta["n_total"],
                     "n_talento": meta["n_talento"]},
            "preprocess": PRE, "model": model_spec,
            "feature_names": feats,
            "p75_puntaje": meta["p75_puntaje"], "indice_alto": meta["indice_alto"],
            "adversas_min": meta["adversas_min"],
            "cond_labels": COND_LABELS}

    # --- Verificación: proba pura == predict_proba del modelo real ----------
    proba_real = modelo.predict_proba(ctx["Xte"])[:, 1]
    proba_pura = np.array([_predict_proba(list(x), model_spec) for x in ctx["Xte"]])
    max_diff = float(np.max(np.abs(proba_real - proba_pura)))
    log(f"    verificación proba pura vs modelo real: máx|Δ| = {max_diff:.3g}")

    # --- CSV de scores ------------------------------------------------------
    out = pd.DataFrame({
        "numero_documento": df["numero_documento"].astype(str),
        "probabilidad_talento": np.round(
            [_predict_proba(_features_clf(r, PRE), model_spec)
             for r in df.to_dict("records")], 4),
        "es_talento_oculto": df["talento_oculto"].astype(bool),
        "n_condiciones_adversas": df["n_condiciones_adversas"].astype(int),
        "condiciones_detalle": df["condiciones_detalle"],
    })
    destino = DEPLOY_DIR / "talento_oculto_scores.csv"
    out.to_csv(destino, index=False, encoding="utf-8-sig")
    log(f"    scores → models/deploy/{destino.name} ({len(out):,} filas)")

    # --- Ejemplo (un talento oculto real, si existe) ------------------------
    tal = df[df["talento_oculto"]]
    fila = (tal.sort_values("puntaje_obtenido", ascending=False).iloc[0]
            if not tal.empty else df.iloc[0])
    ejemplo = _ejemplo_dict(fila)

    generar_predictor_py(SPEC, ejemplo)
    generar_predictor_js(SPEC, ejemplo)
    return {"model_spec": model_spec, "max_diff": max_diff, "SPEC": SPEC,
            "ejemplo": ejemplo, "scores": out}


def _ejemplo_dict(fila) -> dict:
    campos = ["puntaje_obtenido", "indice_potencial", "grado_escolar", "genero",
              "municipio", "tipo_institucion", "estrato", "computador_en_casa",
              "internet_en_casa", "participacion_olimpiadas", "nivel_programacion",
              "nivel_robotica", "interes_prog_robotica", "con_quien_vive",
              "herramientas_conocidas", "areas_interes"]
    ej = {}
    for c in campos:
        if c in fila.index:
            v = fila[c]
            if isinstance(v, float) and v != v:
                ej[c] = None
            elif isinstance(v, float) and c in ("grado_escolar", "estrato"):
                ej[c] = int(v)
            elif isinstance(v, (np.integer,)):
                ej[c] = int(v)
            elif isinstance(v, (np.floating,)):
                ej[c] = float(v)
            else:
                ej[c] = v
    return ej


def generar_predictor_py(SPEC: dict, ejemplo: dict) -> None:
    fns = [_isnan, _to_float, _parse_count, _ord_level, _bin_si, _sigmoid,
           condiciones_adversas, _features_clf, _predict_proba, evaluar_talento]
    fuente = "\n\n".join(inspect.getsource(f) for f in fns)
    js = json.dumps(SPEC, ensure_ascii=True)
    contenido = f'''# -*- coding: utf-8 -*-
"""
Detector PURO de Talento Oculto — Copa STEM 2026 (modelo: {SPEC["meta"]["modelo"]}).
GENERADO por notebooks/06_talento_oculto.py — no editar a mano.

    from talento_oculto_predictor import detectar_talento_oculto
    r = detectar_talento_oculto({{"puntaje_obtenido": 75, "estrato": 1,
                                  "computador_en_casa": "No", ...}})
    # r = {{'probabilidad_talento':.., 'es_talento_oculto':.., ...}}

No requiere sklearn ni librerías de ML: solo la librería estándar (json, math).
"""
import json
import math

SPEC = json.loads(r"""{js}""")


{fuente}


def detectar_talento_oculto(estudiante):
    """Devuelve probabilidad, la bandera determinista y las condiciones adversas."""
    r = evaluar_talento(estudiante, SPEC)
    return {{k: v for k, v in r.items() if not k.startswith("_")}}


if __name__ == "__main__":
    ejemplo = {json.dumps(ejemplo, ensure_ascii=False)}
    import pprint
    pprint.pprint(detectar_talento_oculto(ejemplo))
'''
    (DEPLOY_DIR / "talento_oculto_predictor.py").write_text(contenido, encoding="utf-8")
    log("    predictor Python → models/deploy/talento_oculto_predictor.py")


def generar_predictor_js(SPEC: dict, ejemplo: dict) -> None:
    js_spec = json.dumps(SPEC, ensure_ascii=False)
    ej = json.dumps(ejemplo, ensure_ascii=False)
    contenido = r'''/**
 * Detector PURO de Talento Oculto — Copa STEM 2026.
 * GENERADO por notebooks/06_talento_oculto.py — no editar a mano.
 * Réplica en JavaScript ES6 del predictor Python. Sin dependencias.
 *
 *   import { detectarTalentoOculto } from "./talento_oculto_predictor.js";
 *   const r = detectarTalentoOculto({ puntaje_obtenido: 75, estrato: 1,
 *                                     computador_en_casa: "No" });
 */
const SPEC = __SPEC__;

function _toFloat(v) {
  if (v === null || v === undefined || typeof v === "boolean") return null;
  const f = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isNaN(f) ? null : f;
}

const _isNan = (v) => typeof v === "number" && Number.isNaN(v);

function _parseCount(v) {
  if (v === null || v === undefined) return 0;
  const s = String(v).trim();
  if (s === "" || ["nan", "none", "[]"].includes(s.toLowerCase())) return 0;
  let items = null;
  try { const p = JSON.parse(s); if (Array.isArray(p)) items = p; } catch (e) { items = null; }
  if (items === null) items = s.replace(/[[\]"]/g, "").split(",").filter((x) => x.trim());
  let cnt = 0;
  for (const it of items) {
    const t = String(it).trim().toLowerCase();
    if (t && !["ninguna", "ninguno", "ninguna.", "ninguno."].includes(t)) cnt += 1;
  }
  return cnt;
}

function _ordLevel(v) {
  if (v === null || v === undefined) return null;
  const m = { ninguna: 0, ninguno: 0, "básica": 1, basica: 1, intermedia: 2, avanzada: 3 };
  const s = String(v).trim().toLowerCase();
  return s in m ? m[s] : null;
}

function _binSi(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim().toLowerCase();
  if (["nan", "none", ""].includes(s)) return null;
  if (s.startsWith("s")) return 1;
  if (s.startsWith("n")) return 0;
  return null;
}

function _sigmoid(z) {
  if (z >= 0) return 1.0 / (1.0 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1.0 + e);
}

function condicionesAdversas(raw) {
  const activas = [];
  const est = _toFloat(raw["estrato"]);
  if (est !== null && est <= 2) activas.push("estrato_bajo");
  if (_binSi(raw["computador_en_casa"]) === 0) activas.push("sin_computador");
  if (_binSi(raw["internet_en_casa"]) === 0) activas.push("sin_internet");
  const cqv = raw["con_quien_vive"];
  if (cqv !== null && cqv !== undefined
      && !["nan", "none", ""].includes(String(cqv).trim().toLowerCase())
      && String(cqv).trim() !== "Ambos padres") activas.push("no_ambos_padres");
  if (_binSi(raw["participacion_olimpiadas"]) === 0) activas.push("sin_olimpiadas");
  const prog = raw["nivel_programacion"];
  if (prog !== null && prog !== undefined
      && String(prog).trim().toLowerCase().startsWith("ningun")) activas.push("prog_ninguna");
  return activas;
}

function _featuresClf(raw, PRE) {
  const feats = [];
  for (const name of PRE.numeric) {
    let v = _toFloat(raw[name]);
    if (v === null && name === "n_herramientas") v = _parseCount(raw["herramientas_conocidas"]);
    if (v === null && name === "n_areas_interes") v = _parseCount(raw["areas_interes"]);
    if (v === null) v = PRE.medians[name];
    feats.push(v);
  }
  for (const name of PRE.ordinal) {
    let lv = _ordLevel(raw[name.slice(0, -4)]);
    if (lv === null) lv = PRE.medians[name];
    feats.push(lv);
  }
  for (const name of PRE.binary) {
    let b = _binSi(raw[PRE.binary_src[name]]);
    if (b === null) b = PRE.modes[name];
    feats.push(b);
  }
  for (const col of PRE.onehot_order) {
    let val = raw[col];
    if (val === null || val === undefined || ["nan", "none", ""].includes(String(val).trim().toLowerCase())) {
      val = PRE.onehot_mode[col];
    }
    val = String(val).trim();
    for (const cat of PRE.onehot_cats[col]) feats.push(val === cat ? 1.0 : 0.0);
  }
  return feats;
}

function _predictProba(feats, MODEL) {
  if (MODEL.type === "logistic") {
    let z = MODEL.intercept;
    for (let i = 0; i < MODEL.coef.length; i++) z += MODEL.coef[i] * feats[i];
    return _sigmoid(z);
  }
  const op = MODEL.op;
  let total = 0.0;
  for (const tree of MODEL.trees) {
    let i = 0;
    while (tree[i][0] !== -1) {
      const fi = tree[i][0], thr = tree[i][1];
      const goLeft = op === "lt" ? feats[fi] < thr : feats[fi] <= thr;
      i = goLeft ? tree[i][2] : tree[i][3];
    }
    total += tree[i][4];
  }
  if (MODEL.combine === "mean") total /= MODEL.trees.length;
  if (MODEL.proba_transform === "sigmoid") return _sigmoid(total + MODEL.bias);
  return total;
}

const _round4 = (x) => Math.round(x * 1e4) / 1e4;

export function detectarTalentoOculto(raw) {
  const activas = condicionesAdversas(raw);
  const nAdv = activas.length;
  const punt = _toFloat(raw["puntaje_obtenido"]);
  const indice = _toFloat(raw["indice_potencial"]);
  let alto = false;
  if (punt !== null && punt >= SPEC.p75_puntaje) alto = true;
  if (indice !== null && indice >= SPEC.indice_alto) alto = true;
  const esTalento = Boolean(alto && nAdv >= SPEC.adversas_min);
  const proba = _predictProba(_featuresClf(raw, SPEC.preprocess), SPEC.model);
  return {
    probabilidad_talento: _round4(proba),
    es_talento_oculto: esTalento,
    n_condiciones_adversas: nAdv,
    condiciones_detalle: activas.join("|"),
  };
}

if (typeof process !== "undefined" && Array.isArray(process.argv) && process.argv[1]) {
  const _here = decodeURIComponent(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1");
  const _norm = (p) => p.replace(/\\/g, "/").toLowerCase();
  if (_norm(_here) === _norm(process.argv[1])) console.log(detectarTalentoOculto(__EJEMPLO__));
}
'''
    contenido = contenido.replace("__SPEC__", js_spec).replace("__EJEMPLO__", ej)
    (DEPLOY_DIR / "talento_oculto_predictor.js").write_text(contenido, encoding="utf-8")
    log("    predictor JavaScript → models/deploy/talento_oculto_predictor.js")


# =============================================================================
# INFORME
# =============================================================================

def construir_informe(df, meta, ctx, ana, exp) -> None:
    log("Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    mejor = ctx["mejor"]
    R = REPORT.append

    R("# Detección de Talento Oculto — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Fase 2/3 · Informe: {fecha}\n")
    R("---\n")

    R("## Resumen ejecutivo\n")
    R(dedent(f"""\
        Se identificaron **{meta['n_talento']} estudiantes de talento oculto**
        ({meta['pct']:.1f}% de {meta['n_total']:,}): alto rendimiento **a pesar**
        de condiciones socioeconómicas adversas. Son el retorno social más alto de
        una beca o acompañamiento. Se entrenaron 3 clasificadores; el mejor
        (**{mejor}**, AUC test = {ctx['resultados'][mejor]['test']['auc']:.3f})
        genera una **probabilidad de talento** para priorizar casos límite. La
        réplica pura del modelo (`talento_oculto_predictor.py/.js`) coincide con el
        modelo real dentro de **{exp['max_diff']:.2g}**.\n"""))

    R("## Definición (regla determinista)\n")
    R(dedent(f"""\
        **talento_oculto = alto_rendimiento Y (≥{meta['adversas_min']} condiciones adversas)**

        - **Alto rendimiento** (≥1): `puntaje ≥ P75` (= {meta['p75_puntaje']:.0f})
          **o** `indice_potencial ≥ {meta['indice_alto']}`.
        - **Condiciones adversas** (≥{meta['adversas_min']} de 6): estrato 1-2,
          sin computador, sin internet, no vive con ambos padres, sin olimpiadas
          previas, nivel de programación "Ninguna".

        Los datos faltantes en una condición se tratan como *no adversos*
        (criterio conservador: no se marca talento por falta de información).\n"""))

    R("## Modelos de clasificación\n")
    R(dedent(f"""\
        > **Nota metodológica.** El target es una regla determinista sobre estas
        > mismas variables, por lo que los clasificadores alcanzan métricas muy
        > altas (reconstruyen la regla; fuga de etiqueta esperada). Su valor es
        > (1) la **probabilidad continua** para ordenar casos y (2) confirmar
        > **qué variables pesan** (importancia).\n"""))
    filas = []
    for n, r in ctx["resultados"].items():
        c, t = r["cv"], r["test"]
        filas.append({"Modelo": n + (" ⭐" if n == mejor else ""),
                      "Acc (CV)": f"{c['accuracy'][0]:.3f}",
                      "Prec (CV)": f"{c['precision'][0]:.3f}",
                      "Recall (CV)": f"{c['recall'][0]:.3f}",
                      "F1 (CV)": f"{c['f1'][0]:.3f}",
                      "AUC (CV)": f"{c['auc'][0]:.3f}",
                      "F1 (test)": f"{t['f1']:.3f}",
                      "AUC (test)": f"{t['auc']:.3f}"})
    R(tabla_md(pd.DataFrame(filas)) + "\n")
    R(f"\n{img('roc', 'Curva ROC comparativa')}\n")
    R(f"\n{img('imp', 'Importancia de variables')}\n")

    R("## Análisis descriptivo\n")
    R(f"**Talento oculto por grado:** "
      f"{', '.join(f'{int(k) if isinstance(k,(int,float)) else k}°: {v}' for k, v in ana['por_grado'].items())}.\n")
    R(f"\n**Por género:** "
      f"{', '.join(f'{k}: {v}' for k, v in ana['por_genero'].items())}.\n")
    R(f"\n{img('grado_genero', 'Por grado y género')}\n")
    if "colegios" in FIGURES:
        R(f"\n{img('colegios', 'Talentos por colegio')}\n")
    if "heatmap" in FIGURES:
        R(f"\n{img('heatmap', 'Heatmap colegio × grado')}\n")

    R("### Perfil comparativo: talento oculto vs. resto\n")
    if isinstance(ana.get("perfil"), pd.DataFrame):
        R(tabla_md(ana["perfil"]) + "\n")

    R("### Casos destacados (top 10 por puntaje)\n")
    if isinstance(ana.get("top10"), pd.DataFrame) and not ana["top10"].empty:
        t = ana["top10"].copy()
        ren = {"numero_documento": "Documento", "nombres": "Nombres",
               "apellidos": "Apellidos", "institucion_educativa": "Institución",
               "grado_escolar": "Grado", "genero": "Género",
               "puntaje_obtenido": "Puntaje", "indice_potencial": "Índice",
               "n_condiciones_adversas": "Nº adversas",
               "condiciones_detalle": "Condiciones"}
        t = t.rename(columns={k: v for k, v in ren.items() if k in t.columns})
        R(tabla_md(t) + "\n")

    R("## Exportación para producción\n")
    R(dedent("""\
        - `models/deploy/talento_oculto_scores.csv` — `numero_documento`,
          `probabilidad_talento`, `es_talento_oculto`, `n_condiciones_adversas`,
          `condiciones_detalle`.
        - `models/deploy/talento_oculto_predictor.py` — función pura
          `detectar_talento_oculto(dict)`; sin sklearn.
        - `models/deploy/talento_oculto_predictor.js` — misma función en JS ES6.\n"""))
    R("\n**Ejemplo de entrada (un talento oculto real):**\n")
    R("```json\n" + json.dumps(exp["ejemplo"], ensure_ascii=False, indent=2) + "\n```\n")

    R("## Recomendaciones para la Fundación\n")
    R(dedent(f"""\
        1. **Contactar a los {meta['n_talento']} talentos ocultos** para becas,
           tutoría y rutas STEM: alto potencial que el contexto está frenando.
        2. **Priorizar por probabilidad** (`probabilidad_talento`) y por nº de
           condiciones adversas cuando los recursos sean limitados.
        3. **Focalizar por institución** usando el barplot y el heatmap: algunos
           colegios concentran varios casos.
        4. Tratar la lista como **guía de acción, no veredicto**: validar con los
           colegios (la definición es una regla revisable).\n"""))

    R("## Limitaciones\n")
    R(dedent("""\
        - La **definición es una decisión de política** (umbrales P75, índice ≥ 75,
          ≥2 condiciones); cambiarla cambia la lista.
        - El clasificador **reconstruye la regla** (fuga de etiqueta): sus métricas
          no miden generalización, sino consistencia; la probabilidad sirve para
          ordenar, no como evidencia independiente.
        - Variables **autorreportadas** y ~7% de inscripciones sin datos
          socioeconómicos (tratadas como no adversas).\n"""))
    R("\n---\n_Generado por `notebooks/06_talento_oculto.py` — Copa STEM 2026._\n")

    (REPORTS_DIR / "06_talento_oculto.md").write_text("\n".join(REPORT), encoding="utf-8")
    log("    informe escrito → reports/06_talento_oculto.md")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Detección de Talento Oculto (Fase 2/3)")
    print(" Fundación SapienceLab")
    print("=" * 70)

    df = cargar_datos()
    df, meta = etiquetar_talento(df)
    if meta["n_talento"] < 10:
        log(f"⚠ Muy pocos talentos ocultos ({meta['n_talento']}) para clasificar.")
    ctx = entrenar(df)
    ana = analisis(df, meta)
    exp = exportar(df, ctx, meta)
    construir_informe(df, meta, ctx, ana, exp)

    print("\n" + "=" * 70)
    print(" ✔ TALENTO OCULTO COMPLETADO")
    print(f"   · Talento oculto:    {meta['n_talento']:,} ({meta['pct']:.1f}%)")
    print(f"   · Mejor modelo:      {ctx['mejor']} "
          f"(AUC={ctx['resultados'][ctx['mejor']]['test']['auc']:.3f})")
    print(f"   · Predictor puro:    máx|Δ| = {exp['max_diff']:.3g}")
    print(f"   · Figuras:           {len(FIGURES)} → outputs/")
    print(f"   · Deploy:            models/deploy/talento_oculto_*")
    print(f"   · Informe:           reports/06_talento_oculto.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
