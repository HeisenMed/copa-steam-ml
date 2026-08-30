# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 14: Optimización de hiperparámetros — modelo v2 de producción (Fase 5)
================================================================================

Busca los mejores hiperparámetros del modelo de puntaje sobre
`dataset_C_perfil.csv` (el dataset CON las 5 variables de perfil académico, que
el script 12 demostró superior) y deja el artefacto listo para producción.

Secciones
---------
    A) Preprocesamiento idéntico al del script 03 (medianas/modas del train,
       one-hot), extendido con las 5 variables nuevas.
    B) RandomizedSearchCV sobre RandomForestRegressor (30 candidatos, CV 5).
    C) Evaluación en el hold-out del 20 % — R² y MAE.
    D) Exportación: models/mejor_modelo_puntaje_v2.joblib
                    models/deploy/potencial_stem_predictor_v2.js

Alcance del paso «reentrenar los 4 predictores»
-----------------------------------------------
De los cuatro predictores en producción, **solo uno** es un
RandomForestRegressor y admite esta rejilla de hiperparámetros:

    · potencial STEM   → SÍ. Su componente de rendimiento ES este modelo de
                         puntaje; el resto del índice son fórmulas con pesos
                         fijos. Se re-exporta aquí como _v2.
    · talento oculto   → NO. Es un XGBClassifier (clasificación). `max_features`
                         y `min_samples_leaf` ni siquiera son parámetros suyos.
    · clustering       → NO. Es KMeans (no supervisado). Su único
                         hiperparámetro es k.
    · condiciones      → NO. Es el modelo TEÓRICO del script 10: sus pesos
                         vienen de la literatura educativa y no se entrenan con
                         datos. No tiene hiperparámetros que optimizar.

Reentrenar los tres últimos sobre C sería un cambio de semántica, no una
optimización: cambiaría la definición de los clusters y los umbrales de talento
respecto a los que ya están en `ml_scores`. Queda fuera de este script a
propósito; véase el informe 14.

Formato del export JS
---------------------
Se reutiliza **literalmente** el cuerpo de
`models/deploy/potencial_stem_predictor.js` y solo se sustituye la constante
`SPEC`. Así las funciones de preprocesamiento son las mismas byte a byte y el
contrato con la Edge Function no cambia. Las 5 variables nuevas entran sin tocar
el código JS porque `_featuresPuntaje` itera las listas de `SPEC.puntaje.
preprocess`: basta con que aparezcan en `numeric` / `binary` / `binary_src`.

No modifica ningún fichero existente: todo lo que escribe lleva sufijo `_v2`.
Reproducible: `random_state=42`.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FUENTE = DATA_DIR / "dataset_C_perfil.csv"
JS_V1 = DEPLOY_DIR / "potencial_stem_predictor.js"
OUT_JOBLIB = MODELS_DIR / "mejor_modelo_puntaje_v2.joblib"
OUT_JS = DEPLOY_DIR / "potencial_stem_predictor_v2.js"

TARGET = "puntaje_obtenido"

# --- Bloque de features: el del script 03 + las 5 nuevas --------------------
# El orden importa: es el que replica `_featuresPuntaje` en el JS
# (numeric → ordinal → binary → onehot).
NUMERIC = ["grado_escolar", "estrato", "interes_prog_robotica",
           "n_herramientas", "n_areas_interes",
           # nuevas (4 numéricas: promedio 0–5, horas, y dos Likert 1–5)
           "promedio_academico", "horas_estudio_matematicas",
           "motivacion_participar", "gusto_logica"]
ORDINAL = ["nivel_programacion_ord", "nivel_robotica_ord"]
BINARY = ["computador_bin", "internet_bin", "olimpiadas_bin",
          "clases_extra_bin"]  # nueva
BINARY_SRC = {
    "computador_bin": "computador_en_casa",
    "internet_bin":   "internet_en_casa",
    "olimpiadas_bin": "participacion_olimpiadas",
    "clases_extra_bin": "clases_extra_matematicas",
}
ONEHOT = ["genero", "municipio", "tipo_institucion"]

# Rejilla de búsqueda (la pedida).
GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
}
N_ITER = 30
CV = 5

# Constantes del índice compuesto (idénticas a las del script 04).
PESOS = {"rendimiento": 0.50, "engagement": 0.25, "resiliencia": 0.25}
CATEGORIAS = [
    (85, "Talento destacado"),
    (70, "Alto potencial"),
    (45, "Promedio"),
    (25, "En desarrollo"),
    (0,  "Requiere apoyo"),
]

DOCS_PRUEBA = {"1234", "123456", "123456789", "1234567899"}


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def titulo(txt: str) -> None:
    print("\n" + "=" * 78)
    print(f" {txt}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Transformaciones — misma semántica que el script 03 (y que el JS)
# ---------------------------------------------------------------------------
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
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "[]"):
        return 0
    items = None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            items = parsed
    except Exception:
        items = None
    if items is None:
        items = [x for x in s.strip("[]").replace('"', "").split(",") if x.strip()]
    return sum(1 for it in items
               if str(it).strip().lower() not in
               ("", "ninguna", "ninguno", "ninguna.", "ninguno."))


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
        return 1.0
    if s.startswith("n"):
        return 0.0
    return None


# ---------------------------------------------------------------------------
def cargar() -> pd.DataFrame:
    if not FUENTE.exists():
        print(f"\n  ADVERTENCIA: falta {FUENTE}. Ejecute antes el script 11.\n")
        sys.exit(1)
    df = pd.read_csv(FUENTE, encoding="utf-8", dtype={"numero_documento": str})
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df = df[~df["numero_documento"].isin(DOCS_PRUEBA)]
    df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()]
    return df.reset_index(drop=True)


def ajustar_pre(registros: list[dict]) -> dict:
    """Calcula medianas, modas y categorías one-hot SOBRE EL TRAIN.

    Se guardan dentro del SPEC para que el predictor JS impute exactamente igual
    que el modelo entrenado.
    """
    def col_valores(name):
        vals = []
        for r in registros:
            v = _to_float(r.get(name))
            if v is None and name == "n_herramientas":
                v = _parse_count(r.get("herramientas_conocidas"))
            if v is None and name == "n_areas_interes":
                v = _parse_count(r.get("areas_interes"))
            if v is not None:
                vals.append(v)
        return vals

    medians = {}
    for name in NUMERIC:
        vals = col_valores(name)
        medians[name] = float(np.median(vals)) if vals else 0.0
    for name in ORDINAL:
        src = name[:-4]
        vals = [lv for lv in (_ord_level(r.get(src)) for r in registros)
                if lv is not None]
        medians[name] = float(np.median(vals)) if vals else 0.0

    modes = {}
    for name in BINARY:
        vals = [b for b in (_bin_si(r.get(BINARY_SRC[name])) for r in registros)
                if b is not None]
        modes[name] = float(pd.Series(vals).mode().iloc[0]) if vals else 0.0

    onehot_cats, onehot_mode = {}, {}
    for col in ONEHOT:
        vals = [str(r.get(col)).strip() for r in registros
                if r.get(col) is not None
                and str(r.get(col)).strip().lower() not in ("nan", "none", "")]
        onehot_cats[col] = sorted(set(vals))
        onehot_mode[col] = pd.Series(vals).mode().iloc[0] if vals else ""

    return {
        "numeric": NUMERIC, "ordinal": ORDINAL, "binary": BINARY,
        "binary_src": BINARY_SRC, "onehot_order": ONEHOT,
        "medians": medians, "modes": modes,
        "onehot_mode": onehot_mode, "onehot_cats": onehot_cats,
    }


def features_from_raw(raw: dict, PRE: dict) -> list[float]:
    """Réplica exacta de `_featuresPuntaje` del predictor JS."""
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
        if val is None or str(val).strip().lower() in ("nan", "none", ""):
            val = PRE["onehot_mode"][col]
        val = str(val).strip()
        for cat in PRE["onehot_cats"][col]:
            feats.append(1.0 if val == cat else 0.0)
    return feats


def nombres_features(PRE: dict) -> list[str]:
    nombres = list(PRE["numeric"]) + list(PRE["ordinal"]) + list(PRE["binary"])
    for col in PRE["onehot_order"]:
        nombres += [f"{col}={c}" for c in PRE["onehot_cats"][col]]
    return nombres


def matriz(registros: list[dict], PRE: dict) -> np.ndarray:
    return np.array([features_from_raw(r, PRE) for r in registros], dtype=float)


# ---------------------------------------------------------------------------
def _extract_sklearn_tree(t) -> list:
    """Árbol → lista de nodos [feature, umbral, izq, der, valor]. Idéntico al 03."""
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            nodes.append([-1, 0.0, -1, -1, float(t.value[i].ravel()[0])])
        else:
            nodes.append([int(t.feature[i]), float(t.threshold[i]),
                          int(t.children_left[i]), int(t.children_right[i]), 0.0])
    return nodes


def _extract_rf(modelo) -> dict:
    return {"type": "ensemble", "combine": "mean", "op": "le", "bias": 0.0,
            "trees": [_extract_sklearn_tree(e.tree_) for e in modelo.estimators_]}


def construir_spec(registros: list[dict], PRE: dict, MODEL: dict) -> dict:
    """SPEC del índice compuesto, misma estructura que la del script 04."""
    def conteos(name, key):
        vals = []
        for r in registros:
            c = _to_float(r.get(name))
            if c is None:
                c = _parse_count(r.get(key))
            if c is not None:
                vals.append(c)
        return (min(vals), max(vals)) if vals else (0.0, 1.0)

    lo_h, hi_h = conteos("n_herramientas", "herramientas_conocidas")
    lo_a, hi_a = conteos("n_areas_interes", "areas_interes")

    # Referencia de rendimiento: puntaje real si presentó; si no, el estimado.
    rend = []
    for r in registros:
        real = _to_float(r.get(TARGET))
        rend.append(real if real is not None
                    else _predecir(features_from_raw(r, PRE), MODEL))
    ref = sorted(round(float(x), 4) for x in rend)

    return {
        "meta": {"generado": datetime.now().isoformat(timespec="seconds"),
                 "n_cohorte": len(registros),
                 "modelo_puntaje": "Random Forest v2 (optimizado)",
                 "version": "v2"},
        "puntaje": {"preprocess": PRE, "model": MODEL},
        "engagement": {"n_herramientas": {"lo": float(lo_h), "hi": float(hi_h)},
                       "n_areas_interes": {"lo": float(lo_a), "hi": float(hi_a)}},
        "ref_rendimiento": ref,
        "pesos": PESOS,
        "categorias": [list(c) for c in CATEGORIAS],
    }


def _predecir(feats: list[float], MODEL: dict) -> float:
    total = 0.0
    for tree in MODEL["trees"]:
        i = 0
        while tree[i][0] != -1:
            fi, thr = tree[i][0], tree[i][1]
            i = tree[i][2] if feats[fi] <= thr else tree[i][3]
        total += tree[i][4]
    y = total / len(MODEL["trees"]) + MODEL["bias"]
    return min(100.0, max(0.0, y))


def exportar_js(SPEC: dict) -> None:
    """Reutiliza el cuerpo del JS v1 y sustituye SOLO la constante SPEC.

    Garantiza que las funciones de preprocesamiento sean idénticas byte a byte y
    que el contrato con la Edge Function no cambie.
    """
    if not JS_V1.exists():
        print(f"\n  ADVERTENCIA: falta {JS_V1}; no se puede clonar el formato.\n")
        return

    original = JS_V1.read_text(encoding="utf-8")
    nueva = "const SPEC = " + json.dumps(SPEC, ensure_ascii=False) + ";"
    # La SPEC es una única línea que empieza por `const SPEC = ` y acaba en `;`.
    nuevo, n = re.subn(r"^const SPEC = .*;$", lambda _: nueva, original,
                       count=1, flags=re.MULTILINE)
    if n != 1:
        print("\n  ADVERTENCIA: no se localizó la constante SPEC en el JS v1.\n")
        return

    nuevo = nuevo.replace(
        "GENERADO por notebooks/04_indice_potencial_stem.py — no editar a mano.",
        "GENERADO por notebooks/14_optimizacion_hiperparametros.py — no editar a mano.\n"
        " * MODELO v2: Random Forest con hiperparámetros optimizados, entrenado\n"
        " * sobre dataset_C_perfil.csv (incluye las 5 variables de perfil académico).", 1)

    OUT_JS.write_text(nuevo, encoding="utf-8")
    kb = OUT_JS.stat().st_size / 1024
    log(f"predictor JS v2 → models/deploy/{OUT_JS.name} ({kb:,.0f} KB)")


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(" COPA STEM 2026 — Optimización de hiperparámetros (modelo v2)")
    print(" Fundación SapienceLab · Script 14")
    print("=" * 78)

    # --- Paso 1 -------------------------------------------------------------
    log(f"Paso 1 — Carga de {FUENTE.name}")
    df = cargar()
    registros = df.to_dict("records")
    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    log(f"    {len(df):,} filas con puntaje")

    # Partición ANTES de ajustar el preprocesamiento, para no filtrar el test.
    idx = np.arange(len(df))
    estratos = pd.qcut(pd.Series(y), q=5, labels=False, duplicates="drop")
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_STATE, stratify=estratos)

    reg_tr = [registros[i] for i in idx_tr]
    PRE = ajustar_pre(reg_tr)
    Xtr, Xte = matriz(reg_tr, PRE), matriz([registros[i] for i in idx_te], PRE)
    ytr, yte = y[idx_tr], y[idx_te]
    feat_names = nombres_features(PRE)
    log(f"    train {len(Xtr):,} × {Xtr.shape[1]} features | test {len(Xte):,}")

    # --- Paso 2 -------------------------------------------------------------
    titulo("PASO 2 — RandomizedSearchCV")
    log(f"{N_ITER} candidatos × CV {CV} = {N_ITER * CV} ajustes · scoring='r2'")
    busqueda = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        param_distributions=GRID, n_iter=N_ITER, cv=CV, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    busqueda.fit(Xtr, ytr)

    print("\n  Mejores hiperparámetros:")
    for k, v in sorted(busqueda.best_params_.items()):
        print(f"    {k:<20} {v}")
    print(f"\n  R² CV (mejor candidato): {busqueda.best_score_:+.4f}")

    # --- Paso 3 -------------------------------------------------------------
    titulo("PASO 3 — Evaluación en el hold-out del 20 %")
    pred = busqueda.best_estimator_.predict(Xte)
    r2_te = float(r2_score(yte, pred))
    mae_te = float(mean_absolute_error(yte, pred))
    print(f"  n test : {len(yte):,}")
    print(f"  R²     : {r2_te:+.4f}")
    print(f"  MAE    : {mae_te:.2f}")

    # Referencia: los parámetros por defecto del script 03/12 sobre esta misma
    # partición, para saber cuánto aportó realmente la búsqueda.
    base = RandomForestRegressor(n_estimators=300, max_depth=10,
                                 min_samples_leaf=8, random_state=RANDOM_STATE,
                                 n_jobs=-1).fit(Xtr, ytr)
    pred_b = base.predict(Xte)
    r2_b = float(r2_score(yte, pred_b))
    mae_b = float(mean_absolute_error(yte, pred_b))
    print(f"\n  Referencia (hiperparámetros del script 03, sin optimizar):")
    print(f"    R²  : {r2_b:+.4f}   ({r2_te - r2_b:+.4f} por la búsqueda)")
    print(f"    MAE : {mae_b:.2f}   ({mae_te - mae_b:+.2f})")

    # --- Paso 4 -------------------------------------------------------------
    titulo("PASO 4 — Modelo de producción")
    # Para producción se reajusta con TODAS las filas (más datos), usando los
    # hiperparámetros ganadores. Las métricas reportadas son las del hold-out.
    PRE_full = ajustar_pre(registros)
    X_full = matriz(registros, PRE_full)
    modelo_prod = RandomForestRegressor(
        **busqueda.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
    modelo_prod.fit(X_full, y)
    log(f"reajustado sobre las {len(X_full):,} filas completas")

    bundle = {
        "modelo": modelo_prod,
        "preprocess": PRE_full,
        "feature_names": nombres_features(PRE_full),
        "best_params": busqueda.best_params_,
        "metricas_holdout": {"r2": r2_te, "mae": mae_te, "n_test": len(yte)},
        "version": "v2",
        "entrenado": datetime.now().isoformat(timespec="seconds"),
        "dataset": FUENTE.name,
    }
    joblib.dump(bundle, OUT_JOBLIB)
    log(f"modelo → models/{OUT_JOBLIB.name} "
        f"({OUT_JOBLIB.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Paso 5/6 -----------------------------------------------------------
    titulo("PASO 5/6 — Exportación del predictor JS v2")
    MODEL = _extract_rf(modelo_prod)
    SPEC = construir_spec(registros, PRE_full, MODEL)

    # Verificación: el intérprete de árboles debe reproducir a sklearn.
    muestra = registros[:200]
    dif = max(abs(_predecir(features_from_raw(r, PRE_full), MODEL)
                  - float(np.clip(modelo_prod.predict(
                      matriz([r], PRE_full))[0], 0, 100)))
              for r in muestra)
    log(f"auto-verificación intérprete vs sklearn: máx|Δ| = {dif:.2e}")
    if dif > 1e-6:
        log("    ADVERTENCIA: el intérprete NO reproduce al modelo.")

    exportar_js(SPEC)

    print("\n  Los otros 3 predictores NO se re-exportan:")
    print("    · talento_oculto → XGBClassifier; esta rejilla no le aplica.")
    print("    · clustering     → KMeans; su hiperparámetro es k.")
    print("    · condiciones    → modelo teórico; no se entrena con datos.")
    print("  Reentrenarlos sobre C cambiaría su semántica, no sus")
    print("  hiperparámetros. Véase reports/14_optimizacion_v2.md.")

    resumen = {
        "best_params": busqueda.best_params_,
        "cv_r2": float(busqueda.best_score_),
        "holdout_r2": r2_te, "holdout_mae": mae_te,
        "baseline_r2": r2_b, "baseline_mae": mae_b,
        "n_train": len(Xtr), "n_test": len(Xte), "n_features": Xtr.shape[1],
    }
    (OUTPUTS_DIR / "F14_optimizacion_v2.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"resumen → outputs/F14_optimizacion_v2.json")

    print("\n" + "=" * 78)
    print(" OPTIMIZACIÓN COMPLETADA — ningún fichero existente fue modificado")
    print("=" * 78)


if __name__ == "__main__":
    main()
