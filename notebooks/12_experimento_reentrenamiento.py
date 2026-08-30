# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 12: Experimento de reentrenamiento (Fase 5)
================================================================================

Entrena la MISMA familia de modelo, con el MISMO protocolo de validación, sobre
los cuatro datasets que produce `11_preparar_experimento_reentrenamiento.py`, y
compara R², MAE, n_samples y n_features.

Diseño del experimento
----------------------
    A  (baseline)  — cohorte original del primer modelo.
    B  (completo)  — todas las filas con puntaje.
    C  (perfil)    — filas con perfil académico, CON las 5 variables nuevas.
    C' (control)   — las MISMAS filas de C, SIN las 5 variables nuevas.

    A vs B   → ¿ayuda acumular más inscritos?          (varía la muestra)
    C vs C'  → ¿aportan las 5 variables nuevas?        (muestra FIJA)

C vs C' es la comparación decisiva: al mantener las filas y la partición
train/test idénticas, la diferencia de métricas se atribuye únicamente al bloque
de variables nuevas y no a un cambio de población.

Protocolo (idéntico al del script 03, para que la línea base sea comparable)
---------------------------------------------------------------------------
    · Features: mismas transformaciones que 03 (conteos de listas, ordinales
      0–3, binarias Sí/No, one-hot) — SIN telemetría, porque se mide durante el
      examen y no está disponible al momento de predecir, y SIN `porcentaje`,
      que es un duplicado exacto del target.
    · Imputación ajustada SOLO con el train (mediana / moda) dentro de un
      Pipeline, para no filtrar información del test.
    · Partición: hold-out 20 % estratificado por quintil de puntaje.
    · Validación: KFold(5, shuffle) sobre el train + evaluación en el hold-out.
    · Modelo: RandomForestRegressor(300, max_depth=10, min_samples_leaf=8).

Reproducible: `random_state=42`.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "puntaje_obtenido"

# Los cuatro datasets del experimento: (clave, archivo, rol, incluir_nuevas).
#
# `incluir_nuevas` es explícito y NO se deduce de las columnas del archivo: A y B
# salen del mismo export y arrastran físicamente las 5 columnas nuevas (casi
# todas vacías). Si se usaran, el contraste A→B mezclaría "más datos" con "más
# variables" y dejaría de estar controlado. Solo C las activa; A, B y C' corren
# con el MISMO bloque de features base.
DATASETS = [
    ("A_baseline",     "dataset_A_baseline.csv",     "línea base (cohorte original)", False),
    ("B_completo",     "dataset_B_completo.csv",     "efecto de más DATOS",           False),
    ("C_perfil",       "dataset_C_perfil.csv",       "efecto de más VARIABLES",       True),
    ("C_sin_features", "dataset_C_sin_features.csv", "control de C (mismas filas)",   False),
]

# --- Bloque de features base (réplica de la configuración del script 03) -----
NUMERIC = ["grado_escolar", "estrato", "interes_prog_robotica",
           "n_herramientas", "n_areas_interes"]
ORDINAL = ["nivel_programacion_ord", "nivel_robotica_ord"]
BINARY = ["computador_bin", "internet_bin", "olimpiadas_bin"]
ONEHOT = ["genero", "municipio", "tipo_institucion"]

# --- Bloque de features NUEVAS (solo existen en C) ---------------------------
# 4 numéricas (promedio 0–5, horas, y dos Likert 1–5) + 1 binaria Sí/No.
NUEVAS_NUM = ["promedio_academico", "horas_estudio_matematicas",
              "motivacion_participar", "gusto_logica"]
NUEVAS_BIN = ["clases_extra_bin"]
NUEVAS_BIN_SRC = {"clases_extra_bin": "clases_extra_matematicas"}

BINARY_SRC = {
    "computador_bin": "computador_en_casa",
    "internet_bin":   "internet_en_casa",
    "olimpiadas_bin": "participacion_olimpiadas",
}

MODELO_KWARGS = dict(n_estimators=300, max_depth=10, min_samples_leaf=8,
                     random_state=RANDOM_STATE, n_jobs=-1)


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def titulo(txt: str) -> None:
    print("\n" + "=" * 78)
    print(f" {txt}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Transformaciones — misma semántica que el script 03
# ---------------------------------------------------------------------------
def _parse_count(v) -> int:
    """Cuenta elementos de una lista JSON (o CSV) ignorando 'Ninguna/Ninguno'."""
    if v is None:
        return 0
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
    """Ninguna/Básica/Intermedia/Avanzada → 0/1/2/3 (NaN si desconocido)."""
    if v is None:
        return np.nan
    m = {"ninguna": 0, "ninguno": 0, "básica": 1, "basica": 1,
         "intermedia": 2, "avanzada": 3}
    return m.get(str(v).strip().lower(), np.nan)


def _bin_si(v):
    """Sí* → 1, No* → 0 (NaN si desconocido)."""
    if v is None:
        return np.nan
    s = str(v).strip().lower()
    if s in ("nan", "none", ""):
        return np.nan
    if s.startswith("s"):
        return 1.0
    if s.startswith("n"):
        return 0.0
    return np.nan


def construir_crudos(df: pd.DataFrame, incluir_nuevas: bool
                     ) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Deriva las columnas del modelo. Devuelve (X, numéricas, binarias, one-hot).

    El bloque de 5 variables nuevas se añade solo si `incluir_nuevas` es True Y
    las columnas existen. La condición es explícita, no inferida del archivo:
    A y B las traen físicamente pero deben correr sin ellas para que su contraste
    aísle el tamaño de muestra.
    """
    X = pd.DataFrame(index=df.index)

    # Numéricas directas.
    for col in ("grado_escolar", "estrato", "interes_prog_robotica"):
        X[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    # Conteos de listas.
    X["n_herramientas"] = (df["herramientas_conocidas"].map(_parse_count)
                           if "herramientas_conocidas" in df.columns else 0)
    X["n_areas_interes"] = (df["areas_interes"].map(_parse_count)
                            if "areas_interes" in df.columns else 0)

    # Ordinales 0–3.
    for ordcol in ORDINAL:
        src = ordcol[:-4]  # quita el sufijo "_ord"
        X[ordcol] = df[src].map(_ord_level) if src in df.columns else np.nan

    # Binarias Sí/No.
    for bincol, src in BINARY_SRC.items():
        X[bincol] = df[src].map(_bin_si) if src in df.columns else np.nan

    numericas = NUMERIC + ORDINAL
    binarias = list(BINARY)
    onehot = [c for c in ONEHOT if c in df.columns]
    for col in onehot:
        X[col] = df[col].astype(str).str.strip().replace(
            {"nan": np.nan, "None": np.nan, "": np.nan})

    # --- Bloque nuevo, solo cuando el experimento lo pide -------------------
    if incluir_nuevas:
        for col in NUEVAS_NUM:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce")
                numericas.append(col)
        for bincol, src in NUEVAS_BIN_SRC.items():
            if src in df.columns:
                X[bincol] = df[src].map(_bin_si)
                binarias.append(bincol)

    return X, numericas, binarias, onehot


def construir_pipeline(numericas, binarias, onehot) -> Pipeline:
    """Imputación + one-hot + Random Forest, todo dentro del Pipeline para que
    la imputación se ajuste solo con el train de cada fold."""
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numericas),
        ("bin", SimpleImputer(strategy="most_frequent"), binarias),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), onehot),
    ])
    return Pipeline([("pre", pre), ("rf", RandomForestRegressor(**MODELO_KWARGS))])


# ---------------------------------------------------------------------------
def estratos(y: pd.Series) -> pd.Series | None:
    """Quintiles de puntaje para estratificar la partición (como en 03)."""
    try:
        q = pd.qcut(y, q=5, labels=False, duplicates="drop")
    except ValueError:
        return None
    return q if q.nunique() > 1 else None


def evaluar(clave: str, archivo: str, rol: str, incluir_nuevas: bool) -> dict | None:
    ruta = DATA_DIR / archivo
    if not ruta.exists():
        log(f"{clave:<15} FALTA data/{archivo} — ejecute antes el script 11")
        return None

    df = pd.read_csv(ruta, encoding="utf-8", dtype={"numero_documento": str})
    df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")

    X, numericas, binarias, onehot = construir_crudos(df, incluir_nuevas)

    # Hold-out 20 % estratificado. Con las mismas filas y el mismo random_state,
    # C y C' reciben EXACTAMENTE la misma partición (eso es lo que los hace
    # comparables como tratamiento y control).
    idx_tr, idx_te = train_test_split(
        X.index, test_size=0.20, random_state=RANDOM_STATE, stratify=estratos(y))
    Xtr, Xte = X.loc[idx_tr], X.loc[idx_te]
    ytr, yte = y.loc[idx_tr], y.loc[idx_te]

    modelo = construir_pipeline(numericas, binarias, onehot)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cvres = cross_validate(modelo, Xtr, ytr, cv=cv,
                           scoring={"r2": "r2", "mae": "neg_mean_absolute_error"})

    modelo.fit(Xtr, ytr)
    pred = modelo.predict(Xte)

    # Dimensión real de entrada al modelo (one-hot ya expandido).
    n_feat = modelo.named_steps["pre"].transform(Xtr.head(1)).shape[1]

    res = {
        "dataset": clave,
        "rol": rol,
        "n_samples": len(df),
        "n_predictores": len(numericas) + len(binarias) + len(onehot),
        "n_features": int(n_feat),
        "cv_r2": float(cvres["test_r2"].mean()),
        "cv_r2_std": float(cvres["test_r2"].std()),
        "cv_mae": float(-cvres["test_mae"].mean()),
        "test_r2": float(r2_score(yte, pred)),
        "test_mae": float(mean_absolute_error(yte, pred)),
        "nuevas_incluidas": incluir_nuevas,
    }
    log(f"{clave:<15} n={res['n_samples']:>5,}  feats={res['n_features']:>3}  "
        f"CV R²={res['cv_r2']:+.3f}  CV MAE={res['cv_mae']:.2f}  "
        f"test R²={res['test_r2']:+.3f}  test MAE={res['test_mae']:.2f}")
    return res


# ---------------------------------------------------------------------------
def imprimir_tabla(res: list[dict]) -> None:
    cab = ["Dataset", "Rol", "n_samples", "n_features",
           "CV R²", "CV MAE", "test R²", "test MAE"]
    filas = [[r["dataset"], r["rol"], f"{r['n_samples']:,}", str(r["n_features"]),
              f"{r['cv_r2']:+.3f}", f"{r['cv_mae']:.2f}",
              f"{r['test_r2']:+.3f}", f"{r['test_mae']:.2f}"] for r in res]
    anchos = [max(len(cab[i]), *(len(f[i]) for f in filas)) for i in range(len(cab))]
    sep = "  ".join("-" * a for a in anchos)
    print("  " + "  ".join(c.ljust(anchos[i]) for i, c in enumerate(cab)))
    print("  " + sep)
    for f in filas:
        celdas = [f[0].ljust(anchos[0]), f[1].ljust(anchos[1])]
        celdas += [f[i].rjust(anchos[i]) for i in range(2, len(cab))]
        print("  " + "  ".join(celdas))
    print("  " + sep)


def contraste(res: dict[str, dict], a: str, b: str, pregunta: str) -> None:
    if a not in res or b not in res:
        return
    ra, rb = res[a], res[b]
    d_r2 = rb["cv_r2"] - ra["cv_r2"]
    d_mae = rb["cv_mae"] - ra["cv_mae"]
    print(f"\n  {pregunta}")
    print(f"    {a} → {b}")
    print(f"      CV R²  : {ra['cv_r2']:+.3f} → {rb['cv_r2']:+.3f}  ({d_r2:+.3f})")
    print(f"      CV MAE : {ra['cv_mae']:.2f} → {rb['cv_mae']:.2f}  ({d_mae:+.2f} pts)")
    # El MAE baja cuando mejora; el R² sube.
    if d_r2 > 0.01 and d_mae < 0:
        print("      → mejora")
    elif d_r2 < -0.01 and d_mae > 0:
        print("      → empeora")
    else:
        print("      → sin efecto apreciable")


def comparar_familias() -> list[dict] | None:
    """Compara Random Forest con una regresión lineal y un XGBoost sobre C, con
    el MISMO protocolo, para que la elección de modelo sea empírica y no un
    argumento teórico. Las configuraciones son las del script 03.

    Ojo: esto NO es una búsqueda de hiperparámetros ni una selección de modelo
    por métrica. Es la evidencia de cuánto se está sacrificando —si es que se
    sacrifica algo— al preferir el modelo interpretable.
    """
    ruta = DATA_DIR / "dataset_C_perfil.csv"
    if not ruta.exists():
        return None

    df = pd.read_csv(ruta, encoding="utf-8", dtype={"numero_documento": str})
    df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X, numericas, binarias, onehot = construir_crudos(df, incluir_nuevas=True)

    familias = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(**MODELO_KWARGS),
    }
    try:
        from xgboost import XGBRegressor
        familias["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    except ImportError:
        log("    XGBoost no disponible: se omite de la comparación.")

    idx_tr, _ = train_test_split(
        X.index, test_size=0.20, random_state=RANDOM_STATE, stratify=estratos(y))
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    out = []
    for nombre, estimador in familias.items():
        pre = construir_pipeline(numericas, binarias, onehot).named_steps["pre"]
        pipe = Pipeline([("pre", pre), ("mod", estimador)])
        r = cross_validate(pipe, X.loc[idx_tr], y.loc[idx_tr], cv=cv,
                           scoring={"r2": "r2", "mae": "neg_mean_absolute_error"})
        out.append({"modelo": nombre,
                    "cv_r2": float(r["test_r2"].mean()),
                    "cv_r2_std": float(r["test_r2"].std()),
                    "cv_mae": float(-r["test_mae"].mean())})
    return out


def descomponer_bloques() -> list[dict] | None:
    """Sobre las MISMAS filas de C, mide cuánto explica cada bloque por separado:
    solo socioeconómicas, solo las 5 académicas, y ambas juntas.

    Responde a la pregunta de fondo del experimento: ¿el rendimiento se explica
    mejor por el ORIGEN socioeconómico o por el COMPORTAMIENTO académico? El
    contraste C vs C' dice que las nuevas aportan, pero no cuánto aportan
    comparadas con el bloque socioeconómico; esto sí.
    """
    ruta = DATA_DIR / "dataset_C_perfil.csv"
    if not ruta.exists():
        return None

    df = pd.read_csv(ruta, encoding="utf-8", dtype={"numero_documento": str})
    df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X, numericas, binarias, onehot = construir_crudos(df, incluir_nuevas=True)

    socio_num = [c for c in numericas if c not in NUEVAS_NUM]
    socio_bin = [c for c in binarias if c not in NUEVAS_BIN]

    configs = [
        ("solo socioeconómicas", socio_num, socio_bin, onehot),
        ("solo académicas (5)",  NUEVAS_NUM, NUEVAS_BIN, []),
        ("ambas",                numericas, binarias, onehot),
    ]

    # Misma partición y mismos folds que el resto del experimento.
    idx_tr, _ = train_test_split(
        X.index, test_size=0.20, random_state=RANDOM_STATE, stratify=estratos(y))
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    out = []
    for nombre, num, binr, oh in configs:
        pipe = construir_pipeline(num, binr, oh)
        r = cross_validate(pipe, X.loc[idx_tr], y.loc[idx_tr], cv=cv,
                           scoring={"r2": "r2", "mae": "neg_mean_absolute_error"})
        pipe.fit(X.loc[idx_tr], y.loc[idx_tr])
        n_feat = pipe.named_steps["pre"].transform(X.head(1)).shape[1]
        out.append({"bloque": nombre, "n_features": int(n_feat),
                    "cv_r2": float(r["test_r2"].mean()),
                    "cv_mae": float(-r["test_mae"].mean())})
    return out


def main() -> None:
    print("=" * 78)
    print(" COPA STEM 2026 — Experimento de reentrenamiento")
    print(" Fundación SapienceLab · Script 12")
    print("=" * 78)
    print(" Modelo: RandomForestRegressor(n_estimators=300, max_depth=10,")
    print("         min_samples_leaf=8) · KFold(5) + hold-out 20 % estratificado")

    titulo("Entrenamiento de los cuatro modelos")
    resultados = [r for r in (evaluar(*d) for d in DATASETS) if r is not None]

    if not resultados:
        print("\n  No hay datasets que evaluar. Ejecute antes el script 11.\n")
        sys.exit(1)

    titulo("Tabla comparativa")
    imprimir_tabla(resultados)
    print("  n_features = dimensión de entrada al modelo (one-hot ya expandido).")
    print("  CV = media de 5 folds sobre el train; test = hold-out del 20 %.")

    por_clave = {r["dataset"]: r for r in resultados}

    titulo("Contrastes del experimento")
    contraste(por_clave, "A_baseline", "B_completo",
              "¿Ayuda acumular más inscritos? (varía la muestra)")
    contraste(por_clave, "C_sin_features", "C_perfil",
              "¿Aportan las 5 variables nuevas? (muestra FIJA — decisivo)")

    # A y B son poblaciones distintas: el MAE está en puntos del target, así que
    # una población menos dispersa produce un MAE más bajo sin que el modelo sea
    # mejor. El R² sí se normaliza por la varianza de cada muestra, y es el único
    # comparable entre poblaciones. En C vs C' la muestra es la misma y ambas
    # métricas son directamente comparables.
    print("\n  Nota: entre A y B cambia la población, así que el MAE no es")
    print("  comparable (depende de la dispersión del target). Use el R² para")
    print("  ese contraste. En C vs C' la muestra es fija: ambas métricas valen.")

    # Verificación de que C y C' son realmente tratamiento y control.
    if {"C_perfil", "C_sin_features"} <= por_clave.keys():
        c, cs = por_clave["C_perfil"], por_clave["C_sin_features"]
        if c["n_samples"] != cs["n_samples"]:
            print(f"\n  OJO: C y C' no tienen las mismas filas "
                  f"({c['n_samples']:,} vs {cs['n_samples']:,}); "
                  f"el contraste no es un control limpio.")
        else:
            print(f"\n  Control verificado: C y C' comparten las mismas "
                  f"{c['n_samples']:,} filas y la misma partición; "
                  f"solo difieren en {c['n_features'] - cs['n_features']} features.")

    # --- ¿Es Random Forest la elección correcta? ----------------------------
    familias = comparar_familias()
    if familias:
        titulo("Comparación de familias de modelo (dataset C, mismo protocolo)")
        print(f"  {'Modelo':<20} {'CV R²':>8} {'±':>6} {'CV MAE':>8}")
        print(f"  {'-' * 20} {'-' * 8} {'-' * 6} {'-' * 8}")
        for f in familias:
            print(f"  {f['modelo']:<20} {f['cv_r2']:>+8.3f} "
                  f"{f['cv_r2_std']:>6.3f} {f['cv_mae']:>8.2f}")
        mejor = max(familias, key=lambda f: f["cv_r2"])
        rf = next((f for f in familias if f["modelo"] == "Random Forest"), None)
        if rf and mejor["modelo"] != "Random Forest":
            print(f"\n  {mejor['modelo']} supera a Random Forest por "
                  f"{mejor['cv_r2'] - rf['cv_r2']:+.3f} de R². Ver el informe 12:")
            print("  la diferencia se pondera contra la pérdida de interpretabilidad.")
        salida_f = OUTPUTS_DIR / "F12_comparacion_familias.csv"
        pd.DataFrame(familias).to_csv(salida_f, index=False, encoding="utf-8")
        print(f"\n>>> comparación guardada → outputs/{salida_f.name}")

    # --- ¿Origen socioeconómico o comportamiento académico? -----------------
    bloques = descomponer_bloques()
    if bloques:
        titulo("Descomposición por bloque (mismas filas de C)")
        print(f"  {'Bloque':<22} {'features':>8} {'CV R²':>8} {'CV MAE':>8}")
        print(f"  {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 8}")
        for b in bloques:
            print(f"  {b['bloque']:<22} {b['n_features']:>8} "
                  f"{b['cv_r2']:>+8.3f} {b['cv_mae']:>8.2f}")
        socio, acad = bloques[0], bloques[1]
        if acad["cv_r2"] > socio["cv_r2"]:
            print(f"\n  {acad['n_features']} variables de comportamiento académico explican MÁS")
            print(f"  que {socio['n_features']} de origen socioeconómico "
                  f"({acad['cv_r2']:+.3f} vs {socio['cv_r2']:+.3f}).")
        salida_b = OUTPUTS_DIR / "F12_descomposicion_bloques.csv"
        pd.DataFrame(bloques).to_csv(salida_b, index=False, encoding="utf-8")
        print(f"\n>>> descomposición guardada → outputs/{salida_b.name}")

    salida = OUTPUTS_DIR / "F12_experimento_reentrenamiento.csv"
    pd.DataFrame(resultados).to_csv(salida, index=False, encoding="utf-8")
    print(f">>> resultados guardados → outputs/{salida.name}")

    print("\n" + "=" * 78)
    print(" EXPERIMENTO COMPLETADO")
    print("=" * 78)


if __name__ == "__main__":
    main()
