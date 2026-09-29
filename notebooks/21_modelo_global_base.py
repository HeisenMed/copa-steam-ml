# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 21: Modelo global base (sin perfil académico) — reemplazo de v1
================================================================================

Entrena UN solo modelo de producción que puntúa a todos los estudiantes de los
tres municipios con el MISMO conjunto de variables: el bloque base del dataset A
(sin las 5 preguntas de perfil académico). Motivo: comparabilidad entre
municipios. La cobertura del perfil es 98 % en Bello, 37 % en Girardota y 3 % en
Copacabana; un modelo que dependa del perfil solo puede servir a una fracción
sesgada de la población, y un enrutamiento por municipio mete en la misma
columna números de procedencia distinta.

Sin offset por municipio, sin constante de calibración, sin caso especial para
Copacabana: el efecto de cada municipio lo aprende el modelo de sus propios
estudiantes. v2 queda intacto como artefacto de investigación.

Pasos
-----
    1. Inventario: carga del export de agosto, retiro de PII, documentos de
       prueba, deduplicación por documento (se conserva la sesión completada,
       identificada por `tiempo_usado_segundos`) y exclusión de los resultados
       descartados por sospecha (lista privada, solo en memoria).
    2. Features: las de `12_experimento_reentrenamiento.py` vía importlib, igual
       que el script 13. Bloque base, `incluir_nuevas=False`.
    3. Métricas: out-of-fold (KFold 5) sobre TODO el conjunto + hold-out 20 %
       estratificado. Modelo final ajustado con todas las filas.
    4. Efecto municipio: residuos medios por grupo y dependencia parcial sobre
       `municipio`. Es un HALLAZGO; no se fuerza ningún orden.
    5. Limitaciones de v2 (solo lectura): cruce de `dataset_C_perfil.csv` con la
       lista de descartados.

Privacidad
----------
`nombres`, `apellidos` e `institucion_educativa` se eliminan al cargar.
`numero_documento` se usa solo como clave en memoria (deduplicar y excluir) y se
elimina antes de construir features. La lista de descartados se lee de fuera del
repo y no se copia ni se escribe en ninguna salida: solo se reportan conteos
agregados por municipio.

Salidas (todas nuevas)
----------------------
    models/modelo_global_base.joblib   — pipeline + métricas embebidas
    outputs/F21_modelo_global_base.json

Reproducible: `random_state=42`.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    from sklearn.model_selection import (KFold, cross_val_predict,
                                         train_test_split)
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

FUENTE = DATA_DIR / "copa_stem_dataset_2026-08.csv.csv"
FUENTE_C_V2 = DATA_DIR / "dataset_C_perfil.csv"  # entrenamiento de v2 (lectura)
# Población de v1 (1,748 con puntaje real); solo se lee `puntaje_real`.
V1_SCORES = MODELS_DIR / "deploy" / "puntaje_estimado.csv"
# Lista privada de resultados descartados. Vive FUERA del repo a propósito.
DESCARTADOS = BASE_DIR.parent / "_datos_privados" / "descartados.csv"

OUT_JOBLIB = MODELS_DIR / "modelo_global_base.joblib"
OUT_JSON = OUTPUTS_DIR / "F21_modelo_global_base.json"

# Misma ingeniería de features que el script 12 (y que el 13): duplicarla aquí
# arriesgaría que el modelo de producción y el experimento se desincronicen.
_spec = importlib.util.spec_from_file_location(
    "s12", Path(__file__).resolve().parent / "12_experimento_reentrenamiento.py")
s12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s12)

TARGET = s12.TARGET
PII = ["nombres", "apellidos", "institucion_educativa"]
DOCS_PRUEBA = {"1234", "123456", "123456789", "1234567899"}
MUNICIPIOS = ["Bello", "Girardota", "Copacabana"]

# Cifras verificadas en Supabase (entregadas con el encargo; no se re-consultan).
ESPERADO = {
    "Bello":      {"total": 729,  "con_perfil": 716, "sin_perfil": 13},
    "Girardota":  {"total": 1088, "con_perfil": 412, "sin_perfil": 676},
    "Copacabana": {"total": 1233, "con_perfil": 0,   "sin_perfil": 1233},
}

# Métrica oficial de v1 (reports/20_reconciliacion_metricas_v1.md).
V1_OFICIAL = {"r2_oof": 0.084, "mae": 18.1, "rmse": 22.1,
              "r2_ic95": [0.053, 0.116], "n": 1748,
              "poblacion": "1,748 estudiantes de Copacabana y Girardota",
              "particion": "out-of-fold KFold(5)"}

N_BOOT = 2000

# Duración por encima de la cual una sesión no pudo ser un examen completado
# por el estudiante: 3 h, el doble del p99 de Copacabana (~5,400 s).
UMBRAL_SESION_ABANDONADA_S = 3 * 3600


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def titulo(txt: str) -> None:
    print("\n" + "=" * 78)
    print(f" {txt}")
    print("=" * 78)


def por_municipio(df: pd.DataFrame) -> dict[str, int]:
    c = df["municipio"].value_counts()
    return {m: int(c.get(m, 0)) for m in MUNICIPIOS}


# ---------------------------------------------------------------------------
# Paso 1 — Inventario y limpieza
# ---------------------------------------------------------------------------
def cargar_descartados() -> set[str]:
    if not DESCARTADOS.exists():
        print(f"\n  ALTO: no existe la lista de descartados:\n     {DESCARTADOS}")
        print("     Sin ella no se puede excluir `sospechoso_descartado`.\n")
        sys.exit(1)
    d = pd.read_csv(DESCARTADOS, dtype=str, encoding="utf-8-sig")
    d.columns = [c.strip().lower() for c in d.columns]
    if "numero_documento" not in d.columns:
        print(f"\n  ALTO: {DESCARTADOS.name} no tiene la columna numero_documento "
              f"(columnas: {list(d.columns)}).\n")
        sys.exit(1)
    docs = d["numero_documento"].astype(str).str.strip()
    docs = docs[~docs.isin(["", "nan", "None"])]
    log(f"lista de descartados: {len(docs):,} filas, {docs.nunique():,} "
        f"documentos distintos (solo en memoria)")
    return set(docs)


def inventario(descartados: set[str]) -> tuple[pd.DataFrame, dict]:
    titulo("PASO 1 — Inventario de datos")
    if not FUENTE.exists():
        print(f"\n  ALTO: falta {FUENTE}.\n")
        sys.exit(1)

    df = pd.read_csv(FUENTE, encoding="utf-8", dtype={"numero_documento": str})
    # La institución solo hace falta para una prueba intra-institución (§ análisis
    # complementario 3). Se reemplaza al cargar por un código entero anónimo y
    # el nombre se elimina; el código nunca sale de memoria.
    if "institucion_educativa" in df.columns:
        df["_inst"] = pd.factorize(df["institucion_educativa"])[0]
    pii_presentes = [c for c in PII if c in df.columns]
    df = df.drop(columns=pii_presentes)
    log(f"{FUENTE.name}: {len(df):,} filas; PII retirada al cargar: "
        f"{pii_presentes or 'ninguna'}")

    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["_perfil"] = pd.to_numeric(df["promedio_academico"],
                                  errors="coerce").notna()

    inv: dict = {"fuente": FUENTE.name, "pii_retirada_al_cargar": pii_presentes}

    # Verificaciones de la escala del target.
    pct = pd.to_numeric(df["porcentaje"], errors="coerce")
    inv["target_nulos"] = int(df[TARGET].isna().sum())
    inv["porcentaje_igual_target"] = bool((pct == df[TARGET]).all())
    inv["target_min_max"] = [float(df[TARGET].min()), float(df[TARGET].max())]
    log(f"target nulos={inv['target_nulos']}  porcentaje==puntaje en todas las "
        f"filas: {inv['porcentaje_igual_target']}  rango={inv['target_min_max']}")

    df = df[df[TARGET].notna()].copy()
    inv["crudo"] = por_municipio(df)
    if inv["crudo"]["Copacabana"] == 0:
        print("\n  ALTO: no hay filas puntuadas de Copacabana en local.\n")
        sys.exit(1)

    # Documentos de prueba (regla estándar del proyecto).
    n0 = len(df)
    prueba = df["numero_documento"].isin(DOCS_PRUEBA)
    inv["prueba_retirados"] = por_municipio(df[prueba])
    df = df[~prueba].copy()
    log(f"documentos de prueba retirados: {n0 - len(df)}")

    # Deduplicación: se conserva la sesión COMPLETADA. El export no trae estado,
    # fecha de finalización ni número de preguntas respondidas; la única señal
    # de sesión es `tiempo_usado_segundos`. En cada par duplicado (mismo
    # created_at) una fila registra 51–56 días de duración — imposible para un
    # examen (p99 de Copacabana ≈ 1.5 h): es una sesión que el estudiante no
    # cerró y que el sistema cerró después. Esa fila se retira. La regla NO mira
    # el puntaje; si un documento no queda con exactamente una fila, el script
    # se detiene en vez de caer en otra regla (ni máximo ni keep='first').
    dup_mask = df.duplicated("numero_documento", keep=False)
    pares = df[dup_mask]
    first_elige_menor = int(
        pares.groupby("numero_documento")[TARGET]
        .apply(lambda s: s.iloc[0] < s.max()).sum()) if len(pares) else 0
    n_antes, muni_antes = len(df), por_municipio(df)
    tiempo = pd.to_numeric(df["tiempo_usado_segundos"], errors="coerce")
    abandonada = dup_mask & (tiempo > UMBRAL_SESION_ABANDONADA_S)
    df = df[~abandonada].copy()
    restantes = df.loc[df["numero_documento"].isin(
        pares["numero_documento"]), "numero_documento"].value_counts()
    if len(restantes) != pares["numero_documento"].nunique() or (restantes != 1).any():
        print("\n  ALTO: la regla de sesión completada no deja exactamente una "
              "fila por documento duplicado. Revise los pares a mano.\n")
        sys.exit(1)
    muni_despues = por_municipio(df)
    inv["dedup"] = {
        "regla": ("sesión completada: en cada documento duplicado se retira la "
                  f"fila con tiempo_usado_segundos > {UMBRAL_SESION_ABANDONADA_S:,} "
                  "(sesión no cerrada por el estudiante); la regla no usa el "
                  "puntaje"),
        "columnas_de_sesion_disponibles": [
            c for c in ("estado", "finalizada_en", "preguntas_respondidas",
                        "tiempo_usado_segundos", "cambios_pestana",
                        "intentos_copiar", "intentos_pegar",
                        "intentos_click_derecho", "created_at")
            if c in df.columns],
        "tiempo_filas_retiradas_s": sorted(
            float(v) for v in tiempo[abandonada]),
        "puntaje_filas_retiradas": sorted(
            float(v) for v in pares.loc[abandonada[dup_mask], TARGET]),
        "coincide_con_maximo_puntaje": bool(
            (df.loc[df["numero_documento"].isin(pares["numero_documento"]),
                    TARGET].to_numpy()
             == pares.groupby("numero_documento", sort=False)[TARGET].max()
             .reindex(df.loc[df["numero_documento"].isin(
                 pares["numero_documento"]), "numero_documento"]).to_numpy()
             ).all()),
        "documentos_duplicados": int(pares["numero_documento"].nunique()),
        "filas_retiradas": int(n_antes - len(df)),
        "filas_retiradas_por_municipio": {
            m: muni_antes[m] - muni_despues[m] for m in MUNICIPIOS},
        "pares_donde_keep_first_elegiria_la_fila_menor": first_elige_menor,
        # Observación, NO se actúa sobre ellas: sesiones igual de largas que no
        # están duplicadas y siguen en el conjunto.
        "sesiones_no_duplicadas_sobre_umbral_por_municipio": por_municipio(
            df[pd.to_numeric(df["tiempo_usado_segundos"], errors="coerce")
               > UMBRAL_SESION_ABANDONADA_S]),
    }
    log(f"dedup: {inv['dedup']['documentos_duplicados']} documentos duplicados, "
        f"{inv['dedup']['filas_retiradas']} filas retiradas "
        f"{inv['dedup']['filas_retiradas_por_municipio']}; keep='first' habría "
        f"conservado la sesión abandonada en {first_elige_menor} pares; "
        f"resultado igual al de máximo puntaje: "
        f"{inv['dedup']['coincide_con_maximo_puntaje']}")

    # Exclusión de descartados por sospecha.
    excl = df["numero_documento"].isin(descartados)
    inv["descartados"] = {
        "documentos_en_lista": len(descartados),
        "excluidos_por_municipio": por_municipio(df[excl]),
        "excluidos_total": int(excl.sum()),
        "excluidos_con_perfil_por_municipio": por_municipio(df[excl & df["_perfil"]]),
        "excluidos_sin_perfil_por_municipio": por_municipio(df[excl & ~df["_perfil"]]),
        "documentos_de_la_lista_sin_fila_puntuada": int(
            len(descartados - set(df["numero_documento"]))),
    }
    # Tasa sobre la cohorte deduplicada ANTES de excluir.
    base = por_municipio(df)
    inv["descartados"]["tasa_por_municipio"] = {
        m: (inv["descartados"]["excluidos_por_municipio"][m] / base[m]
            if base[m] else None) for m in MUNICIPIOS}
    df = df[~excl].copy()
    log(f"descartados excluidos: {inv['descartados']['excluidos_total']} "
        f"{inv['descartados']['excluidos_por_municipio']}; documentos de la lista "
        f"sin fila puntuada: "
        f"{inv['descartados']['documentos_de_la_lista_sin_fila_puntuada']}")

    # Conteo final contra las cifras de Supabase.
    final = {}
    for m in MUNICIPIOS:
        sub = df[df["municipio"] == m]
        final[m] = {"total": int(len(sub)),
                    "con_perfil": int(sub["_perfil"].sum()),
                    "sin_perfil": int((~sub["_perfil"]).sum())}
    inv["final"] = final
    inv["final_total"] = int(len(df))
    inv["discrepancias_vs_supabase"] = {
        m: {k: final[m][k] - ESPERADO[m][k] for k in ESPERADO[m]}
        for m in MUNICIPIOS}

    print(f"\n  {'Municipio':<11} {'local':>7} {'Supabase':>9} {'dif':>5}"
          f"   {'perfil':>6} {'sin perf':>8}")
    for m in MUNICIPIOS:
        f, e = final[m], ESPERADO[m]
        print(f"  {m:<11} {f['total']:>7,} {e['total']:>9,} "
              f"{f['total'] - e['total']:>+5}   {f['con_perfil']:>6,} "
              f"{f['sin_perfil']:>8,}")
    print(f"  {'total':<11} {len(df):>7,}")

    # Medias crudas (para contrastar con el efecto aprendido).
    medias = {}
    for m in MUNICIPIOS:
        sub = df[df["municipio"] == m]
        medias[m] = {
            "todos": {"media": float(sub[TARGET].mean()), "n": int(len(sub))},
            "con_perfil": {"media": float(sub.loc[sub["_perfil"], TARGET].mean())
                           if sub["_perfil"].any() else None,
                           "n": int(sub["_perfil"].sum())},
            "sin_perfil": {"media": float(sub.loc[~sub["_perfil"], TARGET].mean())
                           if (~sub["_perfil"]).any() else None,
                           "n": int((~sub["_perfil"]).sum())},
        }
    inv["medias_crudas"] = medias
    g = medias["Girardota"]
    inv["brecha_seleccion_girardota"] = g["sin_perfil"]["media"] - g["con_perfil"]["media"]
    log(f"Girardota: con perfil {g['con_perfil']['media']:.2f} vs sin perfil "
        f"{g['sin_perfil']['media']:.2f} (brecha "
        f"{inv['brecha_seleccion_girardota']:+.2f})")
    return df, inv


# ---------------------------------------------------------------------------
# Paso 3 — Métricas
# ---------------------------------------------------------------------------
def metricas(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"r2": float(r2_score(y, p)),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "n": int(len(y))}


def ic_bootstrap_r2(y, p) -> list[float]:
    rng = np.random.default_rng(RANDOM_STATE)
    y, p = np.asarray(y, float), np.asarray(p, float)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        vals.append(r2_score(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def entrenar(df: pd.DataFrame) -> tuple[dict, object, pd.DataFrame, pd.Series,
                                         np.ndarray]:
    titulo("PASO 2/3 — Features (script 12, bloque base) y métricas")
    X, numericas, binarias, onehot = s12.construir_crudos(df, incluir_nuevas=False)
    y = df[TARGET].astype(float)
    assert not any(c in X.columns for c in s12.NUEVAS_NUM + s12.NUEVAS_BIN), \
        "el bloque de perfil no debe entrar al modelo global"
    base = s12.construir_pipeline(numericas, binarias, onehot)
    log(f"predictores: {len(numericas) + len(binarias) + len(onehot)} "
        f"({numericas + binarias + onehot})")

    res: dict = {"modelo": "RandomForestRegressor", "hiperparametros":
                 {k: v for k, v in s12.MODELO_KWARGS.items() if k != "n_jobs"},
                 "predictores": {"numericas": numericas, "binarias": binarias,
                                 "onehot": onehot}}

    # OOF sobre todo el conjunto.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(base), X, y, cv=cv, n_jobs=-1)
    m = metricas(y, oof)
    m["r2_ic95_bootstrap"] = ic_bootstrap_r2(y, oof)
    m["particion"] = f"out-of-fold KFold(5, shuffle, random_state=42) sobre las {len(y):,} filas"
    res["oof"] = m
    log(f"OOF      R²={m['r2']:+.4f} IC95 [{m['r2_ic95_bootstrap'][0]:+.3f}, "
        f"{m['r2_ic95_bootstrap'][1]:+.3f}]  MAE={m['mae']:.2f}  "
        f"RMSE={m['rmse']:.2f}  n={m['n']:,}")

    # OOF por municipio y por presencia de perfil (el mismo vector OOF).
    res["oof_por_grupo"] = {}
    for m_ in MUNICIPIOS:
        k = (df["municipio"] == m_).to_numpy()
        res["oof_por_grupo"][m_] = metricas(y[k], oof[k])
    for nombre, k in (("con_perfil", df["_perfil"].to_numpy()),
                      ("sin_perfil", ~df["_perfil"].to_numpy())):
        res["oof_por_grupo"][nombre] = metricas(y[k], oof[k])
    for g, v in res["oof_por_grupo"].items():
        log(f"    OOF {g:<11} R²={v['r2']:+.4f}  MAE={v['mae']:.2f}  n={v['n']:,}")

    # Hold-out 20 % estratificado por quintil (mismo criterio que 12).
    idx_tr, idx_te = train_test_split(X.index, test_size=0.20,
                                      random_state=RANDOM_STATE,
                                      stratify=s12.estratos(y))
    mod_ho = clone(base).fit(X.loc[idx_tr], y.loc[idx_tr])
    p_te = mod_ho.predict(X.loc[idx_te])
    h = metricas(y.loc[idx_te], p_te)
    h["n_train"] = int(len(idx_tr))
    h["particion"] = ("hold-out 20 % estratificado por quintil de puntaje "
                      "(random_state=42); entrenado con el 80 % restante")
    h["n_por_municipio"] = por_municipio(df.loc[idx_te])
    res["holdout"] = h
    log(f"hold-out R²={h['r2']:+.4f}  MAE={h['mae']:.2f}  RMSE={h['rmse']:.2f}  "
        f"n_test={h['n']:,}  n_train={h['n_train']:,}")

    # Importancia por permutación del hold-out (columna cruda, como en 13).
    pi = permutation_importance(mod_ho, X.loc[idx_te], y.loc[idx_te],
                                scoring="r2", n_repeats=20,
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = sorted(({"variable": c, "delta_r2": float(mu), "std": float(sd)}
                  for c, mu, sd in zip(X.columns, pi.importances_mean,
                                       pi.importances_std)),
                 key=lambda d: -d["delta_r2"])
    res["importancia_permutacion_holdout"] = imp
    log("permutación (hold-out) top 5: " + ", ".join(
        f"{d['variable']} {d['delta_r2']:+.3f}" for d in imp[:5]))

    # Modelo de producción: todas las filas.
    final = clone(base).fit(X, y)
    return res, final, X, y, oof


# ---------------------------------------------------------------------------
# Paso 4 — Efecto municipio (hallazgo, no objetivo)
# ---------------------------------------------------------------------------
def efecto_municipio(df, X, y, oof, final, inv) -> dict:
    titulo("PASO 4 — Efecto municipio aprendido")
    out: dict = {"residuo_medio_oof": {}, "dependencia_parcial": {},
                 "dependencia_parcial_sin_perfil": {}}
    resid = y.to_numpy() - oof
    for m in MUNICIPIOS:
        k = (df["municipio"] == m).to_numpy()
        out["residuo_medio_oof"][m] = {"media": float(resid[k].mean()),
                                       "n": int(k.sum())}

    # Dependencia parcial: se fija municipio = m para TODAS las filas y se
    # promedia la predicción del modelo final. Las demás variables quedan tal
    # cual, así que la diferencia entre municipios es el efecto que el modelo
    # atribuye al municipio con la composición de la población fija.
    sin_perfil = ~df["_perfil"].to_numpy()
    for m in MUNICIPIOS:
        Xm = X.copy()
        Xm["municipio"] = m
        p = final.predict(Xm)
        out["dependencia_parcial"][m] = {"media": float(p.mean()), "n": int(len(p))}
        out["dependencia_parcial_sin_perfil"][m] = {
            "media": float(p[sin_perfil].mean()), "n": int(sin_perfil.sum())}

    pd_ = out["dependencia_parcial"]
    orden = sorted(MUNICIPIOS, key=lambda m: -pd_[m]["media"])
    out["orden_aprendido"] = orden
    med = inv["medias_crudas"]
    crudo_sin = {m: med[m]["sin_perfil"]["media"] for m in MUNICIPIOS}
    out["copacabana_menor_que_girardota"] = {
        "modelo_dependencia_parcial":
            pd_["Copacabana"]["media"] < pd_["Girardota"]["media"],
        "diferencia_modelo": pd_["Copacabana"]["media"] - pd_["Girardota"]["media"],
        "medias_crudas_sin_perfil": crudo_sin["Copacabana"] < crudo_sin["Girardota"],
        "diferencia_cruda_sin_perfil": crudo_sin["Copacabana"] - crudo_sin["Girardota"],
    }
    out["coincide_con_medias_crudas"] = (
        out["copacabana_menor_que_girardota"]["modelo_dependencia_parcial"]
        == out["copacabana_menor_que_girardota"]["medias_crudas_sin_perfil"])

    print(f"  {'Municipio':<11} {'media cruda':>11} {'sin perfil':>10} "
          f"{'DP todos':>9} {'DP sin perf':>11} {'resid OOF':>9}")
    for m in MUNICIPIOS:
        sp = crudo_sin[m]
        print(f"  {m:<11} {med[m]['todos']['media']:>11.2f} "
              f"{(f'{sp:.2f}' if sp is not None else '—'):>10} "
              f"{pd_[m]['media']:>9.2f} "
              f"{out['dependencia_parcial_sin_perfil'][m]['media']:>11.2f} "
              f"{out['residuo_medio_oof'][m]['media']:>+9.2f}")
    log(f"orden aprendido (DP): {' > '.join(orden)}; coincide con medias "
        f"crudas en Copacabana vs Girardota: {out['coincide_con_medias_crudas']}")
    return out


# ---------------------------------------------------------------------------
# Paso 5 — Limitaciones de v2 (solo lectura)
# ---------------------------------------------------------------------------
def limitaciones_v2(descartados: set[str]) -> dict:
    titulo("PASO 5 — Limitaciones de v2 (lectura de dataset_C_perfil.csv)")
    c = pd.read_csv(FUENTE_C_V2, encoding="utf-8", dtype={"numero_documento": str},
                    usecols=["numero_documento", "municipio", TARGET])
    c["numero_documento"] = c["numero_documento"].astype(str).str.strip()
    c = c[~c["numero_documento"].isin(DOCS_PRUEBA)]
    c = c[pd.to_numeric(c[TARGET], errors="coerce").notna()]
    k = c["numero_documento"].isin(descartados)
    out = {
        "dataset": FUENTE_C_V2.name,
        "n_entrenamiento_v2": int(len(c)),
        "descartados_incluidos": int(k.sum()),
        "descartados_incluidos_por_municipio": por_municipio(c[k]),
        "n_sin_descartados": int((~k).sum()),
        "columna_sospechoso_descartado_en_export": False,
        "filtros_aplicados_por_scripts_11_y_14": [
            "documentos de prueba", "duplicados por documento (keep='first')",
            "puntaje no nulo"],
    }
    log(f"v2 entrenó con {out['n_entrenamiento_v2']:,} filas; de ellas "
        f"{out['descartados_incluidos']} están en la lista de descartados "
        f"{out['descartados_incluidos_por_municipio']} -> "
        f"{out['n_sin_descartados']:,} sin ellas")
    return out


# ---------------------------------------------------------------------------
# Análisis complementarios (no cambian el modelo oficial)
# ---------------------------------------------------------------------------
def oof_y_dp(d: pd.DataFrame, extra: dict[str, str] | None = None) -> dict:
    """OOF KFold(5) + dependencia parcial de `municipio` para un subconjunto,
    con el mismo pipeline del modelo oficial. `extra` añade columnas
    diagnósticas {columna: 'num'|'cat'} que el modelo oficial NO usa."""
    X, num, binr, oh = s12.construir_crudos(d, incluir_nuevas=False)
    for c, kind in (extra or {}).items():
        X[c] = d[c].to_numpy()
        (num if kind == "num" else oh).append(c)
    y = d[TARGET].astype(float)
    base = s12.construir_pipeline(num, binr, oh)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(base), X, y, cv=cv, n_jobs=-1)
    m = metricas(y, oof)
    m["particion"] = f"out-of-fold KFold(5, shuffle, random_state=42) sobre {len(y):,} filas"
    final = clone(base).fit(X, y)
    dp = {}
    for mu in MUNICIPIOS:
        if mu not in set(d["municipio"]):
            continue
        Xm = X.copy()
        Xm["municipio"] = mu
        dp[mu] = float(final.predict(Xm).mean())
    return {"oof": m, "dependencia_parcial": dp, "n_dp": int(len(X))}


def importancia_municipio(d: pd.DataFrame, extra: dict[str, str] | None = None
                          ) -> dict:
    """Protocolo EXACTO del script 13 (sección B): hold-out 20 % estratificado
    por quintil, RF del script 12, permutación de la columna cruda con
    n_repeats=30 y scoring R²."""
    X, num, binr, oh = s12.construir_crudos(d, incluir_nuevas=False)
    for c, kind in (extra or {}).items():
        X[c] = d[c].to_numpy()
        (num if kind == "num" else oh).append(c)
    y = d[TARGET].astype(float)
    pipe = s12.construir_pipeline(num, binr, oh)
    tr, te = train_test_split(X.index, test_size=0.20, random_state=RANDOM_STATE,
                              stratify=s12.estratos(y))
    pipe.fit(X.loc[tr], y.loc[tr])
    p = permutation_importance(pipe, X.loc[te], y.loc[te], n_repeats=30,
                               random_state=RANDOM_STATE, scoring="r2")
    r = {c: {"delta_r2": float(mu), "std": float(sd)}
         for c, mu, sd in zip(X.columns, p.importances_mean, p.importances_std)}
    out = {"municipio": r["municipio"], "n_filas": int(len(X)),
           "n_test": int(len(te))}
    for c in (extra or {}):
        out[c] = r[c]
    return out


def dispersion_target(df: pd.DataFrame, res: dict) -> dict:
    """SD del target por población y descomposición de la caída de R² A→B con
    la identidad RMSE = SD·sqrt(1 − R²) (SD poblacional, ddof=0)."""
    titulo("COMPLEMENTO 1 — Dispersión del target y caída de R²")
    out: dict = {"sd": {}}

    v1 = pd.read_csv(V1_SCORES, usecols=["puntaje_real"])["puntaje_real"].dropna()
    out["sd"]["v1_1748"] = {"sd": float(v1.std(ddof=0)), "n": int(len(v1)),
                           "fuente": V1_SCORES.relative_to(BASE_DIR).as_posix()}
    for clave, archivo in (("A", "dataset_A_baseline.csv"),
                           ("B", "dataset_B_completo.csv")):
        d = pd.read_csv(DATA_DIR / archivo, encoding="utf-8",
                        dtype={"numero_documento": str},
                        usecols=lambda c: c not in PII)
        d = d[pd.to_numeric(d[TARGET], errors="coerce").notna()].copy()
        y = pd.to_numeric(d[TARGET])
        out["sd"][clave] = {"sd": float(y.std(ddof=0)), "n": int(len(y)),
                            "fuente": f"data/{archivo}"}

        # Réplica del protocolo del script 12: CV sobre el train del 80 %.
        X, num, binr, oh = s12.construir_crudos(d, incluir_nuevas=False)
        tr, _ = train_test_split(X.index, test_size=0.20,
                                 random_state=RANDOM_STATE,
                                 stratify=s12.estratos(y))
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        pipe = s12.construir_pipeline(num, binr, oh)
        ytr = y.loc[tr]
        folds = []
        oof = np.empty(len(tr))
        for itr, ite in cv.split(X.loc[tr]):
            mod = clone(pipe).fit(X.loc[tr].iloc[itr], ytr.iloc[itr])
            oof[ite] = mod.predict(X.loc[tr].iloc[ite])
            folds.append(r2_score(ytr.iloc[ite], oof[ite]))
        m = metricas(ytr, oof)
        out[f"cv_{clave}"] = {
            "r2_media_folds": float(np.mean(folds)),
            "r2_oof_agregado": m["r2"], "rmse_oof": m["rmse"],
            "sd_train": float(ytr.std(ddof=0)), "n_train": int(len(ytr)),
            "particion": "KFold(5) sobre el train del 80 % (protocolo del script 12)"}

    a, b = out["cv_A"], out["cv_B"]
    r2_b_contraf = 1 - (a["rmse_oof"] / b["sd_train"]) ** 2
    caida = a["r2_oof_agregado"] - b["r2_oof_agregado"]
    out["descomposicion_A_B"] = {
        "caida_r2": caida,
        "r2_B_si_rmse_de_A": r2_b_contraf,
        "parte_por_varianza": a["r2_oof_agregado"] - r2_b_contraf,
        "parte_por_error": r2_b_contraf - b["r2_oof_agregado"],
        "rmse_A": a["rmse_oof"], "rmse_B": b["rmse_oof"],
    }

    # Lo mismo entre v1 (oficial) y este modelo.
    sd_v1 = out["sd"]["v1_1748"]["sd"]
    sd_g = float(df[TARGET].std(ddof=0))
    out["sd"]["global_3050"] = {"sd": sd_g, "n": int(len(df)),
                                "fuente": "conjunto de este script"}
    out["v1_vs_global"] = {
        "sd_v1_implicita_por_identidad": V1_OFICIAL["rmse"] / np.sqrt(1 - V1_OFICIAL["r2_oof"]),
        "r2_global_si_sd_de_v1": 1 - (res["oof"]["rmse"] / sd_v1) ** 2,
        "caida_r2": V1_OFICIAL["r2_oof"] - res["oof"]["r2"],
    }
    for k, v in out["sd"].items():
        log(f"SD {k:<12} {v['sd']:6.2f}  n={v['n']:,}")
    dd = out["descomposicion_A_B"]
    log(f"A→B: R² {a['r2_oof_agregado']:.4f}→{b['r2_oof_agregado']:.4f} "
        f"(folds {a['r2_media_folds']:.3f}→{b['r2_media_folds']:.3f}); RMSE "
        f"{a['rmse_oof']:.2f}→{b['rmse_oof']:.2f}; con RMSE de A sobre SD de B "
        f"R²={r2_b_contraf:+.4f} → varianza {dd['parte_por_varianza']:+.4f}, "
        f"error {dd['parte_por_error']:+.4f}")
    log(f"v1→global: con la RMSE global sobre la SD de v1, R² = "
        f"{out['v1_vs_global']['r2_global_si_sd_de_v1']:+.4f}")
    return out


def sensibilidad_sesiones_largas(df: pd.DataFrame) -> dict:
    titulo("COMPLEMENTO 2 — Sensibilidad a las sesiones > 3 h")
    t = pd.to_numeric(df["tiempo_usado_segundos"], errors="coerce")
    largas = t > UMBRAL_SESION_ABANDONADA_S
    out = {"n_largas": int(largas.sum()),
           "largas_por_municipio": por_municipio(df[largas]),
           "puntajes_largas": sorted(float(v) for v in df.loc[largas, TARGET]),
           "con": oof_y_dp(df), "sin": oof_y_dp(df[~largas].copy())}
    for k in ("con", "sin"):
        o = out[k]["oof"]
        log(f"{k:<4} R²={o['r2']:.4f} MAE={o['mae']:.2f} RMSE={o['rmse']:.2f} "
            f"n={o['n']:,}  DP={ {m: round(v, 2) for m, v in out[k]['dependencia_parcial'].items()} }")
    return out


def reconciliar_municipio(df: pd.DataFrame) -> dict:
    """¿Por qué `municipio` pasa de ΔR² 0.081 (F13, dataset A) a ~0.003 aquí?"""
    titulo("COMPLEMENTO 3 — Importancia de municipio: F13 vs este modelo")
    d = df.copy()
    fecha = pd.to_datetime(d["created_at"], utc=True, format="mixed")
    d["mes_inscripcion"] = fecha.dt.strftime("%Y-%m")
    d["dias_desde_inicio"] = (fecha - fecha.min()).dt.total_seconds() / 86400

    a = pd.read_csv(DATA_DIR / "dataset_A_baseline.csv", encoding="utf-8",
                    dtype={"numero_documento": str}, usecols=lambda c: c not in PII)
    a = a[pd.to_numeric(a[TARGET], errors="coerce").notna()].copy()
    fa = pd.to_datetime(a["created_at"], utc=True, format="mixed")
    fin_ventana_a = fa.max()

    cg = d[d["municipio"] != "Bello"]
    escenarios = [
        ("A (réplica F13)", a, None),
        ("B completo (3,072)", pd.read_csv(
            DATA_DIR / "dataset_B_completo.csv", encoding="utf-8",
            dtype={"numero_documento": str}, usecols=lambda c: c not in PII),
         None),
        ("global 3,050", d, None),
        ("global 3,050 + mes", d, {"mes_inscripcion": "cat"}),
        ("global 3,050 + días", d, {"dias_desde_inicio": "num"}),
        ("Copa+Girardota, todos los meses", cg, None),
        ("Copa+Girardota + mes", cg, {"mes_inscripcion": "cat"}),
        ("Copa+Girardota, ventana de A", cg[fecha.loc[cg.index] <= fin_ventana_a],
         None),
    ]
    out: dict = {"ventana_A": [str(fa.min()), str(fin_ventana_a)],
                 "ventanas_por_municipio": {
                     m: [str(fecha[d["municipio"] == m].min()),
                         str(fecha[d["municipio"] == m].max())]
                     for m in MUNICIPIOS},
                 "escenarios": []}
    for nombre, sub, extra in escenarios:
        sub = sub[pd.to_numeric(sub[TARGET], errors="coerce").notna()].copy()
        r = importancia_municipio(sub, extra)
        r["escenario"] = nombre
        out["escenarios"].append(r)
        extra_txt = "".join(f"  {c}={r[c]['delta_r2']:+.4f}" for c in (extra or {}))
        log(f"{nombre:<34} municipio ΔR²={r['municipio']['delta_r2']:+.4f} "
            f"±{r['municipio']['std']:.4f}  n_test={r['n_test']}{extra_txt}")

    # Medias por municipio y mes (cohortes de inscripción).
    g = d.groupby(["municipio", "mes_inscripcion"])[TARGET].agg(["size", "mean"])
    out["media_por_mes"] = {f"{m}|{mes}": {"n": int(r["size"]), "media": float(r["mean"])}
                            for (m, mes), r in g.iterrows()}
    out["perfil_por_mes_girardota"] = {
        mes: {"con_perfil": int(s["_perfil"].sum()), "sin_perfil": int((~s["_perfil"]).sum())}
        for mes, s in d[d["municipio"] == "Girardota"].groupby("mes_inscripcion")}

    # Prueba intra-institución en Girardota: si el mes fuera una condición de
    # medición, el puntaje cambiaría con el mes DENTRO de una misma institución.
    gi = d[(d["municipio"] == "Girardota") & d["_inst"].ge(0)].copy()
    multi = gi[gi.groupby("_inst")["mes_inscripcion"].transform("nunique") > 1].copy()
    multi["desv_intra"] = multi[TARGET] - multi.groupby("_inst")[TARGET].transform("mean")
    out["girardota_intra_institucion"] = {
        "instituciones": int(gi["_inst"].nunique()),
        "instituciones_en_mas_de_un_mes": int(multi["_inst"].nunique()),
        "filas_en_esas_instituciones": int(len(multi)),
        "media_cruda_por_mes": {k: {"n": int(v["size"]), "media": float(v["mean"])}
                                for k, v in multi.groupby("mes_inscripcion")[TARGET]
                                .agg(["size", "mean"]).iterrows()},
        "desviacion_media_intra_por_mes": {
            k: float(v) for k, v in multi.groupby("mes_inscripcion")["desv_intra"].mean().items()},
    }
    # Lo mismo para la brecha con/sin perfil (11.75 puntos).
    ip = gi[gi.groupby("_inst")["_perfil"].transform("nunique") > 1].copy()
    ip["desv_intra"] = ip[TARGET] - ip.groupby("_inst")[TARGET].transform("mean")
    out["girardota_perfil_intra_institucion"] = {
        "instituciones_con_ambos": int(ip["_inst"].nunique()),
        "filas": int(len(ip)),
        "media_cruda": {("con_perfil" if k else "sin_perfil"): float(v)
                        for k, v in ip.groupby("_perfil")[TARGET].mean().items()},
        "desviacion_media_intra": {("con_perfil" if k else "sin_perfil"): float(v)
                                   for k, v in ip.groupby("_perfil")["desv_intra"].mean().items()},
    }
    # Telemetría por cohorte: proporción de sesiones con algún evento registrado.
    ev = (pd.to_numeric(d["cambios_pestana"], errors="coerce").fillna(0)
          + pd.to_numeric(d["intentos_click_derecho"], errors="coerce").fillna(0)) > 0
    out["telemetria_con_eventos_por_mes"] = {
        f"{m}|{mes}": float(v) for (m, mes), v in
        ev.groupby([d["municipio"], d["mes_inscripcion"]]).mean().items()}

    # Efecto municipio condicionado al mes (diagnóstico, no producción).
    out["dp_con_mes"] = oof_y_dp(d, {"mes_inscripcion": "cat"})
    log(f"intra-institución Girardota: {out['girardota_intra_institucion']}")
    log(f"perfil intra-institución Girardota: {out['girardota_perfil_intra_institucion']}")
    log(f"DP con mes: {out['dp_con_mes']['dependencia_parcial']}  "
        f"OOF R²={out['dp_con_mes']['oof']['r2']:.4f}")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(" COPA STEM 2026 — Modelo global base (sin perfil académico)")
    print(" Fundación SapienceLab · Script 21")
    print("=" * 78)

    descartados = cargar_descartados()
    df, inv = inventario(descartados)
    res, final, X, y, oof = entrenar(df)
    efecto = efecto_municipio(df, X, y, oof, final, inv)
    v2 = limitaciones_v2(descartados)
    del descartados  # la lista no sale de memoria ni llega a ninguna salida
    dispersion = dispersion_target(df, res)
    sensibilidad = sensibilidad_sesiones_largas(df)
    reconciliacion = reconciliar_municipio(df)

    poblacion = (f"{len(df):,} estudiantes de Bello, Girardota y Copacabana "
                 f"(export {FUENTE.name}, deduplicado, sin descartados)")
    metricas_embebidas = {
        "oof": res["oof"], "holdout": res["holdout"],
        "oof_por_grupo": res["oof_por_grupo"],
        "poblacion_entrenamiento": poblacion,
        # SD poblacional (ddof=0) del target: sin ella el R² no se puede
        # comparar entre poblaciones (RMSE = SD·sqrt(1 − R²)).
        "sd_target": dispersion["sd"],
        "comparacion_v1": {
            "v1": V1_OFICIAL,
            "este_modelo": {"r2_oof": res["oof"]["r2"], "mae": res["oof"]["mae"],
                            "rmse": res["oof"]["rmse"],
                            "r2_ic95": res["oof"]["r2_ic95_bootstrap"],
                            "n": res["oof"]["n"], "poblacion": poblacion,
                            "particion": "out-of-fold KFold(5)"},
            "nota": "poblaciones distintas: v1 1,748 estudiantes de dos "
                    "municipios; este modelo tres municipios. El MAE depende "
                    "de la dispersión del target de cada población.",
        },
    }

    # --- joblib: el modelo viaja con su métrica ------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "modelo": final,  # Pipeline(ColumnTransformer + RandomForest)
        "version": "global_base",
        "reemplaza": "v1 (models/mejor_modelo_puntaje.joblib) en producción",
        "entrenado": datetime.now().isoformat(timespec="seconds"),
        "dataset": FUENTE.name,
        "n_entrenamiento": int(len(df)),
        "n_por_municipio": {m: inv["final"][m]["total"] for m in MUNICIPIOS},
        "target": TARGET,
        "escala_target": "0-100",
        "predictores": res["predictores"],
        "preprocesamiento_crudo": ("notebooks/12_experimento_reentrenamiento.py"
                                   "::construir_crudos(df, incluir_nuevas=False)"),
        "hiperparametros": res["hiperparametros"],
        "metricas": metricas_embebidas,
        "politica_uso": ("puntaje_estimado nunca como cifra puntual para un "
                         "individuo ni para ordenar individuos. Toda métrica se "
                         "reporta con valor, n y partición."),
        "sin_ajustes_por_municipio": True,
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, OUT_JOBLIB, compress=3)
    log(f"modelo → models/{OUT_JOBLIB.name} "
        f"({OUT_JOBLIB.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- JSON de resultados (solo agregados) ---------------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    salida = {
        "script": "notebooks/21_modelo_global_base.py",
        "generado": bundle["entrenado"],
        "inventario": inv,
        "modelo": {k: res[k] for k in ("modelo", "hiperparametros", "predictores")},
        "metricas": metricas_embebidas,
        "importancia_permutacion_holdout": res["importancia_permutacion_holdout"],
        "efecto_municipio": efecto,
        "limitaciones_v2": v2,
        "complementos": {
            "1_dispersion_target": dispersion,
            "2_sensibilidad_sesiones_largas": sensibilidad,
            "3_reconciliacion_importancia_municipio": reconciliacion,
        },
        "entorno": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                    "numpy": np.__version__},
    }
    OUT_JSON.write_text(json.dumps(salida, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log(f"resultados → outputs/{OUT_JSON.name}")

    print("\n" + "=" * 78)
    print(" MODELO GLOBAL BASE COMPLETADO — ver reports/21_modelo_global_base.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
