# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 17: Corrección de `ref_rendimiento` — población de referencia completa
================================================================================

El informe 15 detectó un artefacto: el SPEC v2 (script 14) percentiliza el
componente de rendimiento contra la cohorte de `dataset_C_perfil.csv`, es decir
solo los 1,148 estudiantes que respondieron las 5 preguntas de perfil académico.
Esa población es más estrecha y más homogénea que la cohorte examinada completa
(σ = 20.53 en C frente a 22.66 en B, 3,072 estudiantes), de modo que los mismos
puntajes reales se estiran hacia los extremos del rango percentil: ~72
estudiantes cambian de categoría de potencial sin que su desempeño haya variado
en nada.

Qué hace este script
--------------------
    1) Recalcula `ref_rendimiento` sobre la cohorte COMPLETA de
       `dataset_B_completo.csv` (3,072 estudiantes con resultado de examen,
       tabla `resultados_prueba_copa_stem`).
    2) Vuelve a puntuar a los 3,072 con esa referencia corregida, manteniendo
       la MISMA estrategia híbrida del script 15 (v2 para los 1,148 con perfil
       académico, v1 para los 1,924 restantes) y los MISMOS modelos.
    3) Escribe `outputs/ml_scores_v2_corrected.csv` con el esquema de 18
       columnas de `outputs/ml_scores_v2.csv`.
    4) Compara los conteos de categoría antes/después, con la misma forma que
       la tabla v1 vs v2 del informe 15.

Qué NO cambia
-------------
Las predicciones del modelo v2 se conservan intactas. `ref_rendimiento` solo
interviene en `_percentil`, no en `_predict_puntaje`: `puntaje_estimado` sale
bit a bit igual al del script 15 y el paso 5 lo verifica fila a fila contra
`outputs/ml_scores_v2.csv`. Tampoco se toca ningún script numerado anterior,
ningún artefacto de `models/deploy/`, ni Supabase / la Edge Function.

Por qué la referencia correcta es B y no C
------------------------------------------
`ref_rendimiento` no es un parámetro del modelo: es la vara de medir con la que
se convierte un puntaje en percentil. La pregunta que responde es "¿cómo le fue
a este estudiante frente a los demás que presentaron la prueba?", y los demás
son los 3,072, no el subconjunto que además contestó el perfil académico. Que C
sea el dataset de ENTRENAMIENTO del modelo v2 es irrelevante para esta decisión:
son dos usos distintos de los datos que el script 14 acopló sin necesidad.

Reproducible: `random_state=42`.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
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

FUENTE = DATA_DIR / "dataset_B_completo.csv"
JS_V2 = DEPLOY_DIR / "potencial_stem_predictor_v2.js"
SCORES_V2 = OUTPUTS_DIR / "ml_scores_v2.csv"            # "antes" (script 15)
COMP_V1_V2 = OUTPUTS_DIR / "F15_comparacion_v1_v2.csv"  # v1 de los 3,072

OUT_CSV = OUTPUTS_DIR / "ml_scores_v2_corrected.csv"
OUT_COMP = OUTPUTS_DIR / "F17_comparacion_categorias.csv"
OUT_REF = OUTPUTS_DIR / "F17_ref_rendimiento_corregido.json"

TARGET = "puntaje_obtenido"
COL_PERFIL = "promedio_academico"

# Esquema de `outputs/ml_scores_v2.csv` (17 columnas de la Edge Function + la
# marca de versión). Las 3 que la Edge Function no escribe (nivel_sospecha,
# n_criterios_sospecha, created_at) las pone el trigger `after_resultado_insert`.
COLS_ML_SCORES = [
    "numero_documento",
    "indice_potencial", "componente_rendimiento", "componente_engagement",
    "componente_resiliencia", "categoria_potencial",
    "es_talento_oculto", "probabilidad_talento", "n_condiciones_adversas",
    "condiciones_detalle",
    "cluster_id", "cluster_nombre",
    "indice_condiciones", "nivel_condiciones",
    "tiene_puntaje_real", "puntaje_estimado", "updated_at",
]

DOCS_PRUEBA = {"1234", "123456", "123456789", "1234567899"}


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def titulo(txt: str) -> None:
    print("\n" + "=" * 78)
    print(f" {txt}")
    print("=" * 78)


def cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spec_desde_js(ruta: Path) -> dict:
    """Extrae la constante SPEC del artefacto JS de despliegue."""
    txt = ruta.read_text(encoding="utf-8")
    m = re.search(r"^const SPEC = (.*);$", txt, flags=re.MULTILINE)
    if not m:
        print(f"\n  ADVERTENCIA: no se encontró SPEC en {ruta.name}\n")
        sys.exit(1)
    return json.loads(m.group(1))


def construir_ref(registros, usa_v2, pot, SPEC_V1: dict, SPEC_V2: dict) -> list:
    """Distribución de referencia sobre la cohorte que se le pase.

    Misma semántica que `construir_spec` del script 14 (puntaje real si
    presentó, estimado si no), pero aplicada a los 3,072 de B en vez de a los
    1,148 de C. Para el estimado se usa el SPEC que le tocaría a ese estudiante
    en la estrategia híbrida del script 15.
    """
    rend, n_estimados = [], 0
    for r, v2 in zip(registros, usa_v2):
        real = pot._to_float(r.get(TARGET))
        if real is not None:
            rend.append(real)
            continue
        spec = SPEC_V2 if v2 else SPEC_V1
        PRE, MODEL = spec["puntaje"]["preprocess"], spec["puntaje"]["model"]
        rend.append(pot._predict_puntaje(pot._features_puntaje(r, PRE), MODEL))
        n_estimados += 1
    if n_estimados:
        log(f"    {n_estimados:,} sin puntaje real → se usó el estimado")
    return sorted(round(float(x), 4) for x in rend)


def conteos(serie: pd.Series, orden: list) -> list:
    vc = serie.value_counts()
    return [int(vc.get(c, 0)) for c in orden]


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(" COPA STEM 2026 — Corrección de la población de referencia")
    print(" Fundación SapienceLab · Script 17 — recálculo LOCAL")
    print("=" * 78)

    # --- Paso 1: datos -------------------------------------------------------
    for f in (FUENTE, SCORES_V2, COMP_V1_V2):
        if not f.exists():
            print(f"\n  ADVERTENCIA: falta {f}. Ejecute antes los scripts 11 y 15.\n")
            sys.exit(1)

    df = pd.read_csv(FUENTE, encoding="utf-8", dtype={"numero_documento": str})
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df = df[~df["numero_documento"].isin(DOCS_PRUEBA)].reset_index(drop=True)
    log(f"Paso 1 — {len(df):,} estudiantes de {FUENTE.name}")

    tiene_perfil = df[COL_PERFIL].notna() if COL_PERFIL in df.columns \
        else pd.Series(False, index=df.index)
    log(f"    con `{COL_PERFIL}` → v2        : {int(tiene_perfil.sum()):,}")
    log(f"    sin perfil       → v1 fallback: {int((~tiene_perfil).sum()):,}")

    # --- Paso 2: predictores y SPECs ----------------------------------------
    titulo("PASO 2 — Carga de predictores y SPECs")
    pot = cargar_modulo("pot_v1", DEPLOY_DIR / "potencial_stem_predictor.py")
    tal = cargar_modulo("talento", DEPLOY_DIR / "talento_oculto_predictor.py")
    cond = cargar_modulo("cond", DEPLOY_DIR / "indice_condiciones_predictor.py")
    clus = cargar_modulo("clus", DEPLOY_DIR / "clustering_predictor.py")
    log("predictores cargados desde models/deploy/ (stdlib puro)")

    SPEC_V1 = pot.SPEC
    SPEC_V2 = spec_desde_js(JS_V2)
    log(f"SPEC v1 → ref_rendimiento de {len(SPEC_V1['ref_rendimiento']):,} puntajes")
    log(f"SPEC v2 → ref_rendimiento de {len(SPEC_V2['ref_rendimiento']):,} puntajes")

    # --- Paso 3: referencia corregida ---------------------------------------
    titulo("PASO 3 — Nueva distribución de referencia (cohorte completa)")
    registros = df.to_dict("records")
    REF_B = construir_ref(registros, tiene_perfil, pot, SPEC_V1, SPEC_V2)

    def desc(ref, nombre):
        a = np.asarray(ref, dtype=float)
        print(f"  {nombre:<28} n = {len(a):>5,} | media {a.mean():5.2f} | "
              f"σ {a.std(ddof=0):5.2f} | p25 {np.percentile(a, 25):5.1f} | "
              f"mediana {np.median(a):5.1f} | p75 {np.percentile(a, 75):5.1f}")

    desc(SPEC_V1["ref_rendimiento"], "v1 (cohorte histórica)")
    desc(SPEC_V2["ref_rendimiento"], "v2 actual (dataset C)")
    desc(REF_B, "v2 corregida (dataset B)")

    # SPECs corregidos: copia exacta con la referencia sustituida. Nada más.
    SPEC_V1_C = copy.deepcopy(SPEC_V1)
    SPEC_V2_C = copy.deepcopy(SPEC_V2)
    SPEC_V1_C["ref_rendimiento"] = REF_B
    SPEC_V2_C["ref_rendimiento"] = REF_B
    log("SPECs clonados: solo cambia `ref_rendimiento`; modelo y pesos intactos")

    OUT_REF.write_text(json.dumps({
        "generado": datetime.now().isoformat(timespec="seconds"),
        "fuente": FUENTE.name,
        "n_cohorte": len(REF_B),
        "sigma": round(float(np.std(REF_B)), 4),
        "reemplaza": {"v1": len(SPEC_V1["ref_rendimiento"]),
                      "v2": len(SPEC_V2["ref_rendimiento"])},
        "ref_rendimiento": REF_B,
    }, ensure_ascii=False), encoding="utf-8")
    log(f"referencia → outputs/{OUT_REF.name}")

    # --- Paso 4: re-scoring con la referencia corregida ----------------------
    titulo("PASO 4 — Re-cálculo del vector de scores")
    filas = []
    ahora = datetime.now(timezone.utc).isoformat()

    for r, usa_v2 in zip(registros, tiene_perfil):
        spec = SPEC_V2_C if usa_v2 else SPEC_V1_C
        version = "v2" if usa_v2 else "v1_fallback"

        ind = pot.calcular_indice(r, spec)

        # `puntaje_estimado`: idéntico al del script 15. La referencia no entra
        # en `_predict_puntaje`, así que esta columna no puede moverse.
        PRE, MODEL = spec["puntaje"]["preprocess"], spec["puntaje"]["model"]
        p_est = pot._predict_puntaje(pot._features_puntaje(r, PRE), MODEL)

        t = tal.detectar_talento_oculto({**r, "indice_potencial":
                                         ind["indice_potencial"]})
        ic = cond.indice_condiciones(r)
        ic = 50.0 if ic is None or (isinstance(ic, float) and ic != ic) else ic
        perfil = clus.predecir_perfil(r)

        real = pot._to_float(r.get(TARGET))
        det = t.get("condiciones_detalle")
        filas.append({
            "numero_documento": r["numero_documento"],
            "indice_potencial": ind["indice_potencial"],
            "componente_rendimiento": ind["componente_rendimiento"],
            "componente_engagement": ind["componente_engagement"],
            "componente_resiliencia": ind["componente_resiliencia"],
            "categoria_potencial": ind["categoria"],
            "es_talento_oculto": t.get("es_talento_oculto"),
            "probabilidad_talento": t.get("probabilidad_talento"),
            "n_condiciones_adversas": t.get("n_condiciones_adversas"),
            "condiciones_detalle": json.dumps(det, ensure_ascii=False)
                                   if isinstance(det, (list, dict)) else det,
            "cluster_id": perfil.get("cluster_id"),
            "cluster_nombre": perfil.get("cluster_nombre"),
            "indice_condiciones": round(float(ic), 2),
            "nivel_condiciones": cond.nivel_condiciones(ic),
            "tiene_puntaje_real": real is not None,
            "puntaje_estimado": round(p_est, 2),
            "updated_at": ahora,
            "modelo_version": version,
        })

    out = pd.DataFrame(filas)[COLS_ML_SCORES + ["modelo_version"]]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    log(f"scores → outputs/{OUT_CSV.name} ({len(out):,} filas × "
        f"{out.shape[1]} columnas)")

    # --- Paso 5: verificación de que el modelo no se movió -------------------
    titulo("PASO 5 — Verificación: las predicciones del modelo no cambian")
    antes = pd.read_csv(SCORES_V2, dtype={"numero_documento": str})
    m = antes.merge(out, on="numero_documento", suffixes=("_antes", "_corr"))
    if len(m) != len(out):
        print(f"  ADVERTENCIA: cruce incompleto ({len(m):,} de {len(out):,})")

    d_est = (m.puntaje_estimado_corr - m.puntaje_estimado_antes).abs().max()
    print(f"  máx |Δ puntaje_estimado|          : {d_est:.6f}")
    print(f"  filas con `modelo_version` distinta: "
          f"{int((m.modelo_version_antes != m.modelo_version_corr).sum()):,}")
    for col in ("componente_engagement", "indice_condiciones", "cluster_id"):
        dif = int((m[f"{col}_antes"] != m[f"{col}_corr"]).sum())
        print(f"  filas con `{col}` distinta: {dif:,}")
    if d_est > 1e-9:
        print("\n  ADVERTENCIA: `puntaje_estimado` se movió; revise el SPEC.\n")
    else:
        log("OK — solo cambió lo que depende del percentil")

    # --- Paso 6: comparación de categorías ----------------------------------
    titulo("PASO 6 — Conteos de categoría: antes vs después")
    orden = [c[1] for c in SPEC_V1["categorias"]]          # de mayor a menor
    rank = {n: i for i, n in enumerate(orden[::-1])}       # 0 = la más baja

    comp15 = pd.read_csv(COMP_V1_V2, dtype={"numero_documento": str})
    base = comp15[["numero_documento", "modelo_version",
                   "categoria_v1", "categoria_v2"]].merge(
        out[["numero_documento", "categoria_potencial"]], on="numero_documento")
    base = base.rename(columns={"categoria_potencial": "categoria_corr"})

    filas_comp = []
    for ambito, sub in (("cohorte_completa_3072", base),
                        ("subgrupo_v2_1148", base[base.modelo_version == "v2"])):
        n_v1 = conteos(sub.categoria_v1, orden)
        n_v2 = conteos(sub.categoria_v2, orden)
        n_c = conteos(sub.categoria_corr, orden)
        print(f"\n  Ámbito: {ambito} (n = {len(sub):,})")
        print(f"  {'Categoría':<20}{'v1':>7}{'v2 (C)':>9}{'v2 corr':>9}"
              f"{'Δ corr-v2':>11}{'Δ corr-v1':>11}")
        for i, cat in enumerate(orden):
            print(f"  {cat:<20}{n_v1[i]:>7,}{n_v2[i]:>9,}{n_c[i]:>9,}"
                  f"{n_c[i] - n_v2[i]:>+11,}{n_c[i] - n_v1[i]:>+11,}")
            filas_comp.append({
                "ambito": ambito, "n_ambito": len(sub), "categoria": cat,
                "v1_ref_1750": n_v1[i],
                "v2_ref_1148_antes": n_v2[i],
                "v2_ref_3072_corregido": n_c[i],
                "delta_corregido_vs_antes": n_c[i] - n_v2[i],
                "delta_corregido_vs_v1": n_c[i] - n_v1[i],
            })

        # Movimiento individual: cuántos estudiantes cambian de etiqueta.
        def mov(a, b):
            ra, rb = sub[a].map(rank), sub[b].map(rank)
            return int((rb > ra).sum()), int((rb < ra).sum())

        s_av, b_av = mov("categoria_v1", "categoria_v2")
        s_cv, b_cv = mov("categoria_v1", "categoria_corr")
        s_ac, b_ac = mov("categoria_v2", "categoria_corr")
        print(f"    v1 → v2 (antes)      : {s_av + b_av:,} cambian "
              f"({s_av:,} suben / {b_av:,} bajan)")
        print(f"    v1 → v2 corregido    : {s_cv + b_cv:,} cambian "
              f"({s_cv:,} suben / {b_cv:,} bajan)")
        print(f"    v2 antes → corregido : {s_ac + b_ac:,} cambian "
              f"({s_ac:,} suben / {b_ac:,} bajan)")
        print(f"    desplazamiento a los extremos vs v1 "
              f"(|Δ| destacado + |Δ| apoyo): antes "
              f"{abs(n_v2[0] - n_v1[0]) + abs(n_v2[-1] - n_v1[-1]):,} · "
              f"corregido "
              f"{abs(n_c[0] - n_v1[0]) + abs(n_c[-1] - n_v1[-1]):,}")

    pd.DataFrame(filas_comp).to_csv(OUT_COMP, index=False, encoding="utf-8")
    log(f"\ncomparación → outputs/{OUT_COMP.name}")

    # --- Paso 7: talento oculto ---------------------------------------------
    titulo("PASO 7 — Talento oculto (depende del índice)")
    t_v1 = int(comp15.talento_v1.sum())
    t_v2 = int(comp15.talento_v2.sum())
    t_c = int(out.es_talento_oculto.sum())
    print(f"  v1               : {t_v1:,}")
    print(f"  v2 (ref C)       : {t_v2:,}   ({t_v2 - t_v1:+,} vs v1)")
    print(f"  v2 corregido (B) : {t_c:,}   ({t_c - t_v1:+,} vs v1)")

    print("\n" + "=" * 78)
    print(" COMPLETADO — recálculo local; no se tocó Supabase, la Edge Function")
    print(" ni ningún script o artefacto anterior")
    print("=" * 78)


if __name__ == "__main__":
    main()
