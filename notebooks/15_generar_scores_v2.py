# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 15: Generación de scores v2 — modelo híbrido (Fase 5)
================================================================================

Ejecuta en SOMBRA el vector completo de `ml_scores` para los 3,072 estudiantes
de `dataset_B_completo.csv`, con una estrategia híbrida:

    · v2  → estudiantes CON `promedio_academico` (modelo optimizado, script 14).
    · v1  → el resto (modelo actual de producción, fallback).

No toca ningún modelo existente ni la tabla `ml_scores`: escribe un CSV en
`outputs/` para poder comparar antes de decidir un despliegue.

Hallazgo que condiciona la lectura
----------------------------------
El índice compuesto usa el puntaje REAL cuando el estudiante presentó el examen:

    real = _to_float(raw.get("puntaje_obtenido"))
    if real is not None:  rend_raw = real          # el modelo NO se invoca
    else:                 rend_raw = _predict_puntaje(...)

Como los 3,072 de `dataset_B_completo` presentaron todos, el modelo de puntaje
**no interviene en `componente_rendimiento`** para ninguno de ellos. Lo que sí
cambia entre v1 y v2 es la **distribución de referencia** (`ref_rendimiento`)
con la que se percentiliza ese puntaje real: v1 la calculó sobre 1,750
estudiantes y v2 sobre 1,148. El script mide ese efecto por separado del efecto
del modelo, que solo se aprecia en `puntaje_estimado`.

Reutilización
-------------
Se importan los predictores puros de `models/deploy/` (stdlib, sin sklearn) y se
les inyecta el SPEC correspondiente, de modo que la lógica sea EXACTAMENTE la
que corre en la Edge Function. El SPEC v2 se extrae del artefacto de despliegue
`potencial_stem_predictor_v2.js`, no de una recomputación paralela.

Reproducible: `random_state=42`.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

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
OUT_CSV = OUTPUTS_DIR / "ml_scores_v2.csv"
OUT_COMP = OUTPUTS_DIR / "F15_comparacion_v1_v2.csv"

TARGET = "puntaje_obtenido"
COL_PERFIL = "promedio_academico"

# Columnas que la Edge Function gestiona en `ml_scores` (ver index.ts).
# Las 3 que NO escribe (nivel_sospecha, n_criterios_sospecha, created_at) las
# pone el trigger `after_resultado_insert` y quedan fuera de esta ejecución.
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


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(" COPA STEM 2026 — Generación de scores v2 (modelo híbrido)")
    print(" Fundación SapienceLab · Script 15 — ejecución en SOMBRA")
    print("=" * 78)

    # --- Paso 1: datos -------------------------------------------------------
    if not FUENTE.exists():
        print(f"\n  ADVERTENCIA: falta {FUENTE}. Ejecute antes el script 11.\n")
        sys.exit(1)
    df = pd.read_csv(FUENTE, encoding="utf-8", dtype={"numero_documento": str})
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df = df[~df["numero_documento"].isin(DOCS_PRUEBA)].reset_index(drop=True)
    log(f"Paso 1 — {len(df):,} estudiantes de {FUENTE.name}")

    tiene_perfil = df[COL_PERFIL].notna() if COL_PERFIL in df.columns \
        else pd.Series(False, index=df.index)
    log(f"    con `{COL_PERFIL}` → v2        : {int(tiene_perfil.sum()):,}")
    log(f"    sin perfil       → v1 fallback: {int((~tiene_perfil).sum()):,}")

    # --- Paso 2/3: predictores ----------------------------------------------
    titulo("PASO 2/3 — Carga de predictores")
    pot = cargar_modulo("pot_v1", DEPLOY_DIR / "potencial_stem_predictor.py")
    tal = cargar_modulo("talento", DEPLOY_DIR / "talento_oculto_predictor.py")
    cond = cargar_modulo("cond", DEPLOY_DIR / "indice_condiciones_predictor.py")
    clus = cargar_modulo("clus", DEPLOY_DIR / "clustering_predictor.py")
    log("predictores v1 cargados desde models/deploy/ (stdlib puro)")

    SPEC_V1 = pot.SPEC
    SPEC_V2 = spec_desde_js(JS_V2)
    log(f"SPEC v1 → cohorte {SPEC_V1['meta']['n_cohorte']:,} | "
        f"{len(SPEC_V1['puntaje']['preprocess']['numeric'])} numéricas")
    log(f"SPEC v2 → cohorte {SPEC_V2['meta']['n_cohorte']:,} | "
        f"{len(SPEC_V2['puntaje']['preprocess']['numeric'])} numéricas")

    # --- Paso 4: scoring -----------------------------------------------------
    titulo("PASO 4 — Cálculo del vector de scores")
    registros = df.to_dict("records")
    filas, comparacion = [], []
    ahora = datetime.now(timezone.utc).isoformat()

    for r, usa_v2 in zip(registros, tiene_perfil):
        spec = SPEC_V2 if usa_v2 else SPEC_V1
        version = "v2" if usa_v2 else "v1_fallback"

        # Índice compuesto con el SPEC que toque.
        ind = pot.calcular_indice(r, spec)

        # `puntaje_estimado`: AQUÍ sí interviene el modelo, siempre, aunque el
        # estudiante ya tenga nota. Es la predicción que la Edge Function hoy
        # descarta (la deja en null).
        PRE, MODEL = spec["puntaje"]["preprocess"], spec["puntaje"]["model"]
        p_est = pot._predict_puntaje(pot._features_puntaje(r, PRE), MODEL)

        # Talento oculto: necesita el índice ya calculado (igual que index.ts).
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

        # Contraparte v1 SIEMPRE, para aislar el efecto del cambio de SPEC.
        ind1 = pot.calcular_indice(r, SPEC_V1)
        t1 = tal.detectar_talento_oculto({**r, "indice_potencial":
                                          ind1["indice_potencial"]})
        p_est1 = pot._predict_puntaje(
            pot._features_puntaje(r, SPEC_V1["puntaje"]["preprocess"]),
            SPEC_V1["puntaje"]["model"])
        comparacion.append({
            "numero_documento": r["numero_documento"],
            "modelo_version": version,
            "indice_v1": ind1["indice_potencial"],
            "indice_v2": ind["indice_potencial"],
            "categoria_v1": ind1["categoria"],
            "categoria_v2": ind["categoria"],
            "rend_v1": ind1["componente_rendimiento"],
            "rend_v2": ind["componente_rendimiento"],
            "talento_v1": t1.get("es_talento_oculto"),
            "talento_v2": t.get("es_talento_oculto"),
            "puntaje_est_v1": round(p_est1, 2),
            "puntaje_est_v2": round(p_est, 2),
            "puntaje_real": real,
        })

    out = pd.DataFrame(filas)[COLS_ML_SCORES + ["modelo_version"]]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    log(f"scores → outputs/{OUT_CSV.name} ({len(out):,} filas × "
        f"{out.shape[1]} columnas)")

    comp = pd.DataFrame(comparacion)
    comp.to_csv(OUT_COMP, index=False, encoding="utf-8")
    log(f"comparación → outputs/{OUT_COMP.name}")

    # --- Paso 6: resumen -----------------------------------------------------
    titulo("PASO 6 — Comparación v1 vs v2")
    n_v2 = int((comp.modelo_version == "v2").sum())
    n_v1 = int((comp.modelo_version == "v1_fallback").sum())
    print(f"  Puntuados con v2          : {n_v2:,}")
    print(f"  Puntuados con v1_fallback : {n_v1:,}")

    sub = comp[comp.modelo_version == "v2"]
    d_ind = sub.indice_v2 - sub.indice_v1
    print(f"\n  Cambio de `indice_potencial` en los {len(sub):,} de v2:")
    print(f"    media   : {d_ind.mean():+.2f}")
    print(f"    mediana : {d_ind.median():+.2f}")
    print(f"    rango   : {d_ind.min():+.2f} … {d_ind.max():+.2f}")
    print(f"    sin cambio (|Δ|<0.01): {int((d_ind.abs() < 0.01).sum()):,}")

    # Cambios de categoría (solo tienen sentido donde se aplicó v2).
    orden = [c[1] for c in SPEC_V1["categorias"]][::-1]  # de menor a mayor
    rank = {n: i for i, n in enumerate(orden)}
    subida = int((sub.categoria_v2.map(rank) > sub.categoria_v1.map(rank)).sum())
    bajada = int((sub.categoria_v2.map(rank) < sub.categoria_v1.map(rank)).sum())
    print(f"\n  Cambios de categoría (entre los de v2):")
    print(f"    suben    : {subida:,}")
    print(f"    bajan    : {bajada:,}")
    print(f"    sin cambio: {len(sub) - subida - bajada:,}")

    print(f"\n  Talentos ocultos (toda la cohorte de {len(comp):,}):")
    print(f"    v1 : {int(comp.talento_v1.sum()):,}")
    print(f"    v2 : {int(comp.talento_v2.sum()):,}")
    print(f"    Δ  : {int(comp.talento_v2.sum()) - int(comp.talento_v1.sum()):+,}")

    # El modelo SÍ se nota aquí: puntaje_estimado es puramente predictivo.
    print(f"\n  `puntaje_estimado` en los {len(sub):,} de v2 "
          f"(aquí sí interviene el modelo):")
    d_p = sub.puntaje_est_v2 - sub.puntaje_est_v1
    print(f"    media del cambio : {d_p.mean():+.2f}")
    print(f"    MAE v1 vs real   : "
          f"{(sub.puntaje_est_v1 - sub.puntaje_real).abs().mean():.2f}")
    print(f"    MAE v2 vs real   : "
          f"{(sub.puntaje_est_v2 - sub.puntaje_real).abs().mean():.2f}")

    print("\n" + "=" * 78)
    print(" COMPLETADO — no se tocó ningún modelo ni la tabla ml_scores")
    print("=" * 78)


if __name__ == "__main__":
    main()
