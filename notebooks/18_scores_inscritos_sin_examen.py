# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 18: Scores de los inscritos que NO presentaron el examen
================================================================================

Por qué existe este script
--------------------------
Los scripts 15 y 17 puntuaron únicamente a los 3,072 estudiantes de
`dataset_B_completo.csv`, es decir los que ya tienen fila en
`resultados_prueba_copa_stem`. En ese grupo el componente de rendimiento usa el
puntaje REAL y el modelo de puntaje ni siquiera se invoca:

    real = _to_float(raw.get("puntaje_obtenido"))
    if real is not None:  rend_raw = real          # el modelo NO interviene
    else:                 rend_raw = _predict_puntaje(...)

La población donde el modelo SÍ manda el índice es la contraria: los inscritos
sin resultado. Para ellos no hay nota, así que `rend_raw` es la estimación del
modelo y `componente_rendimiento` —el 50 % del índice— sale entera de ahí.
Ninguna corrida anterior cubrió a ese grupo. Este script lo cubre.

Qué hace
--------
    1) Lee un export de `inscripciones_copa_stem` (ver la consulta más abajo) y
       se queda con los `numero_documento` que NO aparecen en la población
       examinada (`dataset_B_completo.csv` = `resultados_prueba_copa_stem`).
    2) Los puntúa con la MISMA estrategia híbrida de los scripts 15 y 17:
       SPEC v2 cuando está el bloque de perfil académico, SPEC v1 como fallback.
    3) Percentiliza contra `ref_rendimiento` de 3,072 puntajes —la referencia
       corregida del script 17, leída de `outputs/F17_ref_rendimiento_corregido.json`.
    4) Escribe `outputs/ml_scores_sin_examen.csv` con el mismo esquema de 18
       columnas de `outputs/ml_scores_v2_corrected.csv`, y la distribución de
       categorías comparada contra los examinados en `outputs/F18_distribucion_sin_examen.csv`.

Qué NO hace
-----------
No toca ningún script numerado anterior, ningún artefacto de `models/deploy/`,
ni Supabase, ni la Edge Function. No abre conexión a la base de datos: como
todos los scripts del repositorio, lee un CSV exportado a mano en `data/`.

Advertencia sobre `puntaje_estimado`
------------------------------------
Para este grupo `puntaje_estimado` NO es una nota: es una predicción del modelo
v2, cuyo MAE de validación es ~15 puntos sobre una escala de 0 a 100. Un valor
de 55 significa "probablemente entre 40 y 70", no "55". Nunca se debe publicar
como cifra puntual ni presentarse junto a los puntajes reales sin distinguirlos.
La columna `tiene_puntaje_real` vale False en las 100 % de las filas que produce
este script, precisamente para que esa distinción sea explícita.

Cómo obtener el export de entrada
---------------------------------
En el editor SQL de Supabase, y guardando el resultado como
`data/inscritos_copa_stem.csv`:

    SELECT
        i.numero_documento, i.institucion_educativa, i.grado_escolar,
        i.edad_calculada, i.genero, i.municipio, i.tipo_institucion,
        i.estrato, i.jornada, i.con_quien_vive,
        i.computador_en_casa, i.internet_en_casa,
        i.participacion_olimpiadas, i.nivel_programacion, i.nivel_robotica,
        i.herramientas_conocidas, i.areas_interes, i.interes_prog_robotica,
        i.promedio_academico, i.horas_estudio_matematicas,
        i.motivacion_participar, i.clases_extra_matematicas, i.gusto_logica,
        i.created_at
    FROM inscripciones_copa_stem i
    WHERE NOT EXISTS (
        SELECT 1 FROM resultados_prueba_copa_stem r
        WHERE r.numero_documento = i.numero_documento
    );

El `WHERE NOT EXISTS` es opcional: el script vuelve a hacer el anti-join contra
`dataset_B_completo.csv` de todas formas, así que también sirve un export
completo de `inscripciones_copa_stem` sin filtro. Si además se quiere cubrir
`inscripciones_emergencia`, se añade el mismo SELECT con UNION ALL; esas filas
traen varios campos socioeconómicos en NULL y los predictores los imputan con la
mediana de la cohorte.

Ninguna columna de telemetría del examen hace falta: ningún predictor la usa.

Modo de ensayo
--------------
    py notebooks/18_scores_inscritos_sin_examen.py --ensayo

Corre el pipeline completo sobre los 3,072 de `dataset_B_completo.csv` con el
puntaje real BORRADO, para verificar que la ruta "sin nota" funciona de punta a
punta. No escribe nada en `outputs/`. No es la población objetivo.

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

# La consola de Windows abre en cp1252 y los nombres de perfil llevan tildes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Export de `inscripciones_copa_stem`, en orden de preferencia: se usa el primero
# que exista, o la ruta que se pase como primer argumento.
FUENTES_INSCRITOS = [
    DATA_DIR / "inscritos_copa_stem.csv",
    DATA_DIR / "inscripciones_copa_stem.csv",
    DATA_DIR / "inscritos_sin_examen.csv",
]

EXAMINADOS = DATA_DIR / "dataset_B_completo.csv"      # = resultados_prueba_copa_stem
JS_V2 = DEPLOY_DIR / "potencial_stem_predictor_v2.js"
REF_CORREGIDA = OUTPUTS_DIR / "F17_ref_rendimiento_corregido.json"
SCORES_EXAMINADOS = OUTPUTS_DIR / "ml_scores_v2_corrected.csv"

OUT_CSV = OUTPUTS_DIR / "ml_scores_sin_examen.csv"
OUT_DIST = OUTPUTS_DIR / "F18_distribucion_sin_examen.csv"

TARGET = "puntaje_obtenido"
COL_PERFIL = "promedio_academico"  # centinela del bloque de perfil académico

# Las 5 preguntas de perfil académico que habilitan el SPEC v2.
BLOQUE_PERFIL = [
    "promedio_academico",
    "horas_estudio_matematicas",
    "motivacion_participar",
    "clases_extra_matematicas",
    "gusto_logica",
]

# Columnas del export que los predictores leen. Las que falten se imputan con la
# mediana/moda de la cohorte, pero conviene saber cuáles faltan.
COLS_ESPERADAS = [
    "numero_documento", "institucion_educativa", "grado_escolar",
    "edad_calculada", "genero", "municipio", "tipo_institucion", "estrato",
    "jornada", "con_quien_vive", "computador_en_casa", "internet_en_casa",
    "participacion_olimpiadas", "nivel_programacion", "nivel_robotica",
    "herramientas_conocidas", "areas_interes", "interes_prog_robotica",
] + BLOQUE_PERFIL

# Esquema de `outputs/ml_scores_v2_corrected.csv`: las 17 columnas que la Edge
# Function escribe en `ml_scores` + la marca de versión. Las 3 que no escribe
# (nivel_sospecha, n_criterios_sospecha, created_at) las pone el trigger
# `after_resultado_insert`, que para esta población nunca se dispara.
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

DOCS_PRUEBA = {"1234", "123456", "123456789", "1234567899", "0", "00000000"}

MAE_V2 = 15.00  # MAE de validación del modelo v2 (script 14)


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


def localizar_export(argv: list[str]) -> Path | None:
    """Ruta del export de inscripciones: argumento explícito o autodetección."""
    for a in argv[1:]:
        if not a.startswith("-"):
            p = Path(a)
            return p if p.is_absolute() else BASE_DIR / p
    for cand in FUENTES_INSCRITOS:
        if cand.exists():
            return cand
    return None


def ayuda_export() -> None:
    """Instrucciones para producir el CSV que falta."""
    print("\n  ADVERTENCIA: no se encontró el export de `inscripciones_copa_stem`.")
    print("  Se buscó, en este orden:")
    for c in FUENTES_INSCRITOS:
        print(f"    - {c.relative_to(BASE_DIR)}")
    print("\n  Ejecute en el editor SQL de Supabase y guarde el resultado como")
    print("  `data/inscritos_copa_stem.csv`:\n")
    print("""    SELECT
        i.numero_documento, i.institucion_educativa, i.grado_escolar,
        i.edad_calculada, i.genero, i.municipio, i.tipo_institucion,
        i.estrato, i.jornada, i.con_quien_vive,
        i.computador_en_casa, i.internet_en_casa,
        i.participacion_olimpiadas, i.nivel_programacion, i.nivel_robotica,
        i.herramientas_conocidas, i.areas_interes, i.interes_prog_robotica,
        i.promedio_academico, i.horas_estudio_matematicas,
        i.motivacion_participar, i.clases_extra_matematicas, i.gusto_logica,
        i.created_at
    FROM inscripciones_copa_stem i
    WHERE NOT EXISTS (
        SELECT 1 FROM resultados_prueba_copa_stem r
        WHERE r.numero_documento = i.numero_documento
    );""")
    print("\n  O pase la ruta como argumento:")
    print("    py notebooks/18_scores_inscritos_sin_examen.py ruta/al/export.csv")
    print("\n  Para verificar el pipeline sin el export:")
    print("    py notebooks/18_scores_inscritos_sin_examen.py --ensayo\n")


def cargar_ref(pot, SPEC_V1: dict, SPEC_V2: dict) -> tuple[list, str]:
    """Referencia de percentil: la corregida del script 17 (3,072 examinados).

    Se prefiere el JSON que el script 17 ya dejó escrito. Si no está, se
    reconstruye desde `dataset_B_completo.csv` con la misma semántica que usó
    aquel script, para no depender de un artefacto intermedio.
    """
    if REF_CORREGIDA.exists():
        d = json.loads(REF_CORREGIDA.read_text(encoding="utf-8"))
        ref = sorted(float(x) for x in d["ref_rendimiento"])
        return ref, f"{REF_CORREGIDA.name} (n={len(ref):,}, script 17)"

    log(f"    {REF_CORREGIDA.name} no está; se reconstruye desde {EXAMINADOS.name}")
    dfb = pd.read_csv(EXAMINADOS, encoding="utf-8", dtype={"numero_documento": str})
    dfb["numero_documento"] = dfb["numero_documento"].astype(str).str.strip()
    dfb = dfb[~dfb["numero_documento"].isin(DOCS_PRUEBA)]
    perfil_b = dfb[COL_PERFIL].notna() if COL_PERFIL in dfb.columns \
        else pd.Series(False, index=dfb.index)
    rend = []
    for r, v2 in zip(dfb.to_dict("records"), perfil_b):
        real = pot._to_float(r.get(TARGET))
        if real is not None:
            rend.append(real)
            continue
        spec = SPEC_V2 if v2 else SPEC_V1
        rend.append(pot._predict_puntaje(
            pot._features_puntaje(r, spec["puntaje"]["preprocess"]),
            spec["puntaje"]["model"]))
    ref = sorted(round(float(x), 4) for x in rend)
    return ref, f"reconstruida desde {EXAMINADOS.name} (n={len(ref):,})"


def limpiar_documentos(df: pd.DataFrame, etiqueta: str) -> pd.DataFrame:
    """Regla de limpieza estándar del proyecto sobre `numero_documento`."""
    n0 = len(df)
    df = df.copy()
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()

    vacios = df["numero_documento"].isin(["", "nan", "None", "NaN", "null"])
    df = df[~vacios]
    prueba = df["numero_documento"].isin(DOCS_PRUEBA)
    df = df[~prueba]
    cortos = df["numero_documento"].str.len() < 5
    df = df[~cortos]
    dup = df["numero_documento"].duplicated(keep="first")
    df = df[~dup]

    if n0 != len(df):
        log(f"    {etiqueta}: {n0:,} → {len(df):,} "
            f"(vacíos {int(vacios.sum()):,} · prueba {int(prueba.sum()):,} · "
            f"cortos {int(cortos.sum()):,} · duplicados {int(dup.sum()):,})")
    return df.reset_index(drop=True)


def puntuar(registros, tiene_perfil, pot, tal, cond, clus,
            SPEC_V1: dict, SPEC_V2: dict) -> list[dict]:
    """Vector de scores fila a fila, idéntico al de los scripts 15 y 17.

    La única diferencia es la entrada: aquí `puntaje_obtenido` no existe, así
    que `calcular_indice` toma por sí sola la rama del estimado. No se fuerza
    ninguna rama a mano; es la misma función que corre en la Edge Function.
    """
    filas = []
    ahora = datetime.now(timezone.utc).isoformat()

    for r, usa_v2 in zip(registros, tiene_perfil):
        spec = SPEC_V2 if usa_v2 else SPEC_V1
        version = "v2" if usa_v2 else "v1_fallback"

        ind = pot.calcular_indice(r, spec)

        # Para esta población `puntaje_estimado` y `_rend_raw` son el MISMO
        # número: sin nota real, el estimado ES el insumo del percentil.
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
    return filas


def sin_nota(registros: list[dict]) -> list[dict]:
    """Borra toda huella del resultado del examen de cada registro.

    Defensivo: si el export trae `puntaje_obtenido` o `porcentaje` con NULL,
    pandas los deja como NaN y los predictores ya los tratan como ausentes,
    pero si trajera un 0 real la rama del índice cambiaría en silencio.
    """
    fuera = (TARGET, "porcentaje", "tiempo_usado_segundos", "cambios_pestana",
             "intentos_copiar", "intentos_pegar", "intentos_click_derecho")
    return [{k: v for k, v in r.items() if k not in fuera} for r in registros]


def conteos(serie: pd.Series, orden: list) -> list:
    vc = serie.value_counts()
    return [int(vc.get(c, 0)) for c in orden]


# ---------------------------------------------------------------------------
def main() -> None:
    ensayo = "--ensayo" in sys.argv

    print("=" * 78)
    print(" COPA STEM 2026 — Scores de los inscritos SIN examen")
    print(f" Fundación SapienceLab · Script 18 — "
          f"{'ENSAYO (no escribe nada)' if ensayo else 'ejecución en SOMBRA'}")
    print("=" * 78)

    if not EXAMINADOS.exists():
        print(f"\n  ADVERTENCIA: falta {EXAMINADOS}. Ejecute antes el script 11.\n")
        sys.exit(1)

    # --- Paso 2: predictores y SPECs ----------------------------------------
    titulo("PASO 2 — Carga de predictores y SPECs")
    pot = cargar_modulo("pot_v1", DEPLOY_DIR / "potencial_stem_predictor.py")
    tal = cargar_modulo("talento", DEPLOY_DIR / "talento_oculto_predictor.py")
    cond = cargar_modulo("cond", DEPLOY_DIR / "indice_condiciones_predictor.py")
    clus = cargar_modulo("clus", DEPLOY_DIR / "clustering_predictor.py")
    log("predictores cargados desde models/deploy/ (stdlib puro)")

    SPEC_V1 = pot.SPEC
    SPEC_V2 = spec_desde_js(JS_V2)
    log(f"SPEC v1 → {len(SPEC_V1['puntaje']['preprocess']['numeric'])} numéricas | "
        f"ref propia de {len(SPEC_V1['ref_rendimiento']):,}")
    log(f"SPEC v2 → {len(SPEC_V2['puntaje']['preprocess']['numeric'])} numéricas | "
        f"ref propia de {len(SPEC_V2['ref_rendimiento']):,}")

    # --- Paso 3: referencia corregida (script 17) ---------------------------
    titulo("PASO 3 — Referencia de percentil (fix del script 17)")
    REF_B, origen_ref = cargar_ref(pot, SPEC_V1, SPEC_V2)
    a = np.asarray(REF_B, dtype=float)
    log(f"origen: {origen_ref}")
    print(f"  n = {len(a):,} | media {a.mean():.2f} | σ {a.std(ddof=0):.2f} | "
          f"p25 {np.percentile(a, 25):.1f} | mediana {np.median(a):.1f} | "
          f"p75 {np.percentile(a, 75):.1f}")
    if len(REF_B) != 3072:
        print(f"  ADVERTENCIA: se esperaban 3,072 puntajes de referencia, "
              f"hay {len(REF_B):,}.")

    SPEC_V1_C = copy.deepcopy(SPEC_V1)
    SPEC_V2_C = copy.deepcopy(SPEC_V2)
    SPEC_V1_C["ref_rendimiento"] = REF_B
    SPEC_V2_C["ref_rendimiento"] = REF_B
    log("SPECs clonados: solo cambia `ref_rendimiento`; modelo y pesos intactos")

    # --- Paso 1/4: población objetivo ---------------------------------------
    titulo("PASO 4 — Población: inscritos sin fila en resultados")

    dfb = pd.read_csv(EXAMINADOS, encoding="utf-8", dtype={"numero_documento": str})
    dfb["numero_documento"] = dfb["numero_documento"].astype(str).str.strip()
    # El lado examinado NO se limpia: es la lista de quién tiene fila en
    # `resultados_prueba_copa_stem`. Descartar aquí un documento de prueba o
    # corto lo devolvería a la población objetivo como si nunca hubiera
    # presentado, que es justo lo contrario de lo que se quiere.
    docs_examinados = set(dfb["numero_documento"]) - {"", "nan", "None", "NaN", "null"}
    log(f"examinados (resultados_prueba_copa_stem): {len(docs_examinados):,} "
        f"documentos, sin filtrar")

    if ensayo:
        # Los mismos 3,072, con el resultado del examen borrado. Sirve para
        # verificar la ruta "sin nota" de punta a punta; no es la población real.
        df = limpiar_documentos(dfb, "ensayo")
        log(f"MODO ENSAYO — {len(df):,} examinados con el puntaje real borrado")
        log("            los números de abajo NO son la población objetivo")
    else:
        fuente = localizar_export(sys.argv)
        if fuente is None or not fuente.exists():
            ayuda_export()
            sys.exit(1)
        df = pd.read_csv(fuente, encoding="utf-8", dtype={"numero_documento": str})
        log(f"export leído: {fuente.name} ({len(df):,} filas)")
        df = limpiar_documentos(df, fuente.name)

        faltan = [c for c in COLS_ESPERADAS if c not in df.columns]
        if faltan:
            print(f"  ADVERTENCIA: el export no trae {len(faltan)} columnas que los "
                  f"predictores leen; se imputarán con la mediana/moda de cohorte:")
            for c in faltan:
                print(f"    - {c}")

        antes = len(df)
        df = df[~df["numero_documento"].isin(docs_examinados)].reset_index(drop=True)
        log(f"anti-join contra resultados: {antes:,} inscritos → "
            f"{len(df):,} SIN examen ({antes - len(df):,} ya presentaron)")

    if len(df) == 0:
        print("\n  ADVERTENCIA: no quedó ningún estudiante por puntuar.\n")
        sys.exit(1)

    tiene_perfil = df[COL_PERFIL].notna() if COL_PERFIL in df.columns \
        else pd.Series(False, index=df.index)
    n_v2, n_v1 = int(tiene_perfil.sum()), int((~tiene_perfil).sum())
    log(f"con `{COL_PERFIL}` → v2        : {n_v2:,} "
        f"({n_v2 / len(df) * 100:.1f} %)")
    log(f"sin perfil       → v1 fallback: {n_v1:,} "
        f"({n_v1 / len(df) * 100:.1f} %)")

    presentes = [c for c in BLOQUE_PERFIL if c in df.columns]
    if presentes:
        completo = df[presentes].notna().all(axis=1)
        log(f"con las {len(presentes)} preguntas de perfil completas: "
            f"{int(completo.sum()):,}")

    # --- Paso 5: scoring -----------------------------------------------------
    titulo("PASO 5 — Cálculo del vector de scores")
    registros = sin_nota(df.to_dict("records"))
    assert all(TARGET not in r for r in registros), \
        "quedó `puntaje_obtenido` en algún registro"

    filas = puntuar(registros, tiene_perfil, pot, tal, cond, clus,
                    SPEC_V1_C, SPEC_V2_C)
    out = pd.DataFrame(filas)[COLS_ML_SCORES + ["modelo_version"]]

    # Invariante de esta población: nadie tiene nota, y el insumo del percentil
    # es siempre el estimado del modelo.
    n_real = int(out.tiene_puntaje_real.sum())
    if n_real:
        print(f"  ADVERTENCIA: {n_real:,} filas con `tiene_puntaje_real` = True.")
    else:
        log("verificado: `tiene_puntaje_real` = False en las "
            f"{len(out):,} filas")

    if ensayo:
        titulo("ENSAYO — resumen (no se escribió ningún archivo)")
        print(f"  filas puntuadas          : {len(out):,}")
        print(f"  columnas                 : {out.shape[1]} "
              f"({'OK' if out.shape[1] == 18 else 'NO son 18'})")
        print(f"  `puntaje_estimado` media : {out.puntaje_estimado.mean():.2f} "
              f"| σ {out.puntaje_estimado.std(ddof=0):.2f} "
              f"| rango {out.puntaje_estimado.min():.1f}–"
              f"{out.puntaje_estimado.max():.1f}")
        orden = [c[1] for c in SPEC_V1["categorias"]]
        for cat, n in zip(orden, conteos(out.categoria_potencial, orden)):
            print(f"    {cat:<20}{n:>7,}  {n / len(out) * 100:5.1f} %")
        # Estos sí tienen nota real guardada: sirve para medir cuánto se
        # equivoca el estimado que en la población objetivo será todo lo que hay.
        real = dfb.set_index("numero_documento").loc[out.numero_documento, TARGET]
        mae = (out.puntaje_estimado.values - real.values).__abs__().mean()
        print(f"\n  MAE del estimado contra la nota real de estos mismos "
              f"estudiantes: {mae:.2f}")
        print(f"  σ del estimado {out.puntaje_estimado.std(ddof=0):.2f} frente a "
              f"σ {real.std(ddof=0):.2f} de la nota real: el modelo comprime.")
        print("\n" + "=" * 78)
        print(" ENSAYO COMPLETADO — la ruta 'sin nota' corre de punta a punta")
        print("=" * 78)
        return

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    log(f"scores → outputs/{OUT_CSV.name} ({len(out):,} filas × "
        f"{out.shape[1]} columnas)")

    # --- Paso 6: distribución de categorías ---------------------------------
    titulo("PASO 6 — Distribución de categorías")
    orden = [c[1] for c in SPEC_V1["categorias"]]      # de mayor a menor
    n_sin = conteos(out.categoria_potencial, orden)

    n_exam, tot_exam = None, 0
    if SCORES_EXAMINADOS.exists():
        exam = pd.read_csv(SCORES_EXAMINADOS, dtype={"numero_documento": str})
        n_exam = conteos(exam.categoria_potencial, orden)
        tot_exam = len(exam)
    else:
        log(f"    {SCORES_EXAMINADOS.name} no está; se omite la comparación")

    cab = f"  {'Categoría':<20}{'sin examen':>12}{'%':>8}"
    if n_exam:
        cab += f"{'examinados':>12}{'%':>8}{'Δ pp':>8}"
    print(cab)
    filas_dist = []
    for i, cat in enumerate(orden):
        p_sin = n_sin[i] / len(out) * 100
        linea = f"  {cat:<20}{n_sin[i]:>12,}{p_sin:>7.1f}%"
        fila = {"categoria": cat, "n_sin_examen": n_sin[i],
                "pct_sin_examen": round(p_sin, 2)}
        if n_exam:
            p_ex = n_exam[i] / tot_exam * 100
            linea += f"{n_exam[i]:>12,}{p_ex:>7.1f}%{p_sin - p_ex:>+8.1f}"
            fila.update({"n_examinados": n_exam[i],
                         "pct_examinados": round(p_ex, 2),
                         "delta_pp": round(p_sin - p_ex, 2)})
        print(linea)
        filas_dist.append(fila)

    pd.DataFrame(filas_dist).to_csv(OUT_DIST, index=False, encoding="utf-8")
    log(f"distribución → outputs/{OUT_DIST.name}")

    print(f"\n  Por versión de modelo:")
    for v in ("v2", "v1_fallback"):
        sub = out[out.modelo_version == v]
        if len(sub):
            print(f"    {v:<14} n = {len(sub):>6,} | índice medio "
                  f"{sub.indice_potencial.mean():5.2f} | estimado medio "
                  f"{sub.puntaje_estimado.mean():5.2f}")

    # --- Paso 7: el número que hay que leer con pinzas ----------------------
    titulo("PASO 7 — `puntaje_estimado`: incertidumbre, no medición")
    pe = out.puntaje_estimado
    print(f"  n                : {len(pe):,} (el 100 % son estimaciones)")
    print(f"  media            : {pe.mean():.2f}")
    print(f"  σ                : {pe.std(ddof=0):.2f}")
    print(f"  rango            : {pe.min():.1f} … {pe.max():.1f}")
    print(f"  MAE del modelo v2: ±{MAE_V2:.2f} puntos (validación, script 14)")
    print(f"\n  Un estimado de {pe.mean():.0f} significa 'probablemente entre "
          f"{pe.mean() - MAE_V2:.0f} y {pe.mean() + MAE_V2:.0f}'.")
    print("  Estas cifras NO son notas. No deben publicarse como valor puntual,")
    print("  ni mezclarse en un mismo ranking con los puntajes reales de los")
    print("  examinados sin marcar la diferencia (`tiene_puntaje_real`).")

    print("\n" + "=" * 78)
    print(" COMPLETADO — cálculo local; no se tocó Supabase, la Edge Function")
    print(" ni ningún script o artefacto anterior")
    print("=" * 78)


if __name__ == "__main__":
    main()
