# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 11: Preparación del experimento de reentrenamiento (Fase 5)
================================================================================

Objetivo
--------
Dejar listos los datasets de un experimento CONTROLADO que compara tres
configuraciones de modelo, y reportar el estado real de los datos antes de
entrenar. **Este script NO entrena ningún modelo**: solo prepara y reporta.

Configuraciones del experimento
-------------------------------
    A)  dataset_A_baseline.csv      — la cohorte original (~1.735) con la que se
                                      entrenó el primer modelo. Línea base.
    B)  dataset_B_completo.csv      — todas las filas con `puntaje_obtenido`
                                      (~3.207). Aísla el efecto de MÁS DATOS.
    C)  dataset_C_perfil.csv        — solo filas con perfil académico declarado
                                      (~1.239). Efecto de MÁS VARIABLES.
    C') dataset_C_sin_features.csv  — las MISMAS filas de C pero sin las 5
                                      variables nuevas. GRUPO DE CONTROL: la
                                      comparación C vs C' mantiene la muestra
                                      fija, así que la diferencia de métricas se
                                      atribuye únicamente a las variables nuevas
                                      y no a un cambio de población.

Fuente de datos
---------------
Ningún script del repositorio abre conexión a Supabase: todos leen el CSV
exportado a mano en `data/`. Aquí se mantiene ese patrón — se lee el primer
export que exista de `FUENTES_DEFAULT` (por defecto el de agosto de 2026), o la
ruta que se pase como primer argumento. Los cuatro datasets resultantes se
escriben también en `data/`.

Identificación de la cohorte baseline (A), en cascada:
    1. Columna de fecha (`created_at`) — las N inscripciones más antiguas.
    2. Índice guardado del primer modelo (models/deploy/puntaje_estimado.csv).
    3. Muestra aleatoria de 1.735 filas con `random_state=42`.

Reproducible: `random_state=42`.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
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

# Export de Supabase, en orden de preferencia: se usa el primero que exista.
# El segundo candidato cubre la doble extensión con la que quedó guardado el
# export de agosto de 2026 (`...csv.csv`); si se renombra, el primero lo toma.
FUENTES_DEFAULT = [
    DATA_DIR / "copa_stem_dataset_2026-08.csv",
    DATA_DIR / "copa_stem_dataset_2026-08.csv.csv",
    DATA_DIR / "copa_stem_dataset.csv",
]

OUT_A = DATA_DIR / "dataset_A_baseline.csv"
OUT_B = DATA_DIR / "dataset_B_completo.csv"
OUT_C = DATA_DIR / "dataset_C_perfil.csv"
OUT_C_SIN = DATA_DIR / "dataset_C_sin_features.csv"

TARGET = "puntaje_obtenido"

# Las 5 preguntas de perfil académico añadidas al formulario de inscripción.
# Los primeros ~2.000 inscritos las tienen en NULL (no existían en el formulario).
NUEVAS_FEATURES = [
    "promedio_academico",
    "horas_estudio_matematicas",
    "motivacion_participar",
    "clases_extra_matematicas",
    "gusto_logica",
]
COL_PERFIL = "promedio_academico"  # centinela del bloque de perfil académico

# Documentos de prueba a descartar (regla estándar del proyecto).
DOCS_PRUEBA = {"1234", "123456", "123456789", "1234567899"}

# Columnas que NO cuentan como feature: identificadores, el target y su duplicado
# exacto (`porcentaje` == `puntaje_obtenido`; usarlo sería fuga de información).
NO_FEATURES = {"numero_documento", "nombres", "apellidos", TARGET, "porcentaje"}

# Candidatos a "índice guardado" de la cohorte del primer modelo, por preferencia.
INDICES_BASELINE = [
    MODELS_DIR / "deploy" / "puntaje_estimado.csv",
    MODELS_DIR / "scores_potencial_stem.csv",
    MODELS_DIR / "deploy" / "modelo_teorico_scores.csv",
]

N_BASELINE = 1735  # tamaño nominal de la cohorte del primer modelo

# Nombres posibles de la columna de fecha de inscripción según el export.
COLS_FECHA = ("created_at", "fecha_inscripcion", "creado_en", "fecha_registro")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def titulo(txt: str) -> None:
    print("\n" + "=" * 78)
    print(f" {txt}")
    print("=" * 78)


def n_nulos(serie: pd.Series) -> int:
    """Cuenta faltantes. En columnas de texto, las cadenas vacías o en blanco
    también cuentan como faltante (Supabase exporta '' para campos sin valor)."""
    if serie.dtype == object:
        vacias = serie.astype(str).str.strip().isin(["", "nan", "None", "NaN", "null"])
        return int((serie.isna() | vacias).sum())
    return int(serie.isna().sum())


def n_features(df: pd.DataFrame) -> int:
    return sum(1 for c in df.columns if c not in NO_FEATURES)


def serie_target(df: pd.DataFrame) -> pd.Series:
    if TARGET not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[TARGET], errors="coerce").dropna()


# ---------------------------------------------------------------------------
# Paso 1 — Carga
# ---------------------------------------------------------------------------
def resolver_fuente() -> Path:
    """Primer candidato de FUENTES_DEFAULT que exista en disco."""
    for ruta in FUENTES_DEFAULT:
        if ruta.exists():
            return ruta
    print("\n  ADVERTENCIA: no se encontró ningún export en data/. Se buscó:")
    for ruta in FUENTES_DEFAULT:
        print(f"     · {ruta.name}")
    print("     Exporte la tabla de inscripciones desde Supabase a una de esas rutas.\n")
    sys.exit(1)


def cargar(fuente: Path) -> pd.DataFrame:
    if not fuente.exists():
        print(f"\n  ADVERTENCIA: no se encontró el dataset: {fuente}")
        print("     Exporte la tabla de inscripciones desde Supabase a esa ruta.\n")
        sys.exit(1)

    df = pd.read_csv(fuente, encoding="utf-8", dtype={"numero_documento": str})
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()

    # Coerción numérica solo de las columnas que se usan como número. Las tres
    # variables nuevas restantes pueden venir como texto (escala Likert) y se
    # dejan tal cual para no destruir información.
    for col in (TARGET, "porcentaje", "edad_calculada", "tiempo_usado_segundos",
                "cambios_pestana", "intentos_copiar", "intentos_pegar",
                "intentos_click_derecho", "promedio_academico",
                "horas_estudio_matematicas"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Paso 2 — Reporte del estado actual
# ---------------------------------------------------------------------------
def reportar_estado(df: pd.DataFrame) -> None:
    titulo("PASO 2 — Estado actual del dataset")
    total = len(df)

    print(f"  Filas totales:                 {total:,}")
    print(f"  Columnas:                      {df.shape[1]}")

    con_target = int(df[TARGET].notna().sum()) if TARGET in df.columns else 0
    print(f"  Con `{TARGET}` no nulo:  {con_target:,} "
          f"({100 * con_target / total:.1f}%)")

    # --- Bloque de perfil académico (las 5 variables nuevas) ----------------
    print("\n  Perfil académico — las 5 variables nuevas:")
    ausentes = [c for c in NUEVAS_FEATURES if c not in df.columns]
    for col in NUEVAS_FEATURES:
        if col not in df.columns:
            print(f"    {col:<30} COLUMNA AUSENTE en este export")
            continue
        llenos = total - n_nulos(df[col])
        print(f"    {col:<30} {llenos:>6,} con dato "
              f"({100 * llenos / total:5.1f}%)")

    if COL_PERFIL in df.columns:
        perfil = total - n_nulos(df[COL_PERFIL])
        print(f"\n  Filas con `{COL_PERFIL}` no nulo: {perfil:,} "
              f"({100 * perfil / total:.1f}%)")
        if TARGET in df.columns:
            ambos = int((df[COL_PERFIL].notna() & df[TARGET].notna()).sum())
            print(f"  De ellas, con puntaje (entrenables): {ambos:,}")

    if ausentes:
        print(f"\n  ADVERTENCIA: {len(ausentes)} de las 5 variables nuevas no "
              f"existen en este export.")
        print("     El CSV en `data/` es anterior a la ampliación del "
              "formulario de inscripción.")
        print("     Reexporte desde Supabase para poder construir el dataset C.")

    # --- Faltantes por columna ----------------------------------------------
    print("\n  Faltantes por columna (orden descendente):")
    print(f"    {'columna':<32} {'nulos':>9} {'% nulos':>9}")
    print(f"    {'-' * 32} {'-' * 9} {'-' * 9}")
    faltantes = sorted(((c, n_nulos(df[c])) for c in df.columns),
                       key=lambda x: -x[1])
    for col, nn in faltantes:
        print(f"    {col:<32} {nn:>9,} {100 * nn / total:>8.1f}%")

    # --- Distribución del target --------------------------------------------
    print(f"\n  Distribución de `{TARGET}`:")
    t = serie_target(df)
    if t.empty:
        print("    (sin datos de puntaje)")
        return
    q1, q2, q3 = t.quantile([0.25, 0.50, 0.75])
    print(f"    n          : {len(t):,}")
    print(f"    media      : {t.mean():.2f}")
    print(f"    desv. est. : {t.std():.2f}")
    print(f"    mínimo     : {t.min():.2f}")
    print(f"    Q1 (25%)   : {q1:.2f}")
    print(f"    mediana    : {q2:.2f}")
    print(f"    Q3 (75%)   : {q3:.2f}")
    print(f"    máximo     : {t.max():.2f}")


# ---------------------------------------------------------------------------
# Paso 2b — Limpieza mínima antes de partir
# ---------------------------------------------------------------------------
def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Registros de prueba y duplicados por documento (PK en Supabase). Se hace
    DESPUÉS del reporte de estado para que ese reporte describa el export crudo."""
    titulo("PASO 2b — Limpieza mínima previa a los splits")
    n0 = len(df)

    df = df[~df["numero_documento"].isin(DOCS_PRUEBA)].copy()
    n_prueba = n0 - len(df)

    dupes = int(df.duplicated(subset="numero_documento", keep="first").sum())
    df = df.drop_duplicates(subset="numero_documento", keep="first").copy()

    log(f"    registros de prueba retirados : {n_prueba:,}")
    log(f"    duplicados por documento      : {dupes:,} (se conserva el primero)")
    log(f"    filas tras la limpieza        : {len(df):,} (de {n0:,})")
    return df


# ---------------------------------------------------------------------------
# Paso 3 — Cohorte baseline (dataset A)
# ---------------------------------------------------------------------------
def columna_fecha(df: pd.DataFrame) -> str | None:
    for c in COLS_FECHA:
        if c in df.columns:
            return c
    return None


def identificar_baseline(df_target: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Cascada de identificación de la cohorte con la que se entrenó el primer
    modelo. Devuelve (subconjunto, descripción del criterio usado)."""

    # 1) Por fecha de inscripción: las N más antiguas.
    col_fecha = columna_fecha(df_target)
    if col_fecha is not None:
        fechas = pd.to_datetime(df_target[col_fecha], errors="coerce", utc=True)
        if fechas.notna().any():
            orden = df_target.assign(_fecha=fechas).sort_values(
                "_fecha", kind="mergesort", na_position="last")
            sub = orden.head(N_BASELINE).drop(columns="_fecha").sort_index()
            return sub, (f"las {len(sub):,} inscripciones más antiguas según "
                         f"`{col_fecha}`")

    # 2) Por índice guardado: los documentos que el primer modelo puntuó.
    for ruta in INDICES_BASELINE:
        if not ruta.exists():
            continue
        idx = pd.read_csv(ruta, dtype={"numero_documento": str})
        if "numero_documento" not in idx.columns:
            continue
        docs = set(idx["numero_documento"].astype(str).str.strip())
        sub = df_target[df_target["numero_documento"].isin(docs)].copy()
        if sub.empty:
            continue
        rel = ruta.relative_to(BASE_DIR).as_posix()
        return sub, (f"índice guardado del primer modelo — {rel} "
                     f"({len(docs):,} documentos, {len(sub):,} emparejados)")

    # 3) Fallback reproducible.
    n = min(N_BASELINE, len(df_target))
    sub = df_target.sample(n=n, random_state=RANDOM_STATE).sort_index()
    return sub, (f"muestra aleatoria de {n:,} filas "
                 f"(random_state={RANDOM_STATE}) — no se pudo identificar la "
                 f"cohorte original")


# ---------------------------------------------------------------------------
# Paso 5 — Tabla resumen
# ---------------------------------------------------------------------------
def fila_resumen(nombre: str, rol: str, df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return [nombre, rol, "—", "—", "—", "—"]
    t = serie_target(df)
    return [
        nombre, rol, f"{len(df):,}", f"{n_features(df)}",
        f"{t.mean():.2f}" if len(t) else "—",
        f"{t.std():.2f}" if len(t) else "—",
    ]


def imprimir_tabla(filas: list[list[str]]) -> None:
    cab = ["Dataset", "Rol en el experimento", "Filas", "Features",
           "Media target", "Desv. target"]
    anchos = [max(len(cab[i]), *(len(f[i]) for f in filas))
              for i in range(len(cab))]
    sep = "  ".join("-" * a for a in anchos)

    print("  " + "  ".join(c.ljust(anchos[i]) for i, c in enumerate(cab)))
    print("  " + sep)
    for f in filas:
        # Nombre y rol a la izquierda; las métricas numéricas a la derecha.
        celdas = [f[0].ljust(anchos[0]), f[1].ljust(anchos[1])]
        celdas += [f[i].rjust(anchos[i]) for i in range(2, len(cab))]
        print("  " + "  ".join(celdas))
    print("  " + sep)


# ---------------------------------------------------------------------------
def main() -> None:
    fuente = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else resolver_fuente()

    print("=" * 78)
    print(" COPA STEM 2026 — Preparación del experimento de reentrenamiento")
    print(" Fundación SapienceLab · Script 11 — NO entrena ningún modelo")
    print("=" * 78)

    # --- Paso 1 --------------------------------------------------------------
    log(f"Paso 1 — Carga desde {fuente.name}")
    df = cargar(fuente)
    log(f"    {len(df):,} filas x {df.shape[1]} columnas")

    if TARGET not in df.columns:
        print(f"\n  ADVERTENCIA: el export no tiene la columna `{TARGET}`. "
              "No hay experimento posible.\n")
        sys.exit(1)

    # --- Paso 2 --------------------------------------------------------------
    reportar_estado(df)
    df = limpiar(df)

    # --- Paso 3 --------------------------------------------------------------
    titulo("PASO 3 — Construcción de los splits")

    # B: universo de entrenamiento — todo el que presentó el examen.
    ds_b = df[df[TARGET].notna()].copy()
    log(f"B (completo)  : {len(ds_b):,} filas con `{TARGET}` no nulo")

    # A: la cohorte del primer modelo, siempre dentro del universo de B.
    ds_a, criterio = identificar_baseline(ds_b)
    log(f"A (baseline)  : {len(ds_a):,} filas")
    log(f"                criterio -> {criterio}")
    if abs(len(ds_a) - N_BASELINE) > 50:
        log(f"                OJO: difiere del tamaño nominal esperado "
            f"(~{N_BASELINE:,})")

    # C: solo quienes declararon perfil académico. Se define por el enunciado
    # del experimento (`promedio_academico` no nulo), sin exigir el target; más
    # abajo se reporta cuántas de esas filas son realmente entrenables.
    ds_c = ds_c_sin = None
    if COL_PERFIL not in df.columns:
        log(f"C (perfil)    : NO SE GENERA — la columna `{COL_PERFIL}` no "
            f"existe en este export")
    else:
        ds_c = df[df[COL_PERFIL].notna()].copy()
        log(f"C (perfil)    : {len(ds_c):,} filas con `{COL_PERFIL}` no nulo")
        if ds_c.empty:
            log("                OJO: ninguna fila tiene perfil académico; "
                "no se generan C ni C'")
            ds_c = None
        else:
            con_t = int(ds_c[TARGET].notna().sum())
            log(f"                de ellas, {con_t:,} con `{TARGET}` "
                f"(las entrenables)")
            if con_t < len(ds_c):
                log(f"                OJO: {len(ds_c) - con_t:,} filas de C aún "
                    f"no presentaron el examen; descártelas al entrenar")

            # --- Paso 4: grupo de control, mismas filas sin las 5 nuevas -----
            a_quitar = [c for c in NUEVAS_FEATURES if c in ds_c.columns]
            ds_c_sin = ds_c.drop(columns=a_quitar).copy()
            log(f"C' (control)  : {len(ds_c_sin):,} filas, "
                f"{len(a_quitar)} columnas nuevas retiradas "
                f"({', '.join(a_quitar)})")

    # --- Guardado ------------------------------------------------------------
    print()
    guardados: list[tuple[Path, pd.DataFrame]] = [(OUT_A, ds_a), (OUT_B, ds_b)]
    if ds_c is not None:
        guardados += [(OUT_C, ds_c), (OUT_C_SIN, ds_c_sin)]

    for ruta, data in guardados:
        # UTF-8 sin BOM, igual que el resto de CSVs de `data/`.
        data.to_csv(ruta, index=False, encoding="utf-8")
        log(f"guardado -> data/{ruta.name} ({len(data):,} filas x "
            f"{data.shape[1]} columnas)")

    if ds_c is None:
        print(f"\n  ADVERTENCIA: no se generaron data/{OUT_C.name} ni "
              f"data/{OUT_C_SIN.name}.")
        print("     Requieren un export de Supabase que incluya el perfil "
              "académico.")

    # --- Paso 5 --------------------------------------------------------------
    titulo("PASO 5 — Resumen comparativo de los datasets")
    filas = [
        fila_resumen("A_baseline", "línea base (cohorte original)", ds_a),
        fila_resumen("B_completo", "efecto de más DATOS", ds_b),
        fila_resumen("C_perfil", "efecto de más VARIABLES", ds_c),
        fila_resumen("C_sin_features", "control de C (mismas filas)", ds_c_sin),
    ]
    imprimir_tabla(filas)
    print("  Features = columnas menos identificadores, `puntaje_obtenido` y")
    print("  `porcentaje` (duplicado exacto del target -> fuga de información).")
    print("  Media/desv. del target se calculan solo sobre filas con puntaje.")

    print("\n" + "=" * 78)
    print(" PREPARACIÓN COMPLETADA — no se entrenó ningún modelo")
    print(" Siguiente paso: entrenar la misma familia de modelo sobre A, B, C y")
    print(" C' con idéntico protocolo de validación y comparar R2 y MAE.")
    print("   · A vs B  -> ¿ayudan más datos?")
    print("   · C vs C' -> ¿aportan las 5 variables nuevas? (muestra fija)")
    print("=" * 78)


if __name__ == "__main__":
    main()
