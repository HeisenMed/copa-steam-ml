# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 10: Modelo Teórico vs Empírico  (Fase 4 — Robustez / Auditoría)
================================================================================

Contexto
--------
El modelo EMPÍRICO (Random Forest, script 03) aprendió de los datos reales de
Copa STEM. Si esos datos contienen trampa o sesgos, el modelo los aprendió. Este
script construye un modelo TEÓRICO puro (`indice_condiciones`) que NO usa ningún
resultado de Copa STEM: sus pesos vienen SOLO de la literatura educativa
internacional (OECD PISA, UNESCO, meta-análisis de Sirin y Hattie). Comparar
ambos permite auditar los datos y detectar anomalías.

Corrección de dato conocida
---------------------------
En Copacabana, Girardota y Bello el estrato solo va de 1 a 3. El dataset ya viene
corregido en origen (los antiguos valores 4/5/6 de autorreporte fueron
reclasificados a 3), por lo que aquí el estrato se usa directamente en [1, 3].

Secciones
---------
    A) Modelo teórico puro: indice_condiciones (reglas de literatura, [5, 95]).
    B) Comparación triple: teórico vs empírico vs real (R², MAE, correlación).
    C) Detección de anomalías (diferencia_teorica alta + tiempo bajo) y cruce con
       sospecha_trampa.csv (script 05).
    D) Valor del índice de condiciones y cruce con talento oculto (script 06).
    E) Exportar: modelo_teorico_scores.csv + predictor puro .py y .js.
    F) Informe reports/10_modelo_teorico_vs_empirico.md

Reproducible: `random_state=42`.
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

RANDOM_STATE = 42

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
except ImportError as exc:  # pragma: no cover
    print("ERROR: falta una dependencia del entorno.")
    print(f"       Detalle: {exc}")
    sys.exit(1)

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
for _d in (OUTPUTS_DIR, REPORTS_DIR, DEPLOY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

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
STEM = LinearSegmentedColormap.from_list("stem_grad", PALETTE)

sns.set_theme(style="whitegrid")
sns.set_palette(PALETTE)
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.autolayout": True,
})
DPI = 150

# Umbrales.
ANOMALIA_DIF = 30.0       # diferencia_teorica > 30 → "demasiado bueno para el contexto"
TIEMPO_SOSPECHA = 30 * 60  # < 30 min (en segundos)
COND_FAVORABLE, COND_ADVERSA = 60, 45

FIGURES: dict[str, str] = {}
REPORT: list[str] = []


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def savefig(fig, filename: str, key: str) -> None:
    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES[key] = filename
    log(f"    figura guardada → outputs/{filename}")


def img(key: str, alt: str) -> str:
    f = FIGURES.get(key)
    return f"![{alt}](../outputs/{f})" if f else f"_(figura '{alt}' no disponible)_"


def tabla_md(df: pd.DataFrame) -> str:
    enc = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    filas = ["| " + " | ".join(str(v) for v in row) + " |"
             for row in df.itertuples(index=False)]
    return "\n".join([enc, sep] + filas)


def _import_por_ruta(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# A. MODELO TEÓRICO PURO — indice_condiciones
# -----------------------------------------------------------------------------
# Estas funciones NO usan datos de Copa STEM: los pesos vienen de literatura
# educativa. Son PURAS (solo stdlib) y se EMBEBEN en el predictor exportado.
# =============================================================================

def _txt(v):
    """Normaliza a texto en minúsculas sin espacios; None/NaN → ''."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    return str(v).strip().lower()


def _num(v):
    """Convierte a float; None si no es número real."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _si_no(v, w_si, w_no):
    """Sí* → w_si, No* → w_no, desconocido → 0 (neutral)."""
    s = _txt(v)
    if s.startswith("s"):
        return w_si
    if s.startswith("n"):
        return w_no
    return 0


def _estrato_ajuste(v):
    """Estrato 1–3 (máx real en Copacabana/Girardota/Bello). 1=-3, 2=0, 3=+2;
    fuera de rango o NaN → 0."""
    f = _num(v)
    if f is None:
        return 0
    e = int(round(f))
    return {1: -3, 2: 0, 3: 2}.get(e, 0)


def _nivel_ajuste(v, pesos):
    """Mapea Ninguna/Básica/Intermedia/Avanzada según `pesos`; desconocido → 0."""
    s = _txt(v)
    m = {"ninguna": pesos[0], "ninguno": pesos[0], "básica": pesos[1],
         "basica": pesos[1], "intermedia": pesos[2], "avanzada": pesos[3]}
    return m.get(s, 0)


def _interes_ajuste(v):
    """Interés 1–5: bajo(1-2)=-2, medio(3)=0, alto(4-5)=+3; desconocido → 0."""
    f = _num(v)
    if f is None:
        return 0
    if f <= 2:
        return -2
    if f >= 4:
        return 3
    return 0


def _n_herramientas(v):
    """Cuenta herramientas de una lista JSON/CSV, ignorando 'Ninguna/Ninguno'."""
    s = _txt(v)
    if s in ("", "nan", "none", "[]"):
        return 0
    items = None
    try:
        parsed = json.loads(str(v))
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


def _herramientas_ajuste(v):
    """0 herramientas = -2, 1-2 = 0, 3+ = +3."""
    n = _n_herramientas(v)
    if n == 0:
        return -2
    if n >= 3:
        return 3
    return 0


def indice_condiciones(estudiante):
    """Índice de CONDICIONES (0–100) basado SOLO en literatura educativa.

    NO usa municipio, grado, género, tipo de institución ni el colegio (factores
    contaminados en los datos actuales). Mide el contexto socioeconómico y la
    preparación previa, NO la habilidad ni la nota.

        indice = 50 + Σ ajustes,   recortado a [5, 95].
    """
    base = 50.0
    ajuste = 0.0
    ajuste += _si_no(estudiante.get("computador_en_casa"), 3, -3)   # OECD PISA
    ajuste += _si_no(estudiante.get("internet_en_casa"), 2, -2)     # UNESCO
    ajuste += _estrato_ajuste(estudiante.get("estrato"))            # recursos hogar
    # Estabilidad familiar (meta-análisis: efecto pequeño).
    conv = _txt(estudiante.get("con_quien_vive"))
    if conv:
        ajuste += 1 if conv == "ambos padres" else -1
    ajuste += _nivel_ajuste(estudiante.get("nivel_programacion"), (-3, 0, 4, 8))
    ajuste += _nivel_ajuste(estudiante.get("nivel_robotica"), (-1, 0, 2, 4))
    ajuste += _si_no(estudiante.get("participacion_olimpiadas"), 5, 0)
    ajuste += _interes_ajuste(estudiante.get("interes_prog_robotica"))
    ajuste += _herramientas_ajuste(estudiante.get("herramientas_conocidas"))
    val = base + ajuste
    if val < 5.0:
        val = 5.0
    if val > 95.0:
        val = 95.0
    return val


def nivel_condiciones(indice):
    """Etiqueta cualitativa del índice de condiciones."""
    if indice > COND_FAVORABLE:
        return "Favorables"
    if indice >= COND_ADVERSA:
        return "Promedio"
    return "Adversas"


# =============================================================================
# CARGA Y CÁLCULO
# =============================================================================

def cargar() -> pd.DataFrame:
    log("Paso 1 — Carga de datos, estimado empírico y cálculo del índice teórico")
    m03 = _import_por_ruta("modelo03", BASE_DIR / "notebooks" / "03_modelo_predictivo.py")
    df = m03.cargar_y_limpiar()
    df = df.drop_duplicates(subset="numero_documento", keep="first").reset_index(drop=True)
    df["numero_documento"] = df["numero_documento"].astype(str)

    # Modelo teórico por estudiante (independiente de haber presentado).
    df["indice_condiciones"] = [round(indice_condiciones(r), 2)
                                for r in df.to_dict("records")]
    df["nivel_condiciones"] = df["indice_condiciones"].apply(nivel_condiciones)

    # Empírico: del CSV del script 09 (puntaje_estimado del Random Forest).
    est_path = DEPLOY_DIR / "puntaje_estimado.csv"
    if est_path.exists():
        est = pd.read_csv(est_path, dtype={"numero_documento": str})
        df = df.merge(est[["numero_documento", "puntaje_estimado"]],
                      on="numero_documento", how="left")
        df = df.rename(columns={"puntaje_estimado": "puntaje_empirico"})
    else:
        df["puntaje_empirico"] = np.nan
        log("    ⚠ no se encontró puntaje_estimado.csv (empírico quedará vacío)")

    df["puntaje_real"] = pd.to_numeric(df[m03.TARGET], errors="coerce")
    df["tiempo_usado_segundos"] = pd.to_numeric(
        df.get("tiempo_usado_segundos"), errors="coerce")
    df["diferencia_teorica"] = df["puntaje_real"] - df["indice_condiciones"]

    log(f"    estudiantes: {len(df):,} | con puntaje real: "
        f"{df['puntaje_real'].notna().sum():,}")
    return df


# =============================================================================
# B. COMPARACIÓN TRIPLE
# =============================================================================

def _r_pearson(a, b) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def comparacion(df: pd.DataFrame) -> dict:
    log("Paso 2 — Comparación triple (teórico vs empírico vs real)")
    pres = df[df["puntaje_real"].notna() & df["puntaje_empirico"].notna()].copy()
    y = pres["puntaje_real"].to_numpy(float)
    teo = pres["indice_condiciones"].to_numpy(float)
    emp = pres["puntaje_empirico"].to_numpy(float)

    met = {
        "teo_r2": float(r2_score(y, teo)), "teo_mae": float(mean_absolute_error(y, teo)),
        "teo_r": _r_pearson(teo, y),
        "emp_r2": float(r2_score(y, emp)), "emp_mae": float(mean_absolute_error(y, emp)),
        "emp_r": _r_pearson(emp, y),
        "teo_emp_r": _r_pearson(teo, emp),
        "n": int(len(pres)),
    }
    log(f"    teórico: R²={met['teo_r2']:.3f} MAE={met['teo_mae']:.2f} "
        f"r={met['teo_r']:.3f}")
    log(f"    empírico: R²={met['emp_r2']:.3f} MAE={met['emp_mae']:.2f} "
        f"r={met['emp_r']:.3f}")
    log(f"    correlación teórico–empírico: r={met['teo_emp_r']:.3f}")

    # --- Scatters ---
    def _scatter(x, ylab, xlab, titulo, fname, key, color):
        fig, ax = plt.subplots(figsize=(6.8, 6.6))
        ax.scatter(x[0], x[1], s=18, alpha=0.5, color=color,
                   edgecolor=COLORS["dark"], linewidth=0.2)
        lim = [0, 100]
        ax.plot(lim, lim, "--", color=COLORS["dark"], linewidth=1.5, label="45°")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(titulo)
        ax.legend(loc="upper left")
        savefig(fig, fname, key)

    _scatter((y, teo), "Índice teórico (condiciones)", "Puntaje real",
             f"Teórico vs real\nR²={met['teo_r2']:.3f} · r={met['teo_r']:.3f} · "
             f"MAE={met['teo_mae']:.1f}",
             "F10_scatter_teorico_vs_real.png", "teo_real", COLORS["violet"])
    _scatter((y, emp), "Puntaje empírico (Random Forest)", "Puntaje real",
             f"Empírico vs real\nR²={met['emp_r2']:.3f} · r={met['emp_r']:.3f} · "
             f"MAE={met['emp_mae']:.1f}",
             "F10_scatter_empirico_vs_real.png", "emp_real", COLORS["cyan"])
    _scatter((emp, teo), "Índice teórico (condiciones)", "Puntaje empírico (RF)",
             f"¿Miden lo mismo los dos modelos?\nr={met['teo_emp_r']:.3f}",
             "F10_scatter_teorico_vs_empirico.png", "teo_emp", COLORS["amber"])

    # --- Distribución del índice ---
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.histplot(df["indice_condiciones"], bins=30, color=COLORS["violet"],
                 edgecolor="white", ax=ax)
    ax.axvline(COND_ADVERSA, color=COLORS["red"], linestyle="--", linewidth=1.5,
               label=f"< {COND_ADVERSA}: Adversas")
    ax.axvline(COND_FAVORABLE, color=COLORS["green"], linestyle="--", linewidth=1.5,
               label=f"> {COND_FAVORABLE}: Favorables")
    ax.set_title("Distribución del índice de condiciones (modelo teórico)")
    ax.set_xlabel("indice_condiciones (0–100)")
    ax.set_ylabel("N estudiantes")
    ax.legend()
    savefig(fig, "F10_distribucion_indice_condiciones.png", "dist_indice")

    # --- Condiciones vs puntaje real (con recta de tendencia) ---
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.scatter(pres["indice_condiciones"], pres["puntaje_real"], s=18, alpha=0.45,
               color=COLORS["cyan"], edgecolor=COLORS["dark"], linewidth=0.2)
    b1, b0 = np.polyfit(pres["indice_condiciones"], pres["puntaje_real"], 1)
    xs = np.array([pres["indice_condiciones"].min(), pres["indice_condiciones"].max()])
    ax.plot(xs, b0 + b1 * xs, color=COLORS["red"], linewidth=2,
            label=f"tendencia (r={met['teo_r']:.2f})")
    ax.set_title("Índice de condiciones vs puntaje real\n"
                 "(condiciones favorables NO garantizan mejor nota)")
    ax.set_xlabel("indice_condiciones (0–100)")
    ax.set_ylabel("Puntaje real")
    ax.legend()
    savefig(fig, "F10_condiciones_vs_puntaje.png", "cond_punt")

    met["pres"] = pres
    return met


# =============================================================================
# C. DETECCIÓN DE ANOMALÍAS
# =============================================================================

def anomalias(df: pd.DataFrame) -> dict:
    log("Paso 3 — Detección de anomalías y cruce con sospecha_trampa (05)")
    pres = df[df["puntaje_real"].notna()].copy()

    pres["anomalo"] = pres["diferencia_teorica"] > ANOMALIA_DIF
    pres["alta_sospecha"] = (pres["anomalo"]
                             & (pres["tiempo_usado_segundos"] < TIEMPO_SOSPECHA))
    n_anom = int(pres["anomalo"].sum())
    n_alta = int(pres["alta_sospecha"].sum())
    log(f"    'demasiado bueno para el contexto' (dif_teórica > {ANOMALIA_DIF:.0f}): "
        f"{n_anom:,}")
    log(f"    + tiempo < 30 min → alta sospecha: {n_alta:,}")

    # Cruce con sospecha_trampa.csv (script 05).
    coinciden = nuevos = set()
    set_05_anular = set_05_flag = set()
    sp_path = DEPLOY_DIR / "sospecha_trampa.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path, dtype={"numero_documento": str})
        set_05_anular = set(sp.loc[sp["recomendacion"] == "anular", "numero_documento"])
        set_05_flag = set(sp.loc[sp["nivel_sospecha"].isin(["Bajo", "Moderado", "Alto"]),
                                 "numero_documento"])
        docs_alta = set(pres.loc[pres["alta_sospecha"], "numero_documento"])
        coinciden = docs_alta & (set_05_anular | set_05_flag)
        nuevos = docs_alta - set_05_anular - set_05_flag
        log(f"    alta sospecha ∩ señalados por telemetría (05): {len(coinciden)}")
        log(f"    NUEVOS sospechosos (no vistos por telemetría): {len(nuevos)}")

    # --- Scatter anomalías: diferencia_teórica vs tiempo ---
    fig, ax = plt.subplots(figsize=(10, 6))
    normal = pres[~pres["alta_sospecha"]]
    alta = pres[pres["alta_sospecha"]]
    ax.scatter(normal["tiempo_usado_segundos"] / 60, normal["diferencia_teorica"],
               s=14, alpha=0.35, color=COLORS["cyan"], label="Normal",
               edgecolor="none")
    ax.scatter(alta["tiempo_usado_segundos"] / 60, alta["diferencia_teorica"],
               s=45, alpha=0.9, color=COLORS["red"], marker="X",
               edgecolor=COLORS["dark"], linewidth=0.4, label="Alta sospecha")
    ax.axhline(ANOMALIA_DIF, color=COLORS["amber"], linestyle="--", linewidth=1.4,
               label=f"dif. teórica = {ANOMALIA_DIF:.0f}")
    ax.axvline(TIEMPO_SOSPECHA / 60, color=COLORS["violet"], linestyle="--",
               linewidth=1.4, label="30 min")
    ax.set_title("Anomalías: resultado 'demasiado bueno para el contexto' vs tiempo\n"
                 "(cuadrante superior izquierdo = alta sospecha)")
    ax.set_xlabel("Tiempo del examen (minutos)")
    ax.set_ylabel("Diferencia teórica (real − índice de condiciones)")
    ax.legend(loc="upper right", fontsize=8)
    savefig(fig, "F10_anomalias_scatter.png", "anomalias")

    return {"n_anom": n_anom, "n_alta": n_alta,
            "coinciden": coinciden, "nuevos": nuevos,
            "set_05_anular": set_05_anular, "set_05_flag": set_05_flag,
            "pres": pres}


# =============================================================================
# D. ÍNDICE DE CONDICIONES vs TALENTO OCULTO (06)
# =============================================================================

def cruce_talento(df: pd.DataFrame) -> dict:
    log("Paso 4 — Cruce índice de condiciones × talento oculto (06)")
    out = {"n_talento": 0, "n_adversas_alto": 0, "coincidencia": None}
    tal_path = DEPLOY_DIR / "talento_oculto_scores.csv"
    pres = df[df["puntaje_real"].notna()].copy()
    if not tal_path.exists():
        log("    ⚠ no se encontró talento_oculto_scores.csv")
        return {**out, "pres": pres}

    tal = pd.read_csv(tal_path, dtype={"numero_documento": str})
    tal["es_talento_oculto"] = tal["es_talento_oculto"].astype(str).str.lower() == "true"
    pres = pres.merge(tal[["numero_documento", "es_talento_oculto"]],
                      on="numero_documento", how="left")
    pres["es_talento_oculto"] = pres["es_talento_oculto"].fillna(False)

    # "Talento oculto DEFINITIVO": condiciones adversas (<45) y buen puntaje (>=60).
    definitivo = pres[(pres["indice_condiciones"] < COND_ADVERSA)
                      & (pres["puntaje_real"] >= 60)]
    out["n_talento"] = int(pres["es_talento_oculto"].sum())
    out["n_adversas_alto"] = int(len(definitivo))
    # ¿Cuántos de esos "definitivos" también los marcó el script 06?
    if len(definitivo):
        out["coincidencia"] = float(definitivo["es_talento_oculto"].mean())

    log(f"    talento oculto (06): {out['n_talento']:,} | condiciones adversas + "
        f"puntaje≥60: {out['n_adversas_alto']:,}")

    # --- Scatter condiciones vs puntaje coloreado por talento ---
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for flag, c, lab in [(False, COLORS["cyan"], "No talento oculto"),
                         (True, COLORS["amber"], "Talento oculto (06)")]:
        sub = pres[pres["es_talento_oculto"] == flag]
        ax.scatter(sub["indice_condiciones"], sub["puntaje_real"], s=18,
                   alpha=0.55 if flag else 0.3,
                   color=c, edgecolor=COLORS["dark"] if flag else "none",
                   linewidth=0.3, label=lab)
    ax.axvline(COND_ADVERSA, color=COLORS["red"], linestyle="--", linewidth=1.3)
    ax.axhline(60, color=COLORS["green"], linestyle="--", linewidth=1.3)
    ax.axvspan(0, COND_ADVERSA, ymin=0.6, color=COLORS["green"], alpha=0.06)
    ax.text(COND_ADVERSA / 2, 92, "TALENTO OCULTO\n(condiciones adversas + alto puntaje)",
            ha="center", va="top", fontsize=8, color=COLORS["dark"])
    ax.set_title("Índice de condiciones vs puntaje, por talento oculto (06)")
    ax.set_xlabel("indice_condiciones (0–100)")
    ax.set_ylabel("Puntaje real")
    ax.legend(loc="lower right", fontsize=8)
    savefig(fig, "F10_condiciones_vs_talento.png", "cond_talento")

    out["pres"] = pres
    out["definitivo"] = definitivo
    return out


# =============================================================================
# E. EXPORTACIÓN
# =============================================================================

def exportar_csv(df: pd.DataFrame) -> None:
    log("Paso 5 — Exportación del CSV de scores teóricos")
    out = df[df["puntaje_real"].notna()].copy()
    cols = ["numero_documento", "indice_condiciones", "nivel_condiciones",
            "puntaje_real", "puntaje_empirico", "diferencia_teorica"]
    out = out[cols].copy()
    out["puntaje_real"] = out["puntaje_real"].round(2)
    out["puntaje_empirico"] = out["puntaje_empirico"].round(2)
    out["diferencia_teorica"] = out["diferencia_teorica"].round(2)
    destino = DEPLOY_DIR / "modelo_teorico_scores.csv"
    out.to_csv(destino, index=False, encoding="utf-8-sig", na_rep="")
    log(f"    CSV exportado ({len(out):,}) → models/deploy/{destino.name}")


def exportar_predictor_py() -> None:
    log("Paso 5 — Exportación del predictor puro (.py)")
    fns = [_txt, _num, _si_no, _estrato_ajuste, _nivel_ajuste, _interes_ajuste,
           _n_herramientas, _herramientas_ajuste, indice_condiciones,
           nivel_condiciones]
    fuente = "\n\n".join(inspect.getsource(f) for f in fns)
    contenido = f'''# -*- coding: utf-8 -*-
"""
Modelo TEÓRICO de condiciones — Copa STEM 2026 (knowledge-driven).
GENERADO por notebooks/10_modelo_teorico_vs_empirico.py — no editar a mano.

Calcula un `indice_condiciones` (0–100) a partir SOLO de literatura educativa
(OECD PISA, UNESCO, meta-análisis SES). NO usa datos de Copa STEM, ni municipio,
grado, género, tipo de institución o colegio. Mide CONDICIONES, no habilidad.

    from indice_condiciones_predictor import indice_condiciones, nivel_condiciones
    ic = indice_condiciones({{"estrato": 1, "computador_en_casa": "No",
                              "nivel_programacion": "Intermedia", ...}})
    nivel = nivel_condiciones(ic)   # "Favorables" / "Promedio" / "Adversas"

No requiere librerías externas: solo la librería estándar (json).
"""
import json

COND_FAVORABLE = {COND_FAVORABLE}
COND_ADVERSA = {COND_ADVERSA}


{fuente}


if __name__ == "__main__":
    ejemplo = {{"estrato": 1, "computador_en_casa": "No", "internet_en_casa": "Sí",
               "con_quien_vive": "Solo madre", "nivel_programacion": "Ninguna",
               "nivel_robotica": "Ninguna", "participacion_olimpiadas": "No",
               "interes_prog_robotica": 3, "herramientas_conocidas": "[]"}}
    ic = indice_condiciones(ejemplo)
    print("indice_condiciones:", round(ic, 2), "->", nivel_condiciones(ic))
'''
    destino = DEPLOY_DIR / "indice_condiciones_predictor.py"
    destino.write_text(contenido, encoding="utf-8")
    log(f"    predictor puro → models/deploy/{destino.name}")


def exportar_predictor_js() -> None:
    log("Paso 5 — Exportación del predictor (.js ES6)")
    js = f'''// Modelo TEÓRICO de condiciones — Copa STEM 2026 (knowledge-driven).
// GENERADO por notebooks/10_modelo_teorico_vs_empirico.py — no editar a mano.
//
// Calcula un indice_condiciones (0–100) a partir SOLO de literatura educativa.
// No usa datos de Copa STEM ni municipio/grado/género/institución/colegio.

const COND_FAVORABLE = {COND_FAVORABLE};
const COND_ADVERSA = {COND_ADVERSA};

function _txt(v) {{
  if (v === null || v === undefined) return "";
  if (typeof v === "number" && Number.isNaN(v)) return "";
  return String(v).trim().toLowerCase();
}}

function _num(v) {{
  if (v === null || v === undefined || typeof v === "boolean") return null;
  const f = Number(v);
  return Number.isNaN(f) ? null : f;
}}

function _siNo(v, wSi, wNo) {{
  const s = _txt(v);
  if (s.startsWith("s")) return wSi;
  if (s.startsWith("n")) return wNo;
  return 0;
}}

function _estratoAjuste(v) {{
  // Estrato 1-3 (máx real en Copacabana/Girardota/Bello); fuera de rango o NaN -> 0
  const f = _num(v);
  if (f === null) return 0;
  const e = Math.round(f);
  const m = {{1: -3, 2: 0, 3: 2}};
  return (e in m) ? m[e] : 0;
}}

function _nivelAjuste(v, pesos) {{
  const s = _txt(v);
  const m = {{
    "ninguna": pesos[0], "ninguno": pesos[0], "básica": pesos[1],
    "basica": pesos[1], "intermedia": pesos[2], "avanzada": pesos[3]
  }};
  return (s in m) ? m[s] : 0;
}}

function _interesAjuste(v) {{
  const f = _num(v);
  if (f === null) return 0;
  if (f <= 2) return -2;
  if (f >= 4) return 3;
  return 0;
}}

function _nHerramientas(v) {{
  const s = _txt(v);
  if (s === "" || s === "nan" || s === "none" || s === "[]") return 0;
  let items = null;
  try {{
    const parsed = JSON.parse(String(v));
    if (Array.isArray(parsed)) items = parsed;
  }} catch (e) {{ items = null; }}
  if (items === null) {{
    items = s.replace(/[\\[\\]"]/g, "").split(",").filter(x => x.trim());
  }}
  let cnt = 0;
  for (const it of items) {{
    const t = String(it).trim().toLowerCase();
    if (t && !["ninguna", "ninguno", "ninguna.", "ninguno."].includes(t)) cnt++;
  }}
  return cnt;
}}

function _herramientasAjuste(v) {{
  const n = _nHerramientas(v);
  if (n === 0) return -2;
  if (n >= 3) return 3;
  return 0;
}}

export function indiceCondiciones(est) {{
  let ajuste = 0.0;
  ajuste += _siNo(est["computador_en_casa"], 3, -3);   // OECD PISA
  ajuste += _siNo(est["internet_en_casa"], 2, -2);     // UNESCO
  ajuste += _estratoAjuste(est["estrato"]);            // recursos hogar
  const conv = _txt(est["con_quien_vive"]);
  if (conv) ajuste += (conv === "ambos padres") ? 1 : -1;
  ajuste += _nivelAjuste(est["nivel_programacion"], [-3, 0, 4, 8]);
  ajuste += _nivelAjuste(est["nivel_robotica"], [-1, 0, 2, 4]);
  ajuste += _siNo(est["participacion_olimpiadas"], 5, 0);
  ajuste += _interesAjuste(est["interes_prog_robotica"]);
  ajuste += _herramientasAjuste(est["herramientas_conocidas"]);
  let val = 50.0 + ajuste;
  if (val < 5.0) val = 5.0;
  if (val > 95.0) val = 95.0;
  return val;
}}

export function nivelCondiciones(indice) {{
  if (indice > COND_FAVORABLE) return "Favorables";
  if (indice >= COND_ADVERSA) return "Promedio";
  return "Adversas";
}}

// Ejemplo:
//   import {{ indiceCondiciones, nivelCondiciones }} from "./indice_condiciones_predictor.js";
//   const ic = indiceCondiciones({{ estrato: 1, computador_en_casa: "No", ... }});
//   nivelCondiciones(ic); // "Adversas"
'''
    destino = DEPLOY_DIR / "indice_condiciones_predictor.js"
    destino.write_text(js, encoding="utf-8")
    log(f"    predictor JS → models/deploy/{destino.name}")


def verificar_predictor_py(df: pd.DataFrame) -> float:
    """Recarga el .py exportado y confirma que reproduce el índice del notebook."""
    ruta = DEPLOY_DIR / "indice_condiciones_predictor.py"
    mod = _import_por_ruta("indice_check", ruta)
    muestra = df.sample(min(300, len(df)), random_state=RANDOM_STATE)
    diffs = [abs(mod.indice_condiciones(r) - indice_condiciones(r))
             for r in muestra.to_dict("records")]
    return float(max(diffs)) if diffs else 0.0


# =============================================================================
# F. INFORME
# =============================================================================

def construir_informe(df, met, anom, tal, verif) -> None:
    log("Paso 6 — Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    pres = df[df["puntaje_real"].notna()]
    n_pres = len(pres)
    cond_counts = df["nivel_condiciones"].value_counts().to_dict()

    R = REPORT.append
    R("# Modelo Teórico vs Empírico — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Fase 4 · Informe generado: {fecha}\n")
    R("---\n")

    # Resumen ejecutivo
    R("## Resumen ejecutivo\n")
    R(dedent(f"""\
        Se construyó un modelo **teórico** (`indice_condiciones`, 0–100) cuyos pesos
        provienen SOLO de la literatura educativa (OECD PISA, UNESCO, meta-análisis
        de SES), sin usar ningún resultado de Copa STEM. Se comparó con el modelo
        **empírico** (Random Forest, script 03) y con el puntaje **real** de los
        **{met['n']:,} estudiantes** que presentaron:

        - **Empírico vs real:** R² = {met['emp_r2']:.3f}, MAE = {met['emp_mae']:.1f},
          r = {met['emp_r']:.3f}.
        - **Teórico vs real:** R² = {met['teo_r2']:.3f}, MAE = {met['teo_mae']:.1f},
          r = {met['teo_r']:.3f} (el índice mide *condiciones*, no la nota: su R²
          directo no es comparable en escala; la correlación es la lectura justa).
        - **Correlación teórico–empírico:** r = {met['teo_emp_r']:.3f}: ambos modelos
          apuntan en la misma dirección, evidencia de que los datos reflejan en buena
          medida los patrones esperados por la literatura.

        El índice permitió detectar **{anom['n_alta']:,} casos de alta sospecha**
        (resultado "demasiado bueno para el contexto" **y** examen en menos de 30
        min), de los cuales **{len(anom['nuevos'])}** son **nuevos** (no los había
        marcado la telemetría del script 05). Además, el índice sirve independiente de
        la nota: **{tal['n_adversas_alto']:,} estudiantes** con condiciones adversas
        (<{COND_ADVERSA}) sacaron ≥ 60 → talento oculto claro.\n"""))

    # Marco conceptual
    R("## Marco conceptual: ¿por qué dos modelos?\n")
    R(dedent("""\
        El modelo **empírico** (Random Forest) aprendió de los datos reales. Si en
        los datos hay trampa, el modelo aprendió patrones de trampa. Es como un juez
        que aprendió de casos anteriores: si algunos casos fueron fraudulentos, el
        juez puede arrastrar esos sesgos.

        El modelo **teórico** no usa ningún dato de Copa STEM. Se basa en lo que la
        investigación educativa dice sobre qué factores afectan el rendimiento. Es
        como un juez que solo sigue la ley escrita, sin importar los casos anteriores.

        Comparar los dos nos dice:
        - **Si coinciden** → los datos de Copa STEM reflejan los patrones esperados.
        - **Si difieren** → algo en los datos es anómalo (trampa, sesgo o factores
          locales únicos que la literatura global no captura).\n"""))

    # Modelo teórico en ciencia de datos
    R("## ¿Qué es un modelo teórico en ciencia de datos?\n")
    R(dedent("""\
        - **Modelo data-driven (empírico):** aprende los pesos de los datos. Ejemplo:
          el Random Forest descubre solo que "tener computador suma X puntos" mirando
          a los estudiantes. Potente si los datos son buenos; frágil si están sucios.
        - **Modelo knowledge-driven (teórico):** los pesos los fija un experto a
          partir de la literatura. Ejemplo: "según OECD PISA, el acceso a computador
          se asocia con mejor rendimiento → le asigno +3".
        - **Ventajas del teórico:** no se contamina con datos malos, es transparente
          y auditable (cada peso tiene una fuente).
        - **Desventajas:** los pesos son "opiniones educadas" de la literatura, no
          evidencia local; pueden no ajustar la magnitud real en Copacabana/Girardota.
        - **¿Cuándo usar cada uno?** Si confías en tus datos → empírico. Si sospechas
          contaminación (trampa, muestra incompleta) → teórico como contraste.\n"""))

    # Construcción del modelo teórico
    R("## Construcción del modelo teórico\n")
    R(dedent(f"""\
        `indice_condiciones = 50 + Σ ajustes`, recortado a **[5, 95]**. Cada ajuste
        y su sustento:

        | Factor | Ajuste | Fuente / razón |
        | --- | --- | --- |
        | Computador en casa | Sí +3 / No −3 | OECD PISA: el acceso tecnológico se asocia con rendimiento. |
        | Internet en casa | Sí +2 / No −2 | UNESCO: la conectividad da acceso a recursos de estudio. |
        | Estrato (1–3) | 1: −3 · 2: 0 · 3: +2 | Recursos del hogar, alimentación, ambiente de estudio (Sirin 2005). |
        | Vive con ambos padres | Sí +1 / Otro −1 | Meta-análisis: estabilidad familiar, efecto pequeño pero consistente. |
        | Nivel programación | Ninguno −3 · Básico 0 · Intermedio +4 · Avanzado +8 | Preparación previa directa para el examen. |
        | Nivel robótica | Ninguno −1 · Básico 0 · Intermedio +2 · Avanzado +4 | Preparación previa complementaria. |
        | Participó en olimpiadas | Sí +5 / No 0 | Experiencia y familiaridad con el formato. |
        | Interés prog/robótica | bajo(1–2) −2 · medio(3) 0 · alto(4–5) +3 | Motivación intrínseca. |
        | Nº herramientas conocidas | 0 −2 · 1–2 0 · 3+ +3 | Capital tecnológico acumulado. |

        **Lo que asumimos:** que estos factores empujan el rendimiento en la dirección
        y magnitud aproximada que dice la literatura. **Lo que NO asumimos:** nada
        sobre municipio, grado, género, tipo de institución ni colegio — están
        excluidos por estar potencialmente contaminados (trampa, muestra incompleta).
        El estrato viene **corregido en origen**: en Copacabana/Girardota/Bello el
        máximo real es 3, así que los antiguos 4/5/6 (autorreporte) ya fueron
        reclasificados a 3 en el dataset.

        *Ejemplo real:* un estudiante de estrato 1, sin computador, con internet, que
        vive solo con la madre, sin programación ni robótica, sin olimpiadas, interés
        medio y sin herramientas obtiene
        `50 − 3(estrato) − 3(sin PC) + 2(internet) − 1(familia) − 3(prog) − 1(rob) +
        0 + 0 − 2(herr) = {50-3-3+2-1-3-1-2}` → condiciones **adversas**.\n"""))

    # Comparación
    R("## Comparación: ¿cuál predice mejor?\n")
    filas = [
        {"Modelo": "Empírico (Random Forest)", "R² vs real": f"{met['emp_r2']:.3f}",
         "MAE": f"{met['emp_mae']:.1f}", "r (Pearson)": f"{met['emp_r']:.3f}"},
        {"Modelo": "Teórico (indice_condiciones)", "R² vs real": f"{met['teo_r2']:.3f}",
         "MAE": f"{met['teo_mae']:.1f}", "r (Pearson)": f"{met['teo_r']:.3f}"},
    ]
    R(tabla_md(pd.DataFrame(filas)) + "\n")
    R(dedent(f"""\
        **Interpretación.** El empírico gana en R² y MAE porque está *calibrado a la
        escala del puntaje* (aprendió los números exactos de estos datos). El R² del
        teórico es bajo/negativo porque **no está en la escala de la nota**: es un
        índice de condiciones centrado en 50, no un pronóstico del puntaje. Por eso la
        métrica justa para el teórico es la **correlación** (r = {met['teo_r']:.3f}):
        mide si *ordena* bien a los estudiantes, no si acierta el número.

        **¿Se complementan?** Sí. La correlación entre ambos modelos es
        **r = {met['teo_emp_r']:.3f}**: miden algo parecido, pero el teórico es
        inmune a la trampa presente en los datos. Sirve de **red de seguridad**: donde
        empírico y teórico discrepan mucho, hay que mirar de cerca.\n"""))
    R(f"\n{img('emp_real', 'Empírico vs real')}\n")
    R(f"\n{img('teo_real', 'Teórico vs real')}\n")
    R(f"\n{img('teo_emp', 'Teórico vs empírico')}\n")
    R(f"\n{img('cond_punt', 'Condiciones vs puntaje')}\n")

    # Anomalías
    R("## Anomalías encontradas\n")
    R(dedent(f"""\
        `diferencia_teorica = puntaje_real − indice_condiciones`. Un valor **>
        {ANOMALIA_DIF:.0f}** significa "resultado demasiado bueno para el contexto":
        posible trampa **o** talento excepcional. Cruzándolo con el tiempo de examen:

        - Resultados "demasiado buenos para el contexto" (dif > {ANOMALIA_DIF:.0f}):
          **{anom['n_anom']:,}**.
        - De esos, con examen en **< 30 min** → **alta sospecha: {anom['n_alta']:,}**.
        - **Coinciden** con los señalados por la telemetría del script 05:
          **{len(anom['coinciden'])}**.
        - **NUEVOS** sospechosos que la telemetría NO detectó (p. ej. exámenes
          escritos sin telemetría): **{len(anom['nuevos'])}**.

        Los "nuevos" son valiosos: el modelo teórico ve señales que la telemetría no
        puede (no depende de cambios de pestaña ni de copiar/pegar), así que actúa como
        una segunda capa de auditoría independiente.\n"""))
    R(f"\n{img('anomalias', 'Anomalías')}\n")

    # Índice de condiciones como herramienta
    R("## El índice de condiciones como herramienta\n")
    R(dedent(f"""\
        El índice NO predice la nota: mide las **condiciones** del estudiante. Es útil
        *independiente* del rendimiento. Distribución en la cohorte:
        **Favorables (>{COND_FAVORABLE}): {cond_counts.get('Favorables', 0):,}** ·
        **Promedio ({COND_ADVERSA}–{COND_FAVORABLE}): {cond_counts.get('Promedio', 0):,}** ·
        **Adversas (<{COND_ADVERSA}): {cond_counts.get('Adversas', 0):,}**.

        - *"María tiene índice de condiciones 28 (adversas). Sacó 75. Es talento
          oculto."* → prioridad de apoyo y visibilidad.
        - *"Carlos tiene índice 65 (favorables). Sacó 90 en 12 minutos. Sospechoso."*
          → revisar antes de premiar.

        **Cruce con talento oculto (script 06):** de los **{tal['n_adversas_alto']:,}**
        estudiantes con condiciones adversas que sacaron ≥ 60,
        {'el ' + format(tal['coincidencia']*100, '.0f') + '%' if tal['coincidencia'] is not None else 'N/D'}
        también fueron marcados como talento oculto por el modelo del script 06 → los
        dos enfoques (reglas de condiciones y clasificador de talento) se refuerzan.\n"""))
    R(f"\n{img('dist_indice', 'Distribución índice')}\n")
    R(f"\n{img('cond_talento', 'Condiciones vs talento')}\n")

    # Recomendaciones
    R("## Recomendaciones\n")
    R(dedent(f"""\
        - **Para producción:** usar el **empírico** para estimar el puntaje (mejor
          calibrado), pero acompañarlo SIEMPRE del **teórico** como auditoría y del
          `indice_condiciones` para contexto socioeconómico.
        - **Modelo combinado:** un promedio 50/50 (tras llevar ambos a la misma escala)
          puede ser más robusto que cualquiera solo, porque el teórico amortigua la
          contaminación por trampa del empírico. Recomendado evaluarlo formalmente.
        - **Datos a recoger** para mejorar ambos: promedio académico previo, horas de
          estudio, motivación y fecha/hora de inscripción (ver plan del script 08).
        - **Priorización:** revisar los {len(anom['nuevos'])} nuevos sospechosos y
          dar visibilidad a los talentos ocultos con condiciones adversas.\n"""))

    # Limitaciones
    R("## Limitaciones\n")
    R(dedent("""\
        - Los pesos teóricos son **aproximaciones de la literatura general**, no
          calibrados con datos locales; su magnitud puede no ser exacta aquí.
        - La literatura es **global**: Copacabana/Girardota pueden tener dinámicas
          propias que estos pesos no capturan.
        - El **R² del teórico es bajo** (no está en la escala del puntaje); eso **no**
          lo hace peor: cumple otra función (medir condiciones, auditar), y su valor se
          juzga por correlación y por su independencia de la trampa.
        - Un `diferencia_teorica` alta puede ser **talento excepcional**, no trampa: la
          señal es un disparador de revisión, nunca una condena automática.\n"""))

    # Glosario
    R("## Glosario extendido\n")
    R(dedent(f"""\
        - **Modelo data-driven vs knowledge-driven:** *definición* — el primero aprende
          los pesos de los datos; el segundo los fija desde teoría. *Analogía* —
          aprender a cocinar probando (data-driven) vs seguir una receta de un libro
          (knowledge-driven). *Ejemplo Copa STEM* — el RF (empírico) vs el
          indice_condiciones (teórico).
        - **Sesgo de confirmación en datos:** *definición* — un modelo que aprende de
          datos sesgados reproduce y amplifica ese sesgo. *Analogía* — un loro que
          repite lo que oye, incluidos los errores. *Ejemplo* — si los tramposos
          quedaron en los datos, el empírico "aprende" que su perfil rinde más.
        - **Validación cruzada:** *definición* — evaluar el modelo en datos que no vio,
          repartiendo la muestra en pliegues. *Analogía* — estudiar con unas preguntas
          y examinarte con otras distintas. *Ejemplo* — el R² honesto del empírico
          (~0.08) sale de validación cruzada 5-fold (script 08).
        - **Índice de condiciones:** *definición* — puntaje 0–100 del contexto
          socioeconómico y la preparación previa, NO de la habilidad. *Analogía* — el
          "hándicap" en golf: describe las condiciones de partida, no quién es mejor.
          *Ejemplo* — índice 28 = condiciones adversas.
        - **Anomalía estadística:** *definición* — observación que se aparta mucho de
          lo esperado por el modelo. *Analogía* — un termómetro que marca 45 °C en la
          nevera: o está roto o pasa algo raro. *Ejemplo* — sacar 90 con índice 35 en
          12 minutos.\n"""))

    # Referencias
    R("## Referencias bibliográficas\n")
    R(dedent("""\
        - OECD (2015). *Students, Computers and Learning: Making the Connection* (PISA).
        - UNESCO (2020). *Global Education Monitoring Report — technology in education*.
        - Sirin, S. R. (2005). *Socioeconomic Status and Academic Achievement: A
          Meta-Analytic Review of Research*. Review of Educational Research.
        - Hattie, J. (2009). *Visible Learning* (síntesis de meta-análisis de factores
          que afectan el rendimiento).\n"""))
    R(f"\n_Auto-verificación del predictor exportado: máx|Δ| = {verif:.3g} "
      f"(predictor .py vs cálculo del notebook)._\n")
    R("\n---\n_Generado por `notebooks/10_modelo_teorico_vs_empirico.py` — Copa STEM 2026._\n")

    destino = REPORTS_DIR / "10_modelo_teorico_vs_empirico.md"
    destino.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"    informe escrito → reports/{destino.name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Modelo Teórico vs Empírico (Fase 4)")
    print(" Fundación SapienceLab")
    print("=" * 70)

    df = cargar()
    met = comparacion(df)
    anom = anomalias(df)
    tal = cruce_talento(df)

    exportar_csv(df)
    exportar_predictor_py()
    exportar_predictor_js()
    verif = verificar_predictor_py(df)
    log(f"    verificación predictor .py: máx|Δ| = {verif:.3g}")

    construir_informe(df, met, anom, tal, verif)

    print("\n" + "=" * 70)
    print(" ✔ MODELO TEÓRICO vs EMPÍRICO COMPLETADO")
    print(f"   · Empírico R²={met['emp_r2']:.3f} | Teórico r={met['teo_r']:.3f} | "
          f"corr teo-emp={met['teo_emp_r']:.3f}")
    print(f"   · Anomalías alta sospecha: {anom['n_alta']} "
          f"(nuevos vs telemetría: {len(anom['nuevos'])})")
    print(f"   · Talento con condiciones adversas (≥60): {tal['n_adversas_alto']}")
    print(f"   · Exports: modelo_teorico_scores.csv + predictor .py/.js")
    print(f"   · Figuras: {len(FIGURES)} → outputs/F10_*.png")
    print(f"   · Informe: reports/10_modelo_teorico_vs_empirico.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
