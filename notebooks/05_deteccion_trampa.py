# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 05: Detección de Trampa (integridad del examen)  — Fase 3
================================================================================

Contexto
--------
El examen Copa STEM tiene 40 preguntas (16 numéricas de 10 pts + 24 de selección
múltiple de 5 pts) y dura 90 minutos. Las primeras semanas se presentó SIN
sistema anti-cheat. Hay indicios de trampa: puntajes altos en tiempos muy cortos.

**Solo los exámenes de PLATAFORMA tienen telemetría** (`tiempo_usado_segundos`,
`cambios_pestana`). Los exámenes ESCRITOS no la tienen y se marcan como
"Sin telemetría" — quedan fuera del análisis de sospecha.

Secciones
---------
    A) Patrones sospechosos (tiempo vs puntaje, velocidad, criterios A–D)
    B) Análisis detallado (niveles, colegios, distribución, impacto en promedio)
    C) Criterio de anulación recomendado (antes vs. después)
    D) Impacto en el modelo ML (re-entrenar sin sospechosos de nivel "Alto")
    E) Exportación (models/deploy/sospecha_trampa.csv + informe + figuras)

Diseño: reproducible (`random_state=42`), paleta Copa STEM en todos los gráficos.

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
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
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
except ImportError as exc:  # pragma: no cover
    print("ERROR: falta una dependencia del entorno.")
    print(f"       Detalle: {exc}")
    sys.exit(1)

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

DATASET_NAME = "copa_stem_dataset.csv"

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
    if n <= 1:
        return [COLORS["cyan"]]
    return [STEM_GRAD(i / (n - 1)) for i in range(n)]


# --- Parámetros del examen y de los criterios de sospecha --------------------
UMBRAL_APROBACION = 60
# Nivel de sospecha por nº de criterios cumplidos.
NIVELES = {0: "Limpio", 1: "Bajo", 2: "Moderado"}  # 3+ → "Alto"


def nivel_desde_n(n: int) -> str:
    if n >= 3:
        return "Alto"
    return NIVELES.get(n, "Limpio")


ORDEN_NIVEL = ["Limpio", "Bajo", "Moderado", "Alto"]
SUSP_COLOR = {
    "Limpio":         COLORS["green"],
    "Bajo":           COLORS["cyan"],
    "Moderado":       COLORS["amber"],
    "Alto":           COLORS["red"],
    "Sin telemetría": "#9aa0aa",
}

FIGURES: dict[str, str] = {}
REPORT: list[str] = []


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
# CARGA Y LIMPIEZA
# =============================================================================

def cargar_y_limpiar() -> pd.DataFrame:
    log("Carga y limpieza del dataset")
    ruta = DATA_DIR / DATASET_NAME
    if not ruta.exists():
        csvs = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
        if not csvs:
            print(f"\n  ⚠  No se encontró '{DATASET_NAME}' en {DATA_DIR}\n")
            sys.exit(0)
        ruta = csvs[0]

    df = pd.read_csv(ruta, encoding="utf-8")
    docs_prueba = ["1234", "123456", "123456789", "1234567899", "0", "00000000"]
    if "numero_documento" in df.columns:
        df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
        df = df[~df["numero_documento"].isin(docs_prueba)]
        df = df[df["numero_documento"].str.len() >= 5]
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    for c in ["puntaje_obtenido", "grado_escolar", "estrato", "interes_prog_robotica",
              "edad_calculada", "tiempo_usado_segundos", "cambios_pestana",
              "intentos_copiar", "intentos_pegar", "intentos_click_derecho"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["puntaje_obtenido"].notna()].reset_index(drop=True)
    log(f"    presentaron: {len(df):,}")
    return df


# =============================================================================
# A. PATRONES SOSPECHOSOS Y CRITERIOS
# =============================================================================

def calcular_sospecha(df: pd.DataFrame) -> dict:
    log("SECCIÓN A — Patrones sospechosos y criterios A–D")

    # Bandera de telemetría: solo exámenes de plataforma tienen tiempo.
    df["plataforma"] = df["tiempo_usado_segundos"].notna()
    n_plat = int(df["plataforma"].sum())
    n_escrito = int((~df["plataforma"]).sum())
    log(f"    plataforma (con telemetría): {n_plat:,} | "
        f"sin telemetría (escritos): {n_escrito:,}")

    df["tiempo_min"] = df["tiempo_usado_segundos"] / 60.0
    # Velocidad = puntaje por minuto (solo donde hay tiempo > 0).
    df["velocidad"] = np.where(
        df["plataforma"] & (df["tiempo_usado_segundos"] > 0),
        df["puntaje_obtenido"] / df["tiempo_min"].replace(0, np.nan), np.nan)

    plat = df["plataforma"]
    vel_p95 = float(df.loc[plat, "velocidad"].quantile(0.95))
    log(f"    umbral de velocidad (p95): {vel_p95:.2f} puntos/min")

    # --- Criterios de sospecha (solo aplican a exámenes de plataforma) -------
    seg = df["tiempo_usado_segundos"]
    p = df["puntaje_obtenido"]
    camb = df["cambios_pestana"]
    critA = plat & (p >= 60) & (seg < 2100)                 # ≥60 en <35 min
    critB = plat & (camb >= 5) & (p >= 60)                  # ≥5 cambios + ≥60
    critC = plat & (df["velocidad"] > vel_p95)              # velocidad > p95
    critD = plat & (p == 100) & (seg < 2700)                # 100 en <45 min

    crit_cols = {"A": critA, "B": critB, "C": critC, "D": critD}
    for k, v in crit_cols.items():
        df[f"crit_{k}"] = v.fillna(False)

    df["n_criterios"] = sum(df[f"crit_{k}"].astype(int) for k in crit_cols)
    df["criterios_activados"] = df.apply(
        lambda r: "|".join(k for k in crit_cols if r[f"crit_{k}"]), axis=1)

    df["nivel_sospecha"] = np.where(
        ~df["plataforma"], "Sin telemetría",
        df["n_criterios"].apply(nivel_desde_n))

    # Recomendación: anular si (≥2 criterios) Y (puntaje ≥ 60).
    df["recomendacion"] = np.where(
        df["plataforma"] & (df["n_criterios"] >= 2) & (df["puntaje_obtenido"] >= 60),
        "anular", "mantener")

    conteo_criterio = {k: int(df[f"crit_{k}"].sum()) for k in crit_cols}
    conteo_nivel = (df.loc[plat, "nivel_sospecha"]
                    .value_counts().reindex(ORDEN_NIVEL).fillna(0).astype(int).to_dict())

    # --- Tiempo mínimo razonable para ≥60 ----------------------------------
    altos = df.loc[plat & (p >= 60), "tiempo_min"].dropna()
    tiempo_razonable = float(altos.quantile(0.05)) if len(altos) else float("nan")

    # --- Gráfico A.1: tiempo vs puntaje coloreado por nivel -----------------
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sub = df[plat]
    for nivel in ORDEN_NIVEL:
        s = sub[sub["nivel_sospecha"] == nivel]
        if not s.empty:
            ax.scatter(s["tiempo_min"], s["puntaje_obtenido"], s=26,
                       alpha=0.55 if nivel != "Limpio" else 0.30,
                       color=SUSP_COLOR[nivel], label=f"{nivel} (n={len(s)})",
                       edgecolor="white", linewidth=0.3,
                       zorder=5 if nivel in ("Moderado", "Alto") else 2)
    ax.axhline(60, color=COLORS["blue"], linestyle=":", linewidth=1.2)
    ax.axvline(35, color=COLORS["red"], linestyle="--", linewidth=1.2,
               label="35 min (criterio A)")
    ax.set_title("Tiempo vs. puntaje (exámenes de plataforma)\n"
                 "puntaje alto en tiempo corto = sospechoso")
    ax.set_xlabel("Tiempo usado (minutos)")
    ax.set_ylabel("Puntaje")
    ax.legend(title="Nivel de sospecha", fontsize=8)
    savefig(fig, "F05_tiempo_vs_puntaje.png", "scatter")

    # --- Gráfico A.2: distribución de velocidad + outliers ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    vel = df.loc[plat, "velocidad"].dropna()
    sns.histplot(vel, bins=40, color=COLORS["cyan"], edgecolor="white", ax=ax1)
    ax1.axvline(vel_p95, color=COLORS["red"], linestyle="--",
                label=f"p95 = {vel_p95:.2f}")
    ax1.set_title("Distribución de la velocidad (puntos/minuto)")
    ax1.set_xlabel("Velocidad (puntos/min)")
    ax1.set_ylabel("N")
    ax1.legend()
    tr = df.loc[plat & (p >= 60), "tiempo_min"].dropna()
    sns.histplot(tr, bins=30, color=COLORS["violet"], edgecolor="white", ax=ax2)
    ax2.axvline(tiempo_razonable, color=COLORS["red"], linestyle="--",
                label=f"p5 = {tiempo_razonable:.1f} min")
    ax2.set_title("Tiempo empleado por quienes sacaron ≥ 60")
    ax2.set_xlabel("Tiempo (minutos)")
    ax2.set_ylabel("N")
    ax2.legend()
    fig.suptitle("A. Velocidad y tiempo mínimo razonable",
                 fontsize=15, fontweight="bold")
    savefig(fig, "F05_velocidad.png", "velocidad")

    return {"n_plataforma": n_plat, "n_sin_telemetria": n_escrito,
            "vel_p95": vel_p95, "conteo_criterio": conteo_criterio,
            "conteo_nivel": conteo_nivel, "tiempo_razonable": tiempo_razonable}


# =============================================================================
# B. ANÁLISIS DETALLADO
# =============================================================================

def analisis_detallado(df: pd.DataFrame, ctx: dict) -> dict:
    log("SECCIÓN B — Análisis detallado")
    res = {}
    plat = df["plataforma"]

    # --- B.1 Conteo por nivel + boxplot puntaje por nivel -------------------
    conteo = ctx["conteo_nivel"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.bar(list(conteo.keys()), list(conteo.values()),
            color=[SUSP_COLOR[k] for k in conteo], edgecolor="white")
    ax1.set_title("Exámenes por nivel de sospecha (solo plataforma)")
    ax1.set_ylabel("N estudiantes")
    for i, v in enumerate(conteo.values()):
        ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    sub = df[plat]
    sns.boxplot(data=sub, x="nivel_sospecha", y="puntaje_obtenido",
                order=ORDEN_NIVEL, palette=[SUSP_COLOR[n] for n in ORDEN_NIVEL],
                ax=ax2)
    ax2.set_title("Puntaje por nivel de sospecha")
    ax2.set_xlabel("")
    ax2.set_ylabel("Puntaje")
    fig.suptitle("B. Niveles de sospecha", fontsize=15, fontweight="bold")
    savefig(fig, "F05_niveles.png", "niveles")

    # --- B.2 Distribución sospechosos vs limpios ---------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    limpios = df.loc[plat & (df["nivel_sospecha"] == "Limpio"), "puntaje_obtenido"]
    sospech = df.loc[plat & df["nivel_sospecha"].isin(["Moderado", "Alto"]),
                     "puntaje_obtenido"]
    sns.histplot(limpios, bins=25, color=COLORS["green"], alpha=0.55,
                 label=f"Limpios (n={len(limpios)})", stat="density",
                 edgecolor="white", ax=ax)
    sns.histplot(sospech, bins=25, color=COLORS["red"], alpha=0.55,
                 label=f"Sospechosos ≥2 crit. (n={len(sospech)})", stat="density",
                 edgecolor="white", ax=ax)
    ax.set_title("Distribución de puntajes: limpios vs. sospechosos")
    ax.set_xlabel("Puntaje")
    ax.set_ylabel("Densidad")
    ax.legend()
    savefig(fig, "F05_dist_sospechosos.png", "dist_susp")

    # --- B.3 Colegios: nº de sospechosos y tasa de sospecha ----------------
    if "institucion_educativa" in df.columns:
        g = df[plat].groupby("institucion_educativa")
        col = g.agg(n_plataforma=("puntaje_obtenido", "count"),
                    n_sospechosos=("nivel_sospecha",
                                   lambda s: int(s.isin(["Moderado", "Alto"]).sum())),
                    n_alto=("nivel_sospecha", lambda s: int((s == "Alto").sum()))
                    ).reset_index()
        col["tasa_sospecha"] = (col["n_sospechosos"] / col["n_plataforma"] * 100).round(1)
        col = col[col["n_plataforma"] >= 10]
        res["colegios"] = col.sort_values("n_sospechosos", ascending=False)

        # Barplot horizontal: nº de sospechosos por colegio (top 15).
        top = col.sort_values("n_sospechosos", ascending=True).tail(15)
        if not top.empty:
            fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(top))))
            ax.barh(top["institucion_educativa"], top["n_sospechosos"],
                    color=gradient_colors(len(top)), edgecolor="white")
            ax.set_title("Nº de sospechosos (≥2 criterios) por institución (N ≥ 10)")
            ax.set_xlabel("Nº de estudiantes sospechosos")
            for i, (_, f) in enumerate(top.iterrows()):
                ax.text(f["n_sospechosos"], i,
                        f" {int(f['n_sospechosos'])} de {int(f['n_plataforma'])} "
                        f"({f['tasa_sospecha']:.0f}%)", va="center", fontsize=8)
            savefig(fig, "F05_colegios_sospecha.png", "colegios")

        # Colegios con tasa anormalmente alta (> media + 1σ).
        media_tasa = col["tasa_sospecha"].mean()
        sd_tasa = col["tasa_sospecha"].std()
        umbral_anomalo = media_tasa + sd_tasa
        res["tasa_media"] = float(media_tasa)
        res["umbral_anomalo"] = float(umbral_anomalo)
        res["colegios_anomalos"] = col[col["tasa_sospecha"] > umbral_anomalo] \
            .sort_values("tasa_sospecha", ascending=False)

    # --- B.4 Impacto en el promedio si se anula el nivel "Alto" ------------
    media_antes = float(df["puntaje_obtenido"].mean())
    sin_alto = df[df["nivel_sospecha"] != "Alto"]
    media_sin_alto = float(sin_alto["puntaje_obtenido"].mean())
    res["media_antes"] = media_antes
    res["media_sin_alto"] = media_sin_alto
    res["delta_sin_alto"] = media_sin_alto - media_antes
    res["n_alto"] = int((df["nivel_sospecha"] == "Alto").sum())
    return res


# =============================================================================
# C. CRITERIO DE ANULACIÓN RECOMENDADO
# =============================================================================

def criterio_anulacion(df: pd.DataFrame) -> dict:
    log("SECCIÓN C — Criterio de anulación recomendado")
    anular = df["recomendacion"] == "anular"
    n_anular = int(anular.sum())
    total = len(df)

    media_antes = float(df["puntaje_obtenido"].mean())
    media_despues = float(df.loc[~anular, "puntaje_obtenido"].mean())
    res = {"n_anular": n_anular, "pct_anular": 100 * n_anular / total,
           "media_antes": media_antes, "media_despues": media_despues,
           "delta": media_despues - media_antes}
    log(f"    anular: {n_anular} ({res['pct_anular']:.1f}%) | "
        f"promedio {media_antes:.2f} → {media_despues:.2f} "
        f"({res['delta']:+.2f})")

    # --- Cambio en el ranking de colegios (N ≥ 10) -------------------------
    if "institucion_educativa" in df.columns:
        def ranking(frame):
            r = (frame.groupby("institucion_educativa")["puntaje_obtenido"]
                 .agg(["mean", "count"]))
            return r[r["count"] >= 10]["mean"]
        antes = ranking(df)
        despues = ranking(df.loc[~anular])
        comp = pd.DataFrame({"antes": antes, "despues": despues}).dropna()
        comp["delta"] = (comp["despues"] - comp["antes"]).round(2)
        comp = comp.round(2).sort_values("delta")
        res["ranking_cambio"] = comp

        # Gráfico antes/después (colegios más afectados).
        top = comp.head(12)
        if not top.empty:
            fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(top))))
            y = np.arange(len(top))
            ax.barh(y - 0.2, top["antes"], height=0.4, color=COLORS["amber"],
                    edgecolor="white", label="Antes")
            ax.barh(y + 0.2, top["despues"], height=0.4, color=COLORS["cyan"],
                    edgecolor="white", label="Después de anular")
            ax.set_yticks(y)
            ax.set_yticklabels(top.index)
            ax.set_title("Ranking de colegios: promedio antes vs. después de anular\n"
                         "(12 colegios más afectados)")
            ax.set_xlabel("Puntaje promedio")
            ax.legend()
            savefig(fig, "F05_ranking_antes_despues.png", "ranking")

    # --- Gráfico resumen del impacto en el promedio general ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Antes", "Después"], [media_antes, media_despues],
           color=[COLORS["amber"], COLORS["cyan"]], edgecolor="white", width=0.5)
    ax.set_title(f"Promedio general antes vs. después de anular\n"
                 f"({n_anular} anulados, {res['pct_anular']:.1f}% del total)")
    ax.set_ylabel("Puntaje promedio")
    for i, v in enumerate([media_antes, media_despues]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    savefig(fig, "F05_impacto_promedio.png", "impacto")
    return res


# =============================================================================
# D. IMPACTO EN EL MODELO ML
# =============================================================================

def _cargar_features():
    """Reutiliza el constructor de features y el spec del modelo (script 03)."""
    ruta = MODELS_DIR / "predictor.py"
    if not ruta.exists():
        return None
    spec = importlib.util.spec_from_file_location("predictor_puntaje", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def impacto_modelo(df: pd.DataFrame) -> dict:
    log("SECCIÓN D — Impacto en el modelo ML")
    mod = _cargar_features()
    if mod is None:
        log("    ⚠ models/predictor.py no existe; se omite la sección D.")
        return {}

    PRE = mod._SPEC["preprocess"]
    feat_names = mod._SPEC["feature_names"]

    def build_X(frame):
        return np.array([mod.features_from_raw(r, PRE)
                         for r in frame.to_dict("records")], dtype=float)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Baseline: todos los que presentaron.
    Xb = build_X(df)
    yb = df["puntaje_obtenido"].to_numpy(float)
    r2_base = cross_val_score(LinearRegression(), Xb, yb, cv=cv, scoring="r2")
    lin_base = LinearRegression().fit(Xb, yb)

    # Limpio: sin los sospechosos de nivel "Alto".
    limpio = df[df["nivel_sospecha"] != "Alto"]
    Xc = build_X(limpio)
    yc = limpio["puntaje_obtenido"].to_numpy(float)
    r2_clean = cross_val_score(LinearRegression(), Xc, yc, cv=cv, scoring="r2")
    lin_clean = LinearRegression().fit(Xc, yc)

    res = {"r2_base": (float(r2_base.mean()), float(r2_base.std())),
           "r2_clean": (float(r2_clean.mean()), float(r2_clean.std())),
           "n_base": len(df), "n_clean": len(limpio),
           "n_removidos": len(df) - len(limpio)}
    log(f"    R² baseline={r2_base.mean():.3f} | "
        f"sin nivel Alto={r2_clean.mean():.3f} "
        f"(Δ={r2_clean.mean()-r2_base.mean():+.3f})")

    # --- Comparación de importancias (|coef|) antes vs después -------------
    imp_base = np.abs(lin_base.coef_)
    imp_clean = np.abs(lin_clean.coef_)
    orden = np.argsort(imp_base)[::-1][:12][::-1]
    labels = [feat_names[i] for i in orden]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(orden))
    ax.barh(y - 0.2, imp_base[orden], height=0.4, color=COLORS["amber"],
            edgecolor="white", label="Con todos")
    ax.barh(y + 0.2, imp_clean[orden], height=0.4, color=COLORS["cyan"],
            edgecolor="white", label="Sin nivel Alto")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Importancia de variables (|coef| lineal): con todos vs. sin sospechosos")
    ax.set_xlabel("|coeficiente|")
    ax.legend()
    savefig(fig, "F05_importancia_antes_despues.png", "imp")
    return res


# =============================================================================
# E. EXPORTACIÓN
# =============================================================================

def exportar_csv(df: pd.DataFrame) -> Path:
    log("SECCIÓN E — Exportación")
    # n_criterios como entero nullable (vacío para "Sin telemetría", no 0.0).
    n_crit = df["n_criterios"].where(df["plataforma"]).astype("Int64")
    out = pd.DataFrame({
        "numero_documento": df["numero_documento"].astype(str),
        "n_criterios": n_crit,
        "nivel_sospecha": df["nivel_sospecha"],
        "criterios_activados": df["criterios_activados"],
        "recomendacion": df["recomendacion"],
    })
    destino = DEPLOY_DIR / "sospecha_trampa.csv"
    out.to_csv(destino, index=False, encoding="utf-8-sig")
    log(f"    sospecha → models/deploy/{destino.name} ({len(out):,} filas)")
    return destino


# =============================================================================
# INFORME
# =============================================================================

def construir_informe(df, A, B, C, D) -> None:
    log("Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    R = REPORT.append

    R("# Detección de Trampa — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Integridad del examen · Informe: {fecha}\n")
    R("---\n")

    R("## Resumen ejecutivo\n")
    n_susp = A["conteo_nivel"].get("Moderado", 0) + A["conteo_nivel"].get("Alto", 0)
    R(dedent(f"""\
        Se analizó la integridad de los **{A['n_plataforma']:,} exámenes de
        plataforma** (con telemetría). Los **{A['n_sin_telemetria']:,} exámenes
        escritos** no tienen tiempo ni cambios de pestaña y quedan marcados como
        *"Sin telemetría"* (fuera del análisis). Se aplicaron **4 criterios**
        objetivos de sospecha; un examen es sospechoso si cumple **≥ 2**.

        Resultado: **{n_susp:,} exámenes sospechosos** (≥2 criterios), de ellos
        **{A['conteo_nivel'].get('Alto', 0):,} de nivel "Alto"** (≥3 criterios).
        Se recomienda **anular {C['n_anular']:,}** exámenes ({C['pct_anular']:.1f}%
        del total), lo que corrige el promedio general de {C['media_antes']:.2f} a
        {C['media_despues']:.2f} ({C['delta']:+.2f} puntos).\n"""))

    R("## Metodología — criterios de sospecha\n")
    R(dedent(f"""\
        Solo se evalúan exámenes de plataforma. Cada criterio marca un patrón
        difícil de lograr honestamente:

        | Criterio | Condición | Justificación |
        | --- | --- | --- |
        | **A** | puntaje ≥ 60 **y** tiempo < 35 min | 40 preguntas en <35 min con buen puntaje es muy rápido. |
        | **B** | cambios de pestaña ≥ 5 **y** puntaje ≥ 60 | Salir del examen repetidas veces + buen puntaje sugiere consulta externa. |
        | **C** | velocidad > percentil 95 | Velocidad (puntos/min) atípica respecto al grupo (p95 = {A['vel_p95']:.2f}). |
        | **D** | puntaje = 100 **y** tiempo < 45 min | Puntaje perfecto en menos de media prueba: casi imposible sin ayuda. |

        **Nivel de sospecha** por nº de criterios: 0 = *Limpio*, 1 = *Bajo*,
        2 = *Moderado*, ≥3 = *Alto*. El **tiempo mínimo razonable** observado para
        sacar ≥ 60 (percentil 5 de quienes lo lograron) es
        **{A['tiempo_razonable']:.1f} minutos**.\n"""))
    R("\n**Criterios activados (nº de exámenes que cumple cada uno):**\n")
    R(tabla_md(pd.DataFrame([{"Criterio": k, "N exámenes": v}
                             for k, v in A["conteo_criterio"].items()])) + "\n")
    R(f"\n{img('scatter', 'Tiempo vs puntaje')}\n")
    R(f"\n{img('velocidad', 'Velocidad y tiempo')}\n")

    R("## Análisis detallado\n")
    R("**Exámenes por nivel de sospecha (plataforma):**\n")
    R(tabla_md(pd.DataFrame([{"Nivel": k, "N": v}
                             for k, v in A["conteo_nivel"].items()])) + "\n")
    R(f"\n{img('niveles', 'Niveles de sospecha')}\n")
    R(f"\n{img('dist_susp', 'Distribución sospechosos vs limpios')}\n")
    if isinstance(B.get("colegios"), pd.DataFrame):
        R(f"\n{img('colegios', 'Sospechosos por colegio')}\n")
        if isinstance(B.get("colegios_anomalos"), pd.DataFrame) and not B["colegios_anomalos"].empty:
            R(dedent(f"""\

                **Colegios con tasa de sospecha anormalmente alta** (> media +
                1σ = {B['umbral_anomalo']:.1f}%; media global {B['tasa_media']:.1f}%):\n"""))
            ca = B["colegios_anomalos"][["institucion_educativa", "n_plataforma",
                                         "n_sospechosos", "tasa_sospecha"]].copy()
            ca.columns = ["Institución", "N plataforma", "Sospechosos", "Tasa %"]
            R(tabla_md(ca) + "\n")
    R(dedent(f"""\

        **Impacto de anular el nivel "Alto":** el promedio general pasa de
        {B['media_antes']:.2f} a {B['media_sin_alto']:.2f}
        ({B['delta_sin_alto']:+.2f}) al retirar {B['n_alto']} exámenes.\n"""))

    R("## Criterio de anulación recomendado\n")
    R(dedent(f"""\
        **Regla propuesta:** anular si el examen cumple **≥ 2 criterios** de
        sospecha **y** tiene **puntaje ≥ 60** (la nota baja no se beneficia de
        hacer trampa, así que no se penaliza). Esta regla es conservadora:
        exige evidencia múltiple y solo afecta puntajes que "valen la pena".

        - **Exámenes a anular:** {C['n_anular']:,} ({C['pct_anular']:.1f}% del total).
        - **Promedio general:** {C['media_antes']:.2f} → {C['media_despues']:.2f}
          ({C['delta']:+.2f}).\n"""))
    R(f"\n{img('impacto', 'Impacto en el promedio')}\n")
    if isinstance(C.get("ranking_cambio"), pd.DataFrame) and not C["ranking_cambio"].empty:
        R("\n**Colegios más afectados en su promedio (antes → después):**\n")
        rc = C["ranking_cambio"].head(10).reset_index()
        rc.columns = ["Institución", "Antes", "Después", "Δ"]
        R(tabla_md(rc) + "\n")
        R(f"\n{img('ranking', 'Ranking antes vs después')}\n")

    R("## Impacto en el modelo ML\n")
    if D:
        mejora = D["r2_clean"][0] - D["r2_base"][0]
        R(dedent(f"""\
            Se re-entrenó el modelo lineal de la Fase 2 quitando los
            **{D['n_removidos']} exámenes de nivel "Alto"**:

            - R² (CV) con todos:        **{D['r2_base'][0]:.3f} ± {D['r2_base'][1]:.3f}**
            - R² (CV) sin nivel "Alto": **{D['r2_clean'][0]:.3f} ± {D['r2_clean'][1]:.3f}**
            - Cambio: **{mejora:+.3f}**

            {'El R² **mejora** al quitar los sospechosos: los exámenes con trampa introducían ruido (puntajes altos no explicables por las variables del estudiante).' if mejora > 0.002 else 'El R² **no cambia de forma relevante**: los sospechosos de nivel Alto son pocos y no dominan el ajuste global, aunque su anulación sigue siendo correcta por integridad.'}\n"""))
        R(f"\n{img('imp', 'Importancia antes vs después')}\n")
    else:
        R("_(Sección omitida: falta models/predictor.py; ejecute antes el script 03.)_\n")

    R("## Recomendación final para la Fundación\n")
    R(dedent(f"""\
        1. **Anular los {C['n_anular']:,} exámenes** que cumplen la regla (≥2
           criterios y puntaje ≥ 60) y ofrecer **repetición supervisada**.
        2. **Priorizar revisión manual** de los exámenes de nivel "Alto"
           (≥3 criterios) y de los colegios con tasa de sospecha anómala.
        3. **Cerrar la brecha de origen:** exigir el sistema anti-cheat
           (telemetría) para TODOS los exámenes futuros; los escritos sin control
           no son auditables.
        4. Tratar estos criterios como **señal de alerta, no prueba definitiva**:
           la decisión final debe combinar la evidencia estadística con revisión
           humana y derecho de réplica del estudiante.\n"""))

    R("## Limitaciones\n")
    R(dedent("""\
        - Los umbrales (35/45 min, p95) son **heurísticos** calibrados con esta
          cohorte; conviene validarlos con casos confirmados.
        - Un estudiante muy hábil **puede** ser rápido legítimamente: por eso se
          exige evidencia múltiple (≥2 criterios) antes de recomendar anulación.
        - Los **exámenes escritos** no son auditables por falta de telemetría; su
          integridad debe garantizarse por otros medios (supervisión presencial).\n"""))
    R("\n---\n_Generado por `notebooks/05_deteccion_trampa.py` — Copa STEM 2026._\n")

    destino = REPORTS_DIR / "05_deteccion_trampa.md"
    destino.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"    informe escrito → reports/{destino.name}")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Detección de Trampa (integridad del examen)")
    print(" Fundación SapienceLab")
    print("=" * 70)

    df = cargar_y_limpiar()
    A = calcular_sospecha(df)
    B = analisis_detallado(df, A)
    C = criterio_anulacion(df)
    D = impacto_modelo(df)
    exportar_csv(df)
    construir_informe(df, A, B, C, D)

    print("\n" + "=" * 70)
    print(" ✔ DETECCIÓN DE TRAMPA COMPLETADA")
    print(f"   · Plataforma:        {A['n_plataforma']:,} | "
          f"Sin telemetría: {A['n_sin_telemetria']:,}")
    print(f"   · Sospechosos ≥2:    "
          f"{A['conteo_nivel'].get('Moderado', 0) + A['conteo_nivel'].get('Alto', 0):,}"
          f" (Alto: {A['conteo_nivel'].get('Alto', 0):,})")
    print(f"   · Recomendado anular: {C['n_anular']:,} ({C['pct_anular']:.1f}%)")
    print(f"   · Figuras:           {len(FIGURES)} → outputs/")
    print(f"   · Export:            models/deploy/sospecha_trampa.csv")
    print(f"   · Informe:           reports/05_deteccion_trampa.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
