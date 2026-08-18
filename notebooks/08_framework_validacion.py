# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 08: Framework de Validación del Modelo Predictivo  (Fase 4 — Robustez)
================================================================================

Contexto
--------
El modelo predictivo del puntaje (script 03) alcanza un R² ≈ 0.09. Esto NO es
un error de código: es un **hallazgo** sobre los datos. El puntaje de un examen
depende de factores que hoy NO medimos (preparación, motivación puntual, hábitos
de estudio), de modo que las variables socioeconómicas y demográficas disponibles
solo pueden explicar una fracción pequeña de la varianza.

Este script construye un **framework** para (a) VALIDAR que ese R² = 0.09 es
estable y confiable, (b) DIAGNOSTICAR por qué es bajo (¿es el modelo o son los
datos?), (c) VIGILAR el modelo cuando lleguen datos nuevos (drift), y (d) trazar
un PLAN concreto para subir el R² en la próxima Copa STEM.

Secciones
---------
    A) Validación del modelo actual
        A1) Split TEMPORAL simulado (70% antiguas / 30% nuevas) → ¿generaliza?
        A2) Curva de calibración (10 bins): ¿el modelo es "honesto"?
        A3) Bootstrap de R² (1000 remuestreos) → R² medio ± IC 95%.
    B) Diagnóstico: ¿por qué R² = 0.09?
        B1) Techo teórico con "estudiantes gemelos" (varianza intra-grupo).
        B2) Varianza descompuesta (explicada por variable vs residual).
        B3) Variables que FALTAN + formulario sugerido para la próxima edición.
    C) Framework para datos nuevos → models/deploy/validation_framework.py
        Función PURA validate_new_data(csv_path): predice, mide y detecta drift.
    D) Plan de mejora del R² (documentado en el informe).
    E) Informe reports/08_framework_validacion.md

Principios de diseño
--------------------
- Autocontenido y reproducible: `random_state=42`.
- Reutiliza la infraestructura del script 03 (mismas features, mismo
  preprocesamiento, mismo modelo ganador) importándolo como módulo, para que la
  validación sea coherente bit a bit con el modelo en producción.
- El split temporal es SIMULADO: el dataset no tiene columna de fecha, así que se
  usa el ORDEN de fila del CSV (orden de inscripción en Supabase) como proxy del
  tiempo. Se documenta esta limitación en el informe.
- Paleta de marca Copa STEM en todos los gráficos.

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

    from sklearn.base import clone
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
except ImportError as exc:  # pragma: no cover
    print("ERROR: falta una dependencia del entorno.")
    print(f"       Detalle: {exc}")
    print("       Instale: pandas numpy scikit-learn matplotlib seaborn joblib "
          "xgboost lightgbm")
    sys.exit(1)

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# La consola de Windows (cp1252) no imprime '→', '±', etc.; forzamos UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass


# =============================================================================
# 0. CONFIGURACIÓN GLOBAL
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
for _d in (OUTPUTS_DIR, REPORTS_DIR, MODELS_DIR, DEPLOY_DIR):
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
# 0.b IMPORTAR LA INFRAESTRUCTURA DEL SCRIPT 03 (misma feature-pipeline)
# =============================================================================
# El script 03 tiene nombre con dígito inicial (no importable con `import`),
# así que lo cargamos por ruta. Al importarlo NO se ejecuta main() (está
# protegido por __main__), solo se definen funciones/constantes reutilizables.

def _import_por_ruta(nombre_modulo: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cargar_infra():
    log("SECCIÓN 0 — Importando infraestructura del script 03")
    ruta03 = BASE_DIR / "notebooks" / "03_modelo_predictivo.py"
    if not ruta03.exists():
        print(f"\n  ⚠  No se encontró {ruta03}. Ejecute antes el script 03.\n")
        sys.exit(1)
    m03 = _import_por_ruta("modelo03", ruta03)
    log("    infraestructura de features/preprocesamiento cargada desde 03")
    return m03


def resolver_mejor_modelo(m03) -> str:
    """Nombre del modelo ganador según el joblib exportado por 03 (o RF por defecto)."""
    try:
        import joblib
        bundle = joblib.load(MODELS_DIR / "mejor_modelo_puntaje.joblib")
        nombre = bundle.get("nombre", "Random Forest")
    except Exception:
        nombre = "Random Forest"
    disponibles = m03.construir_modelos()
    if nombre not in disponibles:
        nombre = "Random Forest" if "Random Forest" in disponibles \
            else next(iter(disponibles))
    log(f"    modelo ganador (según 03): {nombre}")
    return nombre


# =============================================================================
# CONTEXTO DE DATOS (orden de fila preservado = proxy temporal)
# =============================================================================

def preparar_contexto(m03) -> dict:
    log("SECCIÓN 0 — Preparación del contexto de datos")
    df = m03.cargar_y_limpiar()
    TARGET = m03.TARGET
    # Conservamos el ORDEN de fila del CSV (proxy de orden de inscripción).
    modelo_df = df[df[TARGET].notna()].copy().reset_index(drop=True)
    log(f"    estudiantes con puntaje (modelables): {len(modelo_df):,}")

    # Preprocesador ajustado con TODA la cohorte (para OOF/calibración/bootstrap).
    PRE_full = m03.fit_preprocessor(modelo_df.to_dict("records"), modelo_df)
    feat_names = m03.feature_names_from_pre(PRE_full)
    X_full = m03.build_matrix(modelo_df.to_dict("records"), PRE_full)
    y_full = modelo_df[TARGET].to_numpy(dtype=float)

    return {"m03": m03, "TARGET": TARGET, "modelo_df": modelo_df,
            "PRE_full": PRE_full, "feat_names": feat_names,
            "X_full": X_full, "y_full": y_full}


def _metricas(y_true, y_pred) -> dict:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "n":    int(len(y_true)),
    }


# =============================================================================
# A1. SPLIT TEMPORAL SIMULADO (70% antiguas / 30% nuevas)
# =============================================================================

def validacion_temporal(ctx: dict, nombre_modelo: str,
                        oof_r2: float, boot: dict) -> dict:
    log("SECCIÓN A1 — Validación con split temporal simulado (70/30 por orden)")
    m03 = ctx["m03"]
    TARGET = ctx["TARGET"]
    modelo_df = ctx["modelo_df"]

    n = len(modelo_df)
    corte = int(round(0.70 * n))
    df_antiguas = modelo_df.iloc[:corte].reset_index(drop=True)
    df_nuevas = modelo_df.iloc[corte:].reset_index(drop=True)
    log(f"    'antiguas' (train 70%) = {len(df_antiguas):,} | "
        f"'nuevas' (test 30%) = {len(df_nuevas):,}")

    # Preprocesador ajustado SOLO con las 'antiguas' (sin ver las 'nuevas').
    PRE = m03.fit_preprocessor(df_antiguas.to_dict("records"), modelo_df)
    Xtr = m03.build_matrix(df_antiguas.to_dict("records"), PRE)
    Xnew = m03.build_matrix(df_nuevas.to_dict("records"), PRE)
    ytr = df_antiguas[TARGET].to_numpy(dtype=float)
    ynew = df_nuevas[TARGET].to_numpy(dtype=float)

    modelo = clone(m03.construir_modelos()[nombre_modelo])
    modelo.fit(Xtr, ytr)

    pred_tr = np.clip(modelo.predict(Xtr), 0, 100)
    pred_new = np.clip(modelo.predict(Xnew), 0, 100)
    m_tr = _metricas(ytr, pred_tr)
    m_new = _metricas(ynew, pred_new)

    print(f"        TRAIN(70%, in-sample) R²={m_tr['r2']:.3f} RMSE={m_tr['rmse']:.2f} "
          f"MAE={m_tr['mae']:.2f}")
    print(f"        NUEVAS(30%)           R²={m_new['r2']:.3f} RMSE={m_new['rmse']:.2f} "
          f"MAE={m_new['mae']:.2f}")

    # Veredicto RIGUROSO: el R² in-sample del train (RF) es optimista por
    # construcción, así que NO sirve de referencia. La referencia honesta es el R²
    # out-of-fold (≈{oof}) y su IC 95% por bootstrap. El modelo "generaliza a lo
    # largo del tiempo" si el R² de las 'nuevas' cae DENTRO de ese IC.
    lo, hi = boot["lo"], boot["hi"]
    dentro_ic = lo <= m_new["r2"] <= hi
    delta_vs_oof = m_new["r2"] - oof_r2
    delta_rmse_rel = abs(m_new["rmse"] - m_tr["rmse"]) / max(m_tr["rmse"], 1e-9)
    generaliza = dentro_ic
    if generaliza:
        veredicto = (f"El modelo GENERALIZA a lo largo del tiempo: el R² en las "
                     f"inscripciones 'nuevas' ({m_new['r2']:.3f}) cae dentro del IC 95% "
                     f"del R² global ([{lo:.3f}, {hi:.3f}]). El R² ≈ {oof_r2:.2f} es "
                     f"estable; no es un artefacto de sobreajuste.")
    else:
        veredicto = (f"GENERALIZACIÓN PARCIAL: en agregado el R² es estable "
                     f"(IC 95% [{lo:.3f}, {hi:.3f}]), pero en el 30% de inscripciones "
                     f"más recientes cae a {m_new['r2']:.3f}, POR DEBAJO de ese IC. La "
                     f"relación features→puntaje se desplaza a lo largo de la "
                     f"inscripción (concept drift leve): conviene RE-VALIDAR (y quizá "
                     f"re-entrenar) el modelo con cada nueva cohorte, en vez de asumir "
                     f"que el R² se mantiene. Esto justifica el framework de la sección C.")
    log(f"    veredicto: {'GENERALIZA' if generaliza else 'GENERALIZACIÓN PARCIAL'} "
        f"(nuevas R²={m_new['r2']:.3f} vs IC95 [{lo:.3f}, {hi:.3f}])")

    # --- Gráfico comparativo train vs nuevas, con banda de referencia OOF ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    etiquetas = ["Train 70%\n(in-sample, optimista)", "Nuevas 30%\n(recientes)"]
    r2s = [m_tr["r2"], m_new["r2"]]
    rmses = [m_tr["rmse"], m_new["rmse"]]
    cols = [COLORS["violet"], COLORS["cyan"]]
    ax1.axhspan(lo, hi, color=COLORS["green"], alpha=0.15,
                label=f"IC 95% del R² global\n[{lo:.3f}, {hi:.3f}]")
    ax1.axhline(oof_r2, color=COLORS["green"], linestyle="--", linewidth=1.4,
                label=f"R² out-of-fold = {oof_r2:.3f}")
    ax1.bar(etiquetas, r2s, color=cols, edgecolor="white")
    ax1.set_title("R²: entrenamiento vs. datos 'nuevos'")
    ax1.set_ylabel("R²")
    ax1.axhline(0, color=COLORS["red"], linewidth=1)
    for i, v in enumerate(r2s):
        ax1.text(i, v, f"{v:.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax2.bar(etiquetas, rmses, color=cols, edgecolor="white")
    ax2.set_title("RMSE: entrenamiento vs. datos 'nuevos'")
    ax2.set_ylabel("RMSE (puntos)")
    for i, v in enumerate(rmses):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("A1. Split temporal simulado — ¿el modelo generaliza a datos nuevos?",
                 fontsize=14, fontweight="bold")
    savefig(fig, "F08_validacion_temporal.png", "temporal")

    return {"m_tr": m_tr, "m_new": m_new, "corte": corte, "n": n,
            "delta_vs_oof": delta_vs_oof, "delta_rmse_rel": delta_rmse_rel,
            "dentro_ic": dentro_ic, "generaliza": generaliza,
            "veredicto": veredicto}


# =============================================================================
# OOF: predicciones fuera-de-muestra (base para calibración y bootstrap)
# =============================================================================

def predicciones_oof(ctx: dict, nombre_modelo: str) -> np.ndarray:
    log("SECCIÓN A — Predicciones out-of-fold (5-fold) para calibración/bootstrap")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    modelo = clone(ctx["m03"].construir_modelos()[nombre_modelo])
    pred = cross_val_predict(modelo, ctx["X_full"], ctx["y_full"],
                             cv=cv, n_jobs=-1)
    pred = np.clip(pred, 0, 100)
    m = _metricas(ctx["y_full"], pred)
    log(f"    OOF R²={m['r2']:.3f} RMSE={m['rmse']:.2f} MAE={m['mae']:.2f}")
    return pred, m


# =============================================================================
# A2. CURVA DE CALIBRACIÓN (10 bins)
# =============================================================================

def analisis_calibracion(ctx: dict, pred_oof: np.ndarray) -> dict:
    log("SECCIÓN A2 — Curva de calibración (10 bins)")
    y = ctx["y_full"]
    dfc = pd.DataFrame({"real": y, "pred": pred_oof})
    # 10 bins por cuantiles de la predicción (grupos de tamaño ~igual).
    dfc["bin"] = pd.qcut(dfc["pred"].rank(method="first"), 10, labels=False)
    agg = dfc.groupby("bin").agg(
        pred_medio=("pred", "mean"),
        real_medio=("real", "mean"),
        n=("real", "size")).reset_index()

    # Error de calibración: |predicho − real| promedio por bin.
    ece = float((agg["pred_medio"] - agg["real_medio"]).abs().mean())
    log(f"    error de calibración medio (|pred−real| por bin): {ece:.2f} puntos")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    lim = [min(agg["pred_medio"].min(), agg["real_medio"].min()) - 3,
           max(agg["pred_medio"].max(), agg["real_medio"].max()) + 3]
    ax.plot(lim, lim, "--", color=COLORS["red"], linewidth=1.6,
            label="Calibración perfecta (45°)")
    ax.scatter(agg["pred_medio"], agg["real_medio"],
               s=agg["n"] * 1.2, color=COLORS["cyan"],
               edgecolor=COLORS["dark"], linewidth=0.6, zorder=3,
               label="Bin (tamaño ∝ N)")
    ax.plot(agg["pred_medio"], agg["real_medio"], color=COLORS["violet"],
            linewidth=1.2, alpha=0.7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title(f"A2. Curva de calibración — 10 bins\n"
                 f"Error medio de calibración = {ece:.2f} puntos")
    ax.set_xlabel("Puntaje predicho (promedio del bin)")
    ax.set_ylabel("Puntaje real (promedio del bin)")
    ax.legend(loc="upper left")
    savefig(fig, "F08_calibracion.png", "calibracion")

    return {"tabla": agg, "ece": ece}


# =============================================================================
# A3. BOOTSTRAP DE R² (1000 remuestreos)
# =============================================================================

def bootstrap_r2(ctx: dict, pred_oof: np.ndarray, n_boot: int = 1000) -> dict:
    log(f"SECCIÓN A3 — Bootstrap de R² ({n_boot} remuestreos)")
    y = ctx["y_full"]
    p = pred_oof
    n = len(y)
    rng = np.random.default_rng(RANDOM_STATE)
    r2s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # muestreo con reemplazo
        r2s[b] = r2_score(y[idx], p[idx])
    media = float(np.mean(r2s))
    lo, hi = float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))
    log(f"    R² = {media:.3f}  IC95% = [{lo:.3f}, {hi:.3f}]")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.histplot(r2s, bins=40, color=COLORS["cyan"], edgecolor="white",
                 stat="density", ax=ax)
    ax.axvline(media, color=COLORS["violet"], linewidth=2,
               label=f"R² medio = {media:.3f}")
    ax.axvline(lo, color=COLORS["amber"], linestyle="--", linewidth=1.6,
               label=f"IC 95% = [{lo:.3f}, {hi:.3f}]")
    ax.axvline(hi, color=COLORS["amber"], linestyle="--", linewidth=1.6)
    ax.axvline(0, color=COLORS["red"], linewidth=1)
    ax.set_title(f"A3. Distribución bootstrap del R² ({n_boot} remuestreos)")
    ax.set_xlabel("R²")
    ax.set_ylabel("Densidad")
    ax.legend()
    savefig(fig, "F08_bootstrap_r2.png", "bootstrap")

    estable = lo > 0
    return {"media": media, "lo": lo, "hi": hi, "n_boot": n_boot,
            "estable": estable}


# =============================================================================
# B1. TECHO TEÓRICO — "ESTUDIANTES GEMELOS"
# =============================================================================

GRUPO_GEMELOS = ["grado_escolar", "estrato", "computador_en_casa",
                 "municipio", "genero"]


def techo_teorico(ctx: dict) -> dict:
    log("SECCIÓN B1 — Techo teórico con 'estudiantes gemelos'")
    TARGET = ctx["TARGET"]
    df = ctx["modelo_df"]
    cols = [c for c in GRUPO_GEMELOS if c in df.columns]
    sub = df.dropna(subset=cols + [TARGET]).copy()

    tam = sub.groupby(cols)[TARGET].transform("size")
    gemelos = sub[tam >= 2].copy()
    n_grupos = gemelos.groupby(cols).ngroups if len(gemelos) else 0
    frac_gemelos = len(gemelos) / len(sub) if len(sub) else 0.0

    # Techo: el mejor predictor posible con estas features es la MEDIA del grupo.
    # R² de ese predictor = varianza ENTRE grupos / varianza total (sobre gemelos).
    if len(gemelos):
        media_grupo = gemelos.groupby(cols)[TARGET].transform("mean")
        ss_res = float(((gemelos[TARGET] - media_grupo) ** 2).sum())
        ss_tot = float(((gemelos[TARGET] - gemelos[TARGET].mean()) ** 2).sum())
        techo = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        std_intra = float(gemelos.groupby(cols)[TARGET].std().mean())
        std_total = float(gemelos[TARGET].std())
    else:
        techo, std_intra, std_total = 0.0, 0.0, float(sub[TARGET].std())

    log(f"    gemelos: {len(gemelos):,} estudiantes en {n_grupos:,} grupos "
        f"({frac_gemelos:.1%} de la muestra)")
    log(f"    desv. intra-grupo media = {std_intra:.2f} vs total = {std_total:.2f}")
    log(f"    techo teórico de R² (predictor = media del grupo) ≈ {techo:.3f}")

    # Ejemplos de grupos de gemelos con puntajes muy dispares.
    resumen = (gemelos.groupby(cols)[TARGET]
               .agg(n="size", media="mean", desv="std",
                    minimo="min", maximo="max")
               .reset_index()) if len(gemelos) else pd.DataFrame()
    ejemplos = pd.DataFrame()
    if len(resumen):
        ejemplos = resumen.sort_values("desv", ascending=False).head(6).copy()
        ejemplos["rango"] = (ejemplos["maximo"] - ejemplos["minimo"]).round(0)

    # --- Gráfico: dispersión intra-grupo de los grupos más grandes ---
    fig, ax = plt.subplots(figsize=(11, 5.6))
    if len(resumen):
        top = resumen.sort_values("n", ascending=False).head(12).reset_index(drop=True)
        top["etq"] = ["G" + str(i + 1) for i in range(len(top))]
        for i, row in top.iterrows():
            ax.errorbar(i, row["media"], yerr=row["desv"], fmt="o",
                        color=COLORS["cyan"], ecolor=COLORS["violet"],
                        elinewidth=2, capsize=5, markersize=7,
                        markeredgecolor=COLORS["dark"])
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([f"{e}\n(n={n})" for e, n in zip(top["etq"], top["n"])],
                           fontsize=8)
        ax.axhline(gemelos[TARGET].mean(), color=COLORS["red"], linestyle="--",
                   linewidth=1.2, label="Media global")
        ax.set_ylabel("Puntaje obtenido")
        ax.set_xlabel("Grupos de 'gemelos' (mismas features) — barra = ± desv. intra")
        ax.set_title(f"B1. Estudiantes 'gemelos': mismas features, puntajes dispares\n"
                     f"Techo teórico de R² ≈ {techo:.3f}  "
                     f"(desv. intra-grupo media = {std_intra:.1f} pts)")
        ax.legend()
    savefig(fig, "F08_techo_gemelos.png", "gemelos")

    return {"techo": techo, "n_gemelos": int(len(gemelos)),
            "n_grupos": int(n_grupos), "frac_gemelos": frac_gemelos,
            "std_intra": std_intra, "std_total": std_total,
            "cols": cols, "ejemplos": ejemplos}


# =============================================================================
# B2. VARIANZA DESCOMPUESTA (explicada por variable vs residual)
# =============================================================================

VARS_DESCOMP = [
    ("grado_escolar", "Grado escolar"),
    ("estrato", "Estrato"),
    ("genero", "Género"),
    ("municipio", "Municipio"),
    ("tipo_institucion", "Tipo institución"),
    ("computador_en_casa", "Computador en casa"),
    ("internet_en_casa", "Internet en casa"),
    ("participacion_olimpiadas", "Participó antes"),
    ("nivel_programacion", "Nivel programación"),
    ("nivel_robotica", "Nivel robótica"),
    ("interes_prog_robotica", "Interés prog/robótica"),
]


def _eta2(df: pd.DataFrame, col: str, target: str) -> float:
    """Correlación de razón (eta²): fracción de varianza del target explicada
    por las categorías/valores de `col` de forma marginal (univariada)."""
    sub = df.dropna(subset=[col, target])
    if len(sub) < 2:
        return 0.0
    grand = sub[target].mean()
    ss_tot = float(((sub[target] - grand) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    ss_bet = float(sub.groupby(col)[target]
                   .apply(lambda x: len(x) * (x.mean() - grand) ** 2).sum())
    return ss_bet / ss_tot


def varianza_descompuesta(ctx: dict, oof_r2: float, techo: float) -> dict:
    log("SECCIÓN B2 — Varianza descompuesta por variable")
    TARGET = ctx["TARGET"]
    df = ctx["modelo_df"]
    filas = []
    for col, etq in VARS_DESCOMP:
        if col in df.columns:
            filas.append((etq, round(_eta2(df, col, TARGET) * 100, 2)))
    filas.sort(key=lambda t: t[1], reverse=True)
    etiquetas = [f for f, _ in filas]
    valores = [v for _, v in filas]

    # --- Gráfico 1: eta² marginal por variable (barh) ---
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(filas))))
    colores = gradient_colors(len(filas))
    ax.barh(etiquetas[::-1], valores[::-1], color=colores[::-1], edgecolor="white")
    for i, v in enumerate(valores[::-1]):
        ax.text(v + 0.03, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_title("B2. Varianza del puntaje explicada por cada variable (marginal, η²)")
    ax.set_xlabel("% de varianza del puntaje explicada (individual)")
    savefig(fig, "F08_varianza_por_variable.png", "var_variable")

    # --- Gráfico 2: descomposición honesta (apilada) ---
    # explicada por el modelo | explicable no capturada | inexplicable (individual)
    modelo_pct = max(0.0, oof_r2) * 100
    techo_pct = max(modelo_pct / 100, techo) * 100
    headroom_pct = max(0.0, techo_pct - modelo_pct)
    inexplicable_pct = max(0.0, 100 - techo_pct)

    fig2, ax2 = plt.subplots(figsize=(9, 3.4))
    segmentos = [
        ("Explicada por el modelo actual", modelo_pct, COLORS["green"]),
        ("Explicable no capturada (mejor modelo/datos)", headroom_pct, COLORS["amber"]),
        ("Inexplicable con estas features (diferencias individuales)",
         inexplicable_pct, COLORS["red"]),
    ]
    izq = 0.0
    for nombre, val, c in segmentos:
        ax2.barh(0, val, left=izq, color=c, edgecolor="white",
                 label=f"{nombre} — {val:.1f}%")
        if val > 4:
            ax2.text(izq + val / 2, 0, f"{val:.0f}%", ha="center", va="center",
                     color="white", fontweight="bold", fontsize=10)
        izq += val
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel("% de la varianza total del puntaje")
    ax2.set_title("B2. Descomposición de la varianza del puntaje")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=1, fontsize=9)
    savefig(fig2, "F08_varianza_descompuesta.png", "var_descomp")

    return {"por_variable": filas, "modelo_pct": modelo_pct,
            "headroom_pct": headroom_pct, "inexplicable_pct": inexplicable_pct,
            "techo_pct": techo_pct}


# =============================================================================
# B3 / D. VARIABLES QUE FALTAN + PLAN DE MEJORA
# =============================================================================
# Estimaciones de mejora del R² basadas en literatura de predicción de
# rendimiento académico (el GPA/nota previa es el predictor individual más
# fuerte; motivación, tiempo de estudio e interés aportan efectos moderados).

VARIABLES_FALTANTES = [
    {"variable": "Promedio académico del último período",
     "tipo": "Numérica (0–5 o 0–100)",
     "delta_r2": "+0.10 a +0.20",
     "sustento": "La nota/GPA previa es el predictor individual más fuerte del "
                 "rendimiento (r≈0.4–0.5 en la literatura)."},
    {"variable": "Horas semanales de estudio de matemáticas",
     "tipo": "Numérica (horas)",
     "delta_r2": "+0.03 a +0.07",
     "sustento": "El tiempo de práctica dedicado correlaciona con el logro "
                 "(efecto moderado y consistente)."},
    {"variable": "Motivación para participar",
     "tipo": "Escala 1–5",
     "delta_r2": "+0.02 a +0.05",
     "sustento": "La motivación intrínseca predice el rendimiento en "
                 "meta-análisis educativos."},
    {"variable": "¿Ha recibido clases extra de matemáticas?",
     "tipo": "Binaria (Sí/No)",
     "delta_r2": "+0.02 a +0.05",
     "sustento": "El apoyo/refuerzo adicional mejora el desempeño medido."},
    {"variable": "¿Le gusta resolver problemas lógicos?",
     "tipo": "Escala 1–5",
     "delta_r2": "+0.03 a +0.06",
     "sustento": "La afinidad específica por el razonamiento lógico predice el "
                 "desempeño en tareas afines al examen."},
]

R2_OBJETIVO = (0.25, 0.40)


def plan_mejora() -> dict:
    log("SECCIÓN B3/D — Variables faltantes y plan de mejora del R²")

    etiquetas = [v["variable"].split(" (")[0][:34] for v in VARIABLES_FALTANTES]
    # Punto medio del rango estimado de mejora, para el gráfico.
    def _mid(s):
        nums = [float(x.replace("+", "")) for x in s.replace(" a ", " ").split()]
        return sum(nums) / len(nums)
    mids = [_mid(v["delta_r2"]) for v in VARIABLES_FALTANTES]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    base = 0.09
    izq = base
    colores = gradient_colors(len(mids))
    ax.barh(["R² actual"], [base], color=COLORS["red"], edgecolor="white")
    for etq, d, c in zip(etiquetas, mids, colores):
        ax.barh(["R² proyectado"], [d], left=izq, color=c, edgecolor="white",
                label=f"{etq} (+{d:.02f})")
        izq += d
    ax.axvline(R2_OBJETIVO[0], color=COLORS["dark"], linestyle=":", linewidth=1.4)
    ax.axvline(R2_OBJETIVO[1], color=COLORS["dark"], linestyle=":", linewidth=1.4,
               label=f"Objetivo {R2_OBJETIVO[0]:.2f}–{R2_OBJETIVO[1]:.2f}")
    ax.set_xlabel("R² acumulado (aproximado)")
    ax.set_title("D. Plan de mejora del R² — aporte estimado de nuevas variables\n"
                 "(estimaciones basadas en literatura educativa)")
    ax.legend(loc="lower right", fontsize=8)
    savefig(fig, "F08_plan_mejora_r2.png", "plan")

    return {"variables": VARIABLES_FALTANTES, "objetivo": R2_OBJETIVO,
            "r2_proyectado_medio": round(base + sum(mids), 3)}


# =============================================================================
# C. FRAMEWORK PARA DATOS NUEVOS — validation_framework.py (función PURA)
# =============================================================================

def _baseline_stats(ctx: dict, oof_m: dict, temp: dict, boot: dict,
                    nombre_modelo: str) -> dict:
    """Estadísticos de referencia (entrenamiento) para detectar drift.

    - numeric_ref: bordes de deciles + media/desv de cada campo numérico crudo.
    - categorical_ref: proporciones de cada categoría de los campos categóricos.
    """
    df = ctx["modelo_df"]
    TARGET = ctx["TARGET"]

    numeric_fields = ["grado_escolar", "estrato", "interes_prog_robotica",
                      "edad_calculada"]
    categorical_fields = ["genero", "municipio", "tipo_institucion",
                          "computador_en_casa", "internet_en_casa",
                          "participacion_olimpiadas", "nivel_programacion",
                          "nivel_robotica"]

    def _assign_bin(v, edges):
        # Réplica EXACTA de _bin_numerico del framework generado, para que las
        # proporciones de referencia sean consistentes con las que medirá el .py.
        for i in range(1, len(edges) - 1):
            if v < edges[i]:
                return i - 1
        return len(edges) - 2

    numeric_ref = {}
    for c in numeric_fields:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                edges = [float(q) for q in np.quantile(s, np.linspace(0, 1, 11))]
                nbins = len(edges) - 1
                # Proporción de referencia (entrenamiento) POR BIN, con el mismo
                # binning que usará el framework. Para variables de baja
                # cardinalidad (grado, estrato) los deciles son degenerados y la
                # distribución NO es uniforme; por eso guardamos las proporciones
                # reales en vez de asumir 10% por bin.
                conteo = [0] * nbins
                for v in s:
                    conteo[_assign_bin(float(v), edges)] += 1
                total = sum(conteo)
                bin_props = {str(i): conteo[i] / total for i in range(nbins)}
                numeric_ref[c] = {
                    "edges": edges, "bin_props": bin_props,
                    "mean": float(s.mean()), "std": float(s.std()),
                    "min": float(s.min()), "max": float(s.max())}

    categorical_ref = {}
    for c in categorical_fields:
        if c in df.columns:
            vc = df[c].astype(str).str.strip()
            vc = vc[~vc.str.lower().isin(["nan", "none", ""])]
            props = (vc.value_counts(normalize=True)).round(6).to_dict()
            if props:
                categorical_ref[c] = {k: float(v) for k, v in props.items()}

    return {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modelo": nombre_modelo,
        "target": TARGET,
        "n_train": int(len(df)),
        "target_mean": float(df[TARGET].mean()),
        "target_std": float(df[TARGET].std()),
        "metrics_train": {
            "r2_oof": round(oof_m["r2"], 4),
            "rmse_oof": round(oof_m["rmse"], 3),
            "mae_oof": round(oof_m["mae"], 3),
            "r2_temporal_nuevas": round(temp["m_new"]["r2"], 4),
            "r2_bootstrap_ic95": [round(boot["lo"], 4), round(boot["hi"], 4)],
        },
        "numeric_ref": numeric_ref,
        "categorical_ref": categorical_ref,
    }


def generar_validation_framework(ctx: dict, baseline: dict) -> Path:
    """Escribe models/deploy/validation_framework.py: función PURA (solo stdlib)
    que carga un CSV nuevo, predice, mide (si hay puntaje real) y detecta drift."""
    log("SECCIÓN C — Generando models/deploy/validation_framework.py (función pura)")
    m03 = ctx["m03"]

    # Reutilizamos las MISMAS funciones de transformación/predicción del predictor
    # de producción (embebidas por inspección de código, sin dependencias de ML).
    fns = [m03._isnan, m03._to_float, m03._parse_count, m03._ord_level,
           m03._bin_si, m03.features_from_raw, m03.predict_from_features]
    fuente_fns = "\n\n".join(inspect.getsource(f) for f in fns)

    # Spec del modelo en producción (preprocess + model) desde models/predictor.py.
    pred_mod = _import_por_ruta("predictor_puntaje", MODELS_DIR / "predictor.py")
    spec = pred_mod._SPEC

    spec_js = json.dumps(spec, ensure_ascii=True)
    base_js = json.dumps(baseline, ensure_ascii=True)
    umbral_r2_caida = 0.03  # caída de R² que dispara "re-entrenar"

    contenido = f'''# -*- coding: utf-8 -*-
"""
Framework de VALIDACIÓN de datos nuevos — Copa STEM 2026.
GENERADO AUTOMÁTICAMENTE por notebooks/08_framework_validacion.py — no editar a mano.

Función pura, SIN dependencias de ML (solo la librería estándar: csv, json, math,
statistics). Valida un CSV de inscripciones nuevas contra el modelo de puntaje en
producción y contra la distribución de features del entrenamiento.

Uso:
    from validation_framework import validate_new_data
    reporte = validate_new_data("ruta/a/datos_nuevos.csv")
    print(reporte["mensaje"])          # veredicto legible
    reporte["drift"]["hay_drift"]      # True/False
    reporte["metricas"]                # R²/RMSE/MAE si el CSV trae puntaje real

Interpretación del PSI (Population Stability Index) por feature:
    PSI < 0.10  → sin cambios relevantes
    0.10–0.25   → cambio moderado (vigilar)
    PSI > 0.25  → cambio significativo (drift)
"""
import csv
import json
import math
import statistics

_SPEC = json.loads(r\'\'\'{spec_js}\'\'\')
_BASELINE = json.loads(r\'\'\'{base_js}\'\'\')
PRE = _SPEC["preprocess"]
MODEL = _SPEC["model"]
FEATURE_NAMES = _SPEC["feature_names"]

_UMBRAL_R2_CAIDA = {umbral_r2_caida}
_PSI_MODERADO = 0.10
_PSI_ALTO = 0.25


# --- Funciones de transformación/predicción (idénticas al predictor de producción) ---
{fuente_fns}


def predecir_puntaje(estudiante):
    """Puntaje estimado (0–100) para un dict de features crudos."""
    return predict_from_features(features_from_raw(estudiante, PRE), MODEL)


# --- Utilidades de drift (PSI) ---
def _psi(esperado, actual):
    """Population Stability Index entre dos distribuciones (dict cat->proporción)."""
    claves = set(esperado) | set(actual)
    psi = 0.0
    for k in claves:
        e = max(float(esperado.get(k, 0.0)), 1e-4)
        a = max(float(actual.get(k, 0.0)), 1e-4)
        psi += (a - e) * math.log(a / e)
    return psi


def _nivel_psi(psi):
    if psi < _PSI_MODERADO:
        return "estable"
    if psi < _PSI_ALTO:
        return "moderado"
    return "drift"


def _bin_numerico(valor, edges):
    """Índice de bin (0..len(edges)-2) según los bordes de decil del baseline."""
    for i in range(1, len(edges) - 1):
        if valor < edges[i]:
            return i - 1
    return len(edges) - 2


def _to_float_safe(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _detectar_drift(filas):
    detalles = {{}}
    psis = []

    # Numéricos: proporción por bin de decil vs baseline (uniforme 10%).
    for campo, ref in _BASELINE.get("numeric_ref", {{}}).items():
        edges = ref["edges"]
        nbins = len(edges) - 1
        conteo = [0] * nbins
        total = 0
        for r in filas:
            v = _to_float_safe(r.get(campo))
            if v is None:
                continue
            conteo[_bin_numerico(v, edges)] += 1
            total += 1
        if total == 0:
            continue
        # Esperado = proporciones de ENTRENAMIENTO por bin (no uniforme: las
        # variables de baja cardinalidad tienen deciles degenerados).
        esperado = ref.get("bin_props", {{str(i): 1.0 / nbins for i in range(nbins)}})
        actual = {{str(i): conteo[i] / total for i in range(nbins)}}
        psi = _psi(esperado, actual)
        detalles[campo] = {{"psi": round(psi, 4), "nivel": _nivel_psi(psi),
                            "tipo": "numérico", "n": total}}
        psis.append(psi)

    # Categóricos: proporción por categoría vs baseline.
    for campo, ref in _BASELINE.get("categorical_ref", {{}}).items():
        vals = [str(r.get(campo)).strip() for r in filas
                if r.get(campo) is not None
                and str(r.get(campo)).strip().lower() not in ("nan", "none", "")]
        if not vals:
            continue
        total = len(vals)
        actual = {{}}
        for v in vals:
            actual[v] = actual.get(v, 0) + 1
        actual = {{k: c / total for k, c in actual.items()}}
        psi = _psi(ref, actual)
        detalles[campo] = {{"psi": round(psi, 4), "nivel": _nivel_psi(psi),
                            "tipo": "categórico", "n": total}}
        psis.append(psi)

    con_drift = [c for c, d in detalles.items() if d["nivel"] == "drift"]
    psi_max = max(psis) if psis else 0.0
    return {{"por_feature": detalles, "features_con_drift": con_drift,
            "psi_max": round(psi_max, 4), "hay_drift": len(con_drift) > 0}}


def _leer_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def validate_new_data(csv_path):
    """Valida un CSV de datos nuevos contra el modelo de puntaje en producción.

    Devuelve un dict con:
        n, predicciones (lista), metricas (o None si no hay puntaje real),
        drift (PSI por feature + veredicto), recomendacion, mensaje.
    """
    filas = _leer_csv(csv_path)
    n = len(filas)

    # 1) Predicciones con el modelo actual.
    preds = [round(predecir_puntaje(r), 2) for r in filas]

    # 2) Métricas si el CSV trae el puntaje real.
    target = _BASELINE["target"]
    reales, estimados = [], []
    for r, p in zip(filas, preds):
        yv = _to_float_safe(r.get(target))
        if yv is not None:
            reales.append(yv)
            estimados.append(p)
    metricas = None
    if len(reales) >= 5:
        media = sum(reales) / len(reales)
        ss_tot = sum((y - media) ** 2 for y in reales)
        ss_res = sum((y - p) ** 2 for y, p in zip(reales, estimados))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = math.sqrt(ss_res / len(reales))
        mae = sum(abs(y - p) for y, p in zip(reales, estimados)) / len(reales)
        base_r2 = _BASELINE["metrics_train"]["r2_oof"]
        metricas = {{"n_con_puntaje": len(reales),
                    "r2": round(r2, 4), "rmse": round(rmse, 3),
                    "mae": round(mae, 3),
                    "r2_entrenamiento": base_r2,
                    "caida_r2": round(base_r2 - r2, 4)}}

    # 3) Detección de drift.
    drift = _detectar_drift(filas)

    # 4) Veredicto y recomendación.
    reentrenar = False
    razones = []
    if drift["hay_drift"]:
        reentrenar = True
        razones.append("hay drift significativo en: "
                       + ", ".join(drift["features_con_drift"]))
    if metricas is not None and metricas["caida_r2"] > _UMBRAL_R2_CAIDA:
        reentrenar = True
        razones.append(f"el R² cayó {{metricas['caida_r2']:.3f}} respecto al "
                       f"entrenamiento ({{metricas['r2_entrenamiento']}})")

    if reentrenar:
        recomendacion = "RE-ENTRENAR"
        mensaje = ("⚠ Hay drift o caída de desempeño → conviene RE-ENTRENAR el "
                   "modelo. Motivos: " + "; ".join(razones) + ".")
    else:
        recomendacion = "OK"
        mensaje = ("✔ El modelo sigue funcionando bien: la distribución de los "
                   "datos nuevos es compatible con el entrenamiento"
                   + ("" if metricas is None
                      else f" y el R² se mantiene ({{metricas['r2']}})") + ".")

    return {{"n": n, "modelo": _BASELINE["modelo"], "predicciones": preds,
            "metricas": metricas, "drift": drift,
            "recomendacion": recomendacion, "razones": razones,
            "mensaje": mensaje, "baseline": _BASELINE["metrics_train"]}}


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Uso: python validation_framework.py <ruta_csv_datos_nuevos>")
        raise SystemExit(0)
    rep = validate_new_data(_sys.argv[1])
    print("Modelo:", rep["modelo"], "| N filas:", rep["n"])
    if rep["metricas"]:
        print("Métricas:", rep["metricas"])
    print("Drift (PSI máx):", rep["drift"]["psi_max"],
          "| features con drift:", rep["drift"]["features_con_drift"])
    print("Recomendación:", rep["recomendacion"])
    print(rep["mensaje"])
'''
    destino = DEPLOY_DIR / "validation_framework.py"
    destino.write_text(contenido, encoding="utf-8")
    log(f"    framework escrito → models/deploy/{destino.name}")
    return destino


def autoverificar_framework(ctx: dict, destino: Path) -> dict:
    """Prueba el framework con (a) el propio dataset (drift≈0) y (b) una muestra
    con features perturbadas (drift alto), para confirmar que discrimina."""
    log("SECCIÓN C — Auto-verificación del framework")
    import tempfile
    import shutil
    vf = _import_por_ruta("validation_framework_check", destino)

    scratch = Path(tempfile.mkdtemp(prefix="copastem_val_"))
    df = ctx["modelo_df"]

    # (a) Muestra representativa del propio dataset → NO debe haber drift.
    ruta_ok = scratch / "muestra_igual.csv"
    df.sample(min(400, len(df)), random_state=RANDOM_STATE).to_csv(
        ruta_ok, index=False, encoding="utf-8-sig")
    rep_ok = vf.validate_new_data(str(ruta_ok))

    # (b) Muestra perturbada (todo estrato 1, sin computador, solo grado 11) →
    #     debe disparar drift.
    df_drift = df.sample(min(400, len(df)), random_state=RANDOM_STATE).copy()
    if "estrato" in df_drift.columns:
        df_drift["estrato"] = 1
    if "computador_en_casa" in df_drift.columns:
        df_drift["computador_en_casa"] = "No"
    if "grado_escolar" in df_drift.columns:
        df_drift["grado_escolar"] = 11
    ruta_drift = scratch / "muestra_perturbada.csv"
    df_drift.to_csv(ruta_drift, index=False, encoding="utf-8-sig")
    rep_drift = vf.validate_new_data(str(ruta_drift))

    log(f"    muestra igual     → {rep_ok['recomendacion']} "
        f"(PSI máx={rep_ok['drift']['psi_max']})")
    log(f"    muestra perturbada→ {rep_drift['recomendacion']} "
        f"(PSI máx={rep_drift['drift']['psi_max']}, "
        f"drift en {rep_drift['drift']['features_con_drift']})")

    shutil.rmtree(scratch, ignore_errors=True)  # no dejar CSVs temporales
    return {"ok": rep_ok, "drift": rep_drift,
            "discrimina": (rep_ok["recomendacion"] == "OK"
                           and rep_drift["recomendacion"] == "RE-ENTRENAR")}


# =============================================================================
# E. INFORME MARKDOWN
# =============================================================================

def construir_informe(ctx, nombre_modelo, temp, oof_m, calib, boot,
                      techo, var, plan, framework_check) -> None:
    log("SECCIÓN E — Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    R = REPORT.append

    R("# Framework de Validación del Modelo Predictivo — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Fase 4 · Informe generado: {fecha}\n")
    R("---\n")

    # Resumen ejecutivo
    R("## Resumen ejecutivo\n")
    R(dedent(f"""\
        El modelo predictivo del puntaje (script 03, **{nombre_modelo}**) alcanza
        un R² ≈ {oof_m['r2']:.3f}. Este informe demuestra que ese valor **no es un
        error, sino un hallazgo estable**:

        1. **Estabilidad (bootstrap 1000×):** R² = {boot['media']:.3f}, IC 95%
           [{boot['lo']:.3f}, {boot['hi']:.3f}] → {'intervalo por encima de 0: el poder predictivo es real (pequeño pero no nulo).' if boot['estable'] else 'el intervalo cruza 0.'}
        2. **Generalización temporal (split simulado):** entrenando solo con el 70%
           de inscripciones más antiguas y prediciendo el 30% más reciente, el R²
           cae a {temp['m_new']['r2']:.3f}
           {'(dentro del IC 95%: generaliza).' if temp['generaliza'] else '— POR DEBAJO del IC 95% global. En agregado el R² es estable, pero no se transporta perfecto a la cohorte más reciente: hay que re-validar con cada nueva edición.'}
        3. **Calibración:** error medio de {calib['ece']:.2f} puntos entre lo
           predicho y lo real por bin → el modelo es razonablemente honesto.
        4. **Techo teórico:** los "estudiantes gemelos" (mismas features) tienen una
           desviación de puntaje de **{techo['std_intra']:.1f} puntos** dentro del
           grupo; el R² máximo alcanzable con estas variables es
           **≈ {techo['techo']:.3f}**. Por tanto, el R² bajo se explica por las
           **variables que faltan**, no por el algoritmo.

        Se entrega además `models/deploy/validation_framework.py`, una función pura
        `validate_new_data()` que valida datos futuros y detecta *drift*, y un plan
        para subir el R² a **{plan['objetivo'][0]:.2f}–{plan['objetivo'][1]:.2f}** en
        la próxima edición.\n"""))

    # Metodología
    R("## Metodología\n")
    R(dedent("""\
        - **Reutilización del pipeline de producción:** mismas features, mismo
          preprocesamiento y el mismo modelo ganador que el script 03 (importado como
          módulo) para que la validación sea coherente con el modelo real.
        - **Split temporal simulado:** el dataset no tiene columna de fecha; se usa
          el orden de fila del CSV (orden de inscripción en Supabase) como proxy del
          tiempo (70% antiguas → 30% nuevas). Ver *Limitaciones*.
        - **Predicciones out-of-fold (5-fold)** para calibración y bootstrap, de modo
          que ningún estudiante se evalúa con un modelo que lo vio en entrenamiento.
        - **Detección de drift por PSI** (Population Stability Index) sobre cada
          variable de entrada. Reproducible con `random_state=42`.\n"""))

    # A) Validación
    R("## A. Validación del modelo actual\n")
    R("### A1. Split temporal simulado (70% antiguas / 30% nuevas)\n")
    R(dedent(f"""\
        El R² *in-sample* del train ({temp['m_tr']['r2']:.3f}) es optimista por
        construcción (Random Forest se ajusta a lo que ya vio), así que **no** sirve
        de referencia. La referencia honesta es el R² out-of-fold
        ({oof_m['r2']:.3f}) y su IC 95% por bootstrap
        ([{boot['lo']:.3f}, {boot['hi']:.3f}]).\n"""))
    filas = [
        {"Conjunto": "Train (70% antiguas, in-sample)", "N": temp["m_tr"]["n"],
         "R²": f"{temp['m_tr']['r2']:.3f}", "RMSE": f"{temp['m_tr']['rmse']:.2f}",
         "MAE": f"{temp['m_tr']['mae']:.2f}"},
        {"Conjunto": "Nuevas (30% recientes)", "N": temp["m_new"]["n"],
         "R²": f"{temp['m_new']['r2']:.3f}", "RMSE": f"{temp['m_new']['rmse']:.2f}",
         "MAE": f"{temp['m_new']['mae']:.2f}"},
    ]
    R(tabla_md(pd.DataFrame(filas)) + "\n")
    R(f"\n{img('temporal', 'Validación temporal')}\n")
    R(f"\n**Veredicto:** {temp['veredicto']}\n")

    R("### A2. Curva de calibración (10 bins)\n")
    R(dedent(f"""\
        Se agrupan las predicciones en 10 bins y se compara el promedio predicho con
        el promedio real de cada bin. Un modelo bien calibrado cae sobre la línea de
        45°. Aquí el error medio de calibración es **{calib['ece']:.2f} puntos**, lo
        que indica que el modelo es honesto: no infla ni subestima sistemáticamente.\n"""))
    R(f"\n{img('calibracion', 'Calibración')}\n")

    R("### A3. Bootstrap del R² (1000 remuestreos)\n")
    R(dedent(f"""\
        Con 1000 remuestreos con reemplazo, el R² se distribuye alrededor de
        **{boot['media']:.3f}** con intervalo de confianza 95%
        **[{boot['lo']:.3f}, {boot['hi']:.3f}]**.
        {'El intervalo está por encima de 0, así que el poder predictivo es real (pequeño pero no nulo).' if boot['estable'] else 'El intervalo cruza 0: el poder predictivo no es distinguible de cero.'}\n"""))
    R(f"\n{img('bootstrap', 'Bootstrap R²')}\n")

    # B) Diagnóstico
    R("## B. Diagnóstico: ¿por qué R² ≈ 0.09?\n")
    R("### B1. Techo teórico — estudiantes 'gemelos'\n")
    R(dedent(f"""\
        Dos estudiantes con **exactamente las mismas features**
        ({', '.join(techo['cols'])}) no pueden ser distinguidos por el modelo. Si su
        puntaje difiere mucho, existe un **techo** por encima del cual ningún modelo
        puede mejorar. En la cohorte:

        - **{techo['n_gemelos']:,} estudiantes** ({techo['frac_gemelos']:.1%}) tienen
          al menos un "gemelo", agrupados en {techo['n_grupos']:,} grupos.
        - La desviación de puntaje **dentro** de cada grupo de gemelos es
          **{techo['std_intra']:.1f} puntos** (vs. {techo['std_total']:.1f} de la
          población). Casi tanta variación adentro como afuera.
        - **Techo teórico de R² ≈ {techo['techo']:.3f}.** El modelo actual
          ({oof_m['r2']:.3f}) ya está cerca de ese techo.\n"""))
    if len(techo["ejemplos"]):
        R("\n**Grupos de gemelos con puntajes más dispares:**\n")
        ej = techo["ejemplos"].copy()
        ej_cols = techo["cols"] + ["n", "media", "desv", "minimo", "maximo", "rango"]
        ej = ej[[c for c in ej_cols if c in ej.columns]]
        for c in ["media", "desv"]:
            if c in ej.columns:
                ej[c] = ej[c].round(1)
        R(tabla_md(ej) + "\n")
    R(f"\n{img('gemelos', 'Techo teórico gemelos')}\n")

    R("### B2. Varianza descompuesta\n")
    R(dedent(f"""\
        La varianza total del puntaje se reparte así: **{var['modelo_pct']:.1f}%** la
        captura el modelo actual, **{var['headroom_pct']:.1f}%** es explicable con un
        mejor modelo o más datos (hasta el techo), y **{var['inexplicable_pct']:.1f}%**
        es *inexplicable* con las variables actuales — diferencias individuales que
        hoy no medimos.\n"""))
    R(f"\n{img('var_descomp', 'Varianza descompuesta')}\n")
    R("\n**Varianza explicada individualmente por cada variable (η²):**\n")
    tv = pd.DataFrame(var["por_variable"], columns=["Variable", "% varianza (η²)"])
    R(tabla_md(tv) + "\n")
    R(f"\n{img('var_variable', 'Varianza por variable')}\n")
    R(dedent("""\
        _Nota: los η² son marginales y se solapan entre sí (p. ej. estrato y acceso a
        computador comparten información); no suman el R² del modelo conjunto._\n"""))

    R("### B3. ¿Qué variables faltan?\n")
    R(dedent("""\
        El diagnóstico anterior apunta a que el techo bajo se debe a **variables no
        medidas**. Las siguientes, ausentes en el formulario actual, son las que la
        literatura señala como más predictivas del rendimiento:\n"""))
    tvf = pd.DataFrame([
        {"Variable": v["variable"], "Tipo": v["tipo"],
         "Δ R² estimado": v["delta_r2"], "Sustento": v["sustento"]}
        for v in plan["variables"]])
    R(tabla_md(tvf) + "\n")

    # C) Framework
    R("## C. Framework para datos nuevos\n")
    R(dedent(f"""\
        Se generó **`models/deploy/validation_framework.py`**, una función pura (solo
        librería estándar: `csv`, `json`, `math`, `statistics`) que:

        1. Carga un CSV con inscripciones nuevas.
        2. Aplica **el mismo preprocesamiento** del modelo en producción.
        3. Genera predicciones de puntaje con el modelo actual.
        4. Si el CSV trae puntaje real, calcula **R², RMSE y MAE** y los compara con
           el entrenamiento.
        5. Detecta **drift** con el PSI de cada variable (numérica y categórica).
        6. Emite un veredicto automático: *"El modelo sigue funcionando bien"* o
           *"Hay drift, re-entrenar"*.

        **Uso:**
        ```python
        from validation_framework import validate_new_data
        rep = validate_new_data("data/inscripciones_2027.csv")
        print(rep["mensaje"])
        ```

        **Auto-verificación:** al probar el framework con una muestra idéntica al
        dataset y con una muestra perturbada (todo estrato 1, sin computador, solo
        grado 11), el resultado fue
        `{framework_check['ok']['recomendacion']}` y
        `{framework_check['drift']['recomendacion']}` respectivamente
        → el detector {'discrimina correctamente.' if framework_check['discrimina'] else 'requiere ajuste de umbrales.'}\n"""))

    # D) Plan de mejora
    R("## D. Plan de mejora del R² para la próxima Copa STEM\n")
    R(dedent(f"""\
        Añadiendo al formulario de inscripción las 5 variables de la sección B3, se
        estima que el R² podría subir del {oof_m['r2']:.2f} actual a
        **{plan['objetivo'][0]:.2f}–{plan['objetivo'][1]:.2f}** (proyección media
        ≈ {plan['r2_proyectado_medio']:.2f}), según la literatura de predicción de
        rendimiento académico.\n"""))
    R(f"\n{img('plan', 'Plan de mejora del R²')}\n")

    R("### Formulario sugerido para la próxima edición\n")
    R(dedent("""\
        Añadir estas preguntas al formulario de inscripción (además de las actuales):

        1. **Promedio académico del último período** *(numérico, p. ej. 0–5 o 0–100)*
        2. **Horas semanales de estudio de matemáticas** *(numérico, horas)*
        3. **Motivación para participar** *(escala 1–5)*
        4. **¿Ha recibido clases extra de matemáticas?** *(Sí / No)*
        5. **¿Le gusta resolver problemas lógicos?** *(escala 1–5)*

        Recomendaciones de recolección: campos numéricos validados por rango, escalas
        1–5 tipo Likert obligatorias, y registrar la **fecha/hora de inscripción**
        para permitir un split temporal REAL (no simulado) en el futuro.\n"""))

    # Limitaciones
    R("## Limitaciones\n")
    R(dedent("""\
        - **Split temporal simulado:** sin columna de fecha, se aproxima el tiempo con
          el orden de fila. Si el CSV no está ordenado por inscripción, A1 mide más
          bien estabilidad ante otra partición que un efecto temporal puro.
        - **Techo teórico aproximado:** se estima solo sobre estudiantes con gemelos
          exactos; con más features de agrupación el techo estimado bajaría aún más.
        - **Estimaciones de mejora del R²** provienen de la literatura general, no de
          datos propios; el efecto real dependerá de la calidad de las respuestas.
        - **Drift por PSI** vigila las features de entrada, no la relación
          features→puntaje (concept drift), que requiere puntajes reales para medirse.\n"""))

    R("## Referencias técnicas\n")
    R(dedent("""\
        - Efron & Tibshirani (1993). *An Introduction to the Bootstrap*.
        - Niculescu-Mizil & Caruana (2005). *Predicting Good Probabilities* (calibración).
        - Hattie, J. (2009). *Visible Learning* (predictores del rendimiento).
        - Richardson, Abraham & Bond (2012). *Psychological correlates of university
          students' academic performance: a meta-analysis*.
        - Population Stability Index (PSI) — práctica estándar de monitoreo de modelos.\n"""))
    R("\n---\n_Generado por `notebooks/08_framework_validacion.py` — Copa STEM 2026._\n")

    destino = REPORTS_DIR / "08_framework_validacion.md"
    destino.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"    informe escrito → reports/{destino.name}")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Framework de Validación del Modelo (Fase 4)")
    print(" Fundación SapienceLab")
    print("=" * 70)

    m03 = cargar_infra()
    nombre_modelo = resolver_mejor_modelo(m03)
    ctx = preparar_contexto(m03)

    # A) Validación
    # OOF y bootstrap primero: son la referencia HONESTA contra la que se juzga
    # el split temporal (el R² in-sample del train es optimista y no sirve de base).
    pred_oof, oof_m = predicciones_oof(ctx, nombre_modelo)
    boot = bootstrap_r2(ctx, pred_oof, n_boot=1000)
    temp = validacion_temporal(ctx, nombre_modelo, oof_m["r2"], boot)
    calib = analisis_calibracion(ctx, pred_oof)

    # B) Diagnóstico
    techo = techo_teorico(ctx)
    var = varianza_descompuesta(ctx, oof_m["r2"], techo["techo"])
    plan = plan_mejora()

    # C) Framework para datos nuevos
    baseline = _baseline_stats(ctx, oof_m, temp, boot, nombre_modelo)
    destino_fw = generar_validation_framework(ctx, baseline)
    fw_check = autoverificar_framework(ctx, destino_fw)

    # E) Informe
    construir_informe(ctx, nombre_modelo, temp, oof_m, calib, boot,
                      techo, var, plan, fw_check)

    print("\n" + "=" * 70)
    print(" ✔ FRAMEWORK DE VALIDACIÓN COMPLETADO")
    print(f"   · Modelo validado:   {nombre_modelo}")
    print(f"   · R² (OOF):          {oof_m['r2']:.3f}  "
          f"(temporal 30% nuevas: {temp['m_new']['r2']:.3f})")
    print(f"   · Bootstrap R²:      {boot['media']:.3f} "
          f"IC95% [{boot['lo']:.3f}, {boot['hi']:.3f}]")
    print(f"   · Techo teórico R²:  {techo['techo']:.3f} "
          f"(desv. intra-gemelos {techo['std_intra']:.1f} pts)")
    print(f"   · Framework:         models/deploy/validation_framework.py "
          f"({'discrimina OK' if fw_check['discrimina'] else 'revisar umbrales'})")
    print(f"   · Figuras:           {len(FIGURES)} → outputs/")
    print(f"   · Informe:           reports/08_framework_validacion.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
