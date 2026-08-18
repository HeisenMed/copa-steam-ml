# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 09b: Informe y gráficos del Puntaje Estimado  (Fase 4 — Despliegue)
================================================================================

Objetivo
--------
El script 09 exportó `models/deploy/puntaje_estimado.csv` pero sin gráficos ni
informe. Este script los produce:

    Gráficos (outputs/, prefijo F09_):
      - F09_scatter_real_vs_estimado.png   (real vs estimado, línea 45°, R²/MAE)
      - F09_distribucion_diferencia.png     (histograma de la diferencia)
      - F09_diferencia_por_colegio.png      (boxplot por institución)
      - F09_diferencia_por_estrato.png      (boxplot por estrato)
      - F09_top_resilientes.png             (top 20 mayor diferencia positiva)

    Informe:
      - reports/09_puntaje_estimado.md  (pedagógico: qué es, cómo se calcula,
        qué tan preciso es, hallazgos, glosario)

Nota metodológica sobre el R²
-----------------------------
El `puntaje_estimado` del CSV proviene del Random Forest de producción, que se
entrenó con el 80% de estos mismos estudiantes. Por eso el R² calculado sobre el
CSV es **in-sample** (optimista). Para la sección "¿qué tan preciso es?" se
reporta también el R² **out-of-fold** (validación cruzada 5-fold), que es la
medida HONESTA de cuán bien predice el modelo a un estudiante nuevo. Ambos se
muestran y se explica la diferencia (es el mismo hallazgo del script 08).

Reproducible: `random_state=42`.
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
    import seaborn as sns

    from sklearn.base import clone
    from sklearn.model_selection import KFold, cross_val_predict
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
for _d in (OUTPUTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CSV_ESTIMADO = DEPLOY_DIR / "puntaje_estimado.csv"
DATASET = DATA_DIR / "copa_stem_dataset_limpio.csv"

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
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.autolayout": True,
})
DPI = 150
UMBRAL = 5.0  # ±5 pts = "dentro de lo esperado"

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
# CARGA Y MERGE
# =============================================================================

def cargar() -> tuple[pd.DataFrame, object]:
    log("Paso 1 — Carga del CSV de estimados y del dataset")
    if not CSV_ESTIMADO.exists():
        print(f"\n  ⚠  No se encontró {CSV_ESTIMADO}. Ejecute antes el script 09.\n")
        sys.exit(1)
    est = pd.read_csv(CSV_ESTIMADO, dtype={"numero_documento": str})

    m03 = _import_por_ruta("modelo03", BASE_DIR / "notebooks" / "03_modelo_predictivo.py")
    df = m03.cargar_y_limpiar()
    df = df.drop_duplicates(subset="numero_documento", keep="first").reset_index(drop=True)
    df["numero_documento"] = df["numero_documento"].astype(str)

    cols_meta = ["numero_documento", "nombres", "apellidos", "institucion_educativa",
                 "estrato", "grado_escolar", "tiempo_usado_segundos"]
    cols_meta = [c for c in cols_meta if c in df.columns]
    merged = est.merge(df[cols_meta], on="numero_documento", how="left")
    merged["nombre_completo"] = (
        merged.get("nombres", "").fillna("").astype(str).str.strip() + " "
        + merged.get("apellidos", "").fillna("").astype(str).str.strip()).str.strip()
    log(f"    estudiantes: {len(merged):,} (con puntaje real: "
        f"{merged['puntaje_real'].notna().sum():,})")
    return merged, m03


# =============================================================================
# MÉTRICAS: in-sample (del CSV) y out-of-fold (honesta)
# =============================================================================

def metricas(merged: pd.DataFrame, m03) -> dict:
    log("Paso 2 — Métricas in-sample y out-of-fold")
    pres = merged[merged["puntaje_real"].notna()].copy()
    y = pres["puntaje_real"].to_numpy(float)
    yhat = pres["puntaje_estimado"].to_numpy(float)

    in_r2 = float(r2_score(y, yhat))
    in_mae = float(mean_absolute_error(y, yhat))
    in_rmse = float(np.sqrt(mean_squared_error(y, yhat)))

    # OOF: reentrenar RF con validación cruzada sobre TODOS los que presentaron.
    df = m03.cargar_y_limpiar()
    df = df.drop_duplicates(subset="numero_documento", keep="first").reset_index(drop=True)
    modelo_df = df[df[m03.TARGET].notna()].reset_index(drop=True)
    PRE = m03.fit_preprocessor(modelo_df.to_dict("records"), modelo_df)
    X = m03.build_matrix(modelo_df.to_dict("records"), PRE)
    yfull = modelo_df[m03.TARGET].to_numpy(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf = clone(m03.construir_modelos()["Random Forest"])
    oof = np.clip(cross_val_predict(rf, X, yfull, cv=cv, n_jobs=-1), 0, 100)
    oof_r2 = float(r2_score(yfull, oof))
    oof_mae = float(mean_absolute_error(yfull, oof))
    oof_rmse = float(np.sqrt(mean_squared_error(yfull, oof)))

    log(f"    in-sample:   R²={in_r2:.3f} MAE={in_mae:.2f} RMSE={in_rmse:.2f}")
    log(f"    out-of-fold: R²={oof_r2:.3f} MAE={oof_mae:.2f} RMSE={oof_rmse:.2f}")
    return {"in_r2": in_r2, "in_mae": in_mae, "in_rmse": in_rmse,
            "oof_r2": oof_r2, "oof_mae": oof_mae, "oof_rmse": oof_rmse,
            "n_pres": int(len(pres))}


# =============================================================================
# GRÁFICOS
# =============================================================================

def _cat_diff(d: float) -> str:
    if d > UMBRAL:
        return "superó"
    if d < -UMBRAL:
        return "por debajo"
    return "dentro"


def graficos(merged: pd.DataFrame, met: dict) -> dict:
    log("Paso 3 — Generación de gráficos")
    pres = merged[merged["puntaje_real"].notna()].copy()
    pres["cat"] = pres["diferencia"].apply(_cat_diff)

    # --- F09_scatter_real_vs_estimado ---
    fig, ax = plt.subplots(figsize=(7.6, 7))
    cmap_cols = pres["diferencia"].clip(-30, 30)
    sc = ax.scatter(pres["puntaje_real"], pres["puntaje_estimado"],
                    c=cmap_cols, cmap="RdYlGn", s=22, alpha=0.7,
                    edgecolor=COLORS["dark"], linewidth=0.2, vmin=-30, vmax=30)
    lim = [0, 100]
    ax.plot(lim, lim, "--", color=COLORS["dark"], linewidth=1.6,
            label="Predicción perfecta (45°)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Puntaje real")
    ax.set_ylabel("Puntaje estimado (Random Forest)")
    ax.set_title(f"Real vs. estimado (in-sample)\n"
                 f"R² = {met['in_r2']:.3f} · MAE = {met['in_mae']:.1f} pts")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Diferencia (real − estimado)")
    ax.legend(loc="upper left")
    savefig(fig, "F09_scatter_real_vs_estimado.png", "scatter")

    # --- F09_distribucion_diferencia ---
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    dif = pres["diferencia"]
    n, bins, patches = ax.hist(dif, bins=40, edgecolor="white")
    for p, b in zip(patches, bins[:-1]):
        centro = b + (bins[1] - bins[0]) / 2
        if centro > UMBRAL:
            p.set_facecolor(COLORS["green"])
        elif centro < -UMBRAL:
            p.set_facecolor(COLORS["red"])
        else:
            p.set_facecolor("#b0b0b0")
    ax.axvline(0, color=COLORS["dark"], linewidth=1.2)
    ax.axvline(float(dif.mean()), color=COLORS["violet"], linewidth=2,
               label=f"Media = {dif.mean():+.2f}")
    ax.axvspan(-UMBRAL, UMBRAL, color="#b0b0b0", alpha=0.15)
    from matplotlib.patches import Patch
    leyenda = [
        Patch(facecolor=COLORS["green"], label=f"Superó (> +{int(UMBRAL)})"),
        Patch(facecolor="#b0b0b0", label=f"Dentro (±{int(UMBRAL)})"),
        Patch(facecolor=COLORS["red"], label=f"Por debajo (< -{int(UMBRAL)})"),
        Patch(facecolor=COLORS["violet"], label=f"Media = {dif.mean():+.2f}"),
    ]
    ax.legend(handles=leyenda)
    ax.set_title("Distribución de la diferencia (real − estimado)")
    ax.set_xlabel("puntaje_real − puntaje_estimado")
    ax.set_ylabel("N estudiantes")
    savefig(fig, "F09_distribucion_diferencia.png", "distribucion")

    # --- F09_diferencia_por_colegio ---
    por_col = {}
    if "institucion_educativa" in pres.columns:
        cnt = pres.groupby("institucion_educativa").size()
        grandes = cnt[cnt >= 20].index
        sub = pres[pres["institucion_educativa"].isin(grandes)].copy()
        orden = (sub.groupby("institucion_educativa")["diferencia"].median()
                 .sort_values().index.tolist())
        por_col = (sub.groupby("institucion_educativa")["diferencia"].median()
                   .round(1).sort_values().to_dict())
        fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(orden))))
        colores = [STEM(i / max(1, len(orden) - 1)) for i in range(len(orden))]
        sns.boxplot(data=sub, y="institucion_educativa", x="diferencia",
                    order=orden, palette=colores, ax=ax, fliersize=2)
        ax.axvline(0, color=COLORS["dark"], linestyle="--", linewidth=1.4)
        ax.set_title("Diferencia (real − estimado) por institución\n"
                     "(a la derecha = rinden por encima de lo estimado)")
        ax.set_xlabel("Diferencia (real − estimado)")
        ax.set_ylabel("")
        savefig(fig, "F09_diferencia_por_colegio.png", "colegio")

    # --- F09_diferencia_por_estrato ---
    por_est = {}
    if "estrato" in pres.columns:
        sub = pres.dropna(subset=["estrato"]).copy()
        sub["estrato"] = sub["estrato"].astype(float).astype(int)
        cnt = sub.groupby("estrato").size()
        estratos_ok = cnt[cnt >= 5].index
        sub = sub[sub["estrato"].isin(estratos_ok)]
        orden = sorted(sub["estrato"].unique())
        por_est = (sub.groupby("estrato")["diferencia"].median()
                   .round(1).to_dict())
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        sns.boxplot(data=sub, x="estrato", y="diferencia", order=orden,
                    palette=PALETTE, ax=ax, fliersize=2)
        ax.axhline(0, color=COLORS["dark"], linestyle="--", linewidth=1.4)
        ax.set_title("Diferencia (real − estimado) por estrato\n"
                     "(¿los de estrato bajo superan más las expectativas?)")
        ax.set_xlabel("Estrato socioeconómico")
        ax.set_ylabel("Diferencia (real − estimado)")
        savefig(fig, "F09_diferencia_por_estrato.png", "estrato")

    # --- F09_top_resilientes ---
    top = pres.sort_values("diferencia", ascending=False).head(20).copy()
    etiquetas = []
    for _, r in top.iterrows():
        nom = (r.get("nombre_completo") or "").strip()
        etiquetas.append((nom[:26] if nom else f"doc {r['numero_documento'][-4:]}"))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    colores = [STEM(i / 19) for i in range(len(top))]
    ax.barh(range(len(top)), top["diferencia"].values[::-1],
            color=colores[::-1], edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(etiquetas[::-1], fontsize=8)
    for i, (d, r) in enumerate(zip(top["diferencia"].values[::-1],
                                   top.iloc[::-1].itertuples())):
        ax.text(d + 0.3, i, f"+{d:.0f}  ({r.puntaje_real:.0f} vs {r.puntaje_estimado:.0f})",
                va="center", fontsize=7)
    ax.set_title("Top 20 estudiantes más resilientes\n"
                 "(mayor diferencia positiva: rindieron muy por encima de lo estimado)")
    ax.set_xlabel("Diferencia (real − estimado)")
    savefig(fig, "F09_top_resilientes.png", "top")

    return {"por_col": por_col, "por_est": por_est, "top": top,
            "cat_counts": pres["cat"].value_counts().to_dict(),
            "mean_dif": float(pres["diferencia"].mean())}


# STEM gradient para barras/boxes.
from matplotlib.colors import LinearSegmentedColormap
STEM = LinearSegmentedColormap.from_list("stem_grad", PALETTE)


# =============================================================================
# INFORME
# =============================================================================

def construir_informe(merged, met, g) -> None:
    log("Paso 4 — Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    pres = merged[merged["puntaje_real"].notna()]
    n_pres = len(pres)
    sup = g["cat_counts"].get("superó", 0)
    den = g["cat_counts"].get("dentro", 0)
    deb = g["cat_counts"].get("por debajo", 0)
    lo = met["oof_mae"]

    R = REPORT.append
    R("# Puntaje Estimado vs Real — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Fase 4 · Informe generado: {fecha}\n")
    R("---\n")

    R("## Resumen ejecutivo\n")
    R(dedent(f"""\
        Se generó, para cada uno de los **{n_pres:,} estudiantes** que presentaron,
        un `puntaje_estimado` con el modelo Random Forest (script 03): lo que el
        modelo *esperaba* que sacara dado su perfil socioeconómico y académico. La
        **diferencia** (`real − estimado`) mide resiliencia cruda: **{sup:,}
        estudiantes ({sup/n_pres:.0%}) superaron su expectativa** (> +{int(UMBRAL)}
        pts), **{den:,} ({den/n_pres:.0%}) quedaron dentro de lo esperado**
        (±{int(UMBRAL)}) y **{deb:,} ({deb/n_pres:.0%}) por debajo**. La diferencia
        media es **{g['mean_dif']:+.2f}**, señal de que el modelo está bien
        calibrado (sin sesgo sistemático). El modelo NO es muy preciso a nivel
        individual (R² out-of-fold = {met['oof_r2']:.3f}, MAE = {met['oof_mae']:.1f}
        pts): las variables de un formulario explican solo una parte pequeña del
        rendimiento. Por eso la diferencia se debe leer por tramos amplios, nunca
        como un juicio exacto sobre un estudiante.\n"""))

    R("## ¿Qué es el puntaje estimado?\n")
    R(dedent("""\
        Es lo que el modelo Random Forest **predice** que un estudiante sacará, a
        partir de su perfil (estrato, acceso a computador/internet, nivel de
        programación/robótica, experiencia previa, etc.). **No es una nota real: es
        una EXPECTATIVA estadística.**

        > Imagina dos estudiantes con exactamente el mismo perfil (mismo estrato,
        > mismo colegio, mismo grado, mismos recursos). El modelo dice: *"estudiantes
        > con este perfil típicamente sacan alrededor de 45 puntos"*. Si uno de ellos
        > saca 70, hay algo especial en ese estudiante —motivación, talento natural,
        > un buen profesor— que el modelo no puede medir pero que nosotros sí podemos
        > **detectar** mirando la diferencia.\n"""))

    R("## ¿Cómo se calcula?\n")
    R(dedent(f"""\
        El modelo es un **Random Forest** ("bosque aleatorio"), que funciona así:

        - Aprende **reglas** a partir de los datos, como un árbol de decisiones
          gigante. Una regla se lee: *"si el estudiante es de Girardota **y** tiene
          computador **y** está en grado 11 → estimar 52 puntos"*.
        - El modelo tiene **cientos de estos árboles** (aquí, 300), cada uno con
          reglas ligeramente distintas, y **promedia** sus respuestas. Promediar
          muchos árboles imperfectos da una predicción más estable que un solo árbol.
        - Todas las reglas se aprendieron de los **{n_pres:,} estudiantes** que ya
          presentaron el examen.\n"""))

    R("## ¿Qué tan preciso es?\n")
    R(dedent(f"""\
        Hay que distinguir dos números:

        | Métrica | Sobre los datos de entrenamiento (in-sample) | Con validación cruzada (honesta) |
        | --- | --- | --- |
        | R² | {met['in_r2']:.3f} | **{met['oof_r2']:.3f}** |
        | MAE | {met['in_mae']:.1f} pts | **{met['oof_mae']:.1f} pts** |
        | RMSE | {met['in_rmse']:.1f} pts | **{met['oof_rmse']:.1f} pts** |

        La columna izquierda mide el modelo sobre estudiantes que **ya vio** al
        entrenarse: siempre se ve mejor de lo que es (como un examen con las
        respuestas a la vista). La columna derecha lo mide sobre estudiantes que
        **no vio** (validación cruzada); ese es el número honesto.

        - **R² = {met['oof_r2']:.3f}** significa que el perfil socioeconómico explica
          alrededor del **{met['oof_r2']*100:.0f}%** de por qué unos sacan más que
          otros. El {100-met['oof_r2']*100:.0f}% restante depende de cosas que el
          formulario no captura.
        - **MAE = {met['oof_mae']:.1f} puntos** significa que, en promedio, *el modelo
          se equivoca por unos {met['oof_mae']:.0f} puntos*. Si el modelo dice 45, el
          estudiante realista puede sacar entre **{45-met['oof_rmse']:.0f} y
          {45+met['oof_rmse']:.0f}** (aprox. ± un RMSE).
        - **¿Por qué no es más preciso?** Porque las variables socioeconómicas solo
          explican una parte pequeña del rendimiento. La motivación, la preparación,
          la calidad del profesor, el talento natural, cómo durmió esa noche — nada
          de eso está en el formulario, pero pesa muchísimo en la nota.\n"""))
    R(f"\n{img('scatter', 'Real vs estimado')}\n")

    R("## La diferencia: ¿superaste las expectativas?\n")
    R(dedent(f"""\
        `diferencia = puntaje_real − puntaje_estimado`

        - **Positiva** → el estudiante rindió **mejor** de lo que su contexto sugería.
        - **Negativa** → rindió **por debajo** de su potencial estimado.
        - **Cerca de 0** → dentro de lo esperado.

        En la cohorte: **{sup:,} superaron** (> +{int(UMBRAL)}), **{den:,} dentro**
        (±{int(UMBRAL)}) y **{deb:,} por debajo** (< -{int(UMBRAL)}). La diferencia
        media es **{g['mean_dif']:+.2f}** — cercana a 0, como se espera de un modelo
        bien calibrado (no infla ni subestima de forma sistemática).\n"""))
    R(f"\n{img('distribucion', 'Distribución de la diferencia')}\n")

    R("## Hallazgos por grupo\n")
    if g["por_col"]:
        peor = min(g["por_col"], key=g["por_col"].get)
        mejor = max(g["por_col"], key=g["por_col"].get)
        R(dedent(f"""\
            **Por institución** (mediana de la diferencia): el colegio donde los
            estudiantes rinden más **por encima** de lo estimado es *{mejor}*
            ({g['por_col'][mejor]:+.1f}); el que más **por debajo** es *{peor}*
            ({g['por_col'][peor]:+.1f}). Diferencias sistemáticas por colegio pueden
            reflejar calidad docente, ambiente o preparación específica.\n"""))
    R(f"\n{img('colegio', 'Diferencia por colegio')}\n")
    if g["por_est"]:
        detalle = "; ".join(f"estrato {k}: {v:+.1f}" for k, v in sorted(g["por_est"].items()))
        R(dedent(f"""\
            **Por estrato** (mediana de la diferencia): {detalle}. Si los estratos
            más bajos muestran diferencias positivas, es evidencia de **resiliencia
            académica**: rinden por encima de lo que su contexto material predeciría.\n"""))
    R(f"\n{img('estrato', 'Diferencia por estrato')}\n")

    R("## Los 20 más resilientes\n")
    R(dedent("""\
        Estudiantes con la mayor diferencia positiva: sacaron muchísimo más de lo
        que su perfil sugería. Son candidatos a **talento oculto** (ver script 06).\n"""))
    top = g["top"]
    cols_show = [c for c in ["nombre_completo", "institucion_educativa",
                             "puntaje_estimado", "puntaje_real", "diferencia"]
                 if c in top.columns]
    tt = top[cols_show].copy()
    tt = tt.rename(columns={"nombre_completo": "Estudiante",
                            "institucion_educativa": "Institución",
                            "puntaje_estimado": "Estimado",
                            "puntaje_real": "Real", "diferencia": "Diferencia"})
    R(tabla_md(tt) + "\n")
    R(f"\n{img('top', 'Top resilientes')}\n")

    R("## ¿Para qué sirve este análisis?\n")
    R(dedent("""\
        - **Para la Fundación:** identificar a quién apoyar. Un estudiante con
          diferencia muy negativa puede estar desmotivado o necesitar ayuda concreta.
        - **Para los colegios:** si *todos* los estudiantes de un colegio superan las
          expectativas, ese colegio tiene algo especial (buenos profesores, buen
          ambiente) que vale la pena estudiar y replicar.
        - **Para los estudiantes:** abre una conversación, no un veredicto —
          *"sacaste 35 pero esperábamos 45: ¿qué pasó?, ¿cómo te ayudamos?"*.\n"""))

    R("## Limitaciones\n")
    R(dedent(f"""\
        - El modelo tiene **R² bajo** ({met['oof_r2']:.2f}): las predicciones
          individuales son imprecisas (± ~{met['oof_rmse']:.0f} pts). Úsese por
          grupos, no como etiqueta individual.
        - La diferencia **no es un juicio de valor**: un estudiante "por debajo" pudo
          tener un mal día, estar enfermo o nervioso.
        - **Faltan variables** decisivas: motivación, horas de estudio, calidad
          docente, promedio académico previo (ver el plan de mejora del script 08).\n"""))

    R("## Glosario\n")
    R(dedent(f"""\
        - **R² (coeficiente de determinación):** *definición* — fracción de la
          varianza del puntaje que el modelo explica (0 = nada, 1 = todo).
          *Analogía* — qué porción del "misterio" de por qué unos sacan más resuelve
          el modelo. *Ejemplo Copa STEM* — R² = {met['oof_r2']:.2f} → explica el
          ~{met['oof_r2']*100:.0f}% del misterio; el resto es lo que no medimos.
        - **MAE (error absoluto medio):** *definición* — promedio de |real − estimado|.
          *Analogía* — cuántos puntos, en promedio, "le erra" el modelo. *Ejemplo* —
          MAE = {met['oof_mae']:.1f} → típicamente se equivoca por ~{met['oof_mae']:.0f}
          puntos.
        - **Random Forest (bosque aleatorio):** *definición* — modelo que promedia
          cientos de árboles de decisión entrenados sobre muestras distintas de los
          datos. *Analogía* — en vez de preguntarle a un solo experto, se pregunta a
          300 y se promedia; el consenso es más robusto. *Ejemplo* — aquí 300 árboles
          entrenados con {n_pres:,} estudiantes.
        - **Resiliencia académica:** *definición* — rendir por encima de lo que las
          condiciones socioeconómicas predicen. *Analogía* — "remar contra la
          corriente y aun así avanzar". *Ejemplo* — un estudiante de estrato 1 con
          estimado 40 que saca 75: diferencia +35.\n"""))
    R("\n---\n_Generado por `notebooks/09b_informe_puntaje_estimado.py` — Copa STEM 2026._\n")

    destino = REPORTS_DIR / "09_puntaje_estimado.md"
    destino.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"    informe escrito → reports/{destino.name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Informe del Puntaje Estimado (09b)")
    print("=" * 70)
    merged, m03 = cargar()
    met = metricas(merged, m03)
    g = graficos(merged, met)
    construir_informe(merged, met, g)
    print("\n" + "=" * 70)
    print(" ✔ INFORME 09b COMPLETADO")
    print(f"   · Figuras: {len(FIGURES)} → outputs/F09_*.png")
    print(f"   · Informe: reports/09_puntaje_estimado.md")
    print(f"   · R² honesto (OOF): {met['oof_r2']:.3f} | MAE: {met['oof_mae']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
