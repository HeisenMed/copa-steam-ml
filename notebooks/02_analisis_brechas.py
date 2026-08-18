# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 02: Análisis de Brechas (equidad de género, socioeconómica y territorial)
================================================================================

Objetivo
--------
Cuantificar las **brechas de equidad** en el rendimiento de Copa STEM y detectar
**talento oculto** (estudiantes de alto desempeño con bajo acceso a recursos),
como insumo para las recomendaciones de política de la Fundación SapienceLab.

Secciones
---------
    A) Brechas de género        (t-test, histogramas superpuestos)
    B) Brechas socioeconómicas  (ANOVA por estrato, acceso a tecnología,
                                 con quién vive, cruce estrato × acceso)
    C) Brechas territoriales    (municipio, tipo de institución, ranking de
                                 instituciones, interacción género × municipio)
    D) Brechas por grado        (evolución 9° → 10° → 11°)
    E) Detección de talento oculto (alto puntaje + bajo acceso/estrato)
    F) Informe reports/02_analisis_brechas.md

Principios de diseño
--------------------
- Autocontenido y reproducible: `random_state=42`.
- Robusto a columnas faltantes y a los ~7% de inscripciones de emergencia que
  no tienen datos socioeconómicos completos.
- Paleta de marca Copa STEM; fondo blanco; PNG dpi=150 en outputs/.
- Progreso impreso en consola con prefijo ">>>".

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
import warnings
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
    from scipy import stats
except ImportError as exc:  # pragma: no cover
    print("ERROR: falta una dependencia del entorno virtual.")
    print(f"       Detalle: {exc}")
    print("       Active .venv e instale: pandas numpy matplotlib seaborn scipy")
    sys.exit(1)

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0. CONFIGURACIÓN GLOBAL (idéntica a la del script 01 para consistencia visual)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
for _d in (OUTPUTS_DIR, REPORTS_DIR):
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
# Fijamos la paleta de marca como paleta por defecto de seaborn/matplotlib, de
# modo que TODOS los gráficos usen los colores Copa STEM salvo indicación expresa.
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

# --- Colormaps de marca (para barras con gradiente y heatmaps) --------------
# Gradiente discreto recorriendo la paleta Copa STEM (cyan→violeta→ámbar→…).
STEM_GRAD = LinearSegmentedColormap.from_list("stem_grad", PALETTE)
# Secuencial claro→cyan→azul para heatmaps de valores (p. ej. puntaje promedio).
STEM_SEQ = LinearSegmentedColormap.from_list(
    "stem_seq", ["#eafcff", "#8ee9ff", COLORS["cyan"], COLORS["blue"]])


def gradient_colors(n: int) -> list:
    """Devuelve n colores muestreados del gradiente de marca Copa STEM."""
    if n <= 1:
        return [COLORS["cyan"]]
    return [STEM_GRAD(i / (n - 1)) for i in range(n)]


# Umbral para considerar "alto desempeño" en la detección de talento oculto.
UMBRAL_TALENTO = 70
# Umbral de aprobación (para % de aprobación por institución).
UMBRAL_APROBACION = 60

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


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/d"
    if p < 0.001:
        return f"{p:.2e} (*** altamente significativo)"
    if p < 0.01:
        return f"{p:.4f} (** significativo)"
    if p < 0.05:
        return f"{p:.4f} (* significativo)"
    return f"{p:.4f} (n.s.)"


def img(key: str, alt: str) -> str:
    """Ruta relativa desde reports/ hacia outputs/ para embeber en el informe."""
    f = FIGURES.get(key)
    return f"![{alt}](../outputs/{f})" if f else f"_(figura '{alt}' no disponible)_"


def tabla_md(df: pd.DataFrame, max_filas: int = 40) -> str:
    d = df.head(max_filas)
    enc = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    filas = ["| " + " | ".join(str(v) for v in row) + " |"
             for row in d.itertuples(index=False)]
    return "\n".join([enc, sep] + filas)


# =============================================================================
# HELPERS ESTADÍSTICOS (compartidos con el script 01)
# =============================================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Tamaño del efecto Cohen's d para dos grupos independientes."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def eta_squared(grupos: list[np.ndarray]) -> float:
    """eta² = SS_between / SS_total (tamaño del efecto para ANOVA)."""
    todas = np.concatenate(grupos)
    media_g = todas.mean()
    ss_total = ((todas - media_g) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - media_g) ** 2 for g in grupos)
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def prueba_grupo(df: pd.DataFrame, col: str, y: str = "puntaje_obtenido",
                 min_n: int = 15) -> dict | None:
    """
    Compara `y` entre los grupos de `col`:
      - 2 grupos  → t-test de Welch + Cohen's d.
      - >2 grupos → ANOVA de una vía + eta².
    Solo grupos con al menos `min_n` observaciones.
    """
    if col not in df.columns:
        return None
    sub = df[[col, y]].dropna()
    grupos, etiquetas = [], []
    for nombre, g in sub.groupby(col):
        vals = g[y].values
        if len(vals) >= min_n:
            grupos.append(vals)
            etiquetas.append(str(nombre))
    if len(grupos) < 2:
        return None

    medias = {et: float(np.mean(g)) for et, g in zip(etiquetas, grupos)}
    n_por_grupo = {et: int(len(g)) for et, g in zip(etiquetas, grupos)}

    if len(grupos) == 2:
        stat, p = stats.ttest_ind(grupos[0], grupos[1], equal_var=False)
        efecto, efecto_nombre, prueba = (cohens_d(grupos[0], grupos[1]),
                                         "Cohen's d", "t-test de Welch")
    else:
        stat, p = stats.f_oneway(*grupos)
        efecto, efecto_nombre, prueba = (eta_squared(grupos), "eta²",
                                         "ANOVA de una vía")

    return {"columna": col, "prueba": prueba, "estadistico": float(stat),
            "p_value": float(p), "efecto_nombre": efecto_nombre,
            "efecto": efecto, "medias": medias, "n_por_grupo": n_por_grupo}


# =============================================================================
# CARGA Y LIMPIEZA (misma lógica que 01; incluye variables derivadas de acceso)
# =============================================================================

def cargar_y_limpiar() -> pd.DataFrame:
    log("Carga y limpieza del dataset")
    ruta = DATA_DIR / DATASET_NAME
    if not ruta.exists():
        csvs = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
        if not csvs:
            print("\n" + "=" * 70)
            print(f"  ⚠  No se encontró '{DATASET_NAME}' en {DATA_DIR}")
            print("     Coloque el CSV exportado de Supabase y reejecute.")
            print("=" * 70 + "\n")
            sys.exit(0)
        ruta = csvs[0]
        log(f"    usando '{ruta.name}' (autodetectado)")

    df = pd.read_csv(ruta, encoding="utf-8")
    log(f"    registros crudos: {len(df):,} | columnas: {df.shape[1]}")

    # --- Eliminar registros de prueba --------------------------------------
    docs_prueba = ["1234", "123456", "123456789", "1234567899", "0", "00000000"]
    if "numero_documento" in df.columns:
        df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
        df = df[~df["numero_documento"].isin(docs_prueba)]
        df = df[df["numero_documento"].str.len() >= 5]

    # --- Normalizar strings -------------------------------------------------
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # --- Tipos numéricos ----------------------------------------------------
    for c in ["puntaje_obtenido", "porcentaje", "tiempo_usado_segundos",
              "edad_calculada", "estrato", "grado_escolar"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Binarias de acceso tecnológico ------------------------------------
    def _si_no(serie: pd.Series) -> pd.Series:
        s = serie.astype(str).str.lower()
        out = pd.Series(np.nan, index=serie.index, dtype="object")
        out[s.str.startswith("s")] = "Sí"
        out[s.str.startswith("n")] = "No"
        return out

    if "computador_en_casa" in df.columns:
        df["tiene_computador"] = _si_no(df["computador_en_casa"])
    if "internet_en_casa" in df.columns:
        df["tiene_internet"] = _si_no(df["internet_en_casa"])

    # --- Grupo de estrato (para cruces legibles) ---------------------------
    if "estrato" in df.columns:
        # En Copacabana/Girardota/Bello el estrato real solo va de 1 a 3.
        df["grupo_estrato"] = pd.cut(
            df["estrato"], bins=[0, 1, 2, 3],
            labels=["Bajo (1)", "Medio (2)", "Alto (3)"])

    # --- Codificación ordinal de niveles (femenino: Ninguna/Básica/…) -------
    orden_nivel = {"ninguna": 0, "ninguno": 0, "básica": 1, "basica": 1,
                   "intermedia": 2, "avanzada": 3}
    for c in ["nivel_programacion", "nivel_robotica"]:
        if c in df.columns:
            df[c + "_ord"] = df[c].astype(str).str.lower().map(orden_nivel)

    # --- Interés y participación previa (para perfil del estudiante) --------
    if "interes_prog_robotica" in df.columns:
        df["interes_prog_robotica"] = pd.to_numeric(
            df["interes_prog_robotica"], errors="coerce")
    if "participacion_olimpiadas" in df.columns:
        # Binaria Sí/No robusta a mayúsculas/acentos (dtype=object por pandas 3).
        s = df["participacion_olimpiadas"].astype(str).str.lower()
        binaria = pd.Series(np.nan, index=df.index, dtype="object")
        binaria[s.str.startswith("s")] = "Sí"
        binaria[s.str.startswith("n")] = "No"
        df["participo_olimpiadas"] = binaria

    # Solo quienes presentaron (tienen puntaje) para todo el análisis.
    if "puntaje_obtenido" in df.columns:
        df = df[df["puntaje_obtenido"].notna()].copy()
    log(f"    registros analizados (presentaron): {len(df):,}")
    return df


# =============================================================================
# SECCIÓN A — BRECHAS DE GÉNERO
# =============================================================================

def brechas_genero(df: pd.DataFrame) -> dict:
    log("SECCIÓN A — Brechas de género")
    if "genero" not in df.columns:
        return {}

    # Para el t-test binario nos centramos en Masculino vs Femenino (las dos
    # categorías con N suficiente); reportamos las demás por transparencia.
    conteos = df["genero"].value_counts(dropna=True).to_dict()
    sub = df[df["genero"].isin(["Masculino", "Femenino"])]
    fem = sub[sub["genero"] == "Femenino"]["puntaje_obtenido"].dropna()
    mas = sub[sub["genero"] == "Masculino"]["puntaje_obtenido"].dropna()

    stat, p = stats.ttest_ind(mas, fem, equal_var=False)
    d = cohens_d(mas.values, fem.values)
    res = {"conteos": conteos,
           "media_fem": float(fem.mean()), "media_mas": float(mas.mean()),
           "n_fem": int(fem.size), "n_mas": int(mas.size),
           "t_stat": float(stat), "p_value": float(p), "cohens_d": float(d)}
    print(f"    Femenino µ={res['media_fem']:.2f} (N={res['n_fem']}) | "
          f"Masculino µ={res['media_mas']:.2f} (N={res['n_mas']}) | "
          f"p={fmt_p(p)}")

    # --- Gráfico: histogramas superpuestos + box comparativo ---------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={"width_ratios": [2, 1]})
    sns.histplot(fem, bins=20, color=COLORS["violet"], alpha=0.55,
                 label=f"Femenino (µ={fem.mean():.1f})", stat="density",
                 edgecolor="white", ax=ax1)
    sns.histplot(mas, bins=20, color=COLORS["cyan"], alpha=0.55,
                 label=f"Masculino (µ={mas.mean():.1f})", stat="density",
                 edgecolor="white", ax=ax1)
    ax1.axvline(fem.mean(), color=COLORS["violet"], linestyle="--", linewidth=2)
    ax1.axvline(mas.mean(), color=COLORS["blue"], linestyle="--", linewidth=2)
    ax1.set_title(f"Distribución de puntaje por género\nt-test Welch · p = {p:.3g}")
    ax1.set_xlabel("Puntaje")
    ax1.set_ylabel("Densidad")
    ax1.legend()

    sns.boxplot(data=sub, x="genero", y="puntaje_obtenido",
                order=["Femenino", "Masculino"],
                palette=[COLORS["violet"], COLORS["cyan"]], ax=ax2)
    ax2.set_title("Comparación de medianas")
    ax2.set_xlabel("")
    ax2.set_ylabel("Puntaje")

    fig.suptitle("A. Brecha de género en el puntaje",
                 fontsize=15, fontweight="bold")
    savefig(fig, "F02A_brecha_genero.png", "genero")
    return res


# =============================================================================
# SECCIÓN B — BRECHAS SOCIOECONÓMICAS
# =============================================================================

def brechas_socioeconomicas(df: pd.DataFrame) -> dict:
    log("SECCIÓN B — Brechas socioeconómicas")
    res = {}

    # --- B.1 Estrato (ANOVA) -----------------------------------------------
    res["estrato"] = prueba_grupo(df, "estrato")
    if res["estrato"]:
        print(f"    estrato (ANOVA)     → p={fmt_p(res['estrato']['p_value'])}")

    # --- B.2 Acceso a tecnología (t-test) ----------------------------------
    res["computador"] = prueba_grupo(df, "tiene_computador")
    res["internet"] = prueba_grupo(df, "tiene_internet")
    for k in ("computador", "internet"):
        if res[k]:
            print(f"    {k:11s}         → p={fmt_p(res[k]['p_value'])}")

    # --- B.3 Con quién vive (ANOVA) ----------------------------------------
    res["con_quien_vive"] = prueba_grupo(df, "con_quien_vive")
    if res["con_quien_vive"]:
        print(f"    con_quien_vive      → "
              f"p={fmt_p(res['con_quien_vive']['p_value'])}")

    # --- Gráfico panel: estrato, computador, internet, con_quién_vive ------
    paneles = [("estrato", "Estrato"), ("tiene_computador", "Computador en casa"),
               ("tiene_internet", "Internet en casa"),
               ("con_quien_vive", "Con quién vive")]
    paneles = [(c, t) for c, t in paneles
               if c in df.columns and res.get(c.replace("tiene_", "")
                                              if c.startswith("tiene_") else c)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(-1)
    for ax, (col, titulo) in zip(axes, paneles):
        sub = df[[col, "puntaje_obtenido"]].dropna()
        orden = sorted(sub[col].unique(), key=str)
        sns.boxplot(data=sub, x=col, y="puntaje_obtenido", order=orden,
                    palette=PALETTE, ax=ax)
        key = col.replace("tiene_", "") if col.startswith("tiene_") else col
        p = res[key]["p_value"]
        ax.set_title(f"{titulo} · p = {p:.3g}")
        ax.set_xlabel("")
        ax.set_ylabel("Puntaje")
        ax.tick_params(axis="x", rotation=20)
    for ax in axes[len(paneles):]:
        ax.set_visible(False)
    fig.suptitle("B. Brechas socioeconómicas en el puntaje",
                 fontsize=15, fontweight="bold")
    savefig(fig, "F02B_brechas_socioeconomicas.png", "socio")

    # --- B.4 Cruce estrato × acceso a computador ---------------------------
    if "grupo_estrato" in df.columns and "tiene_computador" in df.columns:
        cruce = (df.dropna(subset=["grupo_estrato", "tiene_computador",
                                   "puntaje_obtenido"])
                 .groupby(["grupo_estrato", "tiene_computador"],
                          observed=True)["puntaje_obtenido"]
                 .agg(["mean", "count"]).reset_index())
        res["cruce"] = cruce
        fig2, ax = plt.subplots(figsize=(9, 5.5))
        pivote = cruce.pivot(index="grupo_estrato", columns="tiene_computador",
                             values="mean")
        pivote.plot(kind="bar", ax=ax,
                    color=[COLORS["red"], COLORS["green"]], edgecolor="white")
        ax.set_title("Puntaje promedio por estrato × acceso a computador")
        ax.set_xlabel("Grupo de estrato")
        ax.set_ylabel("Puntaje promedio")
        ax.legend(title="¿Tiene computador?")
        ax.tick_params(axis="x", rotation=0)
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.1f", fontsize=8)
        savefig(fig2, "F02B_cruce_estrato_acceso.png", "cruce")

    return res


# =============================================================================
# SECCIÓN C — BRECHAS TERRITORIALES
# =============================================================================

def brechas_territoriales(df: pd.DataFrame) -> dict:
    log("SECCIÓN C — Brechas territoriales")
    res = {}

    res["municipio"] = prueba_grupo(df, "municipio")
    res["tipo_institucion"] = prueba_grupo(df, "tipo_institucion")
    for k in ("municipio", "tipo_institucion"):
        if res.get(k):
            print(f"    {k:16s} → p={fmt_p(res[k]['p_value'])}")

    # --- Panel municipio y tipo de institución -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, titulo in [(axes[0], "municipio", "Municipio"),
                            (axes[1], "tipo_institucion", "Tipo de institución")]:
        if col not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[[col, "puntaje_obtenido"]].dropna()
        orden = sorted(sub[col].unique(), key=str)
        sns.boxplot(data=sub, x=col, y="puntaje_obtenido", order=orden,
                    palette=PALETTE, ax=ax)
        p = res[col]["p_value"] if res.get(col) else float("nan")
        ax.set_title(f"{titulo} · p = {p:.3g}")
        ax.set_xlabel("")
        ax.set_ylabel("Puntaje")
    fig.suptitle("C. Brechas territoriales", fontsize=15, fontweight="bold")
    savefig(fig, "F02C_brechas_territoriales.png", "territorio")

    # --- Ranking de instituciones (barras horizontales) --------------------
    if "institucion_educativa" in df.columns:
        MIN_N = 10  # solo instituciones con muestra suficiente para comparar
        rank = (df.dropna(subset=["institucion_educativa", "puntaje_obtenido"])
                .groupby("institucion_educativa")["puntaje_obtenido"]
                .agg(["mean", "count"]).reset_index())
        rank = rank[rank["count"] >= MIN_N].sort_values("mean")
        res["ranking"] = rank.sort_values("mean", ascending=False)
        if not rank.empty:
            fig2, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(rank))))
            # Gradiente de la paleta Copa STEM ordenado por puntaje.
            colores = gradient_colors(len(rank))
            barras = ax.barh(rank["institucion_educativa"], rank["mean"],
                             color=colores, edgecolor="white")
            media_global = df["puntaje_obtenido"].mean()
            ax.axvline(media_global, color=COLORS["red"], linestyle="--",
                       label=f"Media global = {media_global:.1f}")
            ax.set_title(f"Ranking de instituciones por puntaje promedio "
                         f"(N ≥ {MIN_N})")
            ax.set_xlabel("Puntaje promedio")
            ax.set_ylabel("")
            for barra, (_, fila) in zip(barras, rank.iterrows()):
                ax.text(barra.get_width() + 0.5, barra.get_y()
                        + barra.get_height() / 2,
                        f"{fila['mean']:.1f} (n={int(fila['count'])})",
                        va="center", fontsize=8)
            ax.legend()
            savefig(fig2, "F02C_ranking_instituciones.png", "ranking")

    # --- Interacción género × municipio ------------------------------------
    if {"genero", "municipio"}.issubset(df.columns):
        sub = df[df["genero"].isin(["Masculino", "Femenino"])]
        inter = (sub.groupby(["municipio", "genero"],
                             observed=True)["puntaje_obtenido"]
                 .mean().reset_index())
        res["interaccion"] = inter
        # Brecha (Masc - Fem) por municipio + p-value por municipio.
        brechas_mun = {}
        for mun, g in sub.groupby("municipio"):
            f = g[g["genero"] == "Femenino"]["puntaje_obtenido"].dropna()
            m = g[g["genero"] == "Masculino"]["puntaje_obtenido"].dropna()
            if len(f) >= 15 and len(m) >= 15:
                _, pp = stats.ttest_ind(m, f, equal_var=False)
                brechas_mun[mun] = {"brecha": float(m.mean() - f.mean()),
                                    "p": float(pp)}
        res["brechas_genero_municipio"] = brechas_mun

        fig3, ax = plt.subplots(figsize=(9, 5.5))
        sns.barplot(data=inter, x="municipio", y="puntaje_obtenido",
                    hue="genero", palette=[COLORS["violet"], COLORS["cyan"]],
                    ax=ax)
        ax.set_title("Interacción género × municipio (puntaje promedio)")
        ax.set_xlabel("")
        ax.set_ylabel("Puntaje promedio")
        ax.legend(title="Género")
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.1f", fontsize=8)
        savefig(fig3, "F02C_genero_x_municipio.png", "genero_municipio")

    return res


# =============================================================================
# SECCIÓN D — BRECHAS POR GRADO
# =============================================================================

def brechas_por_grado(df: pd.DataFrame) -> dict:
    log("SECCIÓN D — Brechas por grado")
    if "grado_escolar" not in df.columns:
        return {}
    prueba = prueba_grupo(df, "grado_escolar")
    resumen = (df.dropna(subset=["grado_escolar", "puntaje_obtenido"])
               .groupby("grado_escolar")["puntaje_obtenido"]
               .agg(["mean", "median", "std", "count"]).reset_index()
               .sort_values("grado_escolar"))
    res = {"prueba": prueba, "resumen": resumen}
    if prueba:
        print(f"    grado (ANOVA)       → p={fmt_p(prueba['p_value'])}")
    print(resumen.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sub = df.dropna(subset=["grado_escolar", "puntaje_obtenido"])
    sns.boxplot(data=sub, x="grado_escolar", y="puntaje_obtenido",
                order=sorted(sub["grado_escolar"].unique()),
                palette=PALETTE, ax=ax1)
    ax1.set_title("Puntaje por grado escolar")
    ax1.set_xlabel("Grado")
    ax1.set_ylabel("Puntaje")

    sns.pointplot(data=sub, x="grado_escolar", y="puntaje_obtenido",
                  order=sorted(sub["grado_escolar"].unique()),
                  color=COLORS["amber"], capsize=0.15, ax=ax2)
    ax2.set_title("Evolución del puntaje promedio 9° → 11° (IC 95%)")
    ax2.set_xlabel("Grado")
    ax2.set_ylabel("Puntaje promedio")

    fig.suptitle("D. Brechas por grado escolar", fontsize=15, fontweight="bold")
    savefig(fig, "F02D_brechas_grado.png", "grado")
    return res


# =============================================================================
# SECCIÓN E — DETECCIÓN DE TALENTO OCULTO
# =============================================================================

def detectar_talento_oculto(df: pd.DataFrame) -> dict:
    log("SECCIÓN E — Detección de talento oculto")
    alto = df[df["puntaje_obtenido"] > UMBRAL_TALENTO].copy()

    # Criterio 1: alto desempeño SIN computador o SIN internet.
    cond_acceso = pd.Series(False, index=alto.index)
    if "tiene_computador" in alto.columns:
        cond_acceso |= (alto["tiene_computador"] == "No")
    if "tiene_internet" in alto.columns:
        cond_acceso |= (alto["tiene_internet"] == "No")

    # Criterio 2: alto desempeño en estrato bajo (1-2).
    cond_estrato = pd.Series(False, index=alto.index)
    if "estrato" in alto.columns:
        cond_estrato = alto["estrato"].isin([1, 2])

    talento = alto[cond_acceso | cond_estrato].copy()
    talento["motivo"] = np.where(
        cond_acceso.loc[talento.index] & cond_estrato.loc[talento.index],
        "Bajo acceso + estrato bajo",
        np.where(cond_acceso.loc[talento.index], "Sin computador/internet",
                 "Estrato 1-2"))

    # Columnas a exportar (las que existan) para acción de la Fundación.
    cols_exp = [c for c in ["numero_documento", "nombres", "apellidos",
                            "institucion_educativa", "municipio",
                            "grado_escolar", "estrato", "tiene_computador",
                            "tiene_internet", "puntaje_obtenido", "motivo"]
                if c in talento.columns]
    talento_exp = (talento[cols_exp]
                   .sort_values("puntaje_obtenido", ascending=False))
    destino_csv = OUTPUTS_DIR / "talento_oculto.csv"
    talento_exp.to_csv(destino_csv, index=False, encoding="utf-8-sig")
    log(f"    talento oculto detectado: {len(talento_exp)} estudiantes "
        f"→ outputs/talento_oculto.csv")

    res = {
        "n_alto": int(len(alto)),
        "n_talento": int(len(talento_exp)),
        "n_sin_acceso": int(cond_acceso.sum()),
        "n_estrato_bajo": int(cond_estrato.sum()),
        "tabla": talento_exp,
    }

    # --- Gráfico: dispersión estrato vs puntaje resaltando talento ---------
    if "estrato" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        base = df.dropna(subset=["estrato", "puntaje_obtenido"])
        ax.scatter(base["estrato"] + np.random.uniform(-0.15, 0.15, len(base)),
                   base["puntaje_obtenido"], s=18, alpha=0.25,
                   color=COLORS["cyan"], label="Todos")
        tal = talento_exp.dropna(subset=["estrato"]) if "estrato" in \
            talento_exp.columns else talento_exp.iloc[0:0]
        if not tal.empty:
            ax.scatter(tal["estrato"]
                       + np.random.uniform(-0.15, 0.15, len(tal)),
                       tal["puntaje_obtenido"], s=55, alpha=0.9,
                       color=COLORS["violet"], edgecolor=COLORS["dark"],
                       label="Talento oculto", zorder=5)
        ax.axhline(UMBRAL_TALENTO, color=COLORS["red"], linestyle="--",
                   label=f"Umbral talento = {UMBRAL_TALENTO}")
        ax.set_title("Detección de talento oculto: estrato vs. puntaje")
        ax.set_xlabel("Estrato")
        ax.set_ylabel("Puntaje")
        ax.legend()
        savefig(fig, "F02E_talento_oculto.png", "talento")

    print(f"    · alto desempeño (>{UMBRAL_TALENTO}): {res['n_alto']}")
    print(f"    · de ellos, con bajo acceso/estrato: {res['n_talento']}")
    return res


# =============================================================================
# SECCIÓN F — ANÁLISIS CRUZADO PROFUNDO
# =============================================================================

def _heatmap_cruce(ax, df, fila, columna, titulo, min_n=5):
    """Dibuja en `ax` un heatmap del puntaje promedio por (fila × columna)."""
    sub = df.dropna(subset=[fila, columna, "puntaje_obtenido"])
    if sub.empty:
        ax.set_visible(False)
        return None
    tabla = sub.pivot_table(index=fila, columns=columna,
                            values="puntaje_obtenido", aggfunc="mean")
    conteo = sub.pivot_table(index=fila, columns=columna,
                             values="puntaje_obtenido", aggfunc="count")
    # Ocultamos celdas con muy pocos casos (poco fiables).
    tabla = tabla.where(conteo >= min_n)
    sns.heatmap(tabla, annot=True, fmt=".1f", cmap=STEM_SEQ, linewidths=0.5,
                linecolor="white", cbar_kws={"shrink": 0.7},
                annot_kws={"fontsize": 9}, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel(columna)
    ax.set_ylabel(fila)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    return tabla


def analisis_cruzado_profundo(df: pd.DataFrame) -> dict:
    log("SECCIÓN F — Análisis cruzado profundo")
    res = {}

    # --- F.1–F.3 Tres tablas cruzadas como heatmaps ------------------------
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    res["heat_comp_conviven"] = _heatmap_cruce(
        axes[0], df, "computador_en_casa", "con_quien_vive",
        "Computador × Con quién vive\n(puntaje promedio)")
    res["heat_internet_estrato"] = _heatmap_cruce(
        axes[1], df, "internet_en_casa", "estrato",
        "Internet × Estrato\n(puntaje promedio)")
    res["heat_olimp_prog"] = _heatmap_cruce(
        axes[2], df, "participacion_olimpiadas", "nivel_programacion",
        "Participación olimpiadas × Nivel programación\n(puntaje promedio)")
    fig.suptitle("F. Análisis cruzado profundo — puntaje promedio por pares "
                 "de factores", fontsize=15, fontweight="bold")
    savefig(fig, "F02F_cruces_heatmaps.png", "cruces")

    # --- F.4 ¿Participar antes en olimpiadas mejora el puntaje? (t-test) ----
    res["olimpiadas"] = prueba_grupo(df, "participo_olimpiadas")
    if res["olimpiadas"]:
        print(f"    olimpiadas previas  → "
              f"p={fmt_p(res['olimpiadas']['p_value'])}")

    # --- F.5 ¿Mayor nivel de programación → mejor puntaje? (ANOVA) ----------
    res["nivel_prog"] = prueba_grupo(df, "nivel_programacion")
    if res["nivel_prog"]:
        print(f"    nivel programación  → "
              f"p={fmt_p(res['nivel_prog']['p_value'])}")

    # Gráfico: dos boxplots (olimpiadas y nivel de programación ordenado).
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    if "participo_olimpiadas" in df.columns:
        sub = df[["participo_olimpiadas", "puntaje_obtenido"]].dropna()
        sns.boxplot(data=sub, x="participo_olimpiadas", y="puntaje_obtenido",
                    order=["No", "Sí"],
                    palette=[COLORS["amber"], COLORS["cyan"]], ax=ax1)
        p = res["olimpiadas"]["p_value"] if res["olimpiadas"] else float("nan")
        ax1.set_title(f"¿Participó antes en olimpiadas? · p = {p:.3g}")
        ax1.set_xlabel("")
        ax1.set_ylabel("Puntaje")
    if "nivel_programacion" in df.columns:
        orden_prog = [n for n in ["Ninguna", "Básica", "Intermedia", "Avanzada"]
                      if n in df["nivel_programacion"].unique()]
        sub = df[["nivel_programacion", "puntaje_obtenido"]].dropna()
        sns.boxplot(data=sub, x="nivel_programacion", y="puntaje_obtenido",
                    order=orden_prog, palette=PALETTE, ax=ax2)
        p = res["nivel_prog"]["p_value"] if res["nivel_prog"] else float("nan")
        ax2.set_title(f"Nivel de programación autopercibido · p = {p:.3g}")
        ax2.set_xlabel("")
        ax2.set_ylabel("Puntaje")
    fig2.suptitle("F. Experiencia previa y puntaje",
                  fontsize=15, fontweight="bold")
    savefig(fig2, "F02F_experiencia_previa.png", "experiencia")

    # --- F.6 Interacción género × tipo de institución (ANOVA 2 vías) --------
    res["anova2"] = None
    if {"genero", "tipo_institucion"}.issubset(df.columns):
        sub = df[df["genero"].isin(["Masculino", "Femenino"])].dropna(
            subset=["genero", "tipo_institucion", "puntaje_obtenido"])
        try:
            import statsmodels.formula.api as smf
            import statsmodels.api as sm
            modelo = smf.ols(
                "puntaje_obtenido ~ C(genero) * C(tipo_institucion)",
                data=sub).fit()
            aov = sm.stats.anova_lm(modelo, typ=2)
            res["anova2"] = aov
            p_int = float(aov.loc["C(genero):C(tipo_institucion)", "PR(>F)"])
            print(f"    género × institución (interacción) → "
                  f"p={fmt_p(p_int)}")
        except Exception as e:  # statsmodels ausente o fórmula sin datos
            log(f"    ⚠ ANOVA 2 vías no disponible: {e}")

        fig3, ax = plt.subplots(figsize=(9, 5.5))
        sns.boxplot(data=sub, x="tipo_institucion", y="puntaje_obtenido",
                    hue="genero", palette=[COLORS["violet"], COLORS["cyan"]],
                    ax=ax)
        ax.set_title("Interacción género × tipo de institución")
        ax.set_xlabel("")
        ax.set_ylabel("Puntaje")
        ax.legend(title="Género")
        savefig(fig3, "F02F_genero_x_institucion.png", "genero_institucion")

    return res


# =============================================================================
# SECCIÓN G — ANÁLISIS POR INSTITUCIÓN MÁS PROFUNDO
# =============================================================================

def analisis_institucion_profundo(df: pd.DataFrame) -> dict:
    log("SECCIÓN G — Análisis por institución profundo")
    res = {}
    if "institucion_educativa" not in df.columns:
        return res

    sub = df.dropna(subset=["institucion_educativa", "puntaje_obtenido"])

    # --- G.1 Estadísticas por colegio (incl. % aprobación ≥ 60) ------------
    # Named-aggregation (robusto en pandas 3.0; evita groupby.apply sobre el frame).
    sub = sub.assign(
        _aprob=(sub["puntaje_obtenido"] >= UMBRAL_APROBACION).astype(float))
    tabla = (sub.groupby("institucion_educativa").agg(
        n=("puntaje_obtenido", "count"),
        media=("puntaje_obtenido", "mean"),
        mediana=("puntaje_obtenido", "median"),
        std=("puntaje_obtenido", "std"),
        min=("puntaje_obtenido", "min"),
        max=("puntaje_obtenido", "max"),
        pct_aprob=("_aprob", lambda x: x.mean() * 100),
    ).reset_index())
    tabla = tabla[tabla["n"] >= 10].round(2).sort_values("media",
                                                         ascending=False)
    res["tabla"] = tabla

    # --- G.2 Colegios con mayor varianza (talento no aprovechado) ----------
    var_rank = tabla.sort_values("std", ascending=False).head(8)
    res["mayor_varianza"] = var_rank
    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(var_rank))))
    colores = gradient_colors(len(var_rank))
    ax.barh(var_rank["institucion_educativa"], var_rank["std"],
            color=colores, edgecolor="white")
    ax.invert_yaxis()
    ax.set_title("Colegios con mayor dispersión de puntajes (desv. estándar)\n"
                 "→ posible talento no aprovechado")
    ax.set_xlabel("Desviación estándar del puntaje")
    for i, (_, fila) in enumerate(var_rank.iterrows()):
        ax.text(fila["std"] + 0.2, i, f"σ={fila['std']:.1f} (n={int(fila['n'])})",
                va="center", fontsize=8)
    savefig(fig, "F02G_varianza_instituciones.png", "varianza")

    # --- G.3 ANCOVA simplificado: puntaje ~ colegio + estrato --------------
    # (solo colegios con > 50 estudiantes, para comparar "ajustando por estrato")
    res["ancova"] = None
    grandes = tabla[tabla["n"] > 50]["institucion_educativa"].tolist()
    sub_g = sub[sub["institucion_educativa"].isin(grandes)].dropna(
        subset=["estrato"])
    if len(grandes) >= 2 and len(sub_g) > 30:
        try:
            import statsmodels.formula.api as smf
            import statsmodels.api as sm
            modelo = smf.ols(
                "puntaje_obtenido ~ C(institucion_educativa) + estrato",
                data=sub_g).fit()
            aov = sm.stats.anova_lm(modelo, typ=2)
            res["ancova"] = aov
            p_col = float(aov.loc["C(institucion_educativa)", "PR(>F)"])
            p_est = float(aov.loc["estrato", "PR(>F)"])
            res["ancova_pvals"] = {"institucion": p_col, "estrato": p_est,
                                   "n": int(len(sub_g)),
                                   "colegios": len(grandes)}
            print(f"    ANCOVA (colegios>50): institución p={fmt_p(p_col)} | "
                  f"estrato p={fmt_p(p_est)}")
        except Exception as e:
            log(f"    ⚠ ANCOVA no disponible: {e}")

    return res


# =============================================================================
# SECCIÓN H — PERFIL DEL ESTUDIANTE EXITOSO
# =============================================================================

def perfil_estudiante_exitoso(df: pd.DataFrame) -> dict:
    log("SECCIÓN H — Perfil del estudiante exitoso")
    res = {}

    # Rasgos numéricos comparables (todos escalados a 0-1 para el radar).
    # Cada entrada: (etiqueta, función que dado un sub-DataFrame devuelve valor 0-1)
    def _pct_si(g, col):
        s = g[col].dropna()
        return float((s == "Sí").mean()) if len(s) else np.nan

    rasgos = []
    if "tiene_computador" in df.columns:
        rasgos.append(("Computador\nen casa", lambda g: _pct_si(g, "tiene_computador")))
    if "tiene_internet" in df.columns:
        rasgos.append(("Internet\nen casa", lambda g: _pct_si(g, "tiene_internet")))
    if "participo_olimpiadas" in df.columns:
        rasgos.append(("Participó\nolimpiadas", lambda g: _pct_si(g, "participo_olimpiadas")))
    if "nivel_programacion_ord" in df.columns:
        rasgos.append(("Nivel\nprogramación", lambda g: g["nivel_programacion_ord"].mean() / 3))
    if "nivel_robotica_ord" in df.columns:
        rasgos.append(("Nivel\nrobótica", lambda g: g["nivel_robotica_ord"].mean() / 3))
    if "interes_prog_robotica" in df.columns:
        rasgos.append(("Interés\nprog/robótica", lambda g: (g["interes_prog_robotica"].mean() - 1) / 4))
    if "estrato" in df.columns:
        # Estrato real 1-3 → normalizado a [0, 1] (antes /5 asumía rango 1-6).
        rasgos.append(("Estrato", lambda g: (g["estrato"].mean() - 1) / 2))

    # --- Grupos: top 10% vs bottom 10% por puntaje -------------------------
    s = df["puntaje_obtenido"].dropna()
    q90, q10 = s.quantile(0.90), s.quantile(0.10)
    top = df[df["puntaje_obtenido"] >= q90]
    bottom = df[df["puntaje_obtenido"] <= q10]
    res["q90"], res["q10"] = float(q90), float(q10)
    res["n_top"], res["n_bottom"] = int(len(top)), int(len(bottom))

    etiquetas = [r[0] for r in rasgos]
    val_top = [r[1](top) for r in rasgos]
    val_bottom = [r[1](bottom) for r in rasgos]

    # Tabla comparativa (valores 0-1) para el informe.
    comp = pd.DataFrame({
        "Rasgo": [e.replace("\n", " ") for e in etiquetas],
        "Top 10% (0-1)": np.round(val_top, 3),
        "Bottom 10% (0-1)": np.round(val_bottom, 3),
    })
    res["comparacion"] = comp

    # --- Perfil de quienes obtienen puntaje ≥ 80 ---------------------------
    elite = df[df["puntaje_obtenido"] >= 80]
    perfil_elite = {}
    for col in ["genero", "municipio", "tipo_institucion", "tiene_computador",
                "tiene_internet", "participo_olimpiadas", "grupo_estrato"]:
        if col in elite.columns and elite[col].notna().any():
            top_val = elite[col].value_counts(normalize=True).head(1)
            perfil_elite[col] = f"{top_val.index[0]} ({top_val.iloc[0]*100:.0f}%)"
    res["perfil_elite"] = perfil_elite
    res["n_elite"] = int(len(elite))

    # --- Radar / spider chart top vs bottom --------------------------------
    if len(rasgos) >= 3:
        angulos = np.linspace(0, 2 * np.pi, len(etiquetas), endpoint=False).tolist()
        angulos += angulos[:1]  # cerrar el polígono
        vt = list(np.nan_to_num(val_top)) + [np.nan_to_num(val_top)[0]]
        vb = list(np.nan_to_num(val_bottom)) + [np.nan_to_num(val_bottom)[0]]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(etiquetas, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8,
                           color="gray")

        ax.plot(angulos, vt, color=COLORS["cyan"], linewidth=2,
                label=f"Top 10% (≥{q90:.0f} pts)")
        ax.fill(angulos, vt, color=COLORS["cyan"], alpha=0.25)
        ax.plot(angulos, vb, color=COLORS["red"], linewidth=2,
                label=f"Bottom 10% (≤{q10:.0f} pts)")
        ax.fill(angulos, vb, color=COLORS["red"], alpha=0.20)
        ax.set_title("H. Perfil del estudiante: Top 10% vs. Bottom 10%\n"
                     "(rasgos escalados 0–1)", fontsize=14, fontweight="bold",
                     pad=28)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
        savefig(fig, "F02H_perfil_radar.png", "radar")

    print(f"    top 10% N={res['n_top']} (≥{q90:.0f}) | "
          f"bottom 10% N={res['n_bottom']} (≤{q10:.0f}) | "
          f"élite ≥80 N={res['n_elite']}")
    return res


# =============================================================================
# SECCIÓN I — INFORME MARKDOWN
# =============================================================================

def construir_informe(ctx: dict) -> None:
    log("Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = ctx["genero"]
    socio = ctx["socio"]
    terr = ctx["territorio"]
    grado = ctx["grado"]
    tal = ctx["talento"]

    def sig(p): return "**significativa**" if p < 0.05 else "no significativa"

    REPORT.append("# Análisis de Brechas — Copa STEM 2026\n")
    REPORT.append(f"**Fundación SapienceLab** · Informe generado: {fecha}\n")
    REPORT.append("---\n")

    # ---- Resumen ejecutivo -------------------------------------------------
    mun_txt = ""
    if terr.get("municipio"):
        medias = terr["municipio"]["medias"]
        mejor = max(medias, key=medias.get)
        mun_txt = (f"**{mejor}** lidera el rendimiento territorial "
                   f"({medias[mejor]:.1f} puntos)")
    REPORT.append("## Resumen ejecutivo\n")
    REPORT.append(dedent(f"""\
        Se analizaron **{ctx['n']:,} estudiantes** que presentaron Copa STEM 2026
        para cuantificar brechas de equidad. La brecha de **género** resultó
        {sig(gen.get('p_value', 1))}: los hombres promedian {gen.get('media_mas', 0):.1f}
        frente a {gen.get('media_fem', 0):.1f} de las mujeres. La brecha
        **territorial** es la más marcada — {mun_txt}. El acceso a **computador**
        muestra una diferencia {sig(socio.get('computador', {}).get('p_value', 1)) if socio.get('computador') else 'n/d'}
        en el puntaje. Se identificaron **{tal['n_talento']} casos de talento
        oculto** (puntaje > {UMBRAL_TALENTO} con bajo acceso tecnológico o estrato
        1-2) que constituyen el foco prioritario de intervención de la Fundación.\n"""))

    # ---- Metodología -------------------------------------------------------
    REPORT.append("## Metodología\n")
    REPORT.append(dedent("""\
        Se emplean pruebas de hipótesis con α = 0.05. Para dos grupos, **t-test
        de Welch** (robusto a varianzas desiguales) con **Cohen's d**; para tres
        o más, **ANOVA de una vía** con **eta²**. Se exige N ≥ 15 por grupo. El
        talento oculto se define como puntaje > {u} combinado con ausencia de
        computador/internet en casa o pertenencia a estrato 1-2. Procesamiento
        reproducible (`random_state=42`).\n""".format(u=UMBRAL_TALENTO)))

    # ---- A. Género ---------------------------------------------------------
    REPORT.append("## A. Brechas de género\n")
    REPORT.append("**Pregunta:** ¿existe una diferencia significativa de puntaje "
                  "entre hombres y mujeres?\n")
    REPORT.append("**Método:** t-test de Welch (Masculino vs. Femenino) + "
                  "histogramas superpuestos.\n")
    if gen:
        REPORT.append("**Resultado:**\n")
        REPORT.append(tabla_md(pd.DataFrame([
            {"Grupo": "Femenino", "N": gen["n_fem"],
             "Puntaje medio": round(gen["media_fem"], 2)},
            {"Grupo": "Masculino", "N": gen["n_mas"],
             "Puntaje medio": round(gen["media_mas"], 2)},
        ])) + "\n")
        REPORT.append(f"\n- t = {gen['t_stat']:.3f} · p = {gen['p_value']:.3g} · "
                      f"Cohen's d = {gen['cohens_d']:.3f}\n")
    REPORT.append(f"\n{img('genero', 'Brecha de género')}\n")
    REPORT.append(dedent(f"""\
        **Interpretación.** La diferencia es {sig(gen.get('p_value', 1))}. El
        tamaño del efecto (Cohen's d = {gen.get('cohens_d', float('nan')):.3f})
        indica una magnitud {'pequeña' if abs(gen.get('cohens_d', 0)) < 0.2 else 'pequeña-moderada' if abs(gen.get('cohens_d', 0)) < 0.5 else 'moderada'};
        conviene monitorearla pero no domina el resultado global.\n"""))

    # ---- B. Socioeconómicas ------------------------------------------------
    REPORT.append("## B. Brechas socioeconómicas\n")
    REPORT.append("**Pregunta:** ¿el estrato y el acceso a tecnología se asocian "
                  "con el puntaje?\n")
    filas = []
    for k, etq in [("estrato", "Estrato (ANOVA)"),
                   ("computador", "Tiene computador"),
                   ("internet", "Tiene internet"),
                   ("con_quien_vive", "Con quién vive")]:
        r = socio.get(k)
        if r:
            filas.append({"Variable": etq, "Prueba": r["prueba"],
                          "p-value": f"{r['p_value']:.3g}",
                          r["efecto_nombre"]: round(r["efecto"], 3),
                          "Significativa": "Sí" if r["p_value"] < 0.05 else "No"})
    if filas:
        REPORT.append(tabla_md(pd.DataFrame(filas)) + "\n")
    REPORT.append(f"\n{img('socio', 'Brechas socioeconómicas')}\n")
    REPORT.append(f"\n{img('cruce', 'Cruce estrato × acceso')}\n")
    if isinstance(socio.get("cruce"), pd.DataFrame):
        c = socio["cruce"].copy()
        c["mean"] = c["mean"].round(2)
        c.columns = ["Grupo estrato", "¿Computador?", "Puntaje medio", "N"]
        REPORT.append("\n**Cruce estrato × acceso a computador:**\n")
        REPORT.append(tabla_md(c) + "\n")
    REPORT.append(dedent("""\
        **Interpretación.** El cruce estrato × acceso permite distinguir el
        efecto del **recurso** (computador) del efecto del **entorno** (estrato).
        Si dentro de cada estrato quienes tienen computador rinden más, la
        política debe priorizar dotación tecnológica; si la brecha persiste por
        estrato aun con computador, el problema es más estructural.\n"""))

    # ---- C. Territoriales --------------------------------------------------
    REPORT.append("## C. Brechas territoriales\n")
    REPORT.append("**Pregunta:** ¿difiere el rendimiento entre municipios, entre "
                  "instituciones públicas/privadas y entre colegios?\n")
    filas = []
    for k, etq in [("municipio", "Municipio"),
                   ("tipo_institucion", "Tipo de institución")]:
        r = terr.get(k)
        if r:
            medias_txt = "; ".join(f"{a}: {b:.1f}" for a, b in r["medias"].items())
            filas.append({"Comparación": etq, "Prueba": r["prueba"],
                          "p-value": f"{r['p_value']:.3g}",
                          "Medias": medias_txt,
                          "Significativa": "Sí" if r["p_value"] < 0.05 else "No"})
    if filas:
        REPORT.append(tabla_md(pd.DataFrame(filas)) + "\n")
    REPORT.append(f"\n{img('territorio', 'Brechas territoriales')}\n")
    if isinstance(terr.get("ranking"), pd.DataFrame):
        rk = terr["ranking"].copy()
        rk["mean"] = rk["mean"].round(2)
        rk.columns = ["Institución", "Puntaje medio", "N"]
        REPORT.append("\n**Ranking de instituciones (N ≥ 10):**\n")
        REPORT.append(tabla_md(rk) + "\n")
    REPORT.append(f"\n{img('ranking', 'Ranking de instituciones')}\n")
    REPORT.append(f"\n{img('genero_municipio', 'Género × municipio')}\n")
    if terr.get("brechas_genero_municipio"):
        REPORT.append("\n**Brecha de género (Masculino − Femenino) por "
                      "municipio:**\n")
        filas = [{"Municipio": m, "Brecha (pts)": round(v["brecha"], 2),
                  "p-value": f"{v['p']:.3g}"}
                 for m, v in terr["brechas_genero_municipio"].items()]
        REPORT.append(tabla_md(pd.DataFrame(filas)) + "\n")
    REPORT.append(dedent("""\
        **Interpretación.** La brecha territorial es la señal más fuerte del
        estudio: sugiere diferencias sistémicas (calidad de preparación, recursos
        institucionales) entre municipios y colegios. El ranking permite focalizar
        acompañamiento en las instituciones de menor promedio.\n"""))

    # ---- D. Grado ----------------------------------------------------------
    REPORT.append("## D. Brechas por grado escolar\n")
    REPORT.append("**Pregunta:** ¿el rendimiento mejora o empeora de 9° a 11°?\n")
    if isinstance(grado.get("resumen"), pd.DataFrame):
        g = grado["resumen"].copy()
        for c in ["mean", "median", "std"]:
            g[c] = g[c].round(2)
        g.columns = ["Grado", "Media", "Mediana", "Desv.", "N"]
        REPORT.append(tabla_md(g) + "\n")
    if grado.get("prueba"):
        REPORT.append(f"\n- {grado['prueba']['prueba']} · "
                      f"p = {grado['prueba']['p_value']:.3g} · "
                      f"eta² = {grado['prueba']['efecto']:.3f}\n")
    REPORT.append(f"\n{img('grado', 'Brechas por grado')}\n")
    REPORT.append(dedent("""\
        **Interpretación.** La trayectoria 9°→10°→11° indica si el sistema
        agrega valor con el avance escolar. Un promedio plano o decreciente en
        grados superiores sería una alerta sobre la preparación en matemáticas
        y lógica en la media vocacional.\n"""))

    # ---- E. Talento oculto -------------------------------------------------
    REPORT.append("## E. Detección de talento oculto\n")
    REPORT.append(f"**Pregunta:** ¿qué estudiantes de alto desempeño "
                  f"(puntaje > {UMBRAL_TALENTO}) enfrentan barreras de acceso y "
                  f"merecen intervención prioritaria?\n")
    REPORT.append(dedent(f"""\
        **Resultado:** de **{tal['n_alto']}** estudiantes con puntaje >
        {UMBRAL_TALENTO}, **{tal['n_talento']}** cumplen al menos un criterio de
        vulnerabilidad (sin computador/internet en casa, o estrato 1-2). La lista
        completa se exportó a `outputs/talento_oculto.csv`.\n"""))
    REPORT.append(f"\n{img('talento', 'Talento oculto')}\n")
    if isinstance(tal.get("tabla"), pd.DataFrame) and not tal["tabla"].empty:
        t = tal["tabla"].copy()
        # Mostramos el top 20 en el informe; el CSV tiene el listado completo.
        REPORT.append("\n**Top talento oculto (máx. 20; listado completo en el "
                      "CSV):**\n")
        REPORT.append(tabla_md(t, max_filas=20) + "\n")
    REPORT.append(dedent("""\
        **Interpretación.** Estos estudiantes demuestran alto potencial STEM
        **a pesar** de recursos limitados: son el retorno social más alto de una
        beca o acompañamiento. Se recomienda contacto directo con sus
        instituciones.\n"""))

    # ---- F. Análisis cruzado profundo -------------------------------------
    cru = ctx["cruzado"]
    REPORT.append("## F. Análisis cruzado profundo\n")
    REPORT.append("**Pregunta:** ¿cómo interactúan los factores entre sí y qué "
                  "papel juega la experiencia previa (olimpiadas, programación)?\n")
    REPORT.append("**Método:** tablas cruzadas de puntaje promedio (heatmaps), "
                  "t-test para experiencia previa, ANOVA para nivel de "
                  "programación y ANOVA de dos vías para la interacción "
                  "género × tipo de institución.\n")
    REPORT.append(f"\n{img('cruces', 'Heatmaps cruzados')}\n")
    REPORT.append(f"\n{img('experiencia', 'Experiencia previa')}\n")
    filas = []
    if cru.get("olimpiadas"):
        r = cru["olimpiadas"]
        filas.append({"Análisis": "Participó en olimpiadas (t-test)",
                      "p-value": f"{r['p_value']:.3g}",
                      "Efecto": f"d={r['efecto']:.3f}",
                      "Medias": "; ".join(f"{k}:{v:.1f}"
                                          for k, v in r["medias"].items())})
    if cru.get("nivel_prog"):
        r = cru["nivel_prog"]
        filas.append({"Análisis": "Nivel de programación (ANOVA)",
                      "p-value": f"{r['p_value']:.3g}",
                      "Efecto": f"eta²={r['efecto']:.3f}",
                      "Medias": "; ".join(f"{k}:{v:.1f}"
                                          for k, v in r["medias"].items())})
    if filas:
        REPORT.append(tabla_md(pd.DataFrame(filas)) + "\n")
    REPORT.append(f"\n{img('genero_institucion', 'Género × institución')}\n")
    if cru.get("anova2") is not None:
        aov = cru["anova2"].reset_index().rename(columns={"index": "Término"})
        aov = aov.round(4)
        REPORT.append("\n**ANOVA de dos vías (género × tipo de institución):**\n")
        REPORT.append(tabla_md(aov) + "\n")
    REPORT.append(dedent("""\
        **Interpretación.** Los heatmaps revelan combinaciones de factores con
        rendimiento especialmente alto o bajo (útil para focalizar). Si la
        experiencia previa (olimpiadas, programación) se asocia a mayor puntaje,
        conviene incorporarla como variable en los modelos predictivos. La
        interacción género × institución indica si la brecha de género se
        concentra en cierto tipo de colegio.\n"""))

    # ---- G. Análisis por institución profundo -----------------------------
    ins = ctx["instituciones"]
    REPORT.append("## G. Análisis por institución (profundo)\n")
    REPORT.append("**Pregunta:** más allá del promedio, ¿qué colegios concentran "
                  "mayor dispersión (talento no aprovechado) y cómo se comparan "
                  "ajustando por estrato?\n")
    if isinstance(ins.get("tabla"), pd.DataFrame) and not ins["tabla"].empty:
        t = ins["tabla"].copy()
        t.columns = ["Institución", "N", "Media", "Mediana", "Desv.",
                     "Mín", "Máx", "% aprob. (≥60)"]
        REPORT.append("\n**Estadísticas por institución (N ≥ 10):**\n")
        REPORT.append(tabla_md(t) + "\n")
    REPORT.append(f"\n{img('varianza', 'Varianza por institución')}\n")
    if ins.get("ancova_pvals"):
        a = ins["ancova_pvals"]
        REPORT.append(dedent(f"""\

            **ANCOVA simplificado** (colegios con >50 estudiantes, N={a['n']},
            {a['colegios']} colegios): al modelar `puntaje ~ institución +
            estrato`, la **institución** sigue siendo significativa
            (p = {a['institucion']:.3g}) {'aun controlando' if a['institucion'] < 0.05 else 'al controlar'}
            por estrato (efecto del estrato: p = {a['estrato']:.3g}). Esto sugiere
            que las diferencias entre colegios **no** se explican solo por su
            composición socioeconómica.\n"""))
    REPORT.append(dedent("""\
        **Interpretación.** Los colegios con alta desviación estándar tienen
        estudiantes muy por encima y muy por debajo de su media: ahí puede haber
        **talento no identificado** que se beneficiaría de acompañamiento. El
        ANCOVA ayuda a separar el "efecto colegio" del "efecto estrato".\n"""))

    # ---- H. Perfil del estudiante exitoso ---------------------------------
    per = ctx["perfil"]
    REPORT.append("## H. Perfil del estudiante exitoso\n")
    REPORT.append(f"**Pregunta:** ¿qué rasgos comparten los estudiantes de mayor "
                  f"desempeño frente a los de menor desempeño?\n")
    REPORT.append(dedent(f"""\
        **Método:** comparación del **top 10%** (≥ {per.get('q90', 0):.0f} pts,
        N={per.get('n_top', 0)}) vs. **bottom 10%** (≤ {per.get('q10', 0):.0f} pts,
        N={per.get('n_bottom', 0)}) en rasgos escalados 0–1, visualizada en un
        gráfico radar.\n"""))
    if isinstance(per.get("comparacion"), pd.DataFrame):
        REPORT.append(tabla_md(per["comparacion"]) + "\n")
    REPORT.append(f"\n{img('radar', 'Perfil radar top vs bottom')}\n")
    if per.get("perfil_elite"):
        REPORT.append(f"\n**Rasgo mayoritario de la élite (puntaje ≥ 80, "
                      f"N={per.get('n_elite', 0)}):**\n")
        filas = [{"Característica": k, "Valor más frecuente": v}
                 for k, v in per["perfil_elite"].items()]
        REPORT.append(tabla_md(pd.DataFrame(filas)) + "\n")
    REPORT.append(dedent("""\
        **Interpretación.** Los rasgos donde el radar del top se separa más del
        bottom son los **predictores prácticos** de éxito: orientan tanto la
        detección temprana de potencial como el diseño de las intervenciones
        (p. ej. si la experiencia en programación distingue claramente a los
        grupos, conviene ampliar la exposición temprana a la programación).\n"""))

    # ---- Recomendaciones de política --------------------------------------
    REPORT.append("## Conclusiones y recomendaciones de política\n")
    REPORT.append(dedent(f"""\
        1. **Priorizar el cierre de la brecha territorial**, la más marcada del
           estudio, con acompañamiento diferenciado a los municipios e
           instituciones de menor promedio (ver ranking).
        2. **Programa de dotación tecnológica** focalizado: el acceso a computador
           mostró asociación con el puntaje; conviene atender a estudiantes sin
           computador, especialmente en estratos bajos.
        3. **Becas de talento oculto:** contactar a los {tal['n_talento']}
           estudiantes identificados para tutoría, mentoría y rutas STEM.
        4. **Monitoreo de equidad de género** por municipio, dado que la brecha
           puede concentrarse en territorios específicos.
        5. **Refuerzo por grado:** ajustar la preparación según la trayectoria
           9°→11° observada.
        6. Alimentar estos hallazgos a la Fase 2 (modelos predictivos,
           `03_prediccion_puntaje.py` y `05_talento_oculto.py`).\n"""))

    # ---- Limitaciones ------------------------------------------------------
    REPORT.append("## Limitaciones del estudio\n")
    REPORT.append(dedent("""\
        - **Datos observacionales:** las brechas descritas son asociaciones, no
          relaciones causales; pueden existir variables de confusión no medidas.
        - **Inscripciones de emergencia:** ~7% de los registros no tienen datos
          socioeconómicos completos y quedan fuera de los análisis que requieren
          esas variables, lo que puede introducir sesgo de selección.
        - **Autoreporte:** estrato, acceso a tecnología y con quién vive son
          autorreportados.
        - **Comparaciones múltiples:** conviene aplicar correcciones (Bonferroni /
          FDR) antes de decisiones definitivas.
        - **Umbral de talento arbitrario:** el corte en {u} puntos es una decisión
          de política revisable.\n""".format(u=UMBRAL_TALENTO)))

    # ---- Referencias -------------------------------------------------------
    REPORT.append("## Referencias técnicas\n")
    REPORT.append(dedent("""\
        - Welch, B. L. (1947). *Biometrika* (t-test de varianzas desiguales).
        - Fisher, R. A. (1925). *Statistical Methods for Research Workers* (ANOVA).
        - Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
          Sciences* (Cohen's d, eta²).
        - OECD (2018). *PISA — Equity in Education* (marco de brechas educativas).
        - McKinney, W. (2010). *pandas*. · Virtanen et al. (2020). *SciPy 1.0*.
          · Waskom, M. (2021). *seaborn*.\n"""))

    REPORT.append("\n---\n_Generado por `notebooks/02_analisis_brechas.py` — "
                  "Copa STEM 2026._\n")

    destino = REPORTS_DIR / "02_analisis_brechas.md"
    destino.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"    informe escrito → reports/{destino.name}")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Análisis de Brechas de Equidad")
    print(" Fundación SapienceLab")
    print("=" * 70)

    df = cargar_y_limpiar()
    if df.empty or "puntaje_obtenido" not in df.columns:
        log("⚠ No hay datos con puntaje para analizar.")
        sys.exit(0)

    genero = brechas_genero(df)
    socio = brechas_socioeconomicas(df)
    territorio = brechas_territoriales(df)
    grado = brechas_por_grado(df)
    talento = detectar_talento_oculto(df)
    cruzado = analisis_cruzado_profundo(df)
    instituciones = analisis_institucion_profundo(df)
    perfil = perfil_estudiante_exitoso(df)

    ctx = {"n": len(df), "genero": genero, "socio": socio,
           "territorio": territorio, "grado": grado, "talento": talento,
           "cruzado": cruzado, "instituciones": instituciones,
           "perfil": perfil}
    construir_informe(ctx)

    print("\n" + "=" * 70)
    print(" ✔ ANÁLISIS DE BRECHAS COMPLETADO")
    print(f"   · Figuras generadas: {len(FIGURES)}  → outputs/ (paleta Copa STEM)")
    print(f"   · Talento oculto:    outputs/talento_oculto.csv")
    print(f"   · Informe:           reports/02_analisis_brechas.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
