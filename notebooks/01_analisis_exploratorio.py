# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 01: Análisis Exploratorio de Datos (EDA)
================================================================================

Objetivo
--------
Realizar un estudio exploratorio de nivel profesional sobre los resultados de la
olimpiada Copa STEM (grados 9°, 10° y 11° en Copacabana y Girardota, Antioquia).

El script está organizado en las secciones metodológicas clásicas de un EDA:

    A) Carga y limpieza de datos
    B) Análisis univariado
    C) Análisis bivariado (con pruebas estadísticas de significancia)
    D) Análisis socioeconómico y de brechas
    E) Análisis de telemetría (comportamiento durante el examen)
    F) (transversal) Todas las figuras se guardan en outputs/ como PNG dpi=150
    G) Generación del informe reports/01_analisis_exploratorio.md

Principios de diseño
--------------------
- Autocontenido y reproducible: `random_state=42` donde aplique.
- Robusto: si una columna no existe, la sección se omite con aviso (no rompe).
- Cada decisión metodológica está comentada en español.
- Se imprime el progreso en consola con el prefijo ">>>".

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from textwrap import dedent

# --- Semilla global de reproducibilidad -------------------------------------
RANDOM_STATE = 42

# --- Dependencias científicas ------------------------------------------------
# Se importan dentro de un try para dar un mensaje claro si falta el entorno.
try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")  # backend sin ventana: solo escribe archivos PNG
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    from scipy import stats
except ImportError as exc:  # pragma: no cover
    print("=" * 70)
    print("ERROR: falta una dependencia del entorno virtual.")
    print(f"       Detalle: {exc}")
    print("       Active el entorno e instale las dependencias:")
    print("       .venv\\Scripts\\activate")
    print("       pip install pandas numpy scikit-learn matplotlib seaborn scipy "
          "statsmodels")
    print("=" * 70)
    sys.exit(1)

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0. CONFIGURACIÓN GLOBAL: rutas, paleta de colores y estilo de gráficos
# =============================================================================

# El script vive en notebooks/; la raíz del proyecto es su carpeta madre.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

for _d in (OUTPUTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Nombre canónico esperado; si no está, se autodetecta cualquier CSV en data/.
DATASET_NAME = "copa_stem_dataset.csv"

# Paleta de marca Copa STEM (fondo BLANCO en los gráficos para impresión).
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

# Colormap DIVERGENTE de marca para matrices de correlación (valores -1..+1):
# azul (correlación negativa) → blanco (cero) → rojo (correlación positiva).
STEM_DIVERGING = LinearSegmentedColormap.from_list(
    "stem_div", [COLORS["blue"], "#ffffff", COLORS["red"]])

# Estilo global: base limpia con fondo blanco y tipografía legible.
sns.set_theme(style="whitegrid")
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
})

DPI = 150  # alta resolución para impresión

# Acumulador global de rutas de figuras generadas (para el informe).
FIGURES: dict[str, str] = {}

# Acumulador de bloques de texto/hallazgos para construir el informe markdown.
REPORT: list[str] = []


def log(msg: str) -> None:
    """Imprime progreso con marca temporal para seguimiento en consola."""
    print(f">>> {msg}", flush=True)


def savefig(fig, filename: str, key: str | None = None) -> str:
    """Guarda una figura en outputs/ a dpi=150 con fondo blanco y la registra."""
    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES[key or filename] = filename
    log(f"    figura guardada → outputs/{filename}")
    return filename


def fmt_p(p: float) -> str:
    """Formatea un p-value con interpretación de significancia."""
    if np.isnan(p):
        return "n/d"
    if p < 0.001:
        return f"{p:.2e} (*** altamente significativo)"
    if p < 0.01:
        return f"{p:.4f} (** significativo)"
    if p < 0.05:
        return f"{p:.4f} (* significativo)"
    return f"{p:.4f} (n.s. — no significativo)"


# =============================================================================
# SECCIÓN A — CARGA Y LIMPIEZA DE DATOS
# =============================================================================

def localizar_dataset() -> Path | None:
    """
    Devuelve la ruta del dataset. Prioriza el nombre canónico; si no existe,
    autodetecta el primer CSV disponible en data/. Devuelve None si no hay CSV.
    """
    canonico = DATA_DIR / DATASET_NAME
    if canonico.exists():
        return canonico
    if DATA_DIR.exists():
        csvs = sorted(DATA_DIR.glob("*.csv"))
        if csvs:
            log(f"No se halló {DATASET_NAME}; usando '{csvs[0].name}' encontrado "
                f"en data/.")
            return csvs[0]
    return None


def cargar_datos() -> pd.DataFrame:
    """Carga el CSV; si no existe, informa claramente y termina el programa."""
    log("SECCIÓN A — Carga y limpieza")
    ruta = localizar_dataset()

    if ruta is None:
        print("\n" + "=" * 70)
        print("  ⚠  NO SE ENCONTRÓ EL DATASET")
        print("=" * 70)
        print(f"  Coloque el archivo '{DATASET_NAME}' en la carpeta:")
        print(f"    {DATA_DIR}")
        print("\n  El archivo debe ser el export de Supabase con las columnas de")
        print("  Copa STEM (numero_documento, puntaje_obtenido, telemetría, etc.).")
        print("  Una vez colocado, vuelva a ejecutar este script.")
        print("=" * 70 + "\n")
        sys.exit(0)

    log(f"    leyendo {ruta.name}")
    df = pd.read_csv(ruta, encoding="utf-8")
    log(f"    registros crudos: {len(df):,} | columnas: {df.shape[1]}")
    return df


def perfil_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla de calidad de datos: tipo y % de nulos por columna."""
    perfil = pd.DataFrame({
        "columna": df.columns,
        "tipo": [str(t) for t in df.dtypes],
        "n_nulos": df.isna().sum().values,
        "pct_nulos": (df.isna().mean().values * 100).round(2),
        "n_unicos": [df[c].nunique(dropna=True) for c in df.columns],
    })
    return perfil


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza estándar Copa STEM:
      - Elimina registros de prueba (documentos ficticios).
      - Normaliza strings (trim de espacios).
      - Convierte a numérico las columnas que lo requieren.
      - Deriva variables auxiliares (binarias de acceso, conteos de JSON, etc.).
    """
    df = df.copy()

    # --- 1. Eliminar registros de prueba -----------------------------------
    # Documentos ficticios usados durante el desarrollo/QA de la plataforma.
    docs_prueba = ["1234", "123456", "123456789", "1234567899", "0", "00000000"]
    if "numero_documento" in df.columns:
        df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
        antes = len(df)
        df = df[~df["numero_documento"].isin(docs_prueba)]
        # También descartamos documentos absurdamente cortos (< 5 dígitos).
        df = df[df["numero_documento"].str.len() >= 5]
        log(f"    registros de prueba eliminados: {antes - len(df)}")

    # --- 2. Normalizar strings (quitar espacios sobrantes) -----------------
    # Seleccionamos las columnas NO numéricas (robusto entre pandas 2 y 3,
    # donde las cadenas pasan a un dtype 'str' dedicado).
    cols_texto = [c for c in df.columns
                  if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cols_texto:
        df[c] = df[c].astype(str).str.strip()
        # Restaurar nulos: tras astype(str), los NaN quedan como "nan".
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # --- 3. Conversión de tipos numéricos ----------------------------------
    for c in ["puntaje_obtenido", "tiempo_usado_segundos", "edad_calculada",
              "estrato", "grado_escolar", "interes_prog_robotica",
              "cambios_pestana", "intentos_copiar", "intentos_pegar",
              "intentos_click_derecho"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 4. Variables derivadas de acceso tecnológico ----------------------
    # En el CSV real "computador_en_casa" toma valores como
    # "No", "Sí, propio", "Sí, compartido"; y "internet_en_casa" como
    # "No", "Sí, estable", "Sí, intermitente". Creamos binarias limpias.
    def _si_no(serie: pd.Series) -> pd.Series:
        # dtype=object para poder mezclar cadenas y NaN sin conflictos en
        # numpy/pandas 3 (que ya no promociona str + float).
        s = serie.astype(str).str.lower()
        out = pd.Series(np.nan, index=serie.index, dtype="object")
        out[s.str.startswith("s")] = "Sí"
        out[s.str.startswith("n")] = "No"
        return out

    if "computador_en_casa" in df.columns:
        df["tiene_computador"] = _si_no(df["computador_en_casa"])
    if "internet_en_casa" in df.columns:
        df["tiene_internet"] = _si_no(df["internet_en_casa"])

    # --- 5. Conteo de herramientas y áreas (columnas JSON-string) ----------
    def _contar_json(serie: pd.Series) -> pd.Series:
        def _n(x):
            if pd.isna(x):
                return np.nan
            try:
                lst = json.loads(x)
                if isinstance(lst, list):
                    # "Ninguna"/"Ninguno" cuenta como 0 herramientas reales.
                    reales = [i for i in lst
                              if str(i).strip().lower() not in ("ninguna", "ninguno")]
                    return len(reales)
            except (json.JSONDecodeError, TypeError):
                return np.nan
            return np.nan
        return serie.apply(_n)

    if "herramientas_conocidas" in df.columns:
        df["n_herramientas"] = _contar_json(df["herramientas_conocidas"])
    if "areas_interes" in df.columns:
        df["n_areas_interes"] = _contar_json(df["areas_interes"])

    # --- 6. Codificación ordinal de niveles (para correlaciones) -----------
    # Los niveles vienen en femenino: Ninguna/Básica/Intermedia/Avanzada.
    orden_nivel = {"ninguna": 0, "ninguno": 0, "básica": 1, "basica": 1,
                   "intermedia": 2, "avanzada": 3}
    for c in ["nivel_programacion", "nivel_robotica"]:
        if c in df.columns:
            df[c + "_ord"] = (df[c].astype(str).str.lower().map(orden_nivel))

    log(f"    registros tras limpieza: {len(df):,}")
    return df


def separar_presentaron(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa quienes presentaron (tienen puntaje) de quienes no."""
    if "puntaje_obtenido" not in df.columns:
        log("    ⚠ no existe 'puntaje_obtenido'; se asume que todos presentaron.")
        return df.copy(), df.iloc[0:0].copy()
    presentaron = df[df["puntaje_obtenido"].notna()].copy()
    pendientes = df[df["puntaje_obtenido"].isna()].copy()
    log(f"    presentaron: {len(presentaron):,} | "
        f"sin puntaje: {len(pendientes):,}")
    return presentaron, pendientes


# =============================================================================
# SECCIÓN B — ANÁLISIS UNIVARIADO
# =============================================================================

def estadisticas_puntaje(df: pd.DataFrame) -> dict:
    """Estadísticas descriptivas de puntaje_obtenido (incl. asimetría/curtosis)."""
    s = df["puntaje_obtenido"].dropna()
    est = {
        "n": int(s.size),
        "media": float(s.mean()),
        "mediana": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "asimetria": float(stats.skew(s)),
        "curtosis": float(stats.kurtosis(s)),  # exceso de curtosis (Fisher)
    }
    return est


def grafico_distribucion_puntaje(df: pd.DataFrame, est: dict) -> None:
    """Histograma + KDE + box plot de puntaje_obtenido."""
    s = df["puntaje_obtenido"].dropna()
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    sns.histplot(s, bins=25, kde=True, color=COLORS["cyan"],
                 edgecolor="white", ax=ax_hist)
    ax_hist.axvline(est["media"], color=COLORS["red"], linestyle="--",
                    linewidth=2, label=f"Media = {est['media']:.1f}")
    ax_hist.axvline(est["mediana"], color=COLORS["violet"], linestyle=":",
                    linewidth=2, label=f"Mediana = {est['mediana']:.1f}")
    ax_hist.set_title(
        f"Distribución del puntaje obtenido (N = {est['n']:,})")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.legend()

    sns.boxplot(x=s, color=COLORS["amber"], ax=ax_box, width=0.5,
                flierprops={"markerfacecolor": COLORS["red"], "markersize": 4})
    ax_box.set_xlabel("Puntaje obtenido (0–100)")
    ax_box.set_ylabel("")

    savefig(fig, "B01_distribucion_puntaje.png", "dist_puntaje")


def grafico_categoricas_univariado(df: pd.DataFrame) -> None:
    """Barras de conteo para las principales variables categóricas."""
    cat_cols = [c for c in ["grado_escolar", "genero", "municipio",
                            "tipo_institucion", "estrato", "jornada"]
                if c in df.columns]
    if not cat_cols:
        return
    n = len(cat_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts(dropna=False).sort_index()
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax,
                    palette=PALETTE[: len(vc)])
        ax.set_title(f"Distribución por {col}")
        ax.set_xlabel("")
        ax.set_ylabel("N estudiantes")
        ax.tick_params(axis="x", rotation=30)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Análisis univariado — variables categóricas",
                 fontsize=15, fontweight="bold")
    savefig(fig, "B02_categoricas_univariado.png", "cat_univariado")


def grafico_tiempo(df: pd.DataFrame) -> None:
    """Distribución del tiempo_usado_segundos (en minutos para legibilidad)."""
    if "tiempo_usado_segundos" not in df.columns:
        return
    s = df["tiempo_usado_segundos"].dropna()
    if s.empty:
        return
    minutos = s / 60.0
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(minutos, bins=30, kde=True, color=COLORS["violet"],
                 edgecolor="white", ax=ax)
    ax.axvline(minutos.median(), color=COLORS["red"], linestyle="--",
               label=f"Mediana = {minutos.median():.1f} min")
    ax.set_title(f"Distribución del tiempo de examen (N = {s.size:,})")
    ax.set_xlabel("Tiempo usado (minutos)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    savefig(fig, "B03_distribucion_tiempo.png", "dist_tiempo")


# =============================================================================
# SECCIÓN C — ANÁLISIS BIVARIADO (con pruebas de significancia)
# =============================================================================

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Tamaño del efecto Cohen's d para dos grupos independientes."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def prueba_grupo(df: pd.DataFrame, col: str, min_n: int = 15) -> dict | None:
    """
    Compara puntaje_obtenido entre los grupos de `col`.
      - 2 grupos  → t-test de Welch (no asume varianzas iguales) + Cohen's d.
      - >2 grupos → ANOVA de una vía + eta² (tamaño del efecto).
    Solo considera grupos con al menos `min_n` observaciones.
    """
    if col not in df.columns:
        return None
    sub = df[[col, "puntaje_obtenido"]].dropna()
    grupos = []
    etiquetas = []
    for nombre, g in sub.groupby(col):
        vals = g["puntaje_obtenido"].values
        if len(vals) >= min_n:
            grupos.append(vals)
            etiquetas.append(str(nombre))
    if len(grupos) < 2:
        return None

    medias = {et: float(np.mean(g)) for et, g in zip(etiquetas, grupos)}
    n_por_grupo = {et: int(len(g)) for et, g in zip(etiquetas, grupos)}

    if len(grupos) == 2:
        stat, p = stats.ttest_ind(grupos[0], grupos[1], equal_var=False)
        efecto = _cohens_d(grupos[0], grupos[1])
        prueba = "t-test de Welch"
        efecto_nombre = "Cohen's d"
    else:
        stat, p = stats.f_oneway(*grupos)
        # eta² = SS_between / SS_total
        todas = np.concatenate(grupos)
        media_g = todas.mean()
        ss_total = ((todas - media_g) ** 2).sum()
        ss_between = sum(len(g) * (g.mean() - media_g) ** 2 for g in grupos)
        efecto = float(ss_between / ss_total) if ss_total > 0 else float("nan")
        prueba = "ANOVA de una vía"
        efecto_nombre = "eta²"

    return {
        "columna": col,
        "prueba": prueba,
        "estadistico": float(stat),
        "p_value": float(p),
        "efecto_nombre": efecto_nombre,
        "efecto": efecto,
        "medias": medias,
        "n_por_grupo": n_por_grupo,
    }


def grafico_boxplots_bivariado(df: pd.DataFrame, pruebas: dict) -> None:
    """Box plots de puntaje por cada variable categórica analizada."""
    cols = list(pruebas.keys())
    if not cols:
        return
    ncols = 2
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        sub = df[[col, "puntaje_obtenido"]].dropna()
        orden = sorted(sub[col].unique(), key=str)
        sns.boxplot(data=sub, x=col, y="puntaje_obtenido", order=orden,
                    palette=PALETTE, ax=ax)
        p = pruebas[col]["p_value"]
        ax.set_title(f"Puntaje por {col}\n{pruebas[col]['prueba']} · "
                     f"p = {p:.3g}")
        ax.set_xlabel("")
        ax.set_ylabel("Puntaje")
        ax.tick_params(axis="x", rotation=25)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Análisis bivariado — puntaje vs. variables categóricas",
                 fontsize=15, fontweight="bold")
    savefig(fig, "C01_boxplots_bivariado.png", "boxplots_bivariado")


def grafico_scatter(df: pd.DataFrame) -> None:
    """Dispersión: tiempo vs puntaje y edad vs puntaje (con regresión)."""
    pares = []
    if "tiempo_usado_segundos" in df.columns:
        pares.append(("tiempo_usado_segundos", "Tiempo (segundos)",
                      COLORS["cyan"]))
    if "edad_calculada" in df.columns:
        pares.append(("edad_calculada", "Edad (años)", COLORS["violet"]))
    if not pares:
        return

    fig, axes = plt.subplots(1, len(pares), figsize=(7 * len(pares), 5))
    axes = np.atleast_1d(axes)
    for ax, (col, etiqueta, color) in zip(axes, pares):
        sub = df[[col, "puntaje_obtenido"]].dropna()
        if sub.empty:
            continue
        r, p = stats.pearsonr(sub[col], sub["puntaje_obtenido"])
        sns.regplot(data=sub, x=col, y="puntaje_obtenido", ax=ax,
                    scatter_kws={"alpha": 0.25, "s": 18, "color": color},
                    line_kws={"color": COLORS["red"]})
        ax.set_title(f"{etiqueta} vs. puntaje\nPearson r = {r:.3f} · p = {p:.3g}")
        ax.set_xlabel(etiqueta)
        ax.set_ylabel("Puntaje")

    savefig(fig, "C02_scatter_numericas.png", "scatter_numericas")


def grafico_correlacion(df: pd.DataFrame) -> pd.DataFrame | None:
    """Heatmap de correlación de Pearson entre variables numéricas."""
    candidatas = ["puntaje_obtenido", "tiempo_usado_segundos", "edad_calculada",
                  "estrato", "grado_escolar", "interes_prog_robotica",
                  "n_herramientas", "n_areas_interes",
                  "nivel_programacion_ord", "nivel_robotica_ord",
                  "cambios_pestana", "intentos_copiar", "intentos_pegar",
                  "intentos_click_derecho"]
    cols = [c for c in candidatas if c in df.columns
            and df[c].notna().sum() > 10]
    if len(cols) < 2:
        return None
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 1.0 * len(cols) + 1))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=STEM_DIVERGING,
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Matriz de correlación (Pearson) — variables numéricas")
    savefig(fig, "C03_correlacion_heatmap.png", "correlacion")
    return corr


# =============================================================================
# SECCIÓN D — ANÁLISIS SOCIOECONÓMICO Y DE BRECHAS
# =============================================================================

def grafico_brechas_acceso(df: pd.DataFrame) -> dict:
    """
    Compara el puntaje según acceso a computador e internet, y por estrato /
    con_quién_vive. Devuelve las pruebas estadísticas asociadas.
    """
    resultados = {}
    paneles = []
    for col, titulo in [("tiene_computador", "Computador en casa"),
                        ("tiene_internet", "Internet en casa"),
                        ("estrato", "Estrato socioeconómico"),
                        ("con_quien_vive", "Con quién vive")]:
        if col in df.columns and df[[col, "puntaje_obtenido"]].dropna().shape[0] > 0:
            prueba = prueba_grupo(df, col)
            if prueba:
                resultados[col] = prueba
                paneles.append((col, titulo))

    if paneles:
        ncols = 2
        nrows = int(np.ceil(len(paneles) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, (col, titulo) in zip(axes, paneles):
            sub = df[[col, "puntaje_obtenido"]].dropna()
            orden = sorted(sub[col].unique(), key=str)
            sns.boxplot(data=sub, x=col, y="puntaje_obtenido", order=orden,
                        palette=PALETTE, ax=ax)
            p = resultados[col]["p_value"]
            ax.set_title(f"{titulo}\np = {p:.3g}")
            ax.set_xlabel("")
            ax.set_ylabel("Puntaje")
            ax.tick_params(axis="x", rotation=20)
        for ax in axes[len(paneles):]:
            ax.set_visible(False)
        fig.suptitle("Análisis socioeconómico — brechas de acceso y puntaje",
                     fontsize=15, fontweight="bold")
        savefig(fig, "D01_brechas_socioeconomicas.png", "brechas")

    return resultados


def grafico_estrato_tendencia(df: pd.DataFrame) -> None:
    """Puntaje promedio por estrato con barra de error (IC aproximado)."""
    if "estrato" not in df.columns:
        return
    sub = df[["estrato", "puntaje_obtenido"]].dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.pointplot(data=sub, x="estrato", y="puntaje_obtenido",
                  color=COLORS["blue"], capsize=0.15, ax=ax)
    ax.set_title("Puntaje promedio por estrato socioeconómico (IC 95%)")
    ax.set_xlabel("Estrato")
    ax.set_ylabel("Puntaje promedio")
    savefig(fig, "D02_estrato_tendencia.png", "estrato_tendencia")


# =============================================================================
# SECCIÓN E — ANÁLISIS DE TELEMETRÍA (comportamiento durante el examen)
# =============================================================================

def analisis_telemetria(df: pd.DataFrame) -> dict:
    """
    Explora la relación entre señales de comportamiento (cambios de pestaña,
    intentos de copiar/pegar/clic derecho) y el puntaje. Estas señales pueden
    indicar posible fraude académico.
    """
    tele_cols = [c for c in ["cambios_pestana", "intentos_copiar",
                             "intentos_pegar", "intentos_click_derecho"]
                 if c in df.columns]
    resultados = {"correlaciones": {}, "comparacion_pestana": None}
    if not tele_cols:
        return resultados

    sub = df[df[tele_cols].notna().any(axis=1)].copy()
    if sub.empty:
        return resultados

    # --- Correlación de cada señal con el puntaje --------------------------
    for c in tele_cols:
        d = sub[[c, "puntaje_obtenido"]].dropna()
        if len(d) > 10 and d[c].nunique() > 1:
            r, p = stats.spearmanr(d[c], d["puntaje_obtenido"])
            resultados["correlaciones"][c] = {"rho": float(r), "p": float(p),
                                              "n": int(len(d))}

    # --- ¿Cambiar de pestaña se asocia a mejor/peor puntaje? ---------------
    if "cambios_pestana" in sub.columns:
        d = sub[["cambios_pestana", "puntaje_obtenido"]].dropna()
        con = d[d["cambios_pestana"] > 0]["puntaje_obtenido"]
        sin = d[d["cambios_pestana"] == 0]["puntaje_obtenido"]
        if len(con) >= 10 and len(sin) >= 10:
            stat, p = stats.ttest_ind(con, sin, equal_var=False)
            resultados["comparacion_pestana"] = {
                "media_con_cambios": float(con.mean()),
                "media_sin_cambios": float(sin.mean()),
                "n_con": int(len(con)),
                "n_sin": int(len(sin)),
                "p_value": float(p),
                "cohens_d": _cohens_d(con.values, sin.values),
            }

    # --- Gráfico: barras de correlación + boxplot pestaña ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if resultados["correlaciones"]:
        etiquetas = list(resultados["correlaciones"].keys())
        rhos = [resultados["correlaciones"][c]["rho"] for c in etiquetas]
        cols_bar = [COLORS["red"] if r > 0 else COLORS["green"] for r in rhos]
        ax1.barh(etiquetas, rhos, color=cols_bar, edgecolor="white")
        ax1.axvline(0, color="black", linewidth=0.8)
        ax1.set_title("Correlación (Spearman ρ) señal de telemetría vs. puntaje")
        ax1.set_xlabel("ρ de Spearman")
    else:
        ax1.set_visible(False)

    if "cambios_pestana" in sub.columns:
        d = sub[["cambios_pestana", "puntaje_obtenido"]].dropna()
        d["cambió_pestaña"] = np.where(d["cambios_pestana"] > 0, "Sí", "No")
        sns.boxplot(data=d, x="cambió_pestaña", y="puntaje_obtenido",
                    order=["No", "Sí"], palette=[COLORS["green"], COLORS["red"]],
                    ax=ax2)
        ax2.set_title("Puntaje según si cambió de pestaña")
        ax2.set_xlabel("¿Cambió de pestaña al menos una vez?")
        ax2.set_ylabel("Puntaje")

    fig.suptitle("Análisis de telemetría — comportamiento durante el examen",
                 fontsize=15, fontweight="bold")
    savefig(fig, "E01_telemetria.png", "telemetria")

    # --- Tiempo (quintiles) vs puntaje: ¿los rápidos rinden distinto? ------
    if "tiempo_usado_segundos" in sub.columns:
        d = sub[["tiempo_usado_segundos", "puntaje_obtenido"]].dropna()
        if len(d) > 25:
            d["quintil_tiempo"] = pd.qcut(d["tiempo_usado_segundos"], 5,
                                          labels=["Q1 (más rápido)", "Q2", "Q3",
                                                  "Q4", "Q5 (más lento)"],
                                          duplicates="drop")
            fig2, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=d, x="quintil_tiempo", y="puntaje_obtenido",
                        palette=PALETTE, ax=ax)
            ax.set_title("Puntaje por quintil de tiempo de examen")
            ax.set_xlabel("Quintil de tiempo usado")
            ax.set_ylabel("Puntaje")
            ax.tick_params(axis="x", rotation=15)
            savefig(fig2, "E02_tiempo_quintiles.png", "tiempo_quintiles")
            resultados["quintiles_tiempo"] = (
                d.groupby("quintil_tiempo")["puntaje_obtenido"]
                .mean().round(2).to_dict())

    return resultados


# =============================================================================
# SECCIÓN G — GENERACIÓN DEL INFORME MARKDOWN
# =============================================================================

def tabla_md(df: pd.DataFrame, max_filas: int = 30) -> str:
    """Convierte un DataFrame a tabla markdown (recortada si es muy larga)."""
    d = df.head(max_filas)
    encabezado = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    filas = ["| " + " | ".join(str(v) for v in row) + " |"
             for row in d.itertuples(index=False)]
    return "\n".join([encabezado, sep] + filas)


def construir_informe(ctx: dict) -> None:
    """Ensambla y escribe reports/01_analisis_exploratorio.md."""
    log("SECCIÓN G — Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    est = ctx["est_puntaje"]
    pruebas_bi = ctx["pruebas_bivariado"]
    pruebas_socio = ctx["pruebas_socio"]
    tele = ctx["telemetria"]

    # Imagen: ruta relativa desde reports/ hacia outputs/
    def img(key: str, alt: str) -> str:
        f = FIGURES.get(key)
        return (f"![{alt}](../outputs/{f})" if f
                else f"_(figura '{alt}' no disponible)_")

    # ---- Resumen ejecutivo: hallazgos clave dinámicos ----------------------
    # Variable categórica con mayor efecto significativo en el puntaje.
    signif = [(k, v) for k, v in {**pruebas_bi, **pruebas_socio}.items()
              if v["p_value"] < 0.05]
    signif.sort(key=lambda kv: kv[1]["p_value"])
    top_factores = ", ".join(f"`{k}`" for k, _ in signif[:4]) or "ninguno claro"

    forma = ("simétrica" if abs(est["asimetria"]) < 0.5
             else ("asimétrica a la derecha" if est["asimetria"] > 0
                   else "asimétrica a la izquierda"))

    resumen = dedent(f"""\
        La muestra analizada corresponde a **{est['n']:,} estudiantes** que
        presentaron la prueba Copa STEM 2026. El puntaje promedio es de
        **{est['media']:.1f}/100** (mediana {est['mediana']:.1f}, desviación
        estándar {est['std']:.1f}), con una distribución **{forma}**
        (asimetría = {est['asimetria']:.2f}). Las variables con diferencias
        estadísticamente significativas en el puntaje fueron: {top_factores}.
        El análisis socioeconómico y de telemetría permite priorizar hipótesis
        sobre brechas de acceso y sobre comportamientos atípicos durante el
        examen, insumos para las siguientes fases de modelado.""")

    REPORT.append(f"# Análisis Exploratorio — Copa STEM 2026\n")
    REPORT.append(f"**Fundación SapienceLab** · Informe generado: {fecha}\n")
    REPORT.append("---\n")
    REPORT.append("## Resumen ejecutivo\n")
    REPORT.append(resumen + "\n")

    # ---- Metodología general ----------------------------------------------
    REPORT.append("## Metodología general\n")
    REPORT.append(dedent(f"""\
        El estudio sigue el flujo estándar de un Análisis Exploratorio de Datos
        (EDA): (1) carga y control de calidad, (2) limpieza y derivación de
        variables, (3) análisis univariado, (4) análisis bivariado con pruebas
        de significancia, (5) análisis socioeconómico de brechas y (6) análisis
        de telemetría de comportamiento. Todo el procesamiento es reproducible
        (`random_state=42`). Las pruebas de hipótesis usan un nivel de
        significancia **α = 0.05**. Para comparaciones entre **dos** grupos se
        emplea el **t-test de Welch** (robusto a varianzas desiguales) con el
        tamaño del efecto **Cohen's d**; para **tres o más** grupos, **ANOVA de
        una vía** con **eta²**. Las correlaciones lineales usan **Pearson** y
        las de telemetría (variables sesgadas/no normales) **Spearman**.\n"""))

    # ---- A. Calidad de datos ----------------------------------------------
    REPORT.append("## A. Calidad de datos y limpieza\n")
    REPORT.append(f"- Registros crudos: **{ctx['n_crudos']:,}**\n"
                  f"- Registros tras eliminar pruebas y limpiar: "
                  f"**{ctx['n_limpio']:,}**\n"
                  f"- Estudiantes que **presentaron** (con puntaje): "
                  f"**{ctx['n_presentaron']:,}**\n"
                  f"- Estudiantes **sin puntaje**: **{ctx['n_pendientes']:,}**\n")
    REPORT.append("\n**Perfil de calidad por columna** (tipo y % de nulos):\n")
    REPORT.append(tabla_md(ctx["perfil"]) + "\n")

    # ---- B. Univariado -----------------------------------------------------
    REPORT.append("## B. Análisis univariado\n")
    REPORT.append("**Pregunta:** ¿Cómo se distribuyen el puntaje y las "
                  "principales variables de la muestra?\n")
    REPORT.append("**Estadísticas del puntaje obtenido:**\n")
    REPORT.append(tabla_md(pd.DataFrame({
        "métrica": ["N", "Media", "Mediana", "Desv. estándar", "Mínimo",
                    "Q1", "Q3", "Máximo", "Asimetría", "Curtosis (exceso)"],
        "valor": [est["n"], round(est["media"], 2), round(est["mediana"], 2),
                  round(est["std"], 2), round(est["min"], 2),
                  round(est["q1"], 2), round(est["q3"], 2), round(est["max"], 2),
                  round(est["asimetria"], 3), round(est["curtosis"], 3)],
    })) + "\n")
    REPORT.append(f"\n{img('dist_puntaje', 'Distribución del puntaje')}\n")
    REPORT.append(f"\n{img('cat_univariado', 'Variables categóricas')}\n")
    if "dist_tiempo" in FIGURES:
        REPORT.append(f"\n{img('dist_tiempo', 'Distribución del tiempo')}\n")
    REPORT.append(dedent(f"""\
        **Interpretación.** La asimetría ({est['asimetria']:.2f}) y la curtosis
        ({est['curtosis']:.2f}) indican cuán alejada está la distribución de la
        normalidad; valores cercanos a 0 sugieren simetría/mesocurtosis. Esto
        justifica el uso de pruebas robustas y correlaciones de Spearman en las
        secciones siguientes.\n"""))

    # ---- C. Bivariado ------------------------------------------------------
    REPORT.append("## C. Análisis bivariado y pruebas de significancia\n")
    REPORT.append("**Pregunta:** ¿El puntaje difiere significativamente según "
                  "las características demográficas y académicas?\n")
    if pruebas_bi:
        filas = []
        for col, r in pruebas_bi.items():
            filas.append({
                "Variable": col,
                "Prueba": r["prueba"],
                "Estadístico": round(r["estadistico"], 3),
                "p-value": f"{r['p_value']:.3g}",
                r["efecto_nombre"]: round(r["efecto"], 3),
                "Significativo (α=0.05)": "Sí" if r["p_value"] < 0.05 else "No",
            })
        REPORT.append(tabla_md(pd.DataFrame(filas)) + "\n")
    REPORT.append(f"\n{img('boxplots_bivariado', 'Box plots bivariados')}\n")
    REPORT.append(f"\n{img('scatter_numericas', 'Dispersión numéricas')}\n")
    REPORT.append(f"\n{img('correlacion', 'Matriz de correlación')}\n")
    if ctx.get("corr_puntaje"):
        top = ctx["corr_puntaje"]
        REPORT.append("\n**Variables más correlacionadas con el puntaje "
                      "(|Pearson r|):**\n")
        REPORT.append(tabla_md(pd.DataFrame(top)) + "\n")
    REPORT.append(dedent("""\
        **Interpretación.** Un p-value < 0.05 indica que la diferencia de
        puntaje entre grupos es poco probable por azar. El tamaño del efecto
        (Cohen's d / eta²) matiza la magnitud práctica: diferencias
        significativas pero con efecto pequeño deben leerse con cautela.\n"""))

    # ---- D. Socioeconómico -------------------------------------------------
    REPORT.append("## D. Análisis socioeconómico y de brechas\n")
    REPORT.append("**Pregunta:** ¿Los estudiantes con menos recursos "
                  "(sin computador/internet, estratos bajos) rinden menos?\n")
    if pruebas_socio:
        filas = []
        for col, r in pruebas_socio.items():
            medias_txt = "; ".join(f"{k}: {v:.1f}"
                                   for k, v in r["medias"].items())
            filas.append({
                "Variable": col,
                "Prueba": r["prueba"],
                "p-value": f"{r['p_value']:.3g}",
                r["efecto_nombre"]: round(r["efecto"], 3),
                "Medias por grupo": medias_txt,
            })
        REPORT.append(tabla_md(pd.DataFrame(filas)) + "\n")
    REPORT.append(f"\n{img('brechas', 'Brechas socioeconómicas')}\n")
    REPORT.append(f"\n{img('estrato_tendencia', 'Tendencia por estrato')}\n")
    REPORT.append(dedent("""\
        **Interpretación.** Si el puntaje crece de forma monótona con el estrato
        o es mayor entre quienes tienen computador/internet, hay evidencia de
        una **brecha de acceso** que la Fundación puede atender con
        intervenciones focalizadas. La magnitud del efecto indica la prioridad.\n"""))

    # ---- E. Telemetría -----------------------------------------------------
    REPORT.append("## E. Análisis de telemetría (comportamiento)\n")
    REPORT.append("**Pregunta:** ¿Las señales de comportamiento (cambios de "
                  "pestaña, copiar/pegar) se asocian con el puntaje? "
                  "¿Los más rápidos rinden distinto?\n")
    if tele.get("correlaciones"):
        filas = [{"Señal": c, "Spearman ρ": round(v["rho"], 3),
                  "p-value": f"{v['p']:.3g}", "N": v["n"]}
                 for c, v in tele["correlaciones"].items()]
        REPORT.append(tabla_md(pd.DataFrame(filas)) + "\n")
    if tele.get("comparacion_pestana"):
        cp = tele["comparacion_pestana"]
        REPORT.append(dedent(f"""\

            **Cambios de pestaña vs. puntaje:** los estudiantes que cambiaron de
            pestaña al menos una vez (N={cp['n_con']}) obtuvieron en promedio
            **{cp['media_con_cambios']:.1f}**, frente a **{cp['media_sin_cambios']:.1f}**
            de quienes no lo hicieron (N={cp['n_sin']}). Diferencia con
            p = {cp['p_value']:.3g} (Cohen's d = {cp['cohens_d']:.3f}).\n"""))
    REPORT.append(f"\n{img('telemetria', 'Telemetría')}\n")
    if "tiempo_quintiles" in FIGURES:
        REPORT.append(f"\n{img('tiempo_quintiles', 'Puntaje por quintil de tiempo')}\n")
    REPORT.append(dedent("""\
        **Interpretación.** Correlaciones positivas entre señales de
        copiar/pegar/cambio de pestaña y el puntaje serían una **señal de alerta
        de posible fraude** a investigar en la Fase 3 (detección de anomalías).
        La relación entre tiempo y puntaje ayuda a distinguir a quienes abandonan
        pronto (bajo puntaje) de quienes resuelven con eficiencia.\n"""))

    # ---- Conclusiones ------------------------------------------------------
    REPORT.append("## Conclusiones y recomendaciones\n")
    REPORT.append(dedent(f"""\
        1. **Factores asociados al rendimiento.** Los factores con diferencias
           significativas ({top_factores}) deben incluirse como variables
           candidatas en el modelo de predicción de puntaje (Fase 2,
           `03_prediccion_puntaje.py`).
        2. **Brechas de acceso.** Priorizar el análisis dedicado de brechas
           (`02_analisis_brechas.py`) para cuantificar el efecto neto del acceso
           a tecnología controlando por otras variables.
        3. **Integridad del examen.** Las señales de telemetría con correlación
           positiva justifican un modelo de detección de anomalías
           (`07_deteccion_anomalias.py`).
        4. **Segmentación.** La heterogeneidad observada sugiere construir
           perfiles de estudiante mediante clustering (`06_clustering_estudiantes.py`).\n"""))

    # ---- Limitaciones ------------------------------------------------------
    REPORT.append("## Limitaciones del estudio\n")
    REPORT.append(dedent("""\
        - **Datos observacionales:** las asociaciones detectadas **no implican
          causalidad**; factores no medidos (calidad docente, motivación) pueden
          confundir los resultados.
        - **Telemetría parcial:** solo existe para exámenes en plataforma; los
          exámenes escritos tienen estos campos vacíos, lo que puede sesgar el
          análisis de comportamiento.
        - **Autoreporte:** variables socioeconómicas (estrato, acceso a
          tecnología, con quién vive) son autorreportadas y sujetas a error.
        - **Comparaciones múltiples:** se realizan varias pruebas de hipótesis;
          conviene aplicar correcciones (p. ej. Bonferroni) antes de conclusiones
          definitivas.
        - **Tamaños de grupo desiguales:** algunas categorías tienen pocos casos,
          reduciendo la potencia estadística (se exigió N≥15 por grupo).\n"""))

    # ---- Referencias -------------------------------------------------------
    REPORT.append("## Referencias técnicas\n")
    REPORT.append(dedent("""\
        - Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
        - Welch, B. L. (1947). "The generalization of 'Student's' problem when
          several different population variances are involved." *Biometrika*.
        - Fisher, R. A. (1925). *Statistical Methods for Research Workers*
          (ANOVA).
        - Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
          Sciences* (Cohen's d, eta²).
        - Spearman, C. (1904). "The proof and measurement of association between
          two things." *American Journal of Psychology*.
        - McKinney, W. (2010). *Data Structures for Statistical Computing in
          Python* (pandas). · Virtanen et al. (2020). *SciPy 1.0* (Nature
          Methods). · Waskom, M. (2021). *seaborn* (JOSS).\n"""))

    REPORT.append("\n---\n_Generado automáticamente por "
                  "`notebooks/01_analisis_exploratorio.py` — Copa STEM 2026._\n")

    destino = REPORTS_DIR / "01_analisis_exploratorio.md"
    destino.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"    informe escrito → reports/{destino.name}")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Análisis Exploratorio de Datos")
    print(" Fundación SapienceLab")
    print("=" * 70)

    # --- A. Carga y limpieza ------------------------------------------------
    df_crudo = cargar_datos()
    n_crudos = len(df_crudo)
    perfil = perfil_calidad(df_crudo)

    print("\n--- Perfil de calidad (crudo) ---")
    print(perfil.to_string(index=False))
    print()

    df = limpiar_datos(df_crudo)
    n_limpio = len(df)
    presentaron, pendientes = separar_presentaron(df)

    if presentaron.empty:
        log("⚠ No hay estudiantes con puntaje; no se puede continuar el EDA.")
        sys.exit(0)

    # A partir de aquí trabajamos con quienes presentaron el examen.
    dfp = presentaron

    # --- B. Univariado ------------------------------------------------------
    log("SECCIÓN B — Análisis univariado")
    est_puntaje = estadisticas_puntaje(dfp)
    print(f"    puntaje: media={est_puntaje['media']:.2f} "
          f"mediana={est_puntaje['mediana']:.2f} std={est_puntaje['std']:.2f} "
          f"asimetría={est_puntaje['asimetria']:.2f} "
          f"curtosis={est_puntaje['curtosis']:.2f}")
    grafico_distribucion_puntaje(dfp, est_puntaje)
    grafico_categoricas_univariado(dfp)
    grafico_tiempo(dfp)

    # --- C. Bivariado -------------------------------------------------------
    log("SECCIÓN C — Análisis bivariado y pruebas estadísticas")
    cols_categoricas = ["municipio", "grado_escolar", "genero",
                        "tipo_institucion", "estrato", "jornada"]
    pruebas_bivariado = {}
    for col in cols_categoricas:
        r = prueba_grupo(dfp, col)
        if r:
            pruebas_bivariado[col] = r
            print(f"    {col:18s} → {r['prueba']:16s} "
                  f"p={fmt_p(r['p_value'])}")
    grafico_boxplots_bivariado(dfp, pruebas_bivariado)
    grafico_scatter(dfp)
    corr = grafico_correlacion(dfp)

    # Top de correlaciones con el puntaje (para el informe).
    corr_puntaje = None
    if corr is not None and "puntaje_obtenido" in corr.columns:
        serie = (corr["puntaje_obtenido"].drop(labels=["puntaje_obtenido"])
                 .dropna().sort_values(key=lambda s: s.abs(), ascending=False))
        corr_puntaje = [{"Variable": k, "Pearson r": round(v, 3)}
                        for k, v in serie.items()]

    # --- D. Socioeconómico --------------------------------------------------
    log("SECCIÓN D — Análisis socioeconómico y de brechas")
    pruebas_socio = grafico_brechas_acceso(dfp)
    for col, r in pruebas_socio.items():
        print(f"    {col:18s} → p={fmt_p(r['p_value'])}")
    grafico_estrato_tendencia(dfp)

    # --- E. Telemetría ------------------------------------------------------
    log("SECCIÓN E — Análisis de telemetría")
    telemetria = analisis_telemetria(dfp)
    for c, v in telemetria.get("correlaciones", {}).items():
        print(f"    {c:22s} → Spearman ρ={v['rho']:+.3f} "
              f"(p={v['p']:.3g}, N={v['n']})")

    # --- G. Informe ---------------------------------------------------------
    ctx = {
        "n_crudos": n_crudos,
        "n_limpio": n_limpio,
        "n_presentaron": len(presentaron),
        "n_pendientes": len(pendientes),
        "perfil": perfil,
        "est_puntaje": est_puntaje,
        "pruebas_bivariado": pruebas_bivariado,
        "pruebas_socio": pruebas_socio,
        "telemetria": telemetria,
        "corr_puntaje": corr_puntaje,
    }
    construir_informe(ctx)

    print("\n" + "=" * 70)
    print(f" ✔ EDA COMPLETADO")
    print(f"   · Figuras generadas: {len(FIGURES)}  → outputs/")
    print(f"   · Informe:            reports/01_analisis_exploratorio.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
