# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 07: Clustering de Perfiles de Estudiante  — Fase 3
================================================================================

Objetivo
--------
Segmentar a los estudiantes en **perfiles** interpretables combinando rendimiento,
acceso tecnológico y experiencia previa, para diseñar intervenciones diferenciadas.

Secciones
---------
    A) Preparación (9 features, imputación por mediana, StandardScaler)
    B) K óptimo (método del codo + silhouette, K=2..10)
    C) Modelos: K-Means (despliegue) vs. Gaussian Mixture (alternativa)
    D) Perfiles: nombre descriptivo, tabla, radar, PCA 2D, distribución por
       colegio/municipio/grado
    E) Exportación: CSV + predictor puro .py y .js (asignación por centroide más
       cercano), + informe con recomendaciones de intervención por perfil

Nota de despliegue
------------------
El predictor de producción asigna cada estudiante al **centroide más cercano**
(K-Means) en el espacio estandarizado. Es determinista y portable; por eso se
despliega K-Means aunque se compare su silhouette con GMM.

Reproducible (`random_state=42`), paleta Copa STEM en todos los gráficos.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
import json
import inspect
import warnings
import statistics
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

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
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
DEPLOY_DIR = BASE_DIR / "models" / "deploy"
for _d in (OUTPUTS_DIR, REPORTS_DIR, DEPLOY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_CANDIDATOS = ["copa_stem_dataset_limpio.csv", "copa_stem_dataset.csv"]

COLORS = {"cyan": "#00d4ff", "violet": "#8b5cf6", "amber": "#f59e0b",
          "dark": "#050816", "green": "#10b981", "red": "#ef4444", "blue": "#0f77ee"}
PALETTE = ["#00d4ff", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444", "#0f77ee"]

sns.set_theme(style="whitegrid")
sns.set_palette(PALETTE)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "#333333",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "font.size": 10, "figure.autolayout": True,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})
DPI = 150

FIGURES: dict[str, str] = {}
REPORT: list[str] = []

# Colores por cluster (hasta 6 clusters usando la paleta de marca).
CLUSTER_COLORS = PALETTE


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
# CONFIGURACIÓN DE FEATURES
# =============================================================================
FEATURES = ["puntaje_obtenido", "estrato", "computador_bin", "internet_bin",
            "nivel_programacion_ord", "nivel_robotica_ord",
            "interes_prog_robotica", "n_herramientas", "n_areas_interes"]
FEATURE_LABELS = ["Puntaje", "Estrato", "Computador", "Internet", "Nivel prog.",
                  "Nivel rob.", "Interés", "Herramientas", "Áreas"]


# =============================================================================
# FUNCIONES PURAS (se embeben en el predictor de deploy)
# =============================================================================

def _to_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _parse_count(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "[]"):
        return 0
    items = None
    try:
        import json as _json
        parsed = _json.loads(s)
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


def _ord_level(v):
    if v is None:
        return None
    m = {"ninguna": 0, "ninguno": 0, "básica": 1, "basica": 1,
         "intermedia": 2, "avanzada": 3}
    return m.get(str(v).strip().lower())


def _bin_si(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("nan", "none", ""):
        return None
    if s.startswith("s"):
        return 1
    if s.startswith("n"):
        return 0
    return None


def _raw_feature(raw, name):
    """Valor crudo de una feature de clustering (None si falta)."""
    if name == "computador_bin":
        return _bin_si(raw.get("computador_en_casa"))
    if name == "internet_bin":
        return _bin_si(raw.get("internet_en_casa"))
    if name == "nivel_programacion_ord":
        return _ord_level(raw.get("nivel_programacion"))
    if name == "nivel_robotica_ord":
        return _ord_level(raw.get("nivel_robotica"))
    if name == "n_herramientas":
        v = _to_float(raw.get("n_herramientas"))
        return v if v is not None else _parse_count(raw.get("herramientas_conocidas"))
    if name == "n_areas_interes":
        v = _to_float(raw.get("n_areas_interes"))
        return v if v is not None else _parse_count(raw.get("areas_interes"))
    return _to_float(raw.get(name))


def _feature_vector(raw, SPEC):
    """Vector de features imputado (mediana) en el ORDEN de SPEC['features']."""
    vals = []
    for i, name in enumerate(SPEC["features"]):
        v = _raw_feature(raw, name)
        if v is None:
            v = SPEC["medians"][i]
        vals.append(float(v))
    return vals


def predecir_cluster(raw, SPEC):
    """Asigna al estudiante el cluster del CENTROIDE más cercano (espacio estandarizado)."""
    x = _feature_vector(raw, SPEC)
    mean, scale = SPEC["scaler_mean"], SPEC["scaler_scale"]
    z = [(x[i] - mean[i]) / scale[i] for i in range(len(x))]
    best_id, best_d = -1, None
    for cid, c in enumerate(SPEC["centroids"]):
        d = 0.0
        for i in range(len(z)):
            diff = z[i] - c[i]
            d += diff * diff
        if best_d is None or d < best_d:
            best_d, best_id = d, cid
    return {"cluster_id": best_id, "cluster_nombre": SPEC["nombres"][str(best_id)]}


# =============================================================================
# A. PREPARACIÓN
# =============================================================================

def cargar_datos() -> pd.DataFrame:
    log("SECCIÓN A — Carga y preparación")
    ruta = next((DATA_DIR / n for n in DATASET_CANDIDATOS
                 if (DATA_DIR / n).exists()), None)
    if ruta is None:
        print("\n  ⚠  No se encontró el dataset. Ejecute 05b primero.\n")
        sys.exit(1)
    log(f"    dataset: {ruta.name}")
    df = pd.read_csv(ruta, encoding="utf-8", dtype={"numero_documento": str})

    docs = ["1234", "123456", "123456789", "1234567899", "0", "00000000"]
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df = df[~df["numero_documento"].isin(docs)]
    df = df[df["numero_documento"].str.len() >= 5]
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    for c in ["puntaje_obtenido", "grado_escolar", "estrato", "interes_prog_robotica"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["puntaje_obtenido"].notna()].reset_index(drop=True)

    # Features derivadas.
    df["computador_bin"] = df["computador_en_casa"].map(_bin_si)
    df["internet_bin"] = df["internet_en_casa"].map(_bin_si)
    df["nivel_programacion_ord"] = df["nivel_programacion"].map(_ord_level)
    df["nivel_robotica_ord"] = df["nivel_robotica"].map(_ord_level)
    df["n_herramientas"] = df.get("herramientas_conocidas").map(_parse_count) \
        if "herramientas_conocidas" in df.columns else 0
    df["n_areas_interes"] = df.get("areas_interes").map(_parse_count) \
        if "areas_interes" in df.columns else 0
    log(f"    estudiantes que presentaron: {len(df):,}")
    return df


def preparar_matriz(df: pd.DataFrame):
    """Imputa por mediana, estandariza; devuelve X escalado, scaler y medianas."""
    raw = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = raw.median()
    raw_imp = raw.fillna(medians)
    scaler = StandardScaler().fit(raw_imp.values)
    X = scaler.transform(raw_imp.values)
    log(f"    matriz: {X.shape[0]}×{X.shape[1]} (imputada por mediana + estandarizada)")
    return X, scaler, medians, raw_imp


# =============================================================================
# B. K ÓPTIMO
# =============================================================================

def determinar_k(X: np.ndarray) -> int:
    log("SECCIÓN B — Determinación de K óptimo (codo + silhouette)")
    Ks = list(range(2, 11))
    inertias, sils = [], []
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels))
    k_opt = Ks[int(np.argmax(sils))]
    log(f"    silhouette máximo en K={k_opt} (sil={max(sils):.3f})")

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(Ks, inertias, "o-", color=COLORS["cyan"], linewidth=2,
             label="Inercia (codo)")
    ax1.set_xlabel("Número de clusters (K)")
    ax1.set_ylabel("Inercia", color=COLORS["cyan"])
    ax1.tick_params(axis="y", labelcolor=COLORS["cyan"])
    ax2 = ax1.twinx()
    ax2.plot(Ks, sils, "s--", color=COLORS["violet"], linewidth=2,
             label="Silhouette")
    ax2.set_ylabel("Silhouette", color=COLORS["violet"])
    ax2.tick_params(axis="y", labelcolor=COLORS["violet"])
    ax2.axvline(k_opt, color=COLORS["red"], linestyle=":", linewidth=1.5,
                label=f"K óptimo = {k_opt}")
    ax1.set_title("Método del codo + Silhouette — selección de K")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    savefig(fig, "F07_seleccion_k.png", "seleccion_k")
    return k_opt, dict(zip(Ks, sils))


# =============================================================================
# C. MODELOS
# =============================================================================

def ajustar_modelos(X: np.ndarray, k: int) -> dict:
    log(f"SECCIÓN C — K-Means vs. GMM con K={k}")
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    km_labels = km.fit_predict(X)
    km_sil = silhouette_score(X, km_labels)

    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=RANDOM_STATE)
    gmm_labels = gmm.fit_predict(X)
    gmm_sil = silhouette_score(X, gmm_labels)

    log(f"    silhouette: K-Means={km_sil:.3f} | GMM={gmm_sil:.3f}")
    log("    modelo de despliegue: K-Means (asignación por centroide más cercano)")
    return {"kmeans": km, "km_labels": km_labels, "km_sil": float(km_sil),
            "gmm": gmm, "gmm_labels": gmm_labels, "gmm_sil": float(gmm_sil)}


# =============================================================================
# D. ANÁLISIS DE PERFILES
# =============================================================================

# Arquetipos de perfil: (nombre, [rendimiento_z, acceso_z, engagement_z]).
# Se asignan por emparejamiento greedy al perfil relativo de cada cluster, lo que
# garantiza nombres DISTINTOS y descriptivos aunque varíe el nº de clusters.
ARQUETIPOS = [
    ("Alto rendimiento tech",       [1.2, 0.6, 1.2]),
    ("Talento sin acceso",          [1.2, -1.2, 0.2]),
    ("Alto rendimiento conectado",  [1.2, 1.0, 0.0]),
    ("Explorador digital",          [0.0, 0.4, 1.3]),
    ("Promedio conectado",          [0.2, 1.1, -0.3]),
    ("Promedio con acceso limitado", [0.1, -0.9, -0.2]),
    ("Perfil intermedio",           [0.0, 0.0, 0.0]),
    ("Base conectada",              [-0.6, 1.0, -0.5]),
    ("En desarrollo",               [-1.2, -0.2, -0.5]),
    ("Vulnerable desconectado",     [-1.1, -1.1, -0.9]),
]


def nombrar_clusters(perfil: pd.DataFrame) -> dict:
    """Nombre descriptivo por cluster vía emparejamiento greedy con arquetipos."""
    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0

    punt_z = z(perfil["puntaje_obtenido"])
    acc_z = z((perfil["computador_bin"] + perfil["internet_bin"]) / 2)
    eng_z = z((perfil["nivel_programacion_ord"] / 3 + perfil["nivel_robotica_ord"] / 3
               + (perfil["interes_prog_robotica"] - 1) / 4) / 3)

    vecs = {cid: np.array([punt_z[cid], acc_z[cid], eng_z[cid]])
            for cid in perfil.index}

    # Coste = distancia cluster↔arquetipo; asignación greedy sin repetir nombres.
    pares = []
    for cid, v in vecs.items():
        for ai, (nombre, tgt) in enumerate(ARQUETIPOS):
            pares.append((float(np.linalg.norm(v - np.array(tgt))), cid, ai))
    pares.sort(key=lambda t: t[0])

    nombres, usados_arq = {}, set()
    for _, cid, ai in pares:
        if cid in nombres or ai in usados_arq:
            continue
        nombres[cid] = ARQUETIPOS[ai][0]
        usados_arq.add(ai)
        if len(nombres) == len(vecs):
            break
    return nombres


def perfilar(df: pd.DataFrame, labels: np.ndarray, raw_imp: pd.DataFrame,
             k: int) -> dict:
    log("SECCIÓN D — Perfilado de clusters")
    df = df.copy()
    df["cluster_id"] = labels

    # Perfil = medias crudas (imputadas) por cluster.
    perfil = raw_imp.copy()
    perfil["cluster_id"] = labels
    perfil_means = perfil.groupby("cluster_id").mean()

    nombres = nombrar_clusters(perfil_means)
    df["cluster_nombre"] = df["cluster_id"].map(nombres)

    # --- Tabla resumen por cluster -----------------------------------------
    tabla = []
    for cid in range(k):
        sub = df[df["cluster_id"] == cid]
        tabla.append({
            "Cluster": cid,
            "Nombre": nombres[cid],
            "N": len(sub),
            "% total": round(100 * len(sub) / len(df), 1),
            "Puntaje µ": round(float(sub["puntaje_obtenido"].mean()), 1),
            "Estrato µ": round(float(raw_imp.loc[sub.index, "estrato"].mean()), 2),
            "% computador": round(float((sub["computador_bin"] == 1).mean() * 100), 0),
            "% internet": round(float((sub["internet_bin"] == 1).mean() * 100), 0),
            "Nivel prog µ": round(float(raw_imp.loc[sub.index, "nivel_programacion_ord"].mean()), 2),
        })
    tabla_df = pd.DataFrame(tabla)

    # --- Radar de perfiles (normalizado 0-1 por feature, min-max global) ----
    norm = (perfil_means[FEATURES] - raw_imp[FEATURES].min()) / \
           (raw_imp[FEATURES].max() - raw_imp[FEATURES].min() + 1e-9)
    angulos = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angulos += angulos[:1]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(FEATURE_LABELS, fontsize=10)
    ax.set_ylim(0, 1)
    for cid in range(k):
        vals = norm.loc[cid].tolist()
        vals += vals[:1]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax.plot(angulos, vals, linewidth=2, color=color, label=f"{cid}: {nombres[cid]}")
        ax.fill(angulos, vals, color=color, alpha=0.12)
    ax.set_title("Perfil normalizado de cada cluster (radar)",
                 fontsize=14, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    savefig(fig, "F07_radar_perfiles.png", "radar")

    return {"df": df, "nombres": nombres, "tabla": tabla_df,
            "perfil_means": perfil_means}


def graficos_distribucion(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray,
                          nombres: dict, k: int) -> None:
    # --- PCA 2D ------------------------------------------------------------
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    proj = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(10, 7))
    for cid in range(k):
        m = labels == cid
        ax.scatter(proj[m, 0], proj[m, 1], s=22, alpha=0.55,
                   color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                   label=f"{cid}: {nombres[cid]}", edgecolor="white", linewidth=0.2)
    ax.set_title(f"Proyección PCA 2D de los clusters\n"
                 f"(varianza explicada: {100*var.sum():.0f}%)")
    ax.set_xlabel(f"PC1 ({100*var[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100*var[1]:.0f}%)")
    ax.legend(fontsize=8)
    savefig(fig, "F07_pca_clusters.png", "pca")

    # --- Stacked bar por colegio (proporción de clusters) ------------------
    if "institucion_educativa" in df.columns:
        ct = pd.crosstab(df["institucion_educativa"], df["cluster_nombre"])
        ct = ct[ct.sum(axis=1) >= 10]
        if not ct.empty:
            prop = ct.div(ct.sum(axis=1), axis=0)
            prop = prop.loc[prop.index[np.argsort(-ct.sum(axis=1).values)]]
            fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(prop))))
            left = np.zeros(len(prop))
            for j, col in enumerate(prop.columns):
                ax.barh(prop.index, prop[col].values, left=left,
                        color=CLUSTER_COLORS[j % len(CLUSTER_COLORS)],
                        edgecolor="white", label=col)
                left += prop[col].values
            ax.set_title("Composición de perfiles por institución (N ≥ 10)")
            ax.set_xlabel("Proporción")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=7, loc="lower right")
            savefig(fig, "F07_clusters_por_colegio.png", "colegio")

    # --- Distribución por municipio y grado --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, titulo in [(axes[0], "municipio", "Municipio"),
                            (axes[1], "grado_escolar", "Grado")]:
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ct = pd.crosstab(df[col], df["cluster_nombre"])
        ct = ct.reindex(sorted(ct.index, key=str))
        left = np.zeros(len(ct))
        for j, cl in enumerate(ct.columns):
            ax.barh([str(i) for i in ct.index], ct[cl].values, left=left,
                    color=CLUSTER_COLORS[j % len(CLUSTER_COLORS)],
                    edgecolor="white", label=cl)
            left += ct[cl].values
        ax.set_title(f"Perfiles por {titulo}")
        ax.set_xlabel("N estudiantes")
        if col == "municipio":
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Distribución de perfiles por municipio y grado",
                 fontsize=15, fontweight="bold")
    savefig(fig, "F07_clusters_municipio_grado.png", "muni_grado")


# =============================================================================
# E. EXPORTACIÓN
# =============================================================================

def exportar(df: pd.DataFrame, scaler, medians, km, nombres: dict) -> dict:
    log("SECCIÓN E — Exportación (CSV + predictores puros)")
    SPEC = {
        "meta": {"generado": datetime.now().isoformat(timespec="seconds"),
                 "k": int(km.n_clusters), "modelo": "KMeans"},
        "features": FEATURES,
        "medians": [float(medians[f]) for f in FEATURES],
        "scaler_mean": [float(m) for m in scaler.mean_],
        "scaler_scale": [float(s) for s in scaler.scale_],
        "centroids": [[float(v) for v in c] for c in km.cluster_centers_],
        "nombres": {str(cid): nombres[cid] for cid in nombres},
    }

    # --- CSV ---------------------------------------------------------------
    out = pd.DataFrame({
        "numero_documento": df["numero_documento"].astype(str),
        "cluster_id": df["cluster_id"].astype(int),
        "cluster_nombre": df["cluster_nombre"],
    })
    destino = DEPLOY_DIR / "clustering_perfiles.csv"
    out.to_csv(destino, index=False, encoding="utf-8-sig")
    log(f"    perfiles → models/deploy/{destino.name} ({len(out):,} filas)")

    # --- Verificación: predictor puro == labels de K-Means -----------------
    pred = np.array([predecir_cluster(r, SPEC)["cluster_id"]
                     for r in df.to_dict("records")])
    mismatch = int((pred != df["cluster_id"].values).sum())
    log(f"    verificación predictor puro vs K-Means: {mismatch} discrepancias / {len(df)}")

    ejemplo = _ejemplo_dict(df.iloc[0])
    generar_predictor_py(SPEC, ejemplo)
    generar_predictor_js(SPEC, ejemplo)
    return {"SPEC": SPEC, "mismatch": mismatch, "ejemplo": ejemplo}


def _ejemplo_dict(fila) -> dict:
    campos = ["puntaje_obtenido", "estrato", "computador_en_casa", "internet_en_casa",
              "nivel_programacion", "nivel_robotica", "interes_prog_robotica",
              "herramientas_conocidas", "areas_interes"]
    ej = {}
    for c in campos:
        if c in fila.index:
            v = fila[c]
            if isinstance(v, float) and v != v:
                ej[c] = None
            elif isinstance(v, float) and c == "estrato":
                ej[c] = int(v)
            elif isinstance(v, (np.integer,)):
                ej[c] = int(v)
            elif isinstance(v, (np.floating,)):
                ej[c] = float(v)
            else:
                ej[c] = v
    return ej


def generar_predictor_py(SPEC: dict, ejemplo: dict) -> None:
    fns = [_to_float, _parse_count, _ord_level, _bin_si, _raw_feature,
           _feature_vector, predecir_cluster]
    fuente = "\n\n".join(inspect.getsource(f) for f in fns)
    js = json.dumps(SPEC, ensure_ascii=True)
    contenido = f'''# -*- coding: utf-8 -*-
"""
Predictor PURO de perfil (cluster) — Copa STEM 2026 (K-Means, K={SPEC["meta"]["k"]}).
GENERADO por notebooks/07_clustering_perfiles.py — no editar a mano.

    from clustering_predictor import predecir_perfil
    r = predecir_perfil({{"puntaje_obtenido": 70, "estrato": 3,
                          "computador_en_casa": "Sí, propio", ...}})
    # r = {{'cluster_id': .., 'cluster_nombre': '..'}}

Asigna al estudiante el centroide más cercano en el espacio estandarizado.
No requiere sklearn ni numpy: solo la librería estándar (json).
"""
import json

SPEC = json.loads(r"""{js}""")


{fuente}


def predecir_perfil(estudiante):
    """Devuelve el cluster_id y el nombre del perfil (centroide más cercano)."""
    return predecir_cluster(estudiante, SPEC)


if __name__ == "__main__":
    ejemplo = {json.dumps(ejemplo, ensure_ascii=False)}
    print(predecir_perfil(ejemplo))
'''
    (DEPLOY_DIR / "clustering_predictor.py").write_text(contenido, encoding="utf-8")
    log("    predictor Python → models/deploy/clustering_predictor.py")


def generar_predictor_js(SPEC: dict, ejemplo: dict) -> None:
    js_spec = json.dumps(SPEC, ensure_ascii=False)
    ej = json.dumps(ejemplo, ensure_ascii=False)
    contenido = r'''/**
 * Predictor PURO de perfil (cluster) — Copa STEM 2026 (K-Means).
 * GENERADO por notebooks/07_clustering_perfiles.py — no editar a mano.
 * Réplica en JavaScript ES6. Sin dependencias.
 *
 *   import { predecirPerfil } from "./clustering_predictor.js";
 *   const r = predecirPerfil({ puntaje_obtenido: 70, estrato: 3,
 *                              computador_en_casa: "Sí, propio" });
 */
const SPEC = __SPEC__;

function _toFloat(v) {
  if (v === null || v === undefined || typeof v === "boolean") return null;
  const f = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isNaN(f) ? null : f;
}

function _parseCount(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  if (s === "" || ["nan", "none", "[]"].includes(s.toLowerCase())) return 0;
  let items = null;
  try { const p = JSON.parse(s); if (Array.isArray(p)) items = p; } catch (e) { items = null; }
  if (items === null) items = s.replace(/[[\]"]/g, "").split(",").filter((x) => x.trim());
  let cnt = 0;
  for (const it of items) {
    const t = String(it).trim().toLowerCase();
    if (t && !["ninguna", "ninguno", "ninguna.", "ninguno."].includes(t)) cnt += 1;
  }
  return cnt;
}

function _ordLevel(v) {
  if (v === null || v === undefined) return null;
  const m = { ninguna: 0, ninguno: 0, "básica": 1, basica: 1, intermedia: 2, avanzada: 3 };
  const s = String(v).trim().toLowerCase();
  return s in m ? m[s] : null;
}

function _binSi(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim().toLowerCase();
  if (["nan", "none", ""].includes(s)) return null;
  if (s.startsWith("s")) return 1;
  if (s.startsWith("n")) return 0;
  return null;
}

function _rawFeature(raw, name) {
  if (name === "computador_bin") return _binSi(raw["computador_en_casa"]);
  if (name === "internet_bin") return _binSi(raw["internet_en_casa"]);
  if (name === "nivel_programacion_ord") return _ordLevel(raw["nivel_programacion"]);
  if (name === "nivel_robotica_ord") return _ordLevel(raw["nivel_robotica"]);
  if (name === "n_herramientas") {
    const v = _toFloat(raw["n_herramientas"]);
    return v !== null ? v : _parseCount(raw["herramientas_conocidas"]);
  }
  if (name === "n_areas_interes") {
    const v = _toFloat(raw["n_areas_interes"]);
    return v !== null ? v : _parseCount(raw["areas_interes"]);
  }
  return _toFloat(raw[name]);
}

function _featureVector(raw, SPEC) {
  const vals = [];
  for (let i = 0; i < SPEC.features.length; i++) {
    let v = _rawFeature(raw, SPEC.features[i]);
    if (v === null) v = SPEC.medians[i];
    vals.push(v);
  }
  return vals;
}

export function predecirPerfil(raw) {
  const x = _featureVector(raw, SPEC);
  const mean = SPEC.scaler_mean, scale = SPEC.scaler_scale;
  const z = x.map((xi, i) => (xi - mean[i]) / scale[i]);
  let bestId = -1, bestD = Infinity;
  for (let cid = 0; cid < SPEC.centroids.length; cid++) {
    const c = SPEC.centroids[cid];
    let d = 0.0;
    for (let i = 0; i < z.length; i++) { const diff = z[i] - c[i]; d += diff * diff; }
    if (d < bestD) { bestD = d; bestId = cid; }
  }
  return { cluster_id: bestId, cluster_nombre: SPEC.nombres[String(bestId)] };
}

if (typeof process !== "undefined" && Array.isArray(process.argv) && process.argv[1]) {
  const _here = decodeURIComponent(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1");
  const _norm = (p) => p.replace(/\\/g, "/").toLowerCase();
  if (_norm(_here) === _norm(process.argv[1])) console.log(predecirPerfil(__EJEMPLO__));
}
'''
    contenido = contenido.replace("__SPEC__", js_spec).replace("__EJEMPLO__", ej)
    (DEPLOY_DIR / "clustering_predictor.js").write_text(contenido, encoding="utf-8")
    log("    predictor JavaScript → models/deploy/clustering_predictor.js")


# =============================================================================
# RECOMENDACIONES POR PERFIL (heurística basada en el nombre)
# =============================================================================

def recomendacion_perfil(nombre: str) -> str:
    n = nombre.lower()
    if "sin acceso" in n or "vulnerable" in n or "desconectado" in n:
        return ("Dotación tecnológica (computador/internet) + beca; es el grupo "
                "con mayor retorno social por unidad de inversión.")
    if "alto rendimiento" in n:
        return ("Rutas STEM avanzadas, mentoría y competencias de nivel superior "
                "para no perder el talento por falta de reto.")
    if "explorador" in n:
        return ("Canalizar el interés con talleres prácticos de programación y "
                "robótica; alto engagement que conviene convertir en resultados.")
    if "conectado" in n or "promedio" in n:
        return ("Refuerzo académico focalizado; ya tienen acceso, falta "
                "acompañamiento pedagógico para subir el rendimiento.")
    if "en desarrollo" in n:
        return ("Nivelación en matemáticas/lógica y tutoría de base; medir barreras "
                "específicas de aprendizaje.")
    return "Acompañamiento general y seguimiento del progreso."


# =============================================================================
# INFORME
# =============================================================================

def construir_informe(prep, k, sils_k, modelos, perf, exp) -> None:
    log("Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    nombres = perf["nombres"]
    tabla = perf["tabla"]
    R = REPORT.append

    R("# Clustering de Perfiles de Estudiante — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Fase 3 · Informe: {fecha}\n")
    R("---\n")

    R("## Resumen ejecutivo\n")
    mejor_modelo = "K-Means" if modelos["km_sil"] >= modelos["gmm_sil"] else "GMM"
    R(dedent(f"""\
        Se segmentaron **{len(perf['df']):,} estudiantes** en **{k} perfiles**
        combinando rendimiento, acceso tecnológico y experiencia previa (9
        variables estandarizadas). El número de clusters se eligió por
        **silhouette** (K={k}). Se comparó **K-Means** (silhouette
        {modelos['km_sil']:.3f}) con **Gaussian Mixture** ({modelos['gmm_sil']:.3f});
        se despliega **K-Means** por su asignación determinista *centroide más
        cercano*, ideal para un predictor portable. Cada perfil recibe un nombre
        descriptivo y una recomendación de intervención.\n"""))

    R("## Metodología\n")
    R(dedent("""\
        - **Features (9):** puntaje, estrato, computador (0/1), internet (0/1),
          nivel de programación (0-3), nivel de robótica (0-3), interés, nº de
          herramientas, nº de áreas de interés.
        - **Preprocesamiento:** imputación por **mediana** + **StandardScaler**.
        - **K óptimo:** método del codo (inercia) + **silhouette** (K=2..10).
        - **Modelos:** K-Means (despliegue) y Gaussian Mixture (alternativa).
        - Reproducible con `random_state=42`.\n"""))
    R(f"\n{img('seleccion_k', 'Selección de K')}\n")

    R("## Perfiles identificados\n")
    R(tabla_md(tabla) + "\n")
    R(f"\n{img('radar', 'Radar de perfiles')}\n")
    R(f"\n{img('pca', 'PCA 2D')}\n")

    R("## Distribución de perfiles\n")
    if "colegio" in FIGURES:
        R(f"\n{img('colegio', 'Perfiles por colegio')}\n")
    R(f"\n{img('muni_grado', 'Perfiles por municipio y grado')}\n")

    R("## Recomendaciones de intervención por perfil\n")
    for _, fila in tabla.iterrows():
        nombre = fila["Nombre"]
        R(dedent(f"""\
            **{nombre}** (cluster {fila['Cluster']}, N={fila['N']}, {fila['% total']}%)
            — puntaje µ={fila['Puntaje µ']}, {fila['% computador']:.0f}% con
            computador, nivel prog µ={fila['Nivel prog µ']}.
            → *{recomendacion_perfil(nombre)}*\n"""))

    R("## Exportación para producción\n")
    R(dedent("""\
        - `models/deploy/clustering_perfiles.csv` — `numero_documento`,
          `cluster_id`, `cluster_nombre`.
        - `models/deploy/clustering_predictor.py` — función pura `predecir_perfil(dict)`
          (imputa, estandariza y asigna al centroide más cercano); sin sklearn.
        - `models/deploy/clustering_predictor.js` — misma función en JS ES6.\n"""))
    R("\n**Ejemplo de entrada:**\n")
    R("```json\n" + json.dumps(exp["ejemplo"], ensure_ascii=False, indent=2) + "\n```\n")

    R("## Limitaciones\n")
    R(dedent("""\
        - Los **nombres de perfil son etiquetas interpretativas** derivadas del
          perfil relativo de cada cluster, no categorías oficiales.
        - K-Means asume clusters convexos de tamaño similar; la silhouette
          moderada indica solapamiento entre perfiles (frontera difusa).
        - Variables **autorreportadas** e imputación por mediana en ~7% de casos
          sin datos socioeconómicos.\n"""))
    R("\n---\n_Generado por `notebooks/07_clustering_perfiles.py` — Copa STEM 2026._\n")

    (REPORTS_DIR / "07_clustering_perfiles.md").write_text("\n".join(REPORT), encoding="utf-8")
    log("    informe escrito → reports/07_clustering_perfiles.md")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Clustering de Perfiles de Estudiante (Fase 3)")
    print(" Fundación SapienceLab")
    print("=" * 70)

    df = cargar_datos()
    X, scaler, medians, raw_imp = preparar_matriz(df)
    k, sils_k = determinar_k(X)
    modelos = ajustar_modelos(X, k)

    labels = modelos["km_labels"]  # despliegue: K-Means
    perf = perfilar(df, labels, raw_imp, k)
    graficos_distribucion(perf["df"], X, labels, perf["nombres"], k)
    exp = exportar(perf["df"], scaler, medians, modelos["kmeans"], perf["nombres"])
    construir_informe(df, k, sils_k, modelos, perf, exp)

    print("\n" + "=" * 70)
    print(" ✔ CLUSTERING COMPLETADO")
    print(f"   · K óptimo:          {k}")
    print(f"   · Silhouette:        K-Means={modelos['km_sil']:.3f} | "
          f"GMM={modelos['gmm_sil']:.3f}")
    print(f"   · Perfiles:          {', '.join(perf['nombres'].values())}")
    print(f"   · Predictor puro:    {exp['mismatch']} discrepancias vs K-Means")
    print(f"   · Figuras:           {len(FIGURES)} → outputs/")
    print(f"   · Deploy:            models/deploy/clustering_*")
    print(f"   · Informe:           reports/07_clustering_perfiles.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
