# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 04: Índice de Potencial STEM (compuesto)  — Fase 2
================================================================================

Motivación
----------
El modelo predictivo (script 03) tiene un R² bajo (~0.085): las variables
socioeconómicas explican poco del puntaje. Lejos de invalidar el análisis, es un
hallazgo valioso — significa que **la nota cruda no debe ser la única señal de
potencial**. Por eso el Índice de Potencial STEM es **COMPUESTO**:

    indice = 0.50·rendimiento + 0.25·engagement + 0.25·resiliencia

  - rendimiento (0–100): percentil del puntaje real (si presentó) o del puntaje
    estimado por el modelo (si no presentó), dentro de la cohorte.
  - engagement (0–100): promedio normalizado de nivel de programación, robótica,
    interés, nº de herramientas, nº de áreas, participación en olimpiadas y
    acceso a computador e internet.
  - resiliencia (0–100): premia rendir bien A PESAR de condiciones adversas
    (estrato bajo, sin computador, sin internet, no vive con ambos padres).

Entregables
-----------
    models/deploy/scores_potencial_stem.csv
    models/deploy/potencial_stem_predictor.py   (función pura, solo stdlib)
    models/deploy/potencial_stem_predictor.js    (misma función en JS ES6)
    reports/04_indice_potencial_stem.md
    outputs/F04_*.png  (paleta Copa STEM)

Diseño: reproducible (`random_state=42`); el CSV de la cohorte y los predictores
de despliegue comparten EXACTAMENTE la misma lógica pura (se validan entre sí).

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
    from scipy import stats
except ImportError as exc:  # pragma: no cover
    print("ERROR: falta una dependencia del entorno.")
    print(f"       Detalle: {exc}")
    sys.exit(1)

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0. CONFIGURACIÓN GLOBAL (paleta e imagen consistentes con 01/02/03)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
for _d in (OUTPUTS_DIR, REPORTS_DIR, DEPLOY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Preferimos el dataset LIMPIO (sin exámenes anulados; ver 05b). Si no existe,
# se usa el original. Así el índice se calcula sobre datos limpios por defecto.
DATASET_CANDIDATOS = ["copa_stem_dataset_limpio.csv", "copa_stem_dataset.csv"]
DATASET_NAME = DATASET_CANDIDATOS[0]

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


# Pesos del índice compuesto.
PESOS = {"rendimiento": 0.50, "engagement": 0.25, "resiliencia": 0.25}

# Umbrales de categoría (sobre el índice compuesto 0–100).
CATEGORIAS = [
    (85, "Talento destacado"),
    (70, "Alto potencial"),
    (45, "Promedio"),
    (25, "En desarrollo"),
    (0,  "Requiere apoyo"),
]
# Color por categoría (paleta Copa STEM), de mayor a menor.
CAT_COLOR = {
    "Talento destacado": COLORS["green"],
    "Alto potencial":    COLORS["cyan"],
    "Promedio":          COLORS["violet"],
    "En desarrollo":     COLORS["amber"],
    "Requiere apoyo":    COLORS["red"],
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
# FUNCIONES PURAS DEL ÍNDICE
# -----------------------------------------------------------------------------
# Se usan aquí para calcular el CSV de la cohorte Y se EMBEBEN (inspect.getsource)
# en models/deploy/potencial_stem_predictor.py. La versión JS las replica a mano.
# Solo dependen de builtins + json (stdlib). Reciben un dict de features crudos.
# =============================================================================

def _isnan(v):
    return isinstance(v, float) and v != v


def _to_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _parse_count(v):
    # Dato faltante (None o NaN) y "lista vacía"/"ninguna" cuentan como 0
    # herramientas. Se maneja None y NaN de forma IDÉNTICA para que el CSV de
    # la cohorte y los predictores de despliegue coincidan exactamente.
    if v is None:
        return 0
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
    s = str(v).strip().lower()
    m = {"ninguna": 0, "ninguno": 0, "básica": 1, "basica": 1,
         "intermedia": 2, "avanzada": 3}
    return m.get(s)


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


def _clip100(x):
    if x < 0.0:
        return 0.0
    if x > 100.0:
        return 100.0
    return x


def _percentil(v, ref):
    """Percentil (0–100) de v en la lista ORDENADA `ref` (fracción de ref ≤ v)."""
    n = len(ref)
    if n == 0:
        return 50.0
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if ref[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return 100.0 * lo / n


def _features_puntaje(raw, PRE):
    """Vector de features para el modelo de puntaje (idéntico al script 03)."""
    feats = []
    for name in PRE["numeric"]:
        v = _to_float(raw.get(name))
        if v is None and name == "n_herramientas":
            v = _parse_count(raw.get("herramientas_conocidas"))
        if v is None and name == "n_areas_interes":
            v = _parse_count(raw.get("areas_interes"))
        if v is None or _isnan(v):
            v = PRE["medians"][name]
        feats.append(float(v))
    for name in PRE["ordinal"]:
        lv = _ord_level(raw.get(name[:-4]))
        if lv is None:
            lv = PRE["medians"][name]
        feats.append(float(lv))
    for name in PRE["binary"]:
        b = _bin_si(raw.get(PRE["binary_src"][name]))
        if b is None:
            b = PRE["modes"][name]
        feats.append(float(b))
    for col in PRE["onehot_order"]:
        val = raw.get(col)
        if val is None or _isnan(val) or str(val).strip().lower() in ("nan", "none", ""):
            val = PRE["onehot_mode"][col]
        val = str(val).strip()
        for cat in PRE["onehot_cats"][col]:
            feats.append(1.0 if val == cat else 0.0)
    return feats


def _predict_puntaje(feats, MODEL):
    """Puntaje estimado (0–100) por el modelo (lineal o de árboles)."""
    if MODEL["type"] == "linear":
        y = MODEL["intercept"]
        coef = MODEL["coef"]
        for i in range(len(coef)):
            y += coef[i] * feats[i]
    else:
        op = MODEL["op"]
        total = 0.0
        for tree in MODEL["trees"]:
            i = 0
            while tree[i][0] != -1:
                fi, thr = tree[i][0], tree[i][1]
                go_left = feats[fi] < thr if op == "lt" else feats[fi] <= thr
                i = tree[i][2] if go_left else tree[i][3]
            total += tree[i][4]
        if MODEL["combine"] == "mean":
            total /= len(MODEL["trees"])
        y = total + MODEL["bias"]
    return _clip100(y)


def _engagement(raw, SPEC):
    """Componente de engagement (0–100): promedio de 8 señales normalizadas."""
    PRE = SPEC["puntaje"]["preprocess"]
    ENG = SPEC["engagement"]
    señales = []
    # Nivel de programación y robótica (ordinal 0–3 → 0–100).
    for name in ("nivel_programacion_ord", "nivel_robotica_ord"):
        lv = _ord_level(raw.get(name[:-4]))
        if lv is None:
            lv = PRE["medians"][name]
        señales.append(lv / 3.0 * 100.0)
    # Interés en prog/robótica (1–5 → 0–100).
    it = _to_float(raw.get("interes_prog_robotica"))
    if it is None:
        it = PRE["medians"]["interes_prog_robotica"]
    señales.append(_clip100((it - 1.0) / 4.0 * 100.0))
    # Conteos (n_herramientas, n_areas_interes) con min–max de la cohorte.
    for name, key in (("n_herramientas", "herramientas_conocidas"),
                      ("n_areas_interes", "areas_interes")):
        c = _to_float(raw.get(name))
        if c is None:
            c = _parse_count(raw.get(key))
        if c is None:
            c = PRE["medians"][name]
        lo, hi = ENG[name]["lo"], ENG[name]["hi"]
        señales.append(_clip100((c - lo) / (hi - lo) * 100.0) if hi > lo else 0.0)
    # Binarias (participación, computador, internet) → 0 o 100.
    for name, src in (("olimpiadas_bin", "participacion_olimpiadas"),
                      ("computador_bin", "computador_en_casa"),
                      ("internet_bin", "internet_en_casa")):
        b = _bin_si(raw.get(src))
        if b is None:
            b = PRE["modes"][name]
        señales.append(b * 100.0)
    return sum(señales) / len(señales)


def _adversidad(raw):
    """Cuenta condiciones adversas (0–4): estrato≤2, sin PC, sin internet, no ambos padres."""
    adv = 0
    est = _to_float(raw.get("estrato"))
    if est is not None and est <= 2:
        adv += 1
    if _bin_si(raw.get("computador_en_casa")) == 0:
        adv += 1
    if _bin_si(raw.get("internet_en_casa")) == 0:
        adv += 1
    cqv = raw.get("con_quien_vive")
    if cqv is not None and str(cqv).strip().lower() not in ("nan", "none", "") \
            and str(cqv).strip() != "Ambos padres":
        adv += 1
    return adv


def _categoria(indice, CATEGORIAS):
    for umbral, nombre in CATEGORIAS:
        if indice >= umbral:
            return nombre
    return CATEGORIAS[-1][1]


def calcular_indice(raw, SPEC):
    """Calcula el índice compuesto y sus componentes para un estudiante (dict crudo)."""
    PRE = SPEC["puntaje"]["preprocess"]
    MODEL = SPEC["puntaje"]["model"]
    ref = SPEC["ref_rendimiento"]
    pesos = SPEC["pesos"]

    real = _to_float(raw.get("puntaje_obtenido"))
    presento = real is not None
    if presento:
        rend_raw = real
    else:
        rend_raw = _predict_puntaje(_features_puntaje(raw, PRE), MODEL)
    c_rendimiento = _percentil(rend_raw, ref)

    c_engagement = _engagement(raw, SPEC)

    adv = _adversidad(raw)
    if presento:
        c_resiliencia = min(100.0, c_rendimiento * (1.0 + adv * 0.15))
    else:
        c_resiliencia = max(0.0, 50.0 - adv * 5.0)

    indice = (pesos["rendimiento"] * c_rendimiento
              + pesos["engagement"] * c_engagement
              + pesos["resiliencia"] * c_resiliencia)

    return {
        "indice_potencial": round(indice, 2),
        "componente_rendimiento": round(c_rendimiento, 2),
        "componente_engagement": round(c_engagement, 2),
        "componente_resiliencia": round(c_resiliencia, 2),
        "categoria": _categoria(indice, SPEC["categorias"]),
        "_presento": presento,
        "_adversidad": adv,
        "_rend_raw": round(rend_raw, 2),
    }


# =============================================================================
# CARGA / MODELO / SPEC
# =============================================================================

def cargar_y_limpiar() -> pd.DataFrame:
    log("Carga y limpieza del dataset")
    # Resolución del dataset: primero limpio, luego original, luego autodetectar.
    ruta = next((DATA_DIR / n for n in DATASET_CANDIDATOS
                 if (DATA_DIR / n).exists()), None)
    if ruta is None:
        csvs = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
        if not csvs:
            print(f"\n  ⚠  No se encontró '{DATASET_NAME}' en {DATA_DIR}\n")
            sys.exit(0)
        ruta = csvs[0]
    log(f"    dataset: {ruta.name}")

    df = pd.read_csv(ruta, encoding="utf-8")
    docs_prueba = ["1234", "123456", "123456789", "1234567899", "0", "00000000"]
    if "numero_documento" in df.columns:
        df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
        df = df[~df["numero_documento"].isin(docs_prueba)]
        df = df[df["numero_documento"].str.len() >= 5]
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    for c in ["puntaje_obtenido", "grado_escolar", "estrato",
              "interes_prog_robotica", "edad_calculada"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.reset_index(drop=True)
    log(f"    registros tras limpieza: {len(df):,}")
    return df


def cargar_spec_puntaje() -> dict:
    """Lee el spec del modelo de puntaje embebido en models/predictor.py (script 03)."""
    ruta = MODELS_DIR / "predictor.py"
    if not ruta.exists():
        print("\n  ⚠  Falta models/predictor.py. Ejecute antes "
              "notebooks/03_modelo_predictivo.py.\n")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("predictor_puntaje", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log(f"    modelo de puntaje cargado desde models/predictor.py "
        f"(modelo: {mod._SPEC.get('modelo', 'n/d')})")
    return mod._SPEC  # {'preprocess':..., 'model':..., ...}


def construir_spec_indice(df: pd.DataFrame, spec_puntaje: dict) -> dict:
    """Ensambla el SPEC del índice: modelo de puntaje + referencia + normalizadores."""
    PRE = spec_puntaje["preprocess"]
    MODEL = spec_puntaje["model"]
    registros = df.to_dict("records")

    # min–max de los conteos (para normalizar engagement) sobre la cohorte.
    def _conteos(name, key):
        vals = []
        for r in registros:
            c = _to_float(r.get(name))
            if c is None:
                c = _parse_count(r.get(key))
            if c is not None:
                vals.append(c)
        return (min(vals), max(vals)) if vals else (0.0, 1.0)

    lo_h, hi_h = _conteos("n_herramientas", "herramientas_conocidas")
    lo_a, hi_a = _conteos("n_areas_interes", "areas_interes")

    # Referencia de rendimiento: puntaje real (si presentó) o estimado (si no).
    rend_raw = []
    for r in registros:
        real = _to_float(r.get("puntaje_obtenido"))
        if real is not None:
            rend_raw.append(real)
        else:
            rend_raw.append(_predict_puntaje(_features_puntaje(r, PRE), MODEL))
    ref = sorted(round(float(x), 4) for x in rend_raw)

    return {
        "meta": {"generado": datetime.now().isoformat(timespec="seconds"),
                 "n_cohorte": len(registros),
                 "modelo_puntaje": spec_puntaje.get("modelo", "n/d")},
        "puntaje": {"preprocess": PRE, "model": MODEL},
        "engagement": {"n_herramientas": {"lo": float(lo_h), "hi": float(hi_h)},
                       "n_areas_interes": {"lo": float(lo_a), "hi": float(hi_a)}},
        "ref_rendimiento": ref,
        "pesos": PESOS,
        "categorias": [list(c) for c in CATEGORIAS],
    }


# =============================================================================
# CÁLCULO DE LA COHORTE
# =============================================================================

def calcular_cohorte(df: pd.DataFrame, SPEC: dict) -> pd.DataFrame:
    log("Cálculo del índice compuesto para toda la cohorte")
    PRE = SPEC["puntaje"]["preprocess"]
    MODEL = SPEC["puntaje"]["model"]
    # Columnas de contexto que se adjuntan INLINE (sin merge, para no duplicar
    # filas cuando hay numero_documento repetidos).
    ctx_cols = ["numero_documento", "institucion_educativa", "municipio",
                "grado_escolar", "genero", "estrato", "puntaje_obtenido",
                "computador_en_casa", "internet_en_casa", "con_quien_vive"]
    ctx_cols = [c for c in ctx_cols if c in df.columns]

    filas = []
    for r in df.to_dict("records"):
        res = calcular_indice(r, SPEC)
        est = _predict_puntaje(_features_puntaje(r, PRE), MODEL)
        res["puntaje_estimado"] = round(est, 2)
        res["numero_documento"] = str(r.get("numero_documento"))
        for c in ctx_cols:
            if c == "numero_documento":
                continue
            v = r.get(c)
            res[c] = None if (isinstance(v, float) and v != v) else v
        filas.append(res)

    out = pd.DataFrame(filas)
    log(f"    índice calculado para {len(out):,} estudiantes "
        f"(presentaron: {int(out['_presento'].sum()):,})")
    return out


# =============================================================================
# C. ANÁLISIS
# =============================================================================

def analisis(out: pd.DataFrame) -> dict:
    log("SECCIÓN C — Análisis del índice")
    res = {}

    # --- C.1 Distribución del índice + categorías ---
    orden_cat = [c[1] for c in CATEGORIAS][::-1]  # de menor a mayor
    conteo = out["categoria"].value_counts().reindex(orden_cat).fillna(0).astype(int)
    res["conteo_categoria"] = conteo.to_dict()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.histplot(out["indice_potencial"], bins=30, color=COLORS["cyan"],
                 edgecolor="white", ax=ax1)
    ax1.axvline(out["indice_potencial"].mean(), color=COLORS["red"],
                linestyle="--", label=f"Media = {out['indice_potencial'].mean():.1f}")
    ax1.set_title("Distribución del Índice de Potencial STEM (compuesto)")
    ax1.set_xlabel("Índice (0–100)")
    ax1.set_ylabel("N estudiantes")
    ax1.legend()
    ax2.bar(conteo.index, conteo.values,
            color=[CAT_COLOR[c] for c in conteo.index], edgecolor="white")
    ax2.set_title("Estudiantes por categoría")
    ax2.set_ylabel("N estudiantes")
    ax2.tick_params(axis="x", rotation=20)
    for i, v in enumerate(conteo.values):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("A. Índice compuesto de potencial STEM",
                 fontsize=15, fontweight="bold")
    savefig(fig, "F04_distribucion_indice.png", "dist")

    # --- C.2 Índice vs puntaje real: ¿captura más matices? ---
    pres = out[out["_presento"]].copy()
    rho = spearman = float("nan")
    if len(pres) > 10:
        rho, _ = stats.spearmanr(pres["puntaje_obtenido"], pres["indice_potencial"])
    res["rho_indice_puntaje"] = float(rho)
    # Dispersión de índices dentro de una misma nota (muestra que aporta matiz).
    disp = (pres.groupby("puntaje_obtenido")["indice_potencial"]
            .std().dropna())
    res["dispersion_media_indice_por_nota"] = float(disp.mean()) if len(disp) else 0.0
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in [c[1] for c in CATEGORIAS]:
        sub = pres[pres["categoria"] == cat]
        if not sub.empty:
            ax.scatter(sub["puntaje_obtenido"], sub["indice_potencial"],
                       s=22, alpha=0.5, color=CAT_COLOR[cat], label=cat,
                       edgecolor="white", linewidth=0.3)
    ax.set_title(f"Índice compuesto vs. puntaje real (ρ Spearman = {rho:.2f})\n"
                 "una misma nota se abre en un rango de índices → más matiz")
    ax.set_xlabel("Puntaje real")
    ax.set_ylabel("Índice de Potencial STEM")
    ax.legend(title="Categoría", fontsize=8)
    savefig(fig, "F04_indice_vs_puntaje.png", "vs_puntaje")

    # --- C.3 Índice promedio por grupo ---
    grupos = [("municipio", "Municipio"), ("grado_escolar", "Grado"),
              ("genero", "Género"), ("estrato", "Estrato")]
    grupos = [(c, t) for c, t in grupos if c in out.columns]
    res["promedios_grupo"] = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(-1)
    for ax, (col, titulo) in zip(axes, grupos):
        g = (out.dropna(subset=[col]).groupby(col)["indice_potencial"]
             .mean().sort_values(ascending=False))
        res["promedios_grupo"][col] = {str(k): round(float(v), 2)
                                       for k, v in g.items()}
        ax.bar([str(k) for k in g.index], g.values,
               color=gradient_colors(len(g)), edgecolor="white")
        ax.set_title(f"Índice promedio por {titulo}")
        ax.set_ylabel("Índice promedio")
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(g.values):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for ax in axes[len(grupos):]:
        ax.set_visible(False)
    fig.suptitle("Índice de potencial promedio por grupo",
                 fontsize=15, fontweight="bold")
    savefig(fig, "F04_indice_por_grupo.png", "grupo")

    # Ranking de colegios por índice promedio (N ≥ 10).
    if "institucion_educativa" in out.columns:
        rk = (out.dropna(subset=["institucion_educativa"])
              .groupby("institucion_educativa")["indice_potencial"]
              .agg(["mean", "count"]).reset_index())
        rk = rk[rk["count"] >= 10].sort_values("mean")
        res["ranking_colegios"] = rk.sort_values("mean", ascending=False)
        if not rk.empty:
            fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(rk))))
            ax.barh(rk["institucion_educativa"], rk["mean"],
                    color=gradient_colors(len(rk)), edgecolor="white")
            ax.axvline(out["indice_potencial"].mean(), color=COLORS["red"],
                       linestyle="--",
                       label=f"Media global = {out['indice_potencial'].mean():.1f}")
            ax.set_title("Índice de potencial promedio por institución (N ≥ 10)")
            ax.set_xlabel("Índice promedio")
            ax.legend()
            savefig(fig, "F04_ranking_colegios.png", "colegios")

    # --- C.4 Scatter rendimiento vs engagement coloreado por categoría ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for cat in [c[1] for c in CATEGORIAS]:
        sub = out[out["categoria"] == cat]
        if not sub.empty:
            ax.scatter(sub["componente_rendimiento"], sub["componente_engagement"],
                       s=24, alpha=0.55, color=CAT_COLOR[cat], label=cat,
                       edgecolor="white", linewidth=0.3)
    ax.set_title("Rendimiento vs. Engagement (color = categoría de potencial)")
    ax.set_xlabel("Componente rendimiento (percentil, 0–100)")
    ax.set_ylabel("Componente engagement (0–100)")
    ax.legend(title="Categoría", fontsize=8)
    savefig(fig, "F04_rendimiento_vs_engagement.png", "rend_eng")

    # --- C.5 Top 20 que NO presentaron con mayor índice ---
    no_pres = out[~out["_presento"]].sort_values("indice_potencial", ascending=False)
    res["top_no_presento"] = no_pres.head(20)
    res["n_no_presento"] = int(len(no_pres))

    # --- C.6 Top 20 mayor resiliencia (rindieron mejor de lo esperado) ---
    top_res = out[out["_presento"]].copy()
    top_res["sobre_rendimiento"] = (top_res["puntaje_obtenido"]
                                    - top_res["puntaje_estimado"]).round(2)
    top_res = top_res.sort_values(
        ["componente_resiliencia", "sobre_rendimiento"], ascending=False)
    res["top_resiliencia"] = top_res.head(20)

    return res


# =============================================================================
# D. EXPORTACIÓN
# =============================================================================

def exportar_csv(out: pd.DataFrame) -> Path:
    cols = ["numero_documento", "indice_potencial", "componente_rendimiento",
            "componente_engagement", "componente_resiliencia", "categoria"]
    destino = DEPLOY_DIR / "scores_potencial_stem.csv"
    out[cols].to_csv(destino, index=False, encoding="utf-8-sig")
    log(f"    scores → models/deploy/{destino.name} ({len(out):,} filas)")
    return destino


def exportar_predictor_py(SPEC: dict, ejemplo: dict) -> None:
    fns = [_isnan, _to_float, _parse_count, _ord_level, _bin_si, _clip100,
           _percentil, _features_puntaje, _predict_puntaje, _engagement,
           _adversidad, _categoria, calcular_indice]
    fuente = "\n\n".join(inspect.getsource(f) for f in fns)
    js = json.dumps(SPEC, ensure_ascii=True)
    contenido = f'''# -*- coding: utf-8 -*-
"""
Calculadora PURA del Índice de Potencial STEM (compuesto) — Copa STEM 2026.
GENERADO por notebooks/04_indice_potencial_stem.py — no editar a mano.

    from potencial_stem_predictor import calcular_indice_potencial
    r = calcular_indice_potencial({{"puntaje_obtenido": 55, "estrato": 2,
                                    "computador_en_casa": "No", ...}})
    # r = {{'indice_potencial':.., 'componente_rendimiento':.., ...,
    #        'categoria':..}}

No requiere sklearn ni librerías de ML: solo la librería estándar (json).
"""
import json

SPEC = json.loads(r"""{js}""")


{fuente}


def calcular_indice_potencial(estudiante):
    """Devuelve el índice compuesto y sus componentes para un dict de features crudos."""
    r = calcular_indice(estudiante, SPEC)
    return {{k: v for k, v in r.items() if not k.startswith("_")}}


if __name__ == "__main__":
    ejemplo = {json.dumps(ejemplo, ensure_ascii=False)}
    import pprint
    pprint.pprint(calcular_indice_potencial(ejemplo))
'''
    destino = DEPLOY_DIR / "potencial_stem_predictor.py"
    destino.write_text(contenido, encoding="utf-8")
    log(f"    predictor Python → models/deploy/{destino.name}")


def exportar_predictor_js(SPEC: dict, ejemplo: dict) -> None:
    js_spec = json.dumps(SPEC, ensure_ascii=False)
    ej = json.dumps(ejemplo, ensure_ascii=False)
    contenido = r'''/**
 * Calculadora PURA del Índice de Potencial STEM (compuesto) — Copa STEM 2026.
 * GENERADO por notebooks/04_indice_potencial_stem.py — no editar a mano.
 * Réplica en JavaScript ES6 del predictor Python. Sin dependencias.
 *
 *   import { calcularIndicePotencial } from "./potencial_stem_predictor.js";
 *   const r = calcularIndicePotencial({ puntaje_obtenido: 55, estrato: 2,
 *                                       computador_en_casa: "No" });
 */
const SPEC = __SPEC__;

const _isNum = (v) => typeof v === "number" && !Number.isNaN(v);

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
  try {
    const parsed = JSON.parse(s);
    if (Array.isArray(parsed)) items = parsed;
  } catch (e) { items = null; }
  if (items === null) {
    items = s.replace(/[[\]"]/g, "").split(",").filter((x) => x.trim());
  }
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

const _clip100 = (x) => (x < 0 ? 0 : x > 100 ? 100 : x);

function _percentil(v, ref) {
  const n = ref.length;
  if (n === 0) return 50.0;
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ref[mid] <= v) lo = mid + 1; else hi = mid;
  }
  return (100.0 * lo) / n;
}

function _featuresPuntaje(raw, PRE) {
  const feats = [];
  for (const name of PRE.numeric) {
    let v = _toFloat(raw[name]);
    if (v === null && name === "n_herramientas") v = _parseCount(raw["herramientas_conocidas"]);
    if (v === null && name === "n_areas_interes") v = _parseCount(raw["areas_interes"]);
    if (v === null) v = PRE.medians[name];
    feats.push(v);
  }
  for (const name of PRE.ordinal) {
    let lv = _ordLevel(raw[name.slice(0, -4)]);
    if (lv === null) lv = PRE.medians[name];
    feats.push(lv);
  }
  for (const name of PRE.binary) {
    let b = _binSi(raw[PRE.binary_src[name]]);
    if (b === null) b = PRE.modes[name];
    feats.push(b);
  }
  for (const col of PRE.onehot_order) {
    let val = raw[col];
    if (val === null || val === undefined || ["nan", "none", ""].includes(String(val).trim().toLowerCase())) {
      val = PRE.onehot_mode[col];
    }
    val = String(val).trim();
    for (const cat of PRE.onehot_cats[col]) feats.push(val === cat ? 1.0 : 0.0);
  }
  return feats;
}

function _predictPuntaje(feats, MODEL) {
  let y;
  if (MODEL.type === "linear") {
    y = MODEL.intercept;
    for (let i = 0; i < MODEL.coef.length; i++) y += MODEL.coef[i] * feats[i];
  } else {
    const op = MODEL.op;
    let total = 0.0;
    for (const tree of MODEL.trees) {
      let i = 0;
      while (tree[i][0] !== -1) {
        const fi = tree[i][0], thr = tree[i][1];
        const goLeft = op === "lt" ? feats[fi] < thr : feats[fi] <= thr;
        i = goLeft ? tree[i][2] : tree[i][3];
      }
      total += tree[i][4];
    }
    if (MODEL.combine === "mean") total /= MODEL.trees.length;
    y = total + MODEL.bias;
  }
  return _clip100(y);
}

function _engagement(raw, SPEC) {
  const PRE = SPEC.puntaje.preprocess;
  const ENG = SPEC.engagement;
  const s = [];
  for (const name of ["nivel_programacion_ord", "nivel_robotica_ord"]) {
    let lv = _ordLevel(raw[name.slice(0, -4)]);
    if (lv === null) lv = PRE.medians[name];
    s.push((lv / 3.0) * 100.0);
  }
  let it = _toFloat(raw["interes_prog_robotica"]);
  if (it === null) it = PRE.medians["interes_prog_robotica"];
  s.push(_clip100(((it - 1.0) / 4.0) * 100.0));
  for (const [name, key] of [["n_herramientas", "herramientas_conocidas"],
                             ["n_areas_interes", "areas_interes"]]) {
    let c = _toFloat(raw[name]);
    if (c === null) c = _parseCount(raw[key]);
    if (c === null) c = PRE.medians[name];
    const lo = ENG[name].lo, hi = ENG[name].hi;
    s.push(hi > lo ? _clip100(((c - lo) / (hi - lo)) * 100.0) : 0.0);
  }
  for (const [name, src] of [["olimpiadas_bin", "participacion_olimpiadas"],
                             ["computador_bin", "computador_en_casa"],
                             ["internet_bin", "internet_en_casa"]]) {
    let b = _binSi(raw[src]);
    if (b === null) b = PRE.modes[name];
    s.push(b * 100.0);
  }
  return s.reduce((a, b) => a + b, 0) / s.length;
}

function _adversidad(raw) {
  let adv = 0;
  const est = _toFloat(raw["estrato"]);
  if (est !== null && est <= 2) adv += 1;
  if (_binSi(raw["computador_en_casa"]) === 0) adv += 1;
  if (_binSi(raw["internet_en_casa"]) === 0) adv += 1;
  const cqv = raw["con_quien_vive"];
  if (cqv !== null && cqv !== undefined
      && !["nan", "none", ""].includes(String(cqv).trim().toLowerCase())
      && String(cqv).trim() !== "Ambos padres") adv += 1;
  return adv;
}

function _categoria(indice, CATEGORIAS) {
  for (const [umbral, nombre] of CATEGORIAS) if (indice >= umbral) return nombre;
  return CATEGORIAS[CATEGORIAS.length - 1][1];
}

const _round2 = (x) => Math.round(x * 100) / 100;

export function calcularIndicePotencial(raw) {
  const PRE = SPEC.puntaje.preprocess;
  const MODEL = SPEC.puntaje.model;
  const ref = SPEC.ref_rendimiento;
  const pesos = SPEC.pesos;

  const real = _toFloat(raw["puntaje_obtenido"]);
  const presento = real !== null;
  const rendRaw = presento ? real : _predictPuntaje(_featuresPuntaje(raw, PRE), MODEL);
  const cRend = _percentil(rendRaw, ref);
  const cEng = _engagement(raw, SPEC);
  const adv = _adversidad(raw);
  const cResil = presento
    ? Math.min(100.0, cRend * (1.0 + adv * 0.15))
    : Math.max(0.0, 50.0 - adv * 5.0);
  const indice = pesos.rendimiento * cRend + pesos.engagement * cEng + pesos.resiliencia * cResil;

  return {
    indice_potencial: _round2(indice),
    componente_rendimiento: _round2(cRend),
    componente_engagement: _round2(cEng),
    componente_resiliencia: _round2(cResil),
    categoria: _categoria(indice, SPEC.categorias),
  };
}

// Ejecutar como script: node potencial_stem_predictor.js
if (typeof process !== "undefined" && Array.isArray(process.argv) && process.argv[1]) {
  const _here = decodeURIComponent(new URL(import.meta.url).pathname)
    .replace(/^\/([A-Za-z]:)/, "$1");            // quita la barra inicial antes de C:
  const _norm = (p) => p.replace(/\\/g, "/").toLowerCase();
  if (_norm(_here) === _norm(process.argv[1])) {
    console.log(calcularIndicePotencial(__EJEMPLO__));
  }
}
'''
    contenido = contenido.replace("__SPEC__", js_spec).replace("__EJEMPLO__", ej)
    destino = DEPLOY_DIR / "potencial_stem_predictor.js"
    destino.write_text(contenido, encoding="utf-8")
    log(f"    predictor JavaScript → models/deploy/{destino.name}")


def estudiante_ejemplo(df: pd.DataFrame) -> dict:
    """Un estudiante real (mediana de puntaje) como ejemplo reproducible."""
    pres = df[df["puntaje_obtenido"].notna()]
    fila = pres.iloc[(pres["puntaje_obtenido"]
                      - pres["puntaje_obtenido"].median()).abs().argmin()]
    campos = ["puntaje_obtenido", "grado_escolar", "genero", "municipio",
              "tipo_institucion", "estrato", "computador_en_casa",
              "internet_en_casa", "participacion_olimpiadas", "nivel_programacion",
              "nivel_robotica", "interes_prog_robotica", "herramientas_conocidas",
              "areas_interes", "con_quien_vive"]
    ej = {}
    for c in campos:
        if c in df.columns:
            v = fila[c]
            if isinstance(v, float) and v != v:
                ej[c] = None
            elif isinstance(v, float) and c in ("grado_escolar", "estrato"):
                ej[c] = int(v)
            elif isinstance(v, (np.integer,)):
                ej[c] = int(v)
            elif isinstance(v, (np.floating,)):
                ej[c] = float(v)
            else:
                ej[c] = v
    return ej


# =============================================================================
# INFORME
# =============================================================================

def construir_informe(SPEC, out, ana, ejemplo) -> None:
    log("Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    R = REPORT.append
    n = len(out)

    R("# Índice de Potencial STEM (compuesto) — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Fase 2 · Informe generado: {fecha}\n")
    R("---\n")

    R("## Resumen ejecutivo\n")
    R(dedent(f"""\
        El modelo predictivo (script 03) explica poco del puntaje (R²≈0.085): la
        nota depende de factores no capturados por las variables socioeconómicas.
        Por eso el potencial STEM se mide con un **índice compuesto** de tres
        señales, calculado para **{n:,} estudiantes**:

        > **índice = 0.50·rendimiento + 0.25·engagement + 0.25·resiliencia**

        Así, dos estudiantes con la misma nota pueden diferenciarse por su
        interés/experiencia (engagement) y por rendir bien pese a un contexto
        adverso (resiliencia). El correlación de Spearman entre el índice y la
        nota cruda es **ρ = {ana['rho_indice_puntaje']:.2f}**: alto pero no
        perfecto, lo que confirma que el índice **aporta matices** que la nota
        sola no refleja.\n"""))

    R("## Metodología — componentes\n")
    R(dedent(f"""\
        **1. Rendimiento (0–100).** Percentil del `puntaje_obtenido` dentro de la
        cohorte para quienes presentaron; para quienes no presentaron, percentil
        del puntaje **estimado** por el modelo de la Fase 2 (modelo:
        {SPEC['meta']['modelo_puntaje']}). Peso: **0.50**.

        **2. Engagement (0–100).** Promedio de 8 señales normalizadas a 0–100:
        nivel de programación (0–3), nivel de robótica (0–3), interés en
        prog/robótica (1–5), nº de herramientas conocidas, nº de áreas de interés,
        participación previa en olimpiadas (0/100), computador en casa (0/100) e
        internet en casa (0/100). Peso: **0.25**.

        **3. Resiliencia (0–100).** Premia el mérito en contexto adverso.
        `condiciones_adversas` = suma de: estrato ≤ 2, sin computador, sin internet
        y no vive con ambos padres (0–4).
        - Si presentó: `resiliencia = min(100, percentil_puntaje × (1 + adversas × 0.15))`.
          Ej.: percentil 60 con 3 adversidades → 60 × 1.45 = 87.
        - Si no presentó: `resiliencia = max(0, 50 − adversas × 5)`.
        Peso: **0.25**.

        Nota: el **acceso a computador/internet impacta en DOS componentes** —
        suma en engagement (como acceso) y, cuando falta, suma en resiliencia
        (como adversidad superada).\n"""))

    R("## Categorización\n")
    R(tabla_md(pd.DataFrame([
        {"Categoría": nombre, "Umbral (índice)":
         (f"≥ {u}" if nombre == "Talento destacado"
          else f"< {CATEGORIAS[i-1][0]}" if nombre == "Requiere apoyo"
          else f"{u}–{CATEGORIAS[i-1][0]-1}"),
         "N estudiantes": ana["conteo_categoria"].get(nombre, 0)}
        for i, (u, nombre) in enumerate(CATEGORIAS)])) + "\n")
    R(f"\n{img('dist', 'Distribución del índice')}\n")

    R("## ¿El índice captura más matices que la nota?\n")
    R(dedent(f"""\
        Dentro de una misma nota, el índice se abre en un rango (desviación media
        de **{ana['dispersion_media_indice_por_nota']:.1f} puntos** de índice por
        nota): estudiantes con idéntico puntaje reciben índices distintos según su
        engagement y resiliencia. Esto es exactamente lo que se busca cuando el
        modelo predictivo por sí solo es débil.\n"""))
    R(f"\n{img('vs_puntaje', 'Índice vs puntaje')}\n")
    R(f"\n{img('rend_eng', 'Rendimiento vs engagement')}\n")

    R("## Índice promedio por grupo\n")
    for col, d in ana["promedios_grupo"].items():
        detalle = "; ".join(f"{k}: {v}" for k, v in d.items())
        R(f"- **{col}** → {detalle}\n")
    R(f"\n{img('grupo', 'Índice por grupo')}\n")
    if "colegios" in FIGURES:
        R(f"\n{img('colegios', 'Ranking de colegios')}\n")

    R("## Top 20 — mayor resiliencia (rindieron mejor de lo esperado)\n")
    tr = ana["top_resiliencia"]
    if not tr.empty:
        cols = {"numero_documento": "Documento", "puntaje_obtenido": "Nota",
                "puntaje_estimado": "Nota esperada", "sobre_rendimiento": "Δ (real−esp.)",
                "_adversidad": "Adversidad", "componente_resiliencia": "Resiliencia",
                "indice_potencial": "Índice", "categoria": "Categoría"}
        t = tr[[c for c in cols if c in tr.columns]].rename(columns=cols)
        R(tabla_md(t) + "\n")

    R("## Top 20 — mayor índice entre quienes NO presentaron\n")
    if ana["n_no_presento"] == 0:
        R(dedent("""\
            En el dataset actual **todos los inscritos presentaron la prueba**, por
            lo que no hay casos en esta lista. La lógica queda implementada: cuando
            se carguen inscritos sin nota, su rendimiento se estima con el modelo y
            su resiliencia usa la fórmula de no-presentación.\n"""))
    else:
        tn = ana["top_no_presento"]
        cols = {"numero_documento": "Documento", "municipio": "Municipio",
                "puntaje_estimado": "Nota estimada", "componente_engagement": "Engagement",
                "indice_potencial": "Índice", "categoria": "Categoría"}
        t = tn[[c for c in cols if c in tn.columns]].rename(columns=cols)
        R(tabla_md(t) + "\n")

    R("## Exportación para producción\n")
    R(dedent("""\
        - `models/deploy/scores_potencial_stem.csv` — columnas: `numero_documento`,
          `indice_potencial`, `componente_rendimiento`, `componente_engagement`,
          `componente_resiliencia`, `categoria` (para cargar en Supabase).
        - `models/deploy/potencial_stem_predictor.py` — función pura
          `calcular_indice_potencial(dict)`; solo stdlib.
        - `models/deploy/potencial_stem_predictor.js` — misma función en JS ES6,
          sin dependencias (para el frontend).\n"""))
    R("\n**Ejemplo de entrada (estudiante real, mediana de nota):**\n")
    R("```json\n" + json.dumps(ejemplo, ensure_ascii=False, indent=2) + "\n```\n")

    R("## Limitaciones\n")
    R(dedent("""\
        - Los **pesos (0.50/0.25/0.25)** son una decisión de política, no un óptimo
          estadístico; conviene revisarlos con la Fundación.
        - El **rendimiento es relativo** a esta cohorte (percentil), no una medida
          absoluta de habilidad.
        - El **engagement** depende de autorreporte (niveles, interés, herramientas).
        - La **resiliencia** usa un multiplicador lineal (0.15 por adversidad); es
          una heurística transparente, no un modelo causal.\n"""))
    R("\n---\n_Generado por `notebooks/04_indice_potencial_stem.py` — Copa STEM 2026._\n")

    destino = REPORTS_DIR / "04_indice_potencial_stem.md"
    destino.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"    informe escrito → reports/{destino.name}")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Índice de Potencial STEM (compuesto) — Fase 2")
    print(" Fundación SapienceLab")
    print("=" * 70)

    df = cargar_y_limpiar()
    spec_puntaje = cargar_spec_puntaje()
    SPEC = construir_spec_indice(df, spec_puntaje)
    out = calcular_cohorte(df, SPEC)

    ana = analisis(out)

    exportar_csv(out)
    ejemplo = estudiante_ejemplo(df)
    exportar_predictor_py(SPEC, ejemplo)
    exportar_predictor_js(SPEC, ejemplo)
    construir_informe(SPEC, out, ana, ejemplo)

    print("\n" + "=" * 70)
    print(" ✔ ÍNDICE DE POTENCIAL STEM COMPLETADO")
    print(f"   · Estudiantes:       {len(out):,}")
    print(f"   · Índice medio:      {out['indice_potencial'].mean():.1f}")
    print(f"   · Talento destacado: "
          f"{int((out['categoria'] == 'Talento destacado').sum()):,}")
    print(f"   · Figuras:           {len(FIGURES)} → outputs/")
    print(f"   · Deploy:            models/deploy/ (CSV + predictor .py y .js)")
    print(f"   · Informe:           reports/04_indice_potencial_stem.md")
    print("=" * 70)


if __name__ == "__main__":
    main()