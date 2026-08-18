# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 09: Exportar puntaje_estimado por estudiante  (Fase 4 — Despliegue)
================================================================================

Objetivo
--------
Generar, para CADA estudiante, el `puntaje_estimado` del modelo predictivo
ganador (Random Forest del script 03) y exportarlo como CSV listo para subir a
Supabase. Para quienes presentaron el examen se añade la diferencia
`puntaje_real - puntaje_estimado` (resiliencia CRUDA): positiva = rindió mejor
de lo que el modelo esperaba dadas sus condiciones socioeconómicas/demográficas.

Qué hace
--------
    1. Carga data/copa_stem_dataset_limpio.csv (mismo criterio de limpieza que 03).
    2. Carga el modelo ganador de models/mejor_modelo_puntaje.joblib
       (Random Forest + preprocesador ajustado en el train de 03).
    3. Prepara las features con la MISMA pipeline del script 03 (importado como
       módulo) — imputación y one-hot idénticas, sin fuga de datos.
    4. Predice el puntaje (0–100) de cada estudiante (presentó o no) y calcula la
       diferencia contra el puntaje real cuando existe.
    5. Exporta models/deploy/puntaje_estimado.csv:
         numero_documento, puntaje_estimado, puntaje_real, diferencia, interpretacion
    6. Imprime estadísticas de resumen.

Notas
-----
- Reproducible: `random_state=42` (heredado del modelo y del preprocesador de 03).
- El `puntaje_estimado` que produce el modelo cargado se verifica contra el
  predictor puro `models/predictor.py` para garantizar consistencia con producción.
- En el dataset LIMPIO todos los inscritos presentaron; el código igualmente
  soporta `puntaje_real` NULL (se exporta vacío y sin diferencia).

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
import warnings
import importlib.util
from datetime import datetime
from pathlib import Path

RANDOM_STATE = 42

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
except ImportError as exc:  # pragma: no cover
    print("ERROR: falta una dependencia del entorno.")
    print(f"       Detalle: {exc}")
    print("       Instale: pandas numpy scikit-learn matplotlib seaborn joblib")
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
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
for _d in (OUTPUTS_DIR, DEPLOY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET = "copa_stem_dataset_limpio.csv"
MODELO_JOBLIB = MODELS_DIR / "mejor_modelo_puntaje.joblib"
PREDICTOR_PURO = MODELS_DIR / "predictor.py"
SALIDA_CSV = DEPLOY_DIR / "puntaje_estimado.csv"

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
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.autolayout": True,
})
DPI = 150

# Umbral de "dentro de lo esperado": ±5 puntos alrededor de la predicción.
UMBRAL = 5.0


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def _import_por_ruta(nombre_modulo: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# 1–3. CARGA DE DATOS, MODELO Y PIPELINE DE 03
# =============================================================================

def cargar_infra():
    """Importa el script 03 para reutilizar EXACTAMENTE su pipeline de features."""
    ruta03 = BASE_DIR / "notebooks" / "03_modelo_predictivo.py"
    if not ruta03.exists():
        print(f"\n  ⚠  No se encontró {ruta03}. Ejecute antes el script 03.\n")
        sys.exit(1)
    return _import_por_ruta("modelo03", ruta03)


def cargar_modelo():
    """Carga el bundle del modelo ganador (modelo + preprocesador + feature_names)."""
    if not MODELO_JOBLIB.exists():
        print(f"\n  ⚠  No se encontró {MODELO_JOBLIB}. Ejecute antes el script 03.\n")
        sys.exit(1)
    bundle = joblib.load(MODELO_JOBLIB)
    log(f"    modelo cargado: {bundle.get('nombre', '¿?')} "
        f"({len(bundle['feature_names'])} features)")
    return bundle


def cargar_datos(m03) -> pd.DataFrame:
    """Carga y limpia el dataset con el MISMO criterio del script 03."""
    log("Paso 1 — Carga y limpieza del dataset")
    # cargar_y_limpiar() de 03 usa por defecto el dataset limpio; forzamos el
    # nombre pedido de forma explícita para dejar constancia.
    ruta = DATA_DIR / DATASET
    if not ruta.exists():
        print(f"\n  ⚠  No se encontró {ruta}.\n")
        sys.exit(1)
    df = m03.cargar_y_limpiar()  # aplica limpieza idéntica a 03
    log(f"    registros tras limpieza: {len(df):,}")

    # Deduplicar por numero_documento (PK en Supabase). Conservamos la 1ª aparición.
    antes = len(df)
    df = df.drop_duplicates(subset="numero_documento", keep="first").reset_index(drop=True)
    if len(df) < antes:
        log(f"    documentos duplicados eliminados: {antes - len(df)} "
            f"(se conservó la primera aparición)")
    return df


# =============================================================================
# 4. PREDICCIÓN POR ESTUDIANTE
# =============================================================================

def predecir_todos(df: pd.DataFrame, m03, bundle) -> pd.DataFrame:
    log("Paso 4 — Predicción del puntaje para cada estudiante")
    modelo = bundle["modelo"]
    PRE = bundle["preprocessor"]     # preprocesador ajustado en el train de 03
    TARGET = m03.TARGET

    registros = df.to_dict("records")
    X = m03.build_matrix(registros, PRE)
    est = np.clip(modelo.predict(X), 0.0, 100.0)

    # Verificación de consistencia con el predictor puro de producción.
    max_diff = _verificar_contra_predictor_puro(registros, est)
    if max_diff is not None:
        log(f"    consistencia con models/predictor.py: máx|Δ| = {max_diff:.4g} pts")

    real = pd.to_numeric(df[TARGET], errors="coerce") if TARGET in df.columns \
        else pd.Series([np.nan] * len(df))
    real = real.to_numpy(dtype=float)

    diferencia = np.where(np.isnan(real), np.nan, real - est)

    out = pd.DataFrame({
        "numero_documento": df["numero_documento"].astype(str).values,
        "puntaje_estimado": np.round(est, 2),
        "puntaje_real": np.round(real, 2),
        "diferencia": np.round(diferencia, 2),
    })
    out["interpretacion"] = [interpretar(d) for d in diferencia]
    return out


def interpretar(diff: float) -> str:
    """Etiqueta legible de la diferencia real − estimado."""
    if diff is None or (isinstance(diff, float) and np.isnan(diff)):
        return "Sin examen presentado"
    d = int(round(diff))
    if diff > UMBRAL:
        return f"Superó expectativas (+{d})"
    if diff < -UMBRAL:
        return f"Por debajo ({d})"
    return f"Dentro de lo esperado (±{int(UMBRAL)})"


def _verificar_contra_predictor_puro(registros: list[dict], est: np.ndarray):
    """Compara la predicción del modelo cargado con el predictor puro (opcional)."""
    if not PREDICTOR_PURO.exists():
        return None
    try:
        pred_mod = _import_por_ruta("predictor_puntaje", PREDICTOR_PURO)
        puro = np.array([pred_mod.predecir_puntaje(r) for r in registros])
        return float(np.max(np.abs(np.asarray(est) - puro)))
    except Exception as exc:  # pragma: no cover
        log(f"    (aviso) no se pudo verificar contra el predictor puro: {exc}")
        return None


# =============================================================================
# 5. EXPORTACIÓN CSV
# =============================================================================

def exportar_csv(out: pd.DataFrame) -> None:
    log("Paso 5 — Exportación del CSV para Supabase")
    # na_rep="" → los NULL (no presentó) quedan como celda vacía en el CSV.
    out.to_csv(SALIDA_CSV, index=False, encoding="utf-8-sig", na_rep="")
    log(f"    CSV exportado ({len(out):,} estudiantes) → "
        f"models/deploy/{SALIDA_CSV.name}")


# =============================================================================
# 6. ESTADÍSTICAS + FIGURA
# =============================================================================

def estadisticas(out: pd.DataFrame) -> dict:
    presentaron = out[out["puntaje_real"].notna()]
    n_total = len(out)
    n_pres = len(presentaron)

    prom_est = float(out["puntaje_estimado"].mean())
    prom_real = float(presentaron["puntaje_real"].mean()) if n_pres else float("nan")
    prom_dif = float(presentaron["diferencia"].mean()) if n_pres else float("nan")

    superaron = int((presentaron["diferencia"] > UMBRAL).sum())
    debajo = int((presentaron["diferencia"] < -UMBRAL).sum())
    dentro = n_pres - superaron - debajo

    print("\n" + "-" * 70)
    print(" ESTADÍSTICAS DEL PUNTAJE ESTIMADO")
    print("-" * 70)
    print(f"  Estudiantes exportados:            {n_total:,}")
    print(f"  Presentaron el examen:             {n_pres:,}")
    print(f"  No presentaron (solo estimado):    {n_total - n_pres:,}")
    print(f"  Promedio puntaje_estimado:         {prom_est:.2f}")
    if n_pres:
        print(f"  Promedio puntaje_real:             {prom_real:.2f}")
        print(f"  Promedio diferencia (real−est):    {prom_dif:+.2f}")
        print(f"  → Superaron expectativas (>+{int(UMBRAL)}):    "
              f"{superaron:,} ({superaron / n_pres:.1%})")
        print(f"  → Dentro de lo esperado (±{int(UMBRAL)}):      "
              f"{dentro:,} ({dentro / n_pres:.1%})")
        print(f"  → Por debajo (<-{int(UMBRAL)}):               "
              f"{debajo:,} ({debajo / n_pres:.1%})")
    print("-" * 70)

    return {"n_total": n_total, "n_pres": n_pres, "prom_est": prom_est,
            "prom_real": prom_real, "prom_dif": prom_dif,
            "superaron": superaron, "dentro": dentro, "debajo": debajo}


def figura(out: pd.DataFrame, stats: dict) -> None:
    presentaron = out[out["puntaje_real"].notna()]
    if not len(presentaron):
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))

    # (a) Distribución de la diferencia real − estimado.
    dif = presentaron["diferencia"]
    sns.histplot(dif, bins=35, color=COLORS["cyan"], edgecolor="white", ax=ax1)
    ax1.axvline(0, color=COLORS["dark"], linewidth=1.2)
    ax1.axvline(UMBRAL, color=COLORS["green"], linestyle="--", linewidth=1.4,
                label=f"±{int(UMBRAL)} pts (esperado)")
    ax1.axvline(-UMBRAL, color=COLORS["red"], linestyle="--", linewidth=1.4)
    ax1.axvline(float(dif.mean()), color=COLORS["violet"], linewidth=2,
                label=f"Media = {dif.mean():+.2f}")
    ax1.set_title("Diferencia real − estimado (resiliencia cruda)")
    ax1.set_xlabel("puntaje_real − puntaje_estimado")
    ax1.set_ylabel("N estudiantes")
    ax1.legend()

    # (b) Conteo por interpretación.
    orden = ["Superó expectativas", "Dentro de lo esperado", "Por debajo"]
    vals = [stats["superaron"], stats["dentro"], stats["debajo"]]
    cols = [COLORS["green"], COLORS["amber"], COLORS["red"]]
    ax2.bar(orden, vals, color=cols, edgecolor="white")
    ax2.set_title("Estudiantes por interpretación")
    ax2.set_ylabel("N estudiantes")
    ax2.tick_params(axis="x", rotation=12)
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Puntaje estimado vs. real — Copa STEM 2026",
                 fontsize=15, fontweight="bold")
    ruta = OUTPUTS_DIR / "F09_puntaje_estimado.png"
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"    figura guardada → outputs/{ruta.name}")


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Exportar puntaje_estimado por estudiante")
    print(" Fundación SapienceLab")
    print("=" * 70)

    m03 = cargar_infra()
    bundle = cargar_modelo()
    df = cargar_datos(m03)
    out = predecir_todos(df, m03, bundle)
    exportar_csv(out)
    stats = estadisticas(out)
    figura(out, stats)

    print("\n" + "=" * 70)
    print(" ✔ EXPORTACIÓN COMPLETADA")
    print(f"   · CSV:      models/deploy/{SALIDA_CSV.name} ({stats['n_total']:,} filas)")
    print(f"   · Columnas: numero_documento, puntaje_estimado, puntaje_real, "
          f"diferencia, interpretacion")
    print(f"   · Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)


if __name__ == "__main__":
    main()