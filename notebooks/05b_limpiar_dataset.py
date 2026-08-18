# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 05b: Limpieza del dataset (retirar exámenes recomendados para anular)
================================================================================

Paso intermedio entre la detección de trampa (script 05) y los análisis
posteriores. A partir de la recomendación de `models/deploy/sospecha_trampa.csv`
marca los exámenes a anular y genera dos versiones del dataset en `data/`:

    a) copa_stem_dataset_completo.csv  — original + columna booleana `anulado`.
    b) copa_stem_dataset_limpio.csv    — SIN los anulados (default para 03/04…).

No borra ningún script ni el CSV original. Reproducible (`random_state=42`).

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DEPLOY_DIR = BASE_DIR / "models" / "deploy"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL = DATA_DIR / "copa_stem_dataset.csv"
SOSPECHA = DEPLOY_DIR / "sospecha_trampa.csv"
OUT_COMPLETO = DATA_DIR / "copa_stem_dataset_completo.csv"
OUT_LIMPIO = DATA_DIR / "copa_stem_dataset_limpio.csv"

COLORS = {"cyan": "#00d4ff", "violet": "#8b5cf6", "amber": "#f59e0b",
          "green": "#10b981", "red": "#ef4444", "blue": "#0f77ee"}
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


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Limpieza del dataset (retirar anulados)")
    print(" Fundación SapienceLab")
    print("=" * 70)

    if not ORIGINAL.exists():
        print(f"\n  ⚠  No se encontró el dataset original: {ORIGINAL}\n")
        sys.exit(1)
    if not SOSPECHA.exists():
        print(f"\n  ⚠  No se encontró {SOSPECHA}. Ejecute antes "
              "notebooks/05_deteccion_trampa.py.\n")
        sys.exit(1)

    # numero_documento como str en ambos lados para un emparejamiento seguro.
    log("Carga del dataset original y de la recomendación de trampa")
    df = pd.read_csv(ORIGINAL, encoding="utf-8", dtype={"numero_documento": str})
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    log(f"    dataset original: {len(df):,} filas | {df.shape[1]} columnas")

    sosp = pd.read_csv(SOSPECHA, dtype={"numero_documento": str})
    sosp["numero_documento"] = sosp["numero_documento"].astype(str).str.strip()

    # Conjunto de documentos recomendados para anular (case-insensitive: 'anular').
    # Se usa un SET (no un merge) para no duplicar filas si hay documentos
    # repetidos; los documentos a anular son únicos, así que es exacto.
    anular_docs = set(sosp.loc[
        sosp["recomendacion"].astype(str).str.strip().str.lower() == "anular",
        "numero_documento"])
    log(f"    documentos recomendados para anular: {len(anular_docs):,}")

    # --- Columna 'anulado' ---------------------------------------------------
    df["anulado"] = df["numero_documento"].isin(anular_docs)
    n_anulados = int(df["anulado"].sum())

    # Chequeo de integridad: cada doc a anular debe existir en el dataset.
    no_encontrados = anular_docs - set(df["numero_documento"])
    if no_encontrados:
        log(f"    ⚠ {len(no_encontrados)} documentos a anular no están en el "
            f"dataset (se ignoran).")

    # --- Guardar versión completa (original + anulado) -----------------------
    # Se usa UTF-8 SIN BOM (como el CSV original) para que 03/04 lo relean con
    # encoding="utf-8" sin ensuciar el nombre de la primera columna.
    df.to_csv(OUT_COMPLETO, index=False, encoding="utf-8")
    log(f"    guardado → data/{OUT_COMPLETO.name} ({len(df):,} filas, con 'anulado')")

    # --- Guardar versión limpia (sin anulados) -------------------------------
    limpio = df[~df["anulado"]].copy()
    limpio.to_csv(OUT_LIMPIO, index=False, encoding="utf-8")
    log(f"    guardado → data/{OUT_LIMPIO.name} ({len(limpio):,} filas, sin anulados)")

    # --- Resumen -------------------------------------------------------------
    pj = pd.to_numeric(df["puntaje_obtenido"], errors="coerce")
    pj_limpio = pd.to_numeric(limpio["puntaje_obtenido"], errors="coerce")
    media_antes = float(pj.mean())
    media_despues = float(pj_limpio.mean())

    print("\n" + "-" * 70)
    print(" RESUMEN DE LIMPIEZA")
    print("-" * 70)
    print(f"  Total original:        {len(df):,}")
    print(f"  Removidos (anulados):  {n_anulados:,} "
          f"({100 * n_anulados / len(df):.1f}%)")
    print(f"  Nuevo total (limpio):  {len(limpio):,}")
    print(f"  Promedio puntaje ANTES:   {media_antes:.2f}")
    print(f"  Promedio puntaje DESPUÉS: {media_despues:.2f} "
          f"({media_despues - media_antes:+.2f})")
    print("-" * 70)

    # --- Gráfico: distribución de puntaje completo vs. limpio ----------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(pj.dropna(), bins=25, color=COLORS["amber"], alpha=0.55,
                 label=f"Completo (n={pj.notna().sum()}, µ={media_antes:.1f})",
                 stat="density", edgecolor="white", ax=ax)
    sns.histplot(pj_limpio.dropna(), bins=25, color=COLORS["cyan"], alpha=0.55,
                 label=f"Limpio (n={pj_limpio.notna().sum()}, µ={media_despues:.1f})",
                 stat="density", edgecolor="white", ax=ax)
    ax.axvline(media_antes, color=COLORS["amber"], linestyle="--", linewidth=1.5)
    ax.axvline(media_despues, color=COLORS["blue"], linestyle="--", linewidth=1.5)
    ax.set_title(f"Distribución de puntaje: completo vs. limpio\n"
                 f"{n_anulados} exámenes anulados retirados")
    ax.set_xlabel("Puntaje")
    ax.set_ylabel("Densidad")
    ax.legend()
    path = OUTPUTS_DIR / "F05b_completo_vs_limpio.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"    figura guardada → outputs/{path.name}")

    print("\n" + "=" * 70)
    print(" ✔ LIMPIEZA COMPLETADA")
    print(f"   · data/{OUT_COMPLETO.name}  (con columna 'anulado')")
    print(f"   · data/{OUT_LIMPIO.name}   (default para análisis siguientes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
