# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 16: Visualizaciones de la comparación v1 vs v2 (Fase 5)
================================================================================

Acompaña al informe `reports/15_scores_v2_comparacion.md`. No entrena ni
recalcula nada: lee los artefactos ya generados por los scripts 13 y 15 y
produce cuatro figuras.

Gráficos
--------
    1) comparacion_categorias_v1_v2.png   — conteo por categoría, v1 vs v2.
    2) distribucion_potencial_v1_v2.png   — histogramas superpuestos del índice.
    3) puntaje_estimado_vs_real.png       — estimado v2 contra real, con
                                            diagonal de predicción perfecta.
    4) feature_importance_comparacion.png — top 10 por permutación en A vs C.

Fuentes
-------
    · `outputs/F15_comparacion_v1_v2.csv` — 3,072 filas; se filtran las 1,148
      del subgrupo v2 (`modelo_version == "v2"`). Las 1,924 restantes son
      fallback v1 y por construcción no cambian.
    · `outputs/F13_importancias.csv` — importancia por permutación (ΔR²) en los
      datasets A y C.

Advertencia de lectura del gráfico 3
------------------------------------
El MAE que se ve en ese scatter es DENTRO de muestra (12.54): el modelo v2 de
producción se reajustó sobre estas mismas 1,148 filas. La cifra honesta es la
del hold-out del script 14 (15.00). Ambas se anotan en la figura para que nadie
tome la optimista por buena.

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import gaussian_kde
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"cyan": "#00d4ff", "violet": "#8b5cf6", "amber": "#f59e0b",
          "dark": "#050816", "green": "#10b981", "red": "#ef4444",
          "blue": "#0f77ee"}
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

# Colores de versión: v1 en cian (statu quo), v2 en violeta (propuesta).
C_V1, C_V2 = COLORS["cyan"], COLORS["violet"]

# Orden fijo de las categorías, de mejor a peor. Se declara explícito para que
# el eje no dependa del orden alfabético ni de la frecuencia observada.
ORDEN_CATEGORIAS = ["Talento destacado", "Alto potencial", "Promedio",
                    "En desarrollo", "Requiere apoyo"]

# Color por categoría para el scatter (verde = mejor, rojo = peor).
COLOR_CATEGORIA = {
    "Talento destacado": COLORS["green"],
    "Alto potencial":    COLORS["cyan"],
    "Promedio":          COLORS["blue"],
    "En desarrollo":     COLORS["amber"],
    "Requiere apoyo":    COLORS["red"],
}

# MAE de hold-out del script 14: la única cifra de v2 comparable con la de v1.
MAE_V2_HOLDOUT = 15.00
MAE_V1_OOS = 18.45


def log(msg: str) -> None:
    print(f">>> {msg}")


def guardar(fig, nombre: str) -> None:
    """Guarda la figura en outputs/ y confirma en consola."""
    ruta = OUTPUTS_DIR / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    ok = ruta.exists()
    kb = ruta.stat().st_size / 1024 if ok else 0
    print(f"    GUARDADO: outputs/{nombre}  ({'OK' if ok else 'FALLO'}, {kb:,.0f} KB)")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
log("Paso 1: Cargando artefactos de los scripts 13 y 15...")

f15 = OUTPUTS_DIR / "F15_comparacion_v1_v2.csv"
f13 = OUTPUTS_DIR / "F13_importancias.csv"
for f in (f15, f13):
    if not f.exists():
        print(f"ERROR: no se encuentra {f}. Corre antes los scripts 13 y 15.")
        sys.exit(1)

comp = pd.read_csv(f15)
imp = pd.read_csv(f13)

# El subgrupo v2 son las 1,148 filas con perfil académico. El resto es fallback
# v1 (no-op exacto, verificado en el informe 15) y no aporta nada a comparar.
sub = comp[comp["modelo_version"] == "v2"].copy()
n = len(sub)
log(f"Subgrupo v2: {n:,} estudiantes (de {len(comp):,} con examen presentado)")


# ---------------------------------------------------------------------------
# Gráfico 1 — Conteo por categoría, v1 vs v2
# ---------------------------------------------------------------------------
log("Paso 2: Gráfico 1 — comparación de categorías...")

c_v1 = sub["categoria_v1"].value_counts().reindex(ORDEN_CATEGORIAS, fill_value=0)
c_v2 = sub["categoria_v2"].value_counts().reindex(ORDEN_CATEGORIAS, fill_value=0)
delta = c_v2 - c_v1

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(ORDEN_CATEGORIAS))
w = 0.38
b1 = ax.bar(x - w / 2, c_v1.values, w, label="v1 (producción actual)",
            color=C_V1, edgecolor="white", linewidth=0.8)
b2 = ax.bar(x + w / 2, c_v2.values, w, label="v2 (híbrido, en sombra)",
            color=C_V2, edgecolor="white", linewidth=0.8)

for barras in (b1, b2):
    ax.bar_label(barras, padding=3, fontsize=9, fontweight="bold")

# La delta es el dato que importa: cuántos estudiantes cambian de etiqueta sin
# que su desempeño real haya cambiado. Se anota sobre cada par.
techo = max(c_v1.max(), c_v2.max())
for i, d_cat in enumerate(delta.values):
    color = COLORS["green"] if d_cat > 0 else (COLORS["red"] if d_cat < 0 else "#777777")
    ax.text(i, techo * 1.10, f"{d_cat:+d}", ha="center", fontsize=11,
            fontweight="bold", color=color)

ax.text(len(ORDEN_CATEGORIAS) / 2 - 0.5, techo * 1.19, "Delta v2 - v1",
        ha="center", fontsize=9, color="#555555", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(ORDEN_CATEGORIAS, fontsize=10)
ax.set_ylabel("Estudiantes")
ax.set_ylim(0, techo * 1.26)
ax.set_title(f"Categoría de potencial STEM: v1 vs v2  (n = {n:,} con perfil académico)")
ax.legend(loc="upper right", frameon=True)
fig.text(0.5, -0.035,
         "El centro se vacía y los extremos se llenan. El desempeño real de estos estudiantes "
         "no cambió:\nlo que cambia es la cohorte de referencia con la que se percentiliza "
         "(1,750 en v1 → 1,148 en v2).",
         ha="center", fontsize=9, color="#555555")
guardar(fig, "comparacion_categorias_v1_v2.png")


# ---------------------------------------------------------------------------
# Gráfico 2 — Distribución del índice de potencial
# ---------------------------------------------------------------------------
log("Paso 3: Gráfico 2 — distribución del índice de potencial...")

fig, ax = plt.subplots(figsize=(11, 6))
# Bins de 4 puntos. Con 40 bins el histograma sale muy dentado y el ruido de
# conteo tapa el mensaje, que es cuánto se solapan las dos distribuciones.
bins = np.linspace(0, 100, 26)

ax.hist(sub["indice_v1"], bins=bins, alpha=0.5, color=C_V1,
        label=f"v1  (media {sub['indice_v1'].mean():.2f} · sigma {sub['indice_v1'].std():.2f})",
        edgecolor="white", linewidth=0.5)
ax.hist(sub["indice_v2"], bins=bins, alpha=0.5, color=C_V2,
        label=f"v2  (media {sub['indice_v2'].mean():.2f} · sigma {sub['indice_v2'].std():.2f})",
        edgecolor="white", linewidth=0.5)

# Curvas de densidad para leer la forma sin el ruido de los conteos. Se escalan
# a la altura del histograma (n × ancho de bin) para que compartan eje.
escala = len(sub) * (bins[1] - bins[0])
rejilla = np.linspace(0, 100, 400)
for serie, color in ((sub["indice_v1"], C_V1), (sub["indice_v2"], C_V2)):
    dens = gaussian_kde(serie.values)(rejilla) * escala
    ax.plot(rejilla, dens, color=color, linewidth=2.4)

ax.axvline(sub["indice_v1"].mean(), color=C_V1, linestyle="--", linewidth=2)
ax.axvline(sub["indice_v2"].mean(), color=C_V2, linestyle="--", linewidth=2)

d = sub["indice_v2"] - sub["indice_v1"]
ax.set_xlabel("Índice de potencial STEM (0–100)")
ax.set_ylabel("Estudiantes")
ax.set_xlim(0, 100)
ax.set_title(f"Distribución del índice de potencial: v1 vs v2  (n = {n:,})")
ax.legend(loc="upper right", frameon=True)

# Recuadro con el resumen del cambio: la magnitud es pequeña, y conviene que se
# lea junto al histograma para no sobreinterpretar el ensanchamiento visual.
sin_cambio_pct = (sub["categoria_v1"] == sub["categoria_v2"]).mean() * 100
resumen = (f"Cambio medio: {d.mean():+.2f} pts\n"
           f"Mediana: {d.median():+.2f}   ·   Rango: {d.min():+.2f} … {d.max():+.2f}\n"
           f"Sin cambio de categoría: {sin_cambio_pct:.1f} %")
ax.text(0.015, 0.97, resumen, transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                  edgecolor="#cccccc"))

fig.text(0.5, -0.035,
         "v2 es ligeramente más ancha: los mismos puntajes reales se reparten sobre un rango "
         "percentil mayor.\nEl modelo v2 no interviene aquí — estos 1,148 presentaron el examen, "
         "así que el índice usa su puntaje REAL.",
         ha="center", fontsize=9, color="#555555")
guardar(fig, "distribucion_potencial_v1_v2.png")


# ---------------------------------------------------------------------------
# Gráfico 3 — Puntaje estimado v2 contra puntaje real
# ---------------------------------------------------------------------------
log("Paso 4: Gráfico 3 — puntaje estimado v2 vs puntaje real...")

fig, ax = plt.subplots(figsize=(9.5, 8.5))

for cat in ORDEN_CATEGORIAS:
    m = sub["categoria_v2"] == cat
    ax.scatter(sub.loc[m, "puntaje_real"], sub.loc[m, "puntaje_est_v2"],
               s=26, alpha=0.62, color=COLOR_CATEGORIA[cat],
               edgecolors="white", linewidths=0.3,
               label=f"{cat} (n = {int(m.sum())})")

# Diagonal de predicción perfecta.
ax.plot([0, 100], [0, 100], color=COLORS["dark"], linestyle="--", linewidth=1.8,
        label="Predicción perfecta (y = x)", zorder=5)

mae_in = (sub["puntaje_est_v2"] - sub["puntaje_real"]).abs().mean()
r = np.corrcoef(sub["puntaje_real"], sub["puntaje_est_v2"])[0, 1]

ax.set_xlabel("Puntaje obtenido real (0–100)")
ax.set_ylabel("Puntaje estimado por el modelo v2 (0–100)")
ax.set_xlim(-3, 103)
ax.set_ylim(-3, 103)
ax.set_aspect("equal", adjustable="box")
ax.set_title(f"Puntaje estimado v2 vs puntaje real  (n = {n:,})")
ax.legend(loc="upper left", frameon=True, fontsize=8.5)

# La nube es horizontal, no diagonal: el modelo comprime hacia la media. Se
# anota explícitamente porque es el hallazgo que el gráfico transmite.
nota = (f"MAE in-sample: {mae_in:.2f}  <- NO usar (el modelo se\n"
        f"   reajusto sobre estas mismas 1,148 filas)\n"
        f"MAE hold-out v2 (script 14): {MAE_V2_HOLDOUT:.2f}  <- cifra honesta\n"
        f"MAE v1 fuera de muestra:     {MAE_V1_OOS:.2f}\n"
        f"r = {r:.3f}  ·  sigma estimado {sub['puntaje_est_v2'].std():.1f} "
        f"vs sigma real {sub['puntaje_real'].std():.1f}")
ax.text(0.985, 0.03, nota, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe8",
                  edgecolor=COLORS["amber"]))

fig.text(0.5, -0.02,
         "La nube es casi horizontal: el modelo predice entre 21 y 71 puntos mientras la realidad "
         "va de 0 a 100.\nCon R² ≈ 0.18 la predicción tiende a la media — sirve para ordenar "
         "grupos, no para pronosticar a un estudiante.",
         ha="center", fontsize=9, color="#555555")
guardar(fig, "puntaje_estimado_vs_real.png")


# ---------------------------------------------------------------------------
# Gráfico 4 — Top 10 de importancia por permutación: A vs C
# ---------------------------------------------------------------------------
log("Paso 5: Gráfico 4 — importancia de variables, modelo A vs modelo C...")

# Las 5 variables del bloque de perfil académico, para resaltarlas en C.
NUEVAS = {"Promedio académico", "Horas de estudio (mat.)",
          "Motivación para participar", "Gusto por la lógica",
          "Toma clases extra de mat."}

casos = [
    ("A", "Modelo A — cohorte baseline\n(n = 1,735 · sin perfil académico)", C_V1),
    ("C", "Modelo C — con perfil académico\n(n = 1,148 · base del modelo v2)", C_V2),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)

for ax, (clave, titulo, color_base) in zip(axes, casos):
    top = (imp[imp["dataset"] == clave]
           .sort_values("puesto")
           .head(10)
           .iloc[::-1])  # invertido: el puesto 1 queda arriba en barh

    colores = [COLORS["amber"] if v in NUEVAS else color_base
               for v in top["variable"]]
    y = np.arange(len(top))
    ax.barh(y, top["delta_r2"], xerr=top["std"], color=colores,
            edgecolor="white", linewidth=0.8,
            error_kw=dict(ecolor="#666666", lw=1.1, capsize=3))
    ax.set_yticks(y)
    ax.set_yticklabels(top["variable"], fontsize=9.5)

    # La etiqueta va detrás de la barra de error, no encima: si se ancla al
    # valor, el bigote la tacha.
    for yi, val, sd in zip(y, top["delta_r2"], top["std"]):
        ax.text(val + sd + 0.005, yi, f"{val:.4f}", va="center", fontsize=8.5,
                color="#333333")

    ax.set_title(titulo, fontsize=11.5)
    ax.set_xlabel("Caída de R² al desordenar la variable (delta R²)")
    ax.axvline(0, color="#999999", linewidth=1)

axes[0].set_xlim(0, 0.15)

fig.suptitle("Top 10 de importancia por permutación: modelo A vs modelo C",
             fontsize=14, fontweight="bold", y=1.05)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=C_V1),
    plt.Rectangle((0, 0), 1, 1, color=C_V2),
    plt.Rectangle((0, 0), 1, 1, color=COLORS["amber"]),
]
fig.legend(handles,
           ["Variable preexistente (modelo A)", "Variable preexistente (modelo C)",
            "Variable nueva de perfil académico"],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
           fontsize=9.5)

fig.text(0.5, -0.05,
         "Sin perfil académico (A) el modelo se apoya en el CONTEXTO: municipio y grado encabezan. "
         "Con perfil (C),\n'Promedio académico' los desplaza y pesa 2.7x más que la segunda "
         "variable. Barras de error = ±1 desv. sobre 30 permutaciones:\n"
         "donde la barra de error supera a la propia barra, la importancia no es distinguible "
         "de cero.",
         ha="center", fontsize=9, color="#555555")
guardar(fig, "feature_importance_comparacion.png")


# ---------------------------------------------------------------------------
log("Listo. 4 figuras generadas en outputs/.")
for nombre in ("comparacion_categorias_v1_v2.png",
               "distribucion_potencial_v1_v2.png",
               "puntaje_estimado_vs_real.png",
               "feature_importance_comparacion.png"):
    p = OUTPUTS_DIR / nombre
    print(f"    - outputs/{nombre}  ->  {'existe' if p.exists() else 'NO EXISTE'}")
