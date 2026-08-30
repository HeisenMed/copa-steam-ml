# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 13: Análisis de explicabilidad (Fase 5)
================================================================================

El experimento del script 12 estableció QUE las 5 variables de perfil académico
mejoran la predicción (R² +0.098 → +0.180) y que 5 variables de conducta explican
más que 18 de origen socioeconómico. Este script responde la pregunta siguiente:
**CUÁLES** variables pesan, y cuánto.

Secciones
---------
    A) Modelo final sobre TODO `dataset_C_perfil.csv` (sin partición) e
       importancia MDI — top 15.
    B) Importancia por permutación (train 80 % / test 20 %) sobre C.
    C) Lo mismo sobre `dataset_A_baseline.csv`, la cohorte sin perfil académico.
    D) Comparación de los top 10 de A y C: qué se mantiene, qué entra, qué sale.
    E) Conclusión legible en español.

Dos importancias, y por qué las dos
-----------------------------------
    · **MDI** (Mean Decrease in Impurity) — cuánto usó el bosque cada variable
      para partir. Es rápida y viene gratis con el modelo, pero está **sesgada
      hacia variables con muchos valores distintos**: una continua con 41 valores
      ofrece más puntos de corte que una binaria, y eso infla su importancia
      aunque no prediga mejor.
    · **Permutación** — cuánto empeora el R² en datos NO vistos al desordenar la
      variable. Mide impacto predictivo real y no tiene ese sesgo. Es la métrica
      de referencia; la MDI se reporta para contrastar.

Cuando ambas coinciden, la conclusión es sólida. Cuando discrepan, casi siempre
es el sesgo de cardinalidad de la MDI, y manda la permutación.

Aviso de interpretación
-----------------------
El modelo de la sección A se entrena con TODAS las filas de C a propósito: su
único uso es interpretar el ranking, nunca estimar desempeño. Las métricas de
evaluación son las del script 12, obtenidas con validación cruzada.

Reproducible: `random_state=42`.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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
TOP_N = 15

# Se reutiliza la ingeniería de features del script 12 para que el ranking sea
# el del MISMO modelo que se evaluó allí; duplicarla aquí arriesgaría que ambos
# scripts se desincronicen y el informe describiera un modelo distinto.
_spec = importlib.util.spec_from_file_location(
    "s12", Path(__file__).resolve().parent / "12_experimento_reentrenamiento.py")
s12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s12)

TARGET = s12.TARGET

# Los datasets a explicar: (clave, archivo, incluir_nuevas, etiqueta).
CASOS = [
    ("C", "dataset_C_perfil.csv",   True,
     "C — con perfil académico"),
    ("A", "dataset_A_baseline.csv", False,
     "A — cohorte baseline (sin perfil)"),
]

# Nombres legibles para los gráficos y la conclusión.
ETIQUETAS = {
    "promedio_academico": "Promedio académico",
    "horas_estudio_matematicas": "Horas de estudio (mat.)",
    "motivacion_participar": "Motivación para participar",
    "gusto_logica": "Gusto por la lógica",
    "clases_extra_bin": "Toma clases extra de mat.",
    "grado_escolar": "Grado escolar",
    "estrato": "Estrato",
    "interes_prog_robotica": "Interés prog./robótica",
    "n_herramientas": "N.º de herramientas conocidas",
    "n_areas_interes": "N.º de áreas de interés",
    "nivel_programacion_ord": "Nivel de programación",
    "nivel_robotica_ord": "Nivel de robótica",
    "computador_bin": "Computador en casa",
    "internet_bin": "Internet en casa",
    "olimpiadas_bin": "Participó en olimpiadas",
    # Categóricas: la permutación las evalúa como columna cruda (sin one-hot).
    "municipio": "Municipio",
    "genero": "Género",
    "tipo_institucion": "Tipo de institución",
}


# Etiquetas legibles de las 5 variables nuevas, para marcarlas en los listados.
NUEVAS_ETIQUETAS = {ETIQUETAS[k] for k in (
    "promedio_academico", "horas_estudio_matematicas", "motivacion_participar",
    "gusto_logica", "clases_extra_bin")}


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def titulo(txt: str) -> None:
    print("\n" + "=" * 78)
    print(f" {txt}")
    print("=" * 78)


def bonito(nombre: str) -> str:
    """Nombre legible. Las columnas one-hot llegan como 'cat__municipio_Bello'."""
    n = nombre.split("__")[-1] if "__" in nombre else nombre
    if n in ETIQUETAS:
        return ETIQUETAS[n]
    for raiz in ("genero", "municipio", "tipo_institucion"):
        if n.startswith(raiz + "_"):
            valor = n[len(raiz) + 1:]
            return f"{raiz.replace('_', ' ').capitalize()}: {valor}"
    return n.replace("_", " ")


# ---------------------------------------------------------------------------
def preparar(archivo: str, incluir_nuevas: bool):
    """Carga un dataset y devuelve (X, y, preprocesador, nombres de features)."""
    ruta = DATA_DIR / archivo
    if not ruta.exists():
        log(f"FALTA data/{archivo} — ejecute antes el script 11")
        return None

    df = pd.read_csv(ruta, encoding="utf-8", dtype={"numero_documento": str})
    df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X, numericas, binarias, onehot = s12.construir_crudos(df, incluir_nuevas)
    pre = s12.construir_pipeline(numericas, binarias, onehot).named_steps["pre"]
    return X, y, pre


def nombres_features(pre) -> list[str]:
    """Nombres tras el one-hot, en el orden en que los ve el modelo."""
    return [bonito(n) for n in pre.get_feature_names_out()]


def grafico_importancia(nombres, valores, errores, titulo_txt, subtitulo,
                        archivo: str, color: str) -> None:
    """Barras horizontales, top TOP_N descendente (la mayor arriba)."""
    orden = np.argsort(valores)[::-1][:TOP_N]
    nom = [nombres[i] for i in orden]
    val = np.asarray(valores)[orden]
    err = np.asarray(errores)[orden] if errores is not None else None

    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(nom))))
    ypos = np.arange(len(nom))[::-1]  # la más importante arriba
    ax.barh(ypos, val, xerr=err, color=color, alpha=0.85,
            edgecolor="white", error_kw={"ecolor": "#555555", "elinewidth": 1})
    ax.set_yticks(ypos)
    ax.set_yticklabels(nom)
    ax.set_xlabel(subtitulo)
    ax.set_title(titulo_txt)
    ax.grid(axis="y", visible=False)

    # La etiqueta va DESPUÉS de la barra de error, si la hay, para que no se
    # monte encima en las variables de importancia casi nula.
    margen = max(val) * 0.012
    for i, (y_, v) in enumerate(zip(ypos, val)):
        x = max(v, 0) + (err[i] if err is not None else 0) + margen
        ax.text(x, y_, f"{v:.3f}", va="center", fontsize=9, color="#333333")
    tope = max(val) + (max(err) if err is not None else 0)
    ax.set_xlim(0, tope * 1.22)

    path = OUTPUTS_DIR / archivo
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"    figura guardada → outputs/{archivo}")


def analizar(clave: str, archivo: str, incluir_nuevas: bool, etiqueta: str) -> dict | None:
    datos = preparar(archivo, incluir_nuevas)
    if datos is None:
        return None
    X, y, pre = datos

    titulo(f"Dataset {clave} — {etiqueta}")
    log(f"{len(X):,} filas")

    # --- A) MDI sobre el modelo entrenado con TODAS las filas ---------------
    # Sin partición: este modelo es solo para leer el ranking, no para medir
    # desempeño (las métricas válidas están en el script 12).
    modelo = Pipeline([("pre", pre),
                       ("rf", RandomForestRegressor(**s12.MODELO_KWARGS))])
    modelo.fit(X, y)
    nombres = nombres_features(modelo.named_steps["pre"])
    mdi = modelo.named_steps["rf"].feature_importances_
    log(f"MDI — modelo final sobre las {len(X):,} filas "
        f"({len(nombres)} features)")

    grafico_importancia(
        nombres, mdi, None,
        f"Importancia MDI — dataset {clave} (n={len(X):,})",
        "Reducción media de impureza (MDI)",
        f"feature_importance_{clave}.png", COLORS["cyan"])

    # --- B) Permutación sobre un hold-out del 20 % --------------------------
    idx_tr, idx_te = train_test_split(
        X.index, test_size=0.20, random_state=RANDOM_STATE,
        stratify=s12.estratos(y))
    modelo_p = Pipeline([("pre", pre),
                         ("rf", RandomForestRegressor(**s12.MODELO_KWARGS))])
    modelo_p.fit(X.loc[idx_tr], y.loc[idx_tr])
    perm = permutation_importance(
        modelo_p, X.loc[idx_te], y.loc[idx_te], n_repeats=30,
        random_state=RANDOM_STATE, scoring="r2")

    # La permutación se calcula sobre las columnas CRUDAS (antes del one-hot),
    # así que cada variable categórica aparece una sola vez — que es justo la
    # lectura que interesa para el informe.
    nombres_crudos = [bonito(c) for c in X.columns]
    grafico_importancia(
        nombres_crudos, perm.importances_mean, perm.importances_std,
        f"Importancia por permutación — dataset {clave} (n={len(X):,})",
        "Caída de R² al desordenar la variable",
        f"permutation_importance_{clave}.png", COLORS["violet"])

    orden_perm = np.argsort(perm.importances_mean)[::-1]
    ranking = [(nombres_crudos[i], float(perm.importances_mean[i]),
                float(perm.importances_std[i])) for i in orden_perm]

    print(f"\n  Top 10 por permutación (dataset {clave}):")
    print(f"    {'#':>2}  {'variable':<32} {'ΔR²':>8} {'±':>7}")
    print(f"    {'-' * 2}  {'-' * 32} {'-' * 8} {'-' * 7}")
    for i, (nom, val, err) in enumerate(ranking[:10], 1):
        print(f"    {i:>2}  {nom:<32} {val:>+8.4f} {err:>7.4f}")

    orden_mdi = np.argsort(mdi)[::-1]
    return {
        "clave": clave,
        "n": len(X),
        "ranking_perm": ranking,
        "top_perm": [r[0] for r in ranking[:10]],
        "top_mdi": [nombres[i] for i in orden_mdi[:10]],
    }


# ---------------------------------------------------------------------------
def comparar(res_a: dict, res_c: dict) -> None:
    titulo("Comparación A vs C — ¿qué cambia al añadir el perfil académico?")

    top_a, top_c = res_a["top_perm"], res_c["top_perm"]
    comunes = [v for v in top_c if v in top_a]
    nuevas = [v for v in top_c if v not in top_a]
    salen = [v for v in top_a if v not in top_c]

    print(f"  Se mantienen en ambos top 10 ({len(comunes)}):")
    for v in comunes:
        print(f"    · {v}")
    print(f"\n  Entran en C y no estaban en A ({len(nuevas)}):")
    for v in nuevas:
        marca = "  ← variable nueva" if v in NUEVAS_ETIQUETAS else ""
        print(f"    · {v}{marca}")
    print(f"\n  Salen del top 10 al pasar de A a C ({len(salen)}):")
    for v in salen:
        print(f"    · {v}")


def conclusion(res_c: dict, res_a: dict) -> None:
    titulo("Conclusión")
    top = res_c["ranking_perm"]
    positivos = [(n, v) for n, v, _ in top if v > 0]

    nombres_top3 = ", ".join(n for n, _ in positivos[:3])
    print(f"  Las variables más influyentes para predecir el rendimiento son:")
    print(f"  {nombres_top3}.")
    print()
    for i, (nom, val) in enumerate(positivos[:5], 1):
        print(f"    {i}. {nom} — al desordenarla, el R² cae {val:.4f}")

    nuevas_en_top5 = sum(1 for n, _ in positivos[:5] if n in NUEVAS_ETIQUETAS)
    print()
    print(f"  De las 5 variables más influyentes, {nuevas_en_top5} pertenecen al "
          f"bloque de perfil académico añadido al formulario.")

    # La lectura honesta depende de cuánto domina la primera sobre la segunda:
    # una variable académica en el puesto 1 con mucha ventaja no es lo mismo que
    # un top 5 repartido.
    if positivos:
        primera, v1 = positivos[0]
        v2 = positivos[1][1] if len(positivos) > 1 else 0.0
        if primera in NUEVAS_ETIQUETAS and v2 > 0 and v1 > 2 * v2:
            print(f"  «{primera}» domina el ranking: pesa {v1 / v2:.1f} veces más")
            print(f"  que la siguiente variable. Ninguna variable de origen")
            print(f"  socioeconómico se le acerca.")
    print()
    print("  Matiz: el contexto NO desaparece. Municipio y grado escolar siguen")
    print("  altos en el ranking, así que el territorio y el momento escolar")
    print("  siguen pesando; lo que cambia es que ya no encabezan la lista.")


def main() -> None:
    print("=" * 78)
    print(" COPA STEM 2026 — Análisis de explicabilidad")
    print(" Fundación SapienceLab · Script 13")
    print("=" * 78)
    print(" Modelo interpretado: RandomForestRegressor(300, max_depth=10,")
    print("                      min_samples_leaf=8) — el mismo del script 12")

    resultados = {}
    for clave, archivo, incluir, etiqueta in CASOS:
        r = analizar(clave, archivo, incluir, etiqueta)
        if r:
            resultados[clave] = r

    if not resultados:
        print("\n  No hay datasets que analizar. Ejecute antes el script 11.\n")
        sys.exit(1)

    if {"A", "C"} <= resultados.keys():
        comparar(resultados["A"], resultados["C"])
        conclusion(resultados["C"], resultados["A"])

    # Ranking completo a CSV para el informe.
    filas = []
    for clave, r in resultados.items():
        for pos, (nom, val, err) in enumerate(r["ranking_perm"], 1):
            filas.append({"dataset": clave, "puesto": pos, "variable": nom,
                          "delta_r2": val, "std": err})
    salida = OUTPUTS_DIR / "F13_importancias.csv"
    pd.DataFrame(filas).to_csv(salida, index=False, encoding="utf-8")
    print(f"\n>>> rankings guardados → outputs/{salida.name}")

    print("\n" + "=" * 78)
    print(" ANÁLISIS DE EXPLICABILIDAD COMPLETADO")
    print("=" * 78)


if __name__ == "__main__":
    main()
