# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 05c: Cruce Computador × Con-quién-vive sobre el puntaje
================================================================================

Pregunta de fondo
-----------------
¿Tener computador en casa + vivir con ambos padres es una **ventaja real** sobre
el puntaje, o es una **correlación espuria** (proxy del estrato)? Se analiza la
interacción entre `computador_en_casa` y `con_quien_vive`.

Contenido
---------
    1) Tabla cruzada (media/mediana/N) + heatmap anotado
    2a) ¿Los que tienen computador rinden SIEMPRE mejor? (t-test + histogramas)
    2b) ¿Ambos padres + computador = combinación ganadora? (ANOVA 2 factores +
        gráfico de interacción)
    2c) Grupo resiliente: bajo acceso + buena nota (¿quiénes? ¿qué colegios?)
    2d) ¿Computador es proxy de estrato? (crosstab + regresión parcial)
    3) Conclusión clara

Datos: data/copa_stem_dataset_limpio.csv. Paleta Copa STEM. `random_state=42`.
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
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
for _d in (OUTPUTS_DIR, REPORTS_DIR):
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
STEM_SEQ = LinearSegmentedColormap.from_list(
    "stem_seq", ["#eafcff", "#8ee9ff", COLORS["cyan"], COLORS["blue"]])

BUENA_NOTA = 60  # umbral de "buena nota" (≈ P75 de los datos limpios)
FIGURES: dict[str, str] = {}
REPORT: list[str] = []


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def hallazgo(msg: str) -> None:
    print(f"    ★ {msg}", flush=True)


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


def cohens_d(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


# =============================================================================
# CARGA
# =============================================================================

def cargar() -> pd.DataFrame:
    log("Carga de datos limpios")
    ruta = next((DATA_DIR / n for n in DATASET_CANDIDATOS
                 if (DATA_DIR / n).exists()), None)
    if ruta is None:
        print("\n  ⚠  No se encontró el dataset limpio. Ejecute 05b primero.\n")
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
    df["puntaje_obtenido"] = pd.to_numeric(df["puntaje_obtenido"], errors="coerce")
    df["estrato"] = pd.to_numeric(df["estrato"], errors="coerce")
    df = df[df["puntaje_obtenido"].notna()].reset_index(drop=True)

    # Binaria Sí/No de computador ("Sí, propio"/"Sí, compartido" → Sí; "No" → No).
    def _tiene(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        return "Sí" if s.startswith("s") else ("No" if s.startswith("n") else np.nan)
    df["tiene_computador"] = df["computador_en_casa"].map(_tiene)
    log(f"    estudiantes que presentaron: {len(df):,}")
    return df


# =============================================================================
# 1. TABLA CRUZADA + HEATMAP
# =============================================================================

def tabla_cruzada(df: pd.DataFrame) -> dict:
    log("1) Tabla cruzada computador × con_quién_vive")
    sub = df.dropna(subset=["tiene_computador", "con_quien_vive", "puntaje_obtenido"])
    g = (sub.groupby(["tiene_computador", "con_quien_vive"])["puntaje_obtenido"]
         .agg(media="mean", mediana="median", n="count").reset_index())

    orden_cqv = (sub.groupby("con_quien_vive")["puntaje_obtenido"].mean()
                 .sort_values(ascending=False).index.tolist())
    pivote_media = g.pivot(index="tiene_computador", columns="con_quien_vive",
                           values="media").reindex(columns=orden_cqv)
    pivote_n = g.pivot(index="tiene_computador", columns="con_quien_vive",
                       values="n").reindex(columns=orden_cqv)

    # Anotación: media \n (n=N); ocultar celdas con N<5 (poco fiables).
    annot = pivote_media.copy().astype(object)
    for i in pivote_media.index:
        for c in pivote_media.columns:
            m, n = pivote_media.loc[i, c], pivote_n.loc[i, c]
            annot.loc[i, c] = "" if pd.isna(m) or n < 5 else f"{m:.1f}\n(n={int(n)})"
    mask = pivote_n.isna() | (pivote_n < 5)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    sns.heatmap(pivote_media, annot=annot.values, fmt="", cmap=STEM_SEQ,
                mask=mask, linewidths=0.6, linecolor="white",
                cbar_kws={"label": "Puntaje promedio", "shrink": 0.8},
                annot_kws={"fontsize": 9}, ax=ax)
    ax.set_title("Puntaje promedio por computador × con quién vive\n"
                 "(celdas con N<5 ocultas)")
    ax.set_xlabel("Con quién vive")
    ax.set_ylabel("¿Tiene computador?")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    savefig(fig, "F05c_heatmap_cruce.png", "heatmap")

    return {"tabla": g, "orden_cqv": orden_cqv,
            "pivote_media": pivote_media, "pivote_n": pivote_n}


# =============================================================================
# 2a. ¿COMPUTADOR = SIEMPRE MEJOR?
# =============================================================================

def pregunta_a(df: pd.DataFrame) -> dict:
    log("2a) ¿Los que tienen computador rinden mejor?")
    con = df.loc[df["tiene_computador"] == "Sí", "puntaje_obtenido"].dropna()
    sin = df.loc[df["tiene_computador"] == "No", "puntaje_obtenido"].dropna()
    t, p = stats.ttest_ind(con, sin, equal_var=False)
    d = cohens_d(con.values, sin.values)
    res = {"mu_con": float(con.mean()), "mu_sin": float(sin.mean()),
           "n_con": int(con.size), "n_sin": int(sin.size),
           "dif": float(con.mean() - sin.mean()), "t": float(t),
           "p": float(p), "d": float(d)}
    magnitud = ("insignificante" if abs(d) < 0.2 else "pequeña" if abs(d) < 0.5
                else "moderada" if abs(d) < 0.8 else "grande")
    res["magnitud"] = magnitud
    sig = "significativa" if p < 0.05 else "NO significativa"
    hallazgo(f"Con computador µ={res['mu_con']:.1f} (n={res['n_con']}) vs "
             f"sin µ={res['mu_sin']:.1f} (n={res['n_sin']}): "
             f"diferencia {res['dif']:+.1f} pts, {sig} (p={p:.3g}), "
             f"efecto {magnitud} (d={d:.2f})")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(con, bins=25, color=COLORS["green"], alpha=0.55, stat="density",
                 edgecolor="white", label=f"Con computador (µ={con.mean():.1f})", ax=ax)
    sns.histplot(sin, bins=25, color=COLORS["red"], alpha=0.55, stat="density",
                 edgecolor="white", label=f"Sin computador (µ={sin.mean():.1f})", ax=ax)
    ax.axvline(con.mean(), color=COLORS["green"], linestyle="--", linewidth=1.5)
    ax.axvline(sin.mean(), color=COLORS["red"], linestyle="--", linewidth=1.5)
    ax.set_title(f"Puntaje: con vs. sin computador\n"
                 f"t-test Welch p={p:.3g} · Cohen's d={d:.2f} (efecto {magnitud})")
    ax.set_xlabel("Puntaje")
    ax.set_ylabel("Densidad")
    ax.legend()
    savefig(fig, "F05c_hist_computador.png", "hist")
    return res


# =============================================================================
# 2b. ¿AMBOS PADRES + COMPUTADOR = COMBINACIÓN GANADORA?
# =============================================================================

def pregunta_b(df: pd.DataFrame, orden_cqv: list) -> dict:
    log("2b) ¿Ambos padres + computador = combinación ganadora? (ANOVA 2 factores)")
    sub = df.dropna(subset=["tiene_computador", "con_quien_vive", "puntaje_obtenido"])
    res = {"anova": None}
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        d = sub.rename(columns={"tiene_computador": "comp", "con_quien_vive": "cqv"})
        modelo = smf.ols("puntaje_obtenido ~ C(comp) * C(cqv)", data=d).fit()
        aov = sm.stats.anova_lm(modelo, typ=2)
        res["anova"] = aov
        p_comp = float(aov.loc["C(comp)", "PR(>F)"])
        p_cqv = float(aov.loc["C(cqv)", "PR(>F)"])
        p_int = float(aov.loc["C(comp):C(cqv)", "PR(>F)"])
        res["p_comp"], res["p_cqv"], res["p_int"] = p_comp, p_cqv, p_int
        hallazgo(f"ANOVA 2 factores → computador p={p_comp:.3g} | "
                 f"con_quién_vive p={p_cqv:.3g} | "
                 f"INTERACCIÓN p={p_int:.3g} "
                 f"({'hay' if p_int < 0.05 else 'NO hay'} interacción significativa)")
    except Exception as e:
        log(f"    ⚠ ANOVA 2 factores no disponible: {e}")

    # Combinación ambos padres + computador vs. resto.
    amb_comp = sub[(sub["con_quien_vive"] == "Ambos padres")
                   & (sub["tiene_computador"] == "Sí")]["puntaje_obtenido"]
    resto = sub[~((sub["con_quien_vive"] == "Ambos padres")
                  & (sub["tiene_computador"] == "Sí"))]["puntaje_obtenido"]
    res["mu_amb_comp"] = float(amb_comp.mean())
    res["mu_resto"] = float(resto.mean())
    res["n_amb_comp"] = int(amb_comp.size)
    hallazgo(f"'Ambos padres + computador' µ={res['mu_amb_comp']:.1f} "
             f"(n={res['n_amb_comp']}) vs resto µ={res['mu_resto']:.1f} "
             f"(dif {res['mu_amb_comp']-res['mu_resto']:+.1f})")

    # Gráfico de interacción: X=con_quien_vive, líneas=computador.
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.pointplot(data=sub, x="con_quien_vive", y="puntaje_obtenido",
                  hue="tiene_computador", order=orden_cqv,
                  hue_order=["Sí", "No"], palette=[COLORS["green"], COLORS["red"]],
                  capsize=0.1, dodge=0.3, ax=ax)
    ax.set_title("Interacción computador × con quién vive (puntaje promedio, IC95%)\n"
                 "líneas paralelas = sin interacción")
    ax.set_xlabel("Con quién vive")
    ax.set_ylabel("Puntaje promedio")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="¿Tiene computador?")
    savefig(fig, "F05c_interaccion.png", "interaccion")
    return res


# =============================================================================
# 2c. GRUPO RESILIENTE: BAJO ACCESO + BUENA NOTA
# =============================================================================

def pregunta_c(df: pd.DataFrame) -> dict:
    log("2c) Grupo resiliente: sin computador + buena nota")
    sin = df[df["tiene_computador"] == "No"]
    resilientes = sin[sin["puntaje_obtenido"] >= BUENA_NOTA]
    res = {"n_sin": int(len(sin)), "n_resilientes": int(len(resilientes)),
           "umbral": BUENA_NOTA}
    res["pct_resilientes_del_sin"] = (100 * len(resilientes) / len(sin)
                                      if len(sin) else 0.0)

    # Caso específico: sin computador + Solo madre.
    sm_sin = df[(df["tiene_computador"] == "No")
                & (df["con_quien_vive"] == "Solo madre")]["puntaje_obtenido"]
    res["mu_solomadre_sincomp"] = float(sm_sin.mean()) if len(sm_sin) else float("nan")
    res["n_solomadre_sincomp"] = int(len(sm_sin))
    hallazgo(f"Sin computador con buena nota (≥{BUENA_NOTA}): {len(resilientes)} "
             f"de {len(sin)} ({res['pct_resilientes_del_sin']:.0f}% de los sin acceso)")
    hallazgo(f"Sin computador + 'Solo madre': n={res['n_solomadre_sincomp']}, "
             f"µ={res['mu_solomadre_sincomp']:.1f}")

    if "institucion_educativa" in resilientes.columns and not resilientes.empty:
        colg = (resilientes.groupby("institucion_educativa").size()
                .sort_values(ascending=False).head(10))
        res["colegios"] = colg.to_dict()
    return res


# =============================================================================
# 2d. ¿COMPUTADOR ES PROXY DE ESTRATO?
# =============================================================================

def pregunta_d(df: pd.DataFrame) -> dict:
    log("2d) ¿Computador es proxy de estrato?")
    sub = df.dropna(subset=["estrato", "tiene_computador", "puntaje_obtenido"]).copy()
    sub["comp01"] = (sub["tiene_computador"] == "Sí").astype(int)

    # Crosstab estrato × computador (proporción con computador por estrato).
    ct = pd.crosstab(sub["estrato"], sub["tiene_computador"])
    prop = ct.div(ct.sum(axis=1), axis=0).round(3)
    pct_comp_por_estrato = (prop["Sí"] * 100).round(1).to_dict() if "Sí" in prop else {}
    r_pb = stats.pointbiserialr(sub["comp01"], sub["estrato"])
    res = {"crosstab": ct, "prop": prop,
           "pct_comp_por_estrato": pct_comp_por_estrato,
           "corr_comp_estrato": float(r_pb.correlation),
           "p_corr": float(r_pb.pvalue)}
    hallazgo(f"Correlación computador↔estrato: r={r_pb.correlation:.2f} "
             f"(p={r_pb.pvalue:.3g})")

    # Regresión parcial: efecto de computador solo vs. controlando por estrato.
    coef_solo = coef_ctrl = coef_estrato = float("nan")
    try:
        import statsmodels.formula.api as smf
        m1 = smf.ols("puntaje_obtenido ~ comp01", data=sub).fit()
        m2 = smf.ols("puntaje_obtenido ~ comp01 + estrato", data=sub).fit()
        coef_solo = float(m1.params["comp01"])
        coef_ctrl = float(m2.params["comp01"])
        coef_estrato = float(m2.params["estrato"])
        res["p_comp_ctrl"] = float(m2.pvalues["comp01"])
        res["p_estrato_ctrl"] = float(m2.pvalues["estrato"])
    except Exception as e:
        log(f"    ⚠ regresión parcial no disponible: {e}")
    res["coef_comp_solo"] = coef_solo
    res["coef_comp_ctrl"] = coef_ctrl
    res["coef_estrato"] = coef_estrato
    if coef_solo == coef_solo and coef_ctrl == coef_ctrl:
        reduccion = (1 - coef_ctrl / coef_solo) * 100 if coef_solo else float("nan")
        res["reduccion_pct"] = float(reduccion)
        hallazgo(f"Efecto computador: solo={coef_solo:+.1f} pts → "
                 f"controlando estrato={coef_ctrl:+.1f} pts "
                 f"(se reduce {reduccion:.0f}%); estrato={coef_estrato:+.2f} pts/nivel")

    # Gráfico: % con computador por estrato + efecto antes/después.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if pct_comp_por_estrato:
        estr = sorted(pct_comp_por_estrato)
        ax1.bar([str(int(e)) for e in estr],
                [pct_comp_por_estrato[e] for e in estr],
                color=[STEM_SEQ(i / max(1, len(estr) - 1)) for i in range(len(estr))],
                edgecolor="white")
        ax1.set_title("% con computador por estrato")
        ax1.set_xlabel("Estrato")
        ax1.set_ylabel("% con computador")
        ax1.set_ylim(0, 100)
        for i, e in enumerate(estr):
            ax1.text(i, pct_comp_por_estrato[e], f"{pct_comp_por_estrato[e]:.0f}%",
                     ha="center", va="bottom", fontsize=8)
    if coef_solo == coef_solo:
        ax2.bar(["Efecto solo", "Controlando\nestrato"], [coef_solo, coef_ctrl],
                color=[COLORS["amber"], COLORS["cyan"]], edgecolor="white", width=0.5)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_title("Efecto de tener computador sobre el puntaje")
        ax2.set_ylabel("Puntos de puntaje")
        for i, v in enumerate([coef_solo, coef_ctrl]):
            ax2.text(i, v, f"{v:+.1f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=10)
    fig.suptitle("D. ¿Computador es proxy de estrato?", fontsize=15, fontweight="bold")
    savefig(fig, "F05c_estrato_computador.png", "estrato")
    return res


# =============================================================================
# INFORME
# =============================================================================

def construir_informe(df, cruz, A, B, C, D) -> None:
    log("Generación del informe markdown")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    R = REPORT.append

    # Conclusión sintetizada a partir de la evidencia.
    inter = (B.get("p_int") is not None and B["p_int"] < 0.05)
    proxy = (D.get("coef_comp_solo") == D.get("coef_comp_solo")
             and abs(D.get("coef_comp_ctrl", 0)) < abs(D.get("coef_comp_solo", 1e9))
             and D.get("reduccion_pct", 0) > 30)

    R("# Cruce Computador × Con-quién-vive — Copa STEM 2026\n")
    R(f"**Fundación SapienceLab** · Análisis específico · {fecha}\n")
    R("---\n")

    R("## Resumen\n")
    sig_a = "significativa" if A["p"] < 0.05 else "no significativa"
    R(dedent(f"""\
        Tener computador se asocia a **{A['dif']:+.1f} puntos** (con µ={A['mu_con']:.1f}
        vs sin µ={A['mu_sin']:.1f}); la diferencia es {sig_a} (p={A['p']:.3g}) pero
        con **efecto {A['magnitud']}** (Cohen's d={A['d']:.2f}). El acceso a
        computador está correlacionado con el estrato (r={D['corr_comp_estrato']:.2f}),
        y al **controlar por estrato el efecto del computador
        {'se reduce notablemente' if proxy else 'se mantiene en su mayor parte'}**
        ({D.get('coef_comp_solo', float('nan')):+.1f} → {D.get('coef_comp_ctrl', float('nan')):+.1f} pts).
        {'NO hay' if not inter else 'Hay'} interacción significativa entre computador
        y con quién vive (p={B.get('p_int', float('nan')):.3g}).\n"""))

    R("## 1. Tabla cruzada (puntaje promedio · N)\n")
    t = cruz["tabla"].copy()
    t["media"] = t["media"].round(1)
    t["mediana"] = t["mediana"].round(1)
    t.columns = ["¿Computador?", "Con quién vive", "Media", "Mediana", "N"]
    R(tabla_md(t) + "\n")
    R(f"\n{img('heatmap', 'Heatmap cruce')}\n")

    R("## 2a. ¿Los que tienen computador rinden siempre mejor?\n")
    R(dedent(f"""\
        - Con computador: **µ={A['mu_con']:.1f}** (n={A['n_con']}).
        - Sin computador: **µ={A['mu_sin']:.1f}** (n={A['n_sin']}).
        - t-test Welch: t={A['t']:.2f}, **p={A['p']:.3g}** ({sig_a}).
        - Tamaño del efecto: **Cohen's d={A['d']:.2f}** → magnitud **{A['magnitud']}**.

        La diferencia existe pero es **pequeña en la práctica**: hay mucho
        solapamiento entre las dos distribuciones.\n"""))
    R(f"\n{img('hist', 'Histograma con vs sin computador')}\n")

    R("## 2b. ¿Ambos padres + computador = combinación ganadora?\n")
    if B.get("anova") is not None:
        R(dedent(f"""\
            ANOVA de 2 factores (`puntaje ~ computador × con_quien_vive`):
            - Computador: p={B['p_comp']:.3g}
            - Con quién vive: p={B['p_cqv']:.3g}
            - **Interacción: p={B['p_int']:.3g}** → {'**sí** hay' if inter else '**no** hay'} interacción.
            """))
    R(dedent(f"""\
        'Ambos padres + computador' promedia **{B['mu_amb_comp']:.1f}**
        (n={B['n_amb_comp']}) vs. **{B['mu_resto']:.1f}** el resto. Como la
        interacción {'es' if inter else 'no es'} significativa, el beneficio de
        tener computador {'depende' if inter else 'NO depende'} de con quién vive:
        las líneas del gráfico son {'no ' if inter else ''}paralelas.\n"""))
    R(f"\n{img('interaccion', 'Gráfico de interacción')}\n")

    R("## 2c. El grupo resiliente (bajo acceso + buena nota)\n")
    R(dedent(f"""\
        **{C['n_resilientes']}** estudiantes **sin computador** sacaron ≥
        {C['umbral']} puntos ({C['pct_resilientes_del_sin']:.0f}% de los que no
        tienen computador). El subgrupo "sin computador + Solo madre" (n=
        {C['n_solomadre_sincomp']}) promedia {C['mu_solomadre_sincomp']:.1f}. La
        adversidad de acceso **no condena** el resultado: hay talento resiliente.\n"""))
    if C.get("colegios"):
        R("\n**Colegios con más resilientes:**\n")
        R(tabla_md(pd.DataFrame(
            [{"Institución": k, "Resilientes": v} for k, v in C["colegios"].items()])) + "\n")

    R("## 2d. ¿Computador es proxy de estrato?\n")
    R(dedent(f"""\
        - Correlación computador↔estrato: **r={D['corr_comp_estrato']:.2f}**
          (p={D['p_corr']:.3g}).
        - % con computador por estrato: """
             + "; ".join(f"E{int(k)}: {v:.0f}%"
                         for k, v in sorted(D['pct_comp_por_estrato'].items())) + "\n"))
    if D.get("coef_comp_solo") == D.get("coef_comp_solo"):
        R(dedent(f"""\
            - Regresión parcial: el efecto de tener computador pasa de
              **{D['coef_comp_solo']:+.1f} pts** (solo) a **{D['coef_comp_ctrl']:+.1f} pts**
              al controlar por estrato (se reduce {D.get('reduccion_pct', 0):.0f}%);
              el estrato aporta {D['coef_estrato']:+.2f} pts por nivel.\n"""))
    R(f"\n{img('estrato', 'Estrato vs computador')}\n")

    R("## 3. Conclusión\n")
    if proxy:
        concl = ("El computador es **en gran parte un proxy del estrato**: al "
                 "controlar por estrato, buena parte de su 'ventaja' se desvanece. "
                 "La ventaja aparente de 'computador + ambos padres' es "
                 "**mayormente una correlación con el nivel socioeconómico**, no un "
                 "efecto causal fuerte del recurso en sí.")
    else:
        concl = ("Aun controlando por estrato, tener computador conserva parte de "
                 "su asociación con el puntaje: **no es solo un proxy del estrato**, "
                 "aunque el efecto es pequeño.")
    R(dedent(f"""\
        {concl}

        En términos prácticos: la diferencia por computador es real pero **de
        magnitud {A['magnitud']}** (d={A['d']:.2f}), {'sin' if not inter else 'con'}
        interacción con la estructura familiar, y **{C['n_resilientes']} estudiantes
        sin computador rinden ≥ {C['umbral']}**. La política más eficiente combina
        **dotación tecnológica focalizada por estrato** con **acompañamiento
        pedagógico**, sin asumir que el computador por sí solo explica el
        rendimiento.\n"""))
    R("\n---\n_Generado por `notebooks/05c_cruce_computador_familia.py` — Copa STEM 2026._\n")

    (REPORTS_DIR / "05c_cruce_computador_familia.md").write_text(
        "\n".join(REPORT), encoding="utf-8")
    log("    informe escrito → reports/05c_cruce_computador_familia.md")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print(" COPA STEM 2026 — Cruce Computador × Con-quién-vive")
    print(" Fundación SapienceLab")
    print("=" * 70)

    df = cargar()
    cruz = tabla_cruzada(df)
    A = pregunta_a(df)
    B = pregunta_b(df, cruz["orden_cqv"])
    C = pregunta_c(df)
    D = pregunta_d(df)
    construir_informe(df, cruz, A, B, C, D)

    print("\n" + "=" * 70)
    print(" ✔ ANÁLISIS COMPLETADO — HALLAZGOS CLAVE")
    print("=" * 70)
    print(f"  · Computador: {A['dif']:+.1f} pts (d={A['d']:.2f}, {A['magnitud']}), "
          f"p={A['p']:.3g}")
    if B.get("p_int") is not None:
        print(f"  · Interacción computador×familia: p={B['p_int']:.3g} "
              f"({'SÍ' if B['p_int'] < 0.05 else 'NO'} significativa)")
    print(f"  · Correlación computador↔estrato: r={D['corr_comp_estrato']:.2f}")
    if D.get("coef_comp_solo") == D.get("coef_comp_solo"):
        print(f"  · Efecto computador: {D['coef_comp_solo']:+.1f} → "
              f"{D['coef_comp_ctrl']:+.1f} pts al controlar estrato")
    print(f"  · Resilientes (sin PC, ≥{BUENA_NOTA}): {C['n_resilientes']}")
    print(f"  · Figuras: {len(FIGURES)} → outputs/ | Informe: "
          f"reports/05c_cruce_computador_familia.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
