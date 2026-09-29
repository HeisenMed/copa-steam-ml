# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 23: Exportación del modelo global base a los formatos de despliegue
            y verificación de paridad Python ↔ JS
================================================================================

Fase 1 (siempre) — ALCANCE del cambio de referencia
---------------------------------------------------
`ref_rendimiento` pasa de los 3,072 de F17 (que incluyen 22 resultados
descartados por sospecha y 3 sesiones abandonadas que `keep="first"` conservó)
a los 3,050 del conjunto limpio del script 21. Es una DECISIÓN DE PRODUCTO
declarada, no un subproducto del entrenamiento. Antes de exportar se mide cuántos
estudiantes cambian de categoría, por municipio y por dirección, manteniendo
fijo TODO lo demás (modelo v2 vigente, pesos, categorías).

Fase 2 (solo con `--exportar`) — exportación y paridad
------------------------------------------------------
Se reutilizan las piezas del script 19 (`_extract_rf`, `construir_spec`,
`features_from_raw`, `_predecir`) y el cuerpo del JS vigente. El SPEC JSON es la
serialización compacta del mismo objeto que se incrusta en el JS; el script lo
comprueba sobre el artefacto vigente antes de escribir el nuevo.

Privacidad: la lista de descartados y `numero_documento` solo existen en memoria;
las salidas contienen únicamente agregados.

Reproducible: `random_state=42`.
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
NB_DIR = BASE_DIR / "notebooks"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
OUTPUTS_DIR = BASE_DIR / "outputs"

SPEC_VIGENTE = DEPLOY_DIR / "spec_potencial_v2_corrected.json"     # solo lectura
JS_VIGENTE = DEPLOY_DIR / "potencial_stem_predictor_v2_corrected.js"  # solo lectura
PRODUCCION_CSV = OUTPUTS_DIR / "ml_scores_v2_corrected.csv"          # solo lectura
INSCRITOS = DATA_DIR / "inscritos_copa_stem.csv"                     # solo lectura
F17_REF = OUTPUTS_DIR / "F17_ref_rendimiento_corregido.json"         # solo lectura

OUT_IMPACTO = OUTPUTS_DIR / "F23_impacto_referencia.json"

TARGET = "puntaje_obtenido"
MUNICIPIOS = ["Bello", "Girardota", "Copacabana"]


def cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def titulo(txt: str) -> None:
    print("\n" + "=" * 78)
    print(f" {txt}")
    print("=" * 78)


s21 = cargar_modulo("s21", NB_DIR / "21_modelo_global_base.py")
pot = cargar_modulo("pot", DEPLOY_DIR / "potencial_stem_predictor.py")


def registros(df: pd.DataFrame) -> list[dict]:
    """Filas como dicts JSON-limpios (NaN → None), igual que `limpiar` del 19."""
    out = []
    for r in df.to_dict("records"):
        fila = {}
        for k, v in r.items():
            if isinstance(v, float) and v != v:
                fila[k] = None
            elif isinstance(v, (np.integer,)):
                fila[k] = int(v)
            elif isinstance(v, (np.floating,)):
                fila[k] = None if v != v else float(v)
            elif isinstance(v, (np.bool_,)):
                fila[k] = bool(v)
            else:
                fila[k] = v
        out.append(fila)
    return out


def rangos_engagement(regs: list[dict]) -> dict:
    """Mismo cálculo que `construir_spec.conteos` del script 19."""
    s19 = cargar_modulo("s19", NB_DIR / "19_regenerar_deploy_v2.py")
    out = {}
    for name, key in (("n_herramientas", "herramientas_conocidas"),
                      ("n_areas_interes", "areas_interes")):
        vals = []
        for r in regs:
            c = s19._to_float(r.get(name))
            if c is None:
                c = s19._parse_count(r.get(key))
            if c is not None:
                vals.append(c)
        out[name] = {"lo": float(min(vals)), "hi": float(max(vals))}
    return out


# ---------------------------------------------------------------------------
# Fase 1 — alcance del cambio de referencia
# ---------------------------------------------------------------------------
ORDEN_CAT = ["Requiere apoyo", "En desarrollo", "Promedio", "Alto potencial",
             "Talento destacado"]


def impacto_referencia() -> tuple[dict, pd.DataFrame, list[float], dict]:
    titulo("FASE 1 — Alcance del cambio de ref_rendimiento (3,072 → 3,050)")
    spec_act = json.loads(SPEC_VIGENTE.read_text(encoding="utf-8"))
    f17 = json.loads(F17_REF.read_text(encoding="utf-8"))
    assert spec_act["ref_rendimiento"] == f17["ref_rendimiento"], \
        "la referencia del SPEC vigente no es la de F17"

    df, _inv = s21.inventario(s21.cargar_descartados())
    regs = registros(df.drop(columns=["_inst"], errors="ignore"))
    ref_nueva = sorted(float(v) for v in df[TARGET])
    eng_nuevo = rangos_engagement(regs)

    ref_v = np.array(f17["ref_rendimiento"], float)
    ref_n = np.array(ref_nueva, float)
    resumen_ref = {
        "vigente_F17": {"n": int(len(ref_v)), "media": float(ref_v.mean()),
                        "sigma_ddof0": float(ref_v.std(ddof=0)),
                        "sigma_ddof1": float(ref_v.std(ddof=1)),
                        "sigma_declarado_F17": f17.get("sigma")},
        "nueva_3050": {"n": int(len(ref_n)), "media": float(ref_n.mean()),
                       "sigma_ddof0": float(ref_n.std(ddof=0)),
                       "sigma_ddof1": float(ref_n.std(ddof=1))},
        "engagement_vigente": spec_act["engagement"],
        "engagement_nuevo": eng_nuevo,
    }
    log(f"ref vigente: n={len(ref_v):,} media={ref_v.mean():.2f} "
        f"σ0={ref_v.std(ddof=0):.4f} σ1={ref_v.std(ddof=1):.4f} (F17 declara {f17.get('sigma')})")
    log(f"ref nueva  : n={len(ref_n):,} media={ref_n.mean():.2f} "
        f"σ0={ref_n.std(ddof=0):.4f} σ1={ref_n.std(ddof=1):.4f}")
    log(f"engagement vigente {spec_act['engagement']} → nuevo {eng_nuevo}")

    # Esquema VIGENTE de producción (script 17): quien tiene perfil se puntúa
    # con el SPEC v2 y quien no, con el SPEC v1 ("v1_fallback"); ambos con la
    # referencia de F17. Se reconstruye tal cual y se controla contra el CSV.
    spec_v2 = spec_act
    spec_v1 = copy.deepcopy(pot.SPEC)
    spec_v1["ref_rendimiento"] = f17["ref_rendimiento"]
    resumen_ref["engagement_vigente_v1_fallback"] = spec_v1["engagement"]

    def con_ref(spec, ref):
        s = copy.deepcopy(spec)
        s["ref_rendimiento"] = ref
        return s

    def enrutado(regs_, perfil, v2, v1):
        return [pot.calcular_indice(r, v2 if p else v1)["categoria"]
                for r, p in zip(regs_, perfil)]

    import joblib
    spec_global = construir_spec_global(joblib.load(JOBLIB_GLOBAL), regs, ref_nueva)

    out: dict = {"referencia": resumen_ref, "poblaciones": {},
                 "pasos": {"A_solo_referencia": "mismo esquema v2/v1_fallback, "
                                                "ref_rendimiento 3,072 → 3,050",
                           "B_spec_global": "SPEC global para todos: ref y engagement "
                                            "de 3,050 + imputación del modelo global "
                                            "(+ su predicción, para quien no presentó)"}}

    # (a) Los 3,050 examinados.
    perfil = df["_perfil"].tolist()
    antes = enrutado(regs, perfil, spec_v2, spec_v1)
    paso_a = enrutado(regs, perfil, con_ref(spec_v2, ref_nueva), con_ref(spec_v1, ref_nueva))
    paso_b = [pot.calcular_indice(r, spec_global)["categoria"] for r in regs]

    # Control: el "antes" reconstruido debe ser lo que se comunicó.
    prod = pd.read_csv(PRODUCCION_CSV, dtype={"numero_documento": str},
                       usecols=["numero_documento", "categoria_potencial", "modelo_version"])
    prod["numero_documento"] = prod["numero_documento"].astype(str).str.strip()
    prod = prod.drop_duplicates("numero_documento").set_index("numero_documento")
    cat_prod = df["numero_documento"].map(prod["categoria_potencial"])
    coincide = cat_prod.to_numpy() == np.array(antes)
    out["control_produccion"] = {
        "con_fila_en_produccion": int(cat_prod.notna().sum()),
        "misma_categoria_que_produccion": int(coincide.sum()),
        "distintas_por_municipio": df.loc[~coincide, "municipio"].value_counts().to_dict(),
        "nota": ("las distintas son documentos duplicados cuya fila en producción "
                 "es la sesión abandonada (keep='first'); aquí se usa la completada"),
    }
    log(f"control: {coincide.sum():,} de {cat_prod.notna().sum():,} coinciden con "
        f"{PRODUCCION_CSV.name}")
    # A y B se miden contra el "antes" reconstruido con la fila correcta, para
    # que solo reflejen el cambio de SPEC. Los documentos cuya fila comunicada
    # era la sesión abandonada se reportan aparte (efecto de la deduplicación).
    munis = df["municipio"].tolist()
    out["poblaciones"]["examinados_3050"] = {
        "A_solo_referencia": tabla_cambios(munis, antes, paso_a),
        "B_spec_global": tabla_cambios(munis, antes, paso_b),
    }
    k = ~coincide
    out["correccion_dedup_comunicado_vs_fila_correcta"] = tabla_cambios(
        [m for m, x in zip(munis, k) if x], list(cat_prod[k]),
        [a for a, x in zip(antes, k) if x])

    # (b) Inscritos sin examen: su rendimiento es un puntaje PREDICHO.
    ins = pd.read_csv(INSCRITOS, dtype={"numero_documento": str},
                      usecols=lambda c: c not in ("numero_documento", "nombres",
                                                  "apellidos", "institucion_educativa"))
    ins_regs = registros(ins)
    ins_perfil = pd.to_numeric(ins["promedio_academico"], errors="coerce").notna().tolist()
    i_antes = enrutado(ins_regs, ins_perfil, spec_v2, spec_v1)
    out["poblaciones"]["inscritos_sin_examen"] = {
        "A_solo_referencia": tabla_cambios(
            ins["municipio"].tolist(), i_antes,
            enrutado(ins_regs, ins_perfil, con_ref(spec_v2, ref_nueva),
                     con_ref(spec_v1, ref_nueva))),
        "B_spec_global": tabla_cambios(
            ins["municipio"].tolist(), i_antes,
            [pot.calcular_indice(r, spec_global)["categoria"] for r in ins_regs]),
    }
    del prod, cat_prod

    for pob, pasos in out["poblaciones"].items():
        for paso, t in pasos.items():
            log(f"{pob} · {paso}: {t['total']['cambian']} de {t['total']['n']} "
                f"cambian (suben {t['total']['suben']}, bajan {t['total']['bajan']})")
            for m in MUNICIPIOS:
                x = t["por_municipio"].get(m)
                if x:
                    log(f"    {m:<11} n={x['n']:>5}  cambian={x['cambian']:>4}  "
                        f"suben={x['suben']:>4}  bajan={x['bajan']:>4}")
    return out, df, ref_nueva, eng_nuevo


def tabla_cambios(munis: list[str], antes: list[str], despues: list[str]) -> dict:
    d = pd.DataFrame({"m": munis, "a": antes, "d": despues})
    d["ia"] = d["a"].map(ORDEN_CAT.index)
    d["id"] = d["d"].map(ORDEN_CAT.index)

    def fila(x):
        return {"n": int(len(x)), "cambian": int((x["ia"] != x["id"]).sum()),
                "suben": int((x["id"] > x["ia"]).sum()),
                "bajan": int((x["id"] < x["ia"]).sum())}

    matriz = pd.crosstab(d["a"], d["d"]).reindex(index=ORDEN_CAT, columns=ORDEN_CAT,
                                                 fill_value=0)
    return {
        "total": fila(d),
        "por_municipio": {m: fila(g) for m, g in d.groupby("m")},
        "transiciones": {f"{a} -> {b}": int(matriz.loc[a, b])
                         for a in ORDEN_CAT for b in ORDEN_CAT
                         if a != b and matriz.loc[a, b] > 0},
        "distribucion_antes": d["a"].value_counts().reindex(ORDEN_CAT, fill_value=0).astype(int).to_dict(),
        "distribucion_despues": d["d"].value_counts().reindex(ORDEN_CAT, fill_value=0).astype(int).to_dict(),
    }


# ---------------------------------------------------------------------------
# Construcción del SPEC del modelo global (en memoria)
# ---------------------------------------------------------------------------
s19 = cargar_modulo("s19", NB_DIR / "19_regenerar_deploy_v2.py")
s12 = s21.s12
JOBLIB_GLOBAL = MODELS_DIR / "modelo_global_base.joblib"
OUT_SPEC = DEPLOY_DIR / "spec_potencial_global_base.json"
OUT_JS = DEPLOY_DIR / "potencial_stem_predictor_global_base.js"
OUT_PARIDAD = OUTPUTS_DIR / "F23_paridad_modelo_global.json"

CABECERA_VIGENTE = (
    " * GENERADO por notebooks/19_regenerar_deploy_v2.py — no editar a mano.\n"
    " * MODELO v2: Random Forest con hiperparámetros optimizados, entrenado\n"
    " * sobre dataset_C_perfil.csv (incluye las 5 variables de perfil académico).\n"
    " * REFERENCIA CORREGIDA (script 17): `ref_rendimiento` percentiliza contra\n"
    " * los 3,072 examinados de dataset_B_completo.csv, no contra los 1,148 del\n"
    " * dataset de entrenamiento. El modelo y el preprocesamiento no cambian\n"
    " * respecto a potencial_stem_predictor_v2.js.\n")
CABECERA_NUEVA = (
    " * GENERADO por notebooks/23_exportar_modelo_global.py — no editar a mano.\n"
    " * MODELO GLOBAL BASE: Random Forest (150 árboles, 13 predictores, sin las\n"
    " * 5 preguntas de perfil) entrenado sobre 3,050 estudiantes de Bello,\n"
    " * Girardota y Copacabana (script 21). Métricas en SPEC.meta.metricas.\n"
    " * REFERENCIA: `ref_rendimiento` y rangos de engagement sobre los mismos\n"
    " * 3,050 — decisión de producto declarada (informe 23), no subproducto.\n")


def construir_pre(pipe) -> dict:
    """PRE del SPEC leído del Pipeline ENTRENADO: medianas y modas son las
    `statistics_` de sus SimpleImputer y el vocabulario one-hot son las
    `categories_` de su OneHotEncoder. Mismo esquema que el PRE de v2."""
    pre = pipe.named_steps["pre"]
    tr = {n: (t, cols) for n, t, cols in pre.transformers_ if n != "remainder"}
    num_t, num_cols = tr["num"]
    bin_t, bin_cols = tr["bin"]
    cat_t, cat_cols = tr["cat"]
    numeric, ordinal = list(s12.NUMERIC), list(s12.ORDINAL)
    assert list(num_cols) == numeric + ordinal, num_cols
    assert list(bin_cols) == list(s12.BINARY), bin_cols
    assert list(cat_cols) == list(s12.ONEHOT), cat_cols
    imp, oh = cat_t.named_steps["imp"], cat_t.named_steps["oh"]
    return {
        "numeric": numeric, "ordinal": ordinal, "binary": list(s12.BINARY),
        "binary_src": dict(s12.BINARY_SRC), "onehot_order": list(s12.ONEHOT),
        "medians": {c: float(v) for c, v in zip(num_cols, num_t.statistics_)},
        "modes": {c: float(v) for c, v in zip(bin_cols, bin_t.statistics_)},
        "onehot_cats": {c: [str(x) for x in cats]
                        for c, cats in zip(cat_cols, oh.categories_)},
        "onehot_mode": {c: str(v) for c, v in zip(cat_cols, imp.statistics_)},
    }


def construir_spec_global(bundle: dict, regs: list[dict], ref: list[float]
                          ) -> dict:
    pipe = bundle["modelo"]
    PRE = construir_pre(pipe)
    MODEL = s19._extract_rf(pipe.named_steps["rf"])
    met = bundle["metricas"]
    meta = {
        "generado": bundle["entrenado"],
        "generado_por": "notebooks/23_exportar_modelo_global.py",
        "n_cohorte": len(ref),
        "n_entrenamiento": bundle["n_entrenamiento"],
        "modelo_puntaje": "Random Forest global base (sin perfil)",
        "version": "global_base",
        "ref_rendimiento_fuente": "notebooks/21_modelo_global_base.py (conjunto limpio)",
        "ref_rendimiento_cohorte": "3,050 examinados deduplicados, sin descartados por sospecha",
        "ref_rendimiento_sigma": round(float(np.std(ref)), 4),
        "ref_rendimiento_decision": (
            "Decisión de producto declarada (informe 23): la referencia de 3,072 "
            "incluía 22 resultados descartados por sospecha y 3 sesiones "
            "abandonadas; no es un subproducto del entrenamiento."),
        "reemplaza": {"artefacto": "potencial_stem_predictor_v2_corrected.js + v1_fallback",
                      "n_ref_anterior": 3072},
        "n_predictores": 13, "n_arboles": len(MODEL["trees"]),
        "metricas": {
            "oof": {k: met["oof"][k] for k in ("r2", "mae", "rmse", "n", "r2_ic95_bootstrap", "particion")},
            "holdout": {k: met["holdout"][k] for k in ("r2", "mae", "rmse", "n", "n_train", "particion")},
            "sd_target_entrenamiento": met["sd_target"]["global_3050"]["sd"],
        },
        "politica_uso": bundle["politica_uso"],
    }
    # Mismo constructor que el script 19: engagement se calcula sobre `regs`.
    return s19.construir_spec(regs, PRE, MODEL, ref, meta)


def verificar_adaptador(bundle: dict, df: pd.DataFrame, regs: list[dict],
                        SPEC: dict) -> dict:
    """El PRE del SPEC aplicado a filas de entrenamiento debe reproducir, valor
    a valor, la matriz que produce el Pipeline; y el intérprete de árboles, su
    predicción."""
    titulo("Verificación del adaptador PRE (filas de entrenamiento)")
    pipe = bundle["modelo"]
    X, *_ = s12.construir_crudos(df, incluir_nuevas=False)
    M_pipe = pipe.named_steps["pre"].transform(X)
    M_spec = s19.matriz(regs, SPEC["puntaje"]["preprocess"])
    iguales = M_pipe.shape == M_spec.shape and bool(np.array_equal(M_pipe, M_spec))
    difs = int((M_pipe != M_spec).any(axis=1).sum()) if M_pipe.shape == M_spec.shape else -1
    p_sk = pipe.predict(X)
    p_sp = np.array([s19._predecir(list(f), SPEC["puntaje"]["model"]) for f in M_spec])
    out = {"n_filas": int(len(df)), "n_columnas": int(M_spec.shape[1]),
           "matriz_identica": iguales, "filas_con_diferencia": difs,
           "pred_max_abs_diff_vs_sklearn": float(np.abs(p_sk - p_sp).max()),
           "pred_exactamente_iguales": int((p_sk == p_sp).sum())}
    log(f"matriz {M_spec.shape}: idéntica={iguales} (filas distintas: {difs}); "
        f"predicción max|Δ| vs sklearn = {out['pred_max_abs_diff_vs_sklearn']:.3e} "
        f"({out['pred_exactamente_iguales']:,} exactas)")
    if not iguales:
        print("\n  ALTO: el PRE del SPEC no reproduce el Pipeline. No se exporta.\n")
        sys.exit(1)
    return out


def exportar(SPEC: dict) -> dict:
    """SPEC JSON = serialización compacta del objeto; JS = cuerpo del JS vigente
    con la línea `const SPEC` sustituida (mecánica de `exportar_js` del 19)."""
    titulo("Exportación")
    # 1) La cadena de formato, comprobada sobre el artefacto vigente.
    vigente_txt = SPEC_VIGENTE.read_text(encoding="utf-8")
    spec_js_vigente = s19.spec_desde_js(JS_VIGENTE)
    cadena_ok = vigente_txt == json.dumps(spec_js_vigente, ensure_ascii=False,
                                          separators=(",", ":"))
    log(f"formato del SPEC JSON vigente == json.dumps(SPEC del JS, compacto): {cadena_ok}")
    if not cadena_ok:
        print("\n  ALTO: el formato del SPEC JSON vigente no es el esperado.\n")
        sys.exit(1)

    for ruta in (OUT_SPEC, OUT_JS):
        if ruta.resolve() in (SPEC_VIGENTE.resolve(), JS_VIGENTE.resolve()):
            print("\n  ALTO: la salida coincide con un artefacto vigente.\n")
            sys.exit(1)

    OUT_SPEC.write_text(json.dumps(SPEC, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")

    original = JS_VIGENTE.read_text(encoding="utf-8")
    nueva = "const SPEC = " + json.dumps(SPEC, ensure_ascii=False) + ";"
    nuevo, n = __import__("re").subn(r"^const SPEC = .*;$", lambda _: nueva,
                                     original, count=1, flags=__import__("re").MULTILINE)
    assert n == 1, "no se localizó la constante SPEC en el JS vigente"
    assert nuevo.count(CABECERA_VIGENTE) == 1, "cabecera del JS vigente inesperada"
    nuevo = nuevo.replace(CABECERA_VIGENTE, CABECERA_NUEVA, 1)
    OUT_JS.write_text(nuevo, encoding="utf-8")

    # El JSON y el SPEC del JS nuevo deben ser el mismo objeto.
    assert json.loads(OUT_SPEC.read_text(encoding="utf-8")) == s19.spec_desde_js(OUT_JS)

    arboles = SPEC["puntaje"]["model"]["trees"]
    nodos = sum(len(t) for t in arboles)
    hojas = sum(1 for t in arboles for nd in t if nd[0] == -1)
    v_trees = spec_js_vigente["puntaje"]["model"]["trees"]
    out = {
        "spec_json": {"ruta": OUT_SPEC.relative_to(BASE_DIR).as_posix(),
                      "bytes": OUT_SPEC.stat().st_size},
        "js": {"ruta": OUT_JS.relative_to(BASE_DIR).as_posix(),
               "bytes": OUT_JS.stat().st_size},
        "vigente": {"spec_json_bytes": SPEC_VIGENTE.stat().st_size,
                    "js_bytes": JS_VIGENTE.stat().st_size,
                    "arboles": len(v_trees),
                    "nodos": sum(len(t) for t in v_trees),
                    "n_features": len(s19.features_from_raw({}, spec_js_vigente["puntaje"]["preprocess"]))},
        "arboles": len(arboles), "nodos": nodos, "hojas": hojas,
        "n_features": len(s19.features_from_raw({}, SPEC["puntaje"]["preprocess"])),
        "cadena_formato_verificada": cadena_ok,
    }
    log(f"SPEC JSON → {out['spec_json']['ruta']} ({out['spec_json']['bytes']:,} B)")
    log(f"JS        → {out['js']['ruta']} ({out['js']['bytes']:,} B)")
    log(f"árboles={out['arboles']} nodos={nodos:,} hojas={hojas:,} features={out['n_features']}"
        f"  | vigente: árboles={out['vigente']['arboles']} nodos={out['vigente']['nodos']:,}")
    return out


# ---------------------------------------------------------------------------
# Paridad Python ↔ JS
# ---------------------------------------------------------------------------
HERRAMIENTAS = ["Python", "Scratch", "Arduino", "Minecraft Education", "Roblox Studio",
                "Micro:bit", "App Inventor", "Tinkercad", "LEGO", "JavaScript",
                "C++", "Unity", "Ninguna"]
AREAS = ["Matemáticas", "Robótica", "Programación", "Inteligencia Artificial",
         "Artes", "Cultura", "Física", "Química", "Biología", "Ninguna"]


def generar_payloads(SPEC: dict, df: pd.DataFrame) -> list[tuple[str, dict]]:
    """≥ 2,000 cargas sintéticas + casos adversariales. Cada una lleva su tipo."""
    rng = np.random.default_rng(RANDOM_STATE)
    PRE = SPEC["puntaje"]["preprocess"]
    ENG = SPEC["engagement"]
    voc = {c: sorted(df[c].dropna().astype(str).unique().tolist())
           for c in ("nivel_programacion", "nivel_robotica", "computador_en_casa",
                     "internet_en_casa", "participacion_olimpiadas", "con_quien_vive")}
    hi_h, hi_a = int(ENG["n_herramientas"]["hi"]), int(ENG["n_areas_interes"]["hi"])

    def lista(items, k):
        return json.dumps(list(rng.choice(items, size=k, replace=False)), ensure_ascii=False)

    def base():
        return {
            "grado_escolar": int(rng.integers(6, 12)),
            "estrato": int(rng.integers(1, 7)),
            "interes_prog_robotica": float(rng.integers(1, 6)),
            "herramientas_conocidas": lista(HERRAMIENTAS[:-1], int(rng.integers(0, 6))),
            "areas_interes": lista(AREAS[:-1], int(rng.integers(1, 6))),
            "nivel_programacion": str(rng.choice(voc["nivel_programacion"])),
            "nivel_robotica": str(rng.choice(voc["nivel_robotica"])),
            "computador_en_casa": str(rng.choice(voc["computador_en_casa"])),
            "internet_en_casa": str(rng.choice(voc["internet_en_casa"])),
            "participacion_olimpiadas": str(rng.choice(voc["participacion_olimpiadas"])),
            "con_quien_vive": str(rng.choice(voc["con_quien_vive"])),
            **{c: str(rng.choice(PRE["onehot_cats"][c])) for c in PRE["onehot_order"]},
        }

    cargas: list[tuple[str, dict]] = []
    for i in range(2000):
        r = base()
        if i % 2 == 0:  # la mitad con puntaje real → índice por percentil
            r["puntaje_obtenido"] = float(rng.integers(0, 21) * 5)
        cargas.append(("sintetica", r))

    # Medianas/modas exactas: todo ausente → cada campo se imputa del SPEC.
    cargas.append(("todo_ausente", {}))
    # Cada municipio (incluida Copacabana) y cada categoría one-hot.
    for col in PRE["onehot_order"]:
        for cat in PRE["onehot_cats"][col]:
            for con_puntaje in (False, True):
                r = base(); r[col] = cat
                if con_puntaje:
                    r["puntaje_obtenido"] = 50.0
                cargas.append((f"onehot:{col}", r))
    # Cada campo opcional ausente, de uno en uno, y en None/"".
    for campo in list(base().keys()):
        for modo in ("ausente", "none", "vacio"):
            r = base()
            if modo == "ausente":
                r.pop(campo)
            else:
                r[campo] = None if modo == "none" else ""
            cargas.append((f"faltante:{modo}", r))
    # Listas: vacías, "Ninguna", en el techo del engagement y por encima.
    for h, a, tag in (("[]", "[]", "vacias"), ('["Ninguna"]', '["Ninguna"]', "ninguna"),
                      (json.dumps(HERRAMIENTAS[:hi_h], ensure_ascii=False),
                       json.dumps(AREAS[:min(hi_a, len(AREAS) - 1)], ensure_ascii=False), "techo"),
                      (json.dumps(HERRAMIENTAS[:-1] + ["X1", "X2", "X3"], ensure_ascii=False),
                       json.dumps(AREAS[:-1] + ["Y1", "Y2"], ensure_ascii=False), "sobre_techo"),
                      ("Python, Scratch", "Artes,Robótica", "csv_plano")):
        for con_puntaje in (False, True):
            r = base(); r["herramientas_conocidas"] = h; r["areas_interes"] = a
            if con_puntaje:
                r["puntaje_obtenido"] = 70.0
            cargas.append((f"listas:{tag}", r))
    # Fuera de vocabulario y tipos raros.
    oov = {"municipio": "Medellín", "genero": "Otro", "tipo_institucion": "Mixta",
           "nivel_programacion": "Experto", "nivel_robotica": "?",
           "computador_en_casa": "Tal vez", "internet_en_casa": "quizá",
           "participacion_olimpiadas": "X", "estrato": "abc", "grado_escolar": "once",
           "interes_prog_robotica": 9.0, "con_quien_vive": "Otro familiar"}
    for campo, valor in oov.items():
        r = base(); r[campo] = valor
        cargas.append((f"oov:{campo}", r))
    for campo, valor in (("municipio", " Copacabana "), ("municipio", "copacabana"),
                         ("estrato", "3"), ("grado_escolar", "10.0"),
                         ("puntaje_obtenido", "65"), ("puntaje_obtenido", -5.0),
                         ("puntaje_obtenido", 130.0), ("estrato", 0)):
        r = base(); r[campo] = valor
        cargas.append((f"tipos:{campo}", r))
    return cargas


HARNESS = """\
import fs from "node:fs";
import { calcularIndicePotencial } from "./pred_intacto.mjs";
import { SPEC, _featuresPuntaje, _predictPuntaje } from "./pred_export.mjs";
const filas = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const PRE = SPEC.puntaje.preprocess, MODEL = SPEC.puntaje.model;
const res = filas.map((r) => ({ crudo: _predictPuntaje(_featuresPuntaje(r, PRE), MODEL),
                               indice: calcularIndicePotencial(r) }));
fs.writeFileSync(process.argv[3], JSON.stringify({ node: process.version, res }));
"""


def correr_node(cargas: list[dict]) -> dict:
    import shutil, subprocess, tempfile
    node = shutil.which("node")
    if node is None:
        print("\n  ALTO: Node.js no está en el PATH.\n")
        sys.exit(1)
    tmp = Path(tempfile.mkdtemp(prefix="copastem_23_"))
    try:
        texto = OUT_JS.read_text(encoding="utf-8")
        (tmp / "pred_intacto.mjs").write_text(texto, encoding="utf-8")
        (tmp / "pred_export.mjs").write_text(
            texto + "\nexport { SPEC, _featuresPuntaje, _predictPuntaje };\n", encoding="utf-8")
        (tmp / "harness.mjs").write_text(HARNESS, encoding="utf-8")
        (tmp / "in.json").write_text(json.dumps(cargas, ensure_ascii=False), encoding="utf-8")
        p = subprocess.run([node, str(tmp / "harness.mjs"), str(tmp / "in.json"),
                            str(tmp / "out.json")], capture_output=True, text=True,
                           encoding="utf-8", timeout=900)
        if p.returncode != 0:
            print(f"\n  ALTO: Node falló:\n{p.stderr[:2000]}\n")
            sys.exit(1)
        return json.loads((tmp / "out.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TOL_CRUDO = 1e-9    # puntaje estimado a precisión completa
TOL_INDICE = 0.01   # salida pública redondeada a 2 decimales (modo de redondeo)


def paridad(SPEC: dict, df: pd.DataFrame, bundle: dict) -> dict:
    titulo("Paridad Python (intérprete del SPEC) ↔ JS")
    cargas = generar_payloads(SPEC, df)
    js = correr_node([c for _, c in cargas])
    PRE, MODEL = SPEC["puntaje"]["preprocess"], SPEC["puntaje"]["model"]

    filas = []
    for (tipo, r), o in zip(cargas, js["res"]):
        py_crudo = pot._predict_puntaje(pot._features_puntaje(r, PRE), MODEL)
        py = pot.calcular_indice(r, SPEC)
        d_crudo = abs(py_crudo - o["crudo"])
        d_ind = max(abs(py[k] - o["indice"][k]) for k in
                    ("indice_potencial", "componente_rendimiento",
                     "componente_engagement", "componente_resiliencia"))
        filas.append({"tipo": tipo.split(":")[0], "d_crudo": d_crudo, "d_ind": d_ind,
                      "cat_ok": py["categoria"] == o["indice"]["categoria"],
                      "crudo_exacto": py_crudo == o["crudo"]})
    f = pd.DataFrame(filas)
    mism = (f["d_crudo"] > TOL_CRUDO) | (f["d_ind"] > TOL_INDICE) | (~f["cat_ok"])
    causa = np.where(f["d_crudo"] > TOL_CRUDO, "lista_ausente_0_vs_mediana",
                     np.where(f["d_ind"] > 0, "redondeo_bancario_round()", "ok"))
    out = {
        "node": js["node"], "n_cargas": int(len(f)),
        "n_comparaciones": int(len(f) * 6),  # crudo + 4 componentes + categoría
        "tolerancias": {"puntaje_crudo": TOL_CRUDO, "indice_y_componentes": TOL_INDICE,
                        "categoria": "igualdad exacta"},
        "cargas_con_discrepancia": int(mism.sum()),
        "max_abs_diff_crudo": float(f["d_crudo"].max()),
        "crudo_bit_a_bit_identico": int(f["crudo_exacto"].sum()),
        "max_abs_diff_indice": float(f["d_ind"].max()),
        "categorias_distintas": int((~f["cat_ok"]).sum()),
        "discrepancias_por_causa": pd.Series(causa[mism.to_numpy()]).value_counts().to_dict(),
        "ruta_python": "models/deploy/potencial_stem_predictor.py (pot.calcular_indice)",
        "por_tipo": f.groupby("tipo").agg(n=("d_crudo", "size"),
                                          max_d_crudo=("d_crudo", "max"),
                                          max_d_indice=("d_ind", "max")).reset_index()
                     .to_dict("records"),
    }
    log(f"{out['n_cargas']:,} cargas ({out['n_comparaciones']:,} comparaciones): "
        f"discrepancias={out['cargas_con_discrepancia']}  max|Δ crudo|={out['max_abs_diff_crudo']:.3e} "
        f"(bit a bit {out['crudo_bit_a_bit_identico']:,})  max|Δ índice|={out['max_abs_diff_indice']:.3e}  "
        f"categorías distintas={out['categorias_distintas']}")

    # Referencia adicional: la ruta de ENTRENAMIENTO (s12 + sklearn) sobre las
    # mismas cargas. No es la ruta de servicio; se reporta la semántica que
    # difiere, sin tocar ningún artefacto.
    X, *_ = s12.construir_crudos(pd.DataFrame([c for _, c in cargas]), incluir_nuevas=False)
    p_sk = bundle["modelo"].predict(X)
    d = np.abs(p_sk - np.array([o["crudo"] for o in js["res"]]))
    tipos = pd.Series([t for t, _ in cargas])
    out["ruta_entrenamiento_vs_js"] = {
        "cargas_con_diferencia_gt_1e-9": int((d > 1e-9).sum()),
        "max_abs_diff": float(d.max()),
        "tipos_con_diferencia": tipos[d > 1e-9].value_counts().to_dict(),
    }
    log(f"ruta de entrenamiento (sklearn) vs JS: {out['ruta_entrenamiento_vs_js']}")
    return out


def paridad_scorer_api(ruta_scorer: Path, SPEC: dict, df: pd.DataFrame) -> dict:
    """Ejecuta `app/models/scorer.py` de la API EN MEMORIA contra el JS.

    Solo lectura: el fichero se lee como texto y se compila con `exec`; no se
    importa el paquete ni se escribe `.pyc`. `app.core.config` se sustituye por
    un stub que apunta `spec_file` al SPEC nuevo. Se corre dos veces: con la
    `sum()` del intérprete actual (compensada desde Python 3.12) y con suma
    ingenua izquierda-a-derecha (la de Python ≤ 3.11 y la del `reduce` del JS).
    """
    import functools, operator, types
    titulo("Paridad scorer de la API (en memoria) ↔ JS")
    sys.dont_write_bytecode = True
    cfg = types.ModuleType("app.core.config")
    cfg.get_settings = lambda: types.SimpleNamespace(spec_file=OUT_SPEC)
    sys.modules.update({"app": types.ModuleType("app"),
                        "app.core": types.ModuleType("app.core"),
                        "app.core.config": cfg})
    src = ruta_scorer.read_text(encoding="utf-8")
    cargas = generar_payloads(SPEC, df)
    js = correr_node([c for _, c in cargas])["res"]
    claves = ("indice_potencial", "componente_rendimiento",
              "componente_engagement", "componente_resiliencia")
    out = {"python": sys.version.split()[0], "n_cargas": len(cargas),
           "n_comparaciones": len(cargas) * 6}
    for modo in ("sum_interprete_actual", "suma_ingenua"):
        api = types.ModuleType(f"scorer_api_{modo}")
        sys.modules[api.__name__] = api
        api.__dict__["__file__"] = str(ruta_scorer)
        if modo == "suma_ingenua":
            api.__dict__["sum"] = lambda it, start=0: functools.reduce(operator.add, it, start)
        exec(compile(src, str(ruta_scorer), "exec"), api.__dict__)
        d_crudo, d_ind, cat_mal = [], [], 0
        for (_, r), o in zip(cargas, js):
            crudo = api._predict_puntaje(api._features_puntaje(
                r, SPEC["puntaje"]["preprocess"], []), SPEC["puntaje"]["model"])
            res = api.calcular_indice_potencial(r)
            d_crudo.append(abs(crudo - o["crudo"]))
            d_ind.append(max(abs(getattr(res, k) - o["indice"][k]) for k in claves))
            cat_mal += res.categoria != o["indice"]["categoria"]
        d_crudo, d_ind = np.array(d_crudo), np.array(d_ind)
        out[modo] = {"crudo_bit_a_bit": int((d_crudo == 0).sum()),
                     "max_abs_diff_crudo": float(d_crudo.max()),
                     "salida_publica_identica": int((d_ind == 0).sum()),
                     "max_abs_diff_salida_publica": float(d_ind.max()),
                     "categorias_distintas": int(cat_mal),
                     "cargas_con_discrepancia": int(((d_crudo > 0) | (d_ind > 0)).sum() + 0)}
        log(f"{modo:<22} {out[modo]}")
    out["version_modelo_leida_por_la_api"] = api.version_modelo()
    out["vocabulario_municipio_leido_por_la_api"] = api.categorias_onehot("municipio")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(" COPA STEM 2026 — Exportación del modelo global base · Script 23")
    print("=" * 78)
    impacto, df, ref_nueva, eng_nuevo = impacto_referencia()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMPACTO.write_text(json.dumps(impacto, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    log(f"impacto → outputs/{OUT_IMPACTO.name}")

    if "--exportar" not in sys.argv:
        print("\n  Fase 2 (exportación) no ejecutada: requiere --exportar.\n")
        return

    import joblib
    bundle = joblib.load(JOBLIB_GLOBAL)
    regs = registros(df.drop(columns=["_inst"], errors="ignore"))
    SPEC = construir_spec_global(bundle, regs, ref_nueva)
    assert SPEC["engagement"] == eng_nuevo
    vig = json.loads(SPEC_VIGENTE.read_text(encoding="utf-8"))
    assert SPEC["pesos"] == vig["pesos"] and SPEC["categorias"] == vig["categorias"]
    assert set(SPEC["puntaje"]["preprocess"]) == set(vig["puntaje"]["preprocess"])
    adaptador = verificar_adaptador(bundle, df, regs, SPEC)
    artefactos = exportar(SPEC)
    par = paridad(SPEC, df, bundle)
    par_api = None
    if "--scorer-api" in sys.argv:
        ruta = Path(sys.argv[sys.argv.index("--scorer-api") + 1]).resolve()
        par_api = paridad_scorer_api(ruta, SPEC, df)
    OUT_PARIDAD.write_text(json.dumps({"adaptador": adaptador, "artefactos": artefactos,
                                       "paridad": par, "paridad_scorer_api": par_api},
                                      ensure_ascii=False, indent=2),
                           encoding="utf-8")
    log(f"paridad → outputs/{OUT_PARIDAD.name}")
    if par["cargas_con_discrepancia"]:
        print("\n  DISCREPANCIAS ENTRE PYTHON Y JS — ver el JSON. No se ajusta nada.\n")


if __name__ == "__main__":
    main()
