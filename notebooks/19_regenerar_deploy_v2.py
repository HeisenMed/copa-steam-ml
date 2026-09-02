# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 19: Regeneración del artefacto de despliegue v2 con la referencia
            corregida (n = 3,072)
================================================================================

El problema
-----------
`models/deploy/potencial_stem_predictor_v2.js` es el artefacto que consume la
Edge Function. Lo generó el script 14 y lleva embebida, dentro de su constante
`SPEC`, la distribución de referencia `ref_rendimiento` calculada sobre los
**1,148** estudiantes de `dataset_C_perfil.csv` — el dataset de ENTRENAMIENTO
del modelo v2.

El script 17 demostró que esa vara de medir es la equivocada (σ 20.53 frente a
22.66) y la corrigió sobre los **3,072** examinados de `dataset_B_completo.csv`,
dejando el resultado en `outputs/F17_ref_rendimiento_corregido.json`. Pero esa
corrección se quedó en el lado Python/CSV: **nunca se propagó al `.js`**.
Desplegar el artefacto tal como está hoy reintroduciría, por la puerta de atrás,
exactamente el problema que el script 17 existió para eliminar.

Qué hace este script
--------------------
    1) Carga `models/mejor_modelo_puntaje_v2.joblib` (el mismo modelo v2 ya
       verificado que usan los scripts 14/15/17) y reconstruye el `MODEL`
       serializable con el mismo extractor de árboles del script 14.
    2) Comprueba que ese modelo es, bit a bit, el que está embebido en el `.js`
       actual: si no coincidiera, el `.js` no vendría de este `.joblib` y no
       tendría sentido regenerarlo desde aquí.
    3) Reconstruye el `SPEC` con el mismo proceso de exportación del script 14
       (mismo cuerpo JS, misma estructura de SPEC, mismo preprocesamiento
       guardado en el bundle), sustituyendo `ref_rendimiento` por la referencia
       corregida de 3,072 de `outputs/F17_ref_rendimiento_corregido.json`.
    4) Verifica que el `SPEC` nuevo difiere del desplegado SOLO en
       `ref_rendimiento` y en `meta`.
    5) Verifica la precisión en dos planos:
         · Python  — intérprete de árboles vs `sklearn.predict`.
         · Node.js — el `.js` GENERADO vs `sklearn.predict`, sobre las mismas
                     filas, y el índice compuesto completo del `.js` contra el
                     predictor Python de `models/deploy/`.
    6) Escribe el resultado como fichero NUEVO:
       `models/deploy/potencial_stem_predictor_v2_corrected.js`.

Qué NO hace
-----------
No sobrescribe `potencial_stem_predictor_v2.js`: el fichero viejo queda intacto
para comparación y rollback (el script aborta si la ruta de salida coincidiera
con él). No toca ningún otro script numerado, ningún `.joblib`, ningún otro
artefacto de `models/deploy/`, ni Supabase, ni la Edge Function. El paso 8
comprueba esa afirmación con hashes SHA-256 tomados antes y después.

Reproducible: `random_state=42`. Requiere Node.js en el PATH para la
verificación cruzada (si falta, el script lo dice y sigue con la parte Python).
Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: falta una dependencia del entorno. Detalle: {exc}")
    sys.exit(1)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# La consola de Windows llega en cp1252 y este script imprime `σ` y `Δ`.
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FUENTE_C = DATA_DIR / "dataset_C_perfil.csv"    # entrenamiento del modelo v2
FUENTE_B = DATA_DIR / "dataset_B_completo.csv"  # cohorte examinada completa
JOBLIB = MODELS_DIR / "mejor_modelo_puntaje_v2.joblib"
JS_ACTUAL = DEPLOY_DIR / "potencial_stem_predictor_v2.js"   # SOLO LECTURA
REF_JSON = OUTPUTS_DIR / "F17_ref_rendimiento_corregido.json"

OUT_JS = DEPLOY_DIR / "potencial_stem_predictor_v2_corrected.js"
OUT_VERIF = OUTPUTS_DIR / "F19_verificacion_deploy_v2.json"

TARGET = "puntaje_obtenido"
COL_PERFIL = "promedio_academico"

# Muestra de verificación: filas CON perfil (ruta v2 normal) + filas SIN perfil
# (ejercitan la imputación por mediana/moda del SPEC, que es donde el JS y
# sklearn podrían divergir si el preprocesamiento no fuera idéntico).
N_MUESTRA_C = 200
N_MUESTRA_B = 100

# Constantes del índice compuesto — idénticas a las de los scripts 04 y 14.
PESOS = {"rendimiento": 0.50, "engagement": 0.25, "resiliencia": 0.25}
CATEGORIAS = [
    (85, "Talento destacado"),
    (70, "Alto potencial"),
    (45, "Promedio"),
    (25, "En desarrollo"),
    (0,  "Requiere apoyo"),
]

DOCS_PRUEBA = {"1234", "123456", "123456789", "1234567899"}


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def titulo(txt: str) -> None:
    print("\n" + "=" * 78)
    print(f" {txt}")
    print("=" * 78)


def aviso(msg: str) -> None:
    print(f"\n  ADVERTENCIA: {msg}\n")


# ---------------------------------------------------------------------------
# Transformaciones — copiadas literalmente del script 14 (y del JS)
# ---------------------------------------------------------------------------
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
        parsed = json.loads(s)
        if isinstance(parsed, list):
            items = parsed
    except Exception:
        items = None
    if items is None:
        items = [x for x in s.strip("[]").replace('"', "").split(",") if x.strip()]
    return sum(1 for it in items
               if str(it).strip().lower() not in
               ("", "ninguna", "ninguno", "ninguna.", "ninguno."))


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
        return 1.0
    if s.startswith("n"):
        return 0.0
    return None


def features_from_raw(raw: dict, PRE: dict) -> list[float]:
    """Réplica exacta de `_featuresPuntaje` del predictor JS."""
    feats = []
    for name in PRE["numeric"]:
        v = _to_float(raw.get(name))
        if v is None and name == "n_herramientas":
            v = _parse_count(raw.get("herramientas_conocidas"))
        if v is None and name == "n_areas_interes":
            v = _parse_count(raw.get("areas_interes"))
        if v is None:
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
        if val is None or str(val).strip().lower() in ("nan", "none", ""):
            val = PRE["onehot_mode"][col]
        val = str(val).strip()
        for cat in PRE["onehot_cats"][col]:
            feats.append(1.0 if val == cat else 0.0)
    return feats


def matriz(registros: list[dict], PRE: dict) -> np.ndarray:
    return np.array([features_from_raw(r, PRE) for r in registros], dtype=float)


def _extract_sklearn_tree(t) -> list:
    """Árbol → lista de nodos [feature, umbral, izq, der, valor]. Idéntico al 14."""
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            nodes.append([-1, 0.0, -1, -1, float(t.value[i].ravel()[0])])
        else:
            nodes.append([int(t.feature[i]), float(t.threshold[i]),
                          int(t.children_left[i]), int(t.children_right[i]), 0.0])
    return nodes


def _extract_rf(modelo) -> dict:
    return {"type": "ensemble", "combine": "mean", "op": "le", "bias": 0.0,
            "trees": [_extract_sklearn_tree(e.tree_) for e in modelo.estimators_]}


def _predecir(feats: list[float], MODEL: dict) -> float:
    total = 0.0
    for tree in MODEL["trees"]:
        i = 0
        while tree[i][0] != -1:
            fi, thr = tree[i][0], tree[i][1]
            i = tree[i][2] if feats[fi] <= thr else tree[i][3]
        total += tree[i][4]
    y = total / len(MODEL["trees"]) + MODEL["bias"]
    return min(100.0, max(0.0, y))


# ---------------------------------------------------------------------------
# Utilidades de este script
# ---------------------------------------------------------------------------
def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def inventario(carpetas: list[Path]) -> dict:
    """SHA-256 de todo lo que este script NO debe tocar."""
    inv = {}
    for c in carpetas:
        for f in sorted(c.glob("*")):
            if f.is_file() and f != OUT_JS:
                inv[str(f.relative_to(BASE_DIR)).replace("\\", "/")] = sha256(f)
    return inv


def cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spec_desde_js(ruta: Path) -> dict:
    """Extrae la constante SPEC del artefacto JS (igual que el script 17)."""
    txt = ruta.read_text(encoding="utf-8")
    m = re.search(r"^const SPEC = (.*);$", txt, flags=re.MULTILINE)
    if not m:
        aviso(f"no se encontró la constante SPEC en {ruta.name}")
        sys.exit(1)
    return json.loads(m.group(1))


def cargar_C() -> pd.DataFrame:
    """Mismos filtros que `cargar()` del script 14."""
    if not FUENTE_C.exists():
        aviso(f"falta {FUENTE_C}. Ejecute antes el script 11.")
        sys.exit(1)
    df = pd.read_csv(FUENTE_C, encoding="utf-8", dtype={"numero_documento": str})
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df = df[~df["numero_documento"].isin(DOCS_PRUEBA)]
    df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()]
    return df.reset_index(drop=True)


def cargar_B() -> pd.DataFrame:
    """Mismos filtros que el script 17."""
    if not FUENTE_B.exists():
        aviso(f"falta {FUENTE_B}. Ejecute antes el script 11.")
        sys.exit(1)
    df = pd.read_csv(FUENTE_B, encoding="utf-8", dtype={"numero_documento": str})
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df = df[~df["numero_documento"].isin(DOCS_PRUEBA)]
    return df.reset_index(drop=True)


def limpiar(registros: list[dict]) -> list[dict]:
    """NaN → None y tipos numpy → tipos nativos.

    Las MISMAS filas limpias se le pasan a Python y a Node, para que la
    comparación mida el código y no el paso por JSON.
    """
    out = []
    for r in registros:
        fila = {}
        for k, v in r.items():
            if v is None:
                fila[k] = None
            elif isinstance(v, (np.integer,)):
                fila[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                f = float(v)
                fila[k] = None if f != f else f
            elif isinstance(v, (np.bool_, bool)):
                fila[k] = bool(v)
            else:
                s = str(v)
                fila[k] = None if s.strip().lower() in ("nan", "none") else s
        out.append(fila)
    return out


def _round2_js(x: float) -> float:
    """`Math.round(x * 100) / 100` del JS. Python `round()` redondea al par."""
    return math.floor(x * 100.0 + 0.5) / 100.0


def componentes_sin_redondear(raw: dict, SPEC: dict, pot) -> dict:
    """Los 4 componentes del índice ANTES de redondear.

    Replica `calcular_indice` del predictor Python de `models/deploy/` usando
    sus propias funciones internas; lo único que se omite es el `round()` final,
    para poder comparar el modo de redondeo por separado del cálculo.
    """
    PRE, MODEL = SPEC["puntaje"]["preprocess"], SPEC["puntaje"]["model"]
    real = pot._to_float(raw.get(TARGET))
    presento = real is not None
    rend_raw = real if presento else \
        pot._predict_puntaje(pot._features_puntaje(raw, PRE), MODEL)
    c_rend = pot._percentil(rend_raw, SPEC["ref_rendimiento"])
    c_eng = pot._engagement(raw, SPEC)
    adv = pot._adversidad(raw)
    c_res = min(100.0, c_rend * (1.0 + adv * 0.15)) if presento \
        else max(0.0, 50.0 - adv * 5.0)
    pesos = SPEC["pesos"]
    return {
        "indice_potencial": (pesos["rendimiento"] * c_rend
                             + pesos["engagement"] * c_eng
                             + pesos["resiliencia"] * c_res),
        "componente_rendimiento": c_rend,
        "componente_engagement": c_eng,
        "componente_resiliencia": c_res,
    }


def construir_spec(registros: list[dict], PRE: dict, MODEL: dict,
                   ref: list, meta: dict) -> dict:
    """SPEC del índice compuesto — misma estructura que la del script 14.

    Única diferencia con `construir_spec` del 14: `ref_rendimiento` llega como
    parámetro (la referencia corregida de 3,072) en vez de calcularse sobre los
    1,148 del dataset de entrenamiento.
    """
    def conteos(name, key):
        vals = []
        for r in registros:
            c = _to_float(r.get(name))
            if c is None:
                c = _parse_count(r.get(key))
            if c is not None:
                vals.append(c)
        return (min(vals), max(vals)) if vals else (0.0, 1.0)

    lo_h, hi_h = conteos("n_herramientas", "herramientas_conocidas")
    lo_a, hi_a = conteos("n_areas_interes", "areas_interes")

    return {
        "meta": meta,
        "puntaje": {"preprocess": PRE, "model": MODEL},
        "engagement": {"n_herramientas": {"lo": float(lo_h), "hi": float(hi_h)},
                       "n_areas_interes": {"lo": float(lo_a), "hi": float(hi_a)}},
        "ref_rendimiento": ref,
        "pesos": PESOS,
        "categorias": [list(c) for c in CATEGORIAS],
    }


def exportar_js(SPEC: dict) -> tuple[int, list[int]]:
    """Clona el cuerpo del JS v2 y sustituye SOLO la constante SPEC.

    Misma mecánica que `exportar_js` del script 14, pero partiendo del v2 (no
    del v1), de modo que el cuerpo del artefacto desplegado se conserva byte a
    byte. Devuelve el tamaño en bytes y las líneas que cambian.
    """
    if OUT_JS.resolve() == JS_ACTUAL.resolve():
        aviso("la ruta de salida coincide con el artefacto vigente. Abortado.")
        sys.exit(1)

    original = JS_ACTUAL.read_text(encoding="utf-8")
    nueva = "const SPEC = " + json.dumps(SPEC, ensure_ascii=False) + ";"
    nuevo, n = re.subn(r"^const SPEC = .*;$", lambda _: nueva, original,
                       count=1, flags=re.MULTILINE)
    if n != 1:
        aviso("no se localizó la constante SPEC en el JS v2.")
        sys.exit(1)

    # Cabecera: el fichero deja de venir del script 14 y la referencia embebida
    # ya no es la del dataset de entrenamiento. Decirlo en el propio artefacto.
    nuevo = nuevo.replace(
        " * GENERADO por notebooks/14_optimizacion_hiperparametros.py — no editar a mano.\n"
        " * MODELO v2: Random Forest con hiperparámetros optimizados, entrenado\n"
        " * sobre dataset_C_perfil.csv (incluye las 5 variables de perfil académico).\n",
        " * GENERADO por notebooks/19_regenerar_deploy_v2.py — no editar a mano.\n"
        " * MODELO v2: Random Forest con hiperparámetros optimizados, entrenado\n"
        " * sobre dataset_C_perfil.csv (incluye las 5 variables de perfil académico).\n"
        " * REFERENCIA CORREGIDA (script 17): `ref_rendimiento` percentiliza contra\n"
        " * los 3,072 examinados de dataset_B_completo.csv, no contra los 1,148 del\n"
        " * dataset de entrenamiento. El modelo y el preprocesamiento no cambian\n"
        " * respecto a potencial_stem_predictor_v2.js.\n", 1)

    OUT_JS.write_text(nuevo, encoding="utf-8")

    # Diff real (no por índice: la cabecera crece 4 líneas y desplazaría todo).
    viejas, nuevas = original.splitlines(), nuevo.splitlines()
    quitadas = [l for l in difflib.unified_diff(viejas, nuevas, n=0, lineterm="")
                if l.startswith("-") and not l.startswith("---")]
    puestas = [l for l in difflib.unified_diff(viejas, nuevas, n=0, lineterm="")
               if l.startswith("+") and not l.startswith("+++")]
    resumen = {
        "lineas_antes": len(viejas), "lineas_despues": len(nuevas),
        "lineas_quitadas": len(quitadas), "lineas_puestas": len(puestas),
        "quitadas": [l[1:80] for l in quitadas],
        "puestas": [l[1:80] for l in puestas],
    }
    return OUT_JS.stat().st_size, resumen


# ---------------------------------------------------------------------------
HARNESS = """\
import fs from "node:fs";
import { calcularIndicePotencial } from "./pred_intacto.mjs";
import { SPEC, _featuresPuntaje, _predictPuntaje } from "./pred_export.mjs";

const filas = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const PRE = SPEC.puntaje.preprocess;
const MODEL = SPEC.puntaje.model;

const resultados = filas.map((r) => ({
  crudo: _predictPuntaje(_featuresPuntaje(r, PRE), MODEL),
  indice: calcularIndicePotencial(r),
}));

fs.writeFileSync(process.argv[3], JSON.stringify({
  node: process.version,
  n_ref: SPEC.ref_rendimiento.length,
  n_arboles: MODEL.trees.length,
  meta: SPEC.meta,
  resultados,
}));
"""


def verificar_en_node(muestra: list[dict]) -> dict | None:
    """Ejecuta el `.js` GENERADO en Node y devuelve sus salidas.

    Se importa el artefacto por dos vías:
      · `pred_intacto.mjs` — copia byte a byte del fichero generado; de ahí sale
        `calcularIndicePotencial`, que es lo que consume la Edge Function.
      · `pred_export.mjs`  — el mismo texto con una línea `export` añadida al
        final, para poder llamar a `_predictPuntaje` a precisión completa (el
        artefacto redondea a 2 decimales en su salida pública).
    """
    node = shutil.which("node")
    if node is None:
        aviso("Node.js no está en el PATH; se omite la verificación cruzada.")
        return None

    tmp = Path(tempfile.mkdtemp(prefix="copastem_19_"))
    try:
        texto = OUT_JS.read_text(encoding="utf-8")
        (tmp / "pred_intacto.mjs").write_text(texto, encoding="utf-8")
        (tmp / "pred_export.mjs").write_text(
            texto + "\nexport { SPEC, _featuresPuntaje, _predictPuntaje, _percentil };\n",
            encoding="utf-8")
        (tmp / "harness.mjs").write_text(HARNESS, encoding="utf-8")
        (tmp / "muestra.json").write_text(
            json.dumps(muestra, ensure_ascii=False), encoding="utf-8")

        proc = subprocess.run(
            [node, str(tmp / "harness.mjs"), str(tmp / "muestra.json"),
             str(tmp / "salida.json")],
            capture_output=True, text=True, encoding="utf-8", timeout=600)
        if proc.returncode != 0:
            aviso(f"Node falló (código {proc.returncode}):\n{proc.stderr[:2000]}")
            return None
        return json.loads((tmp / "salida.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(" COPA STEM 2026 — Regeneración del desplegable v2 (referencia 3,072)")
    print(" Fundación SapienceLab · Script 19")
    print("=" * 78)

    for f in (JOBLIB, JS_ACTUAL, REF_JSON, FUENTE_C, FUENTE_B):
        if not f.exists():
            aviso(f"falta {f}. Ejecute antes los scripts 11, 14 y 17.")
            sys.exit(1)

    # Foto previa de todo lo que NO se debe tocar.
    inv_antes = inventario([DEPLOY_DIR, MODELS_DIR])
    hash_js_antes = sha256(JS_ACTUAL)

    # --- Paso 1: modelo v2 ---------------------------------------------------
    titulo("PASO 1 — Modelo v2 desde el .joblib")
    bundle = joblib.load(JOBLIB)
    modelo = bundle["modelo"]
    PRE = bundle["preprocess"]
    log(f"{JOBLIB.name} — versión {bundle['version']}, entrenado "
        f"{bundle['entrenado']} sobre {bundle['dataset']}")
    log(f"    {len(modelo.estimators_)} árboles · "
        f"{len(bundle['feature_names'])} features")
    mh = bundle["metricas_holdout"]
    log(f"    hold-out: R² {mh['r2']:+.4f} · MAE {mh['mae']:.2f} "
        f"(n = {mh['n_test']:,})")
    MODEL = _extract_rf(modelo)
    log(f"    árboles serializados: {len(MODEL['trees'])}")

    # --- Paso 2: SPEC vigente ------------------------------------------------
    titulo("PASO 2 — SPEC embebido en el artefacto vigente")
    SPEC_ACTUAL = spec_desde_js(JS_ACTUAL)
    ref_antes = SPEC_ACTUAL["ref_rendimiento"]
    a = np.asarray(ref_antes, dtype=float)
    log(f"{JS_ACTUAL.name} ({JS_ACTUAL.stat().st_size / 1024:,.0f} KB)")
    log(f"    meta          : {json.dumps(SPEC_ACTUAL['meta'], ensure_ascii=False)}")
    log(f"    ref_rendimiento: n = {len(ref_antes):,} | media {a.mean():.2f} | "
        f"σ {a.std(ddof=0):.4f}")

    # ¿Viene este .js de este .joblib? Comparación bit a bit de los árboles.
    mismo_modelo = (json.dumps(MODEL, sort_keys=True)
                    == json.dumps(SPEC_ACTUAL["puntaje"]["model"], sort_keys=True))
    mismo_pre = (json.dumps(PRE, sort_keys=True, ensure_ascii=False)
                 == json.dumps(SPEC_ACTUAL["puntaje"]["preprocess"],
                               sort_keys=True, ensure_ascii=False))
    log(f"    modelo del .joblib == modelo embebido en el .js : {mismo_modelo}")
    log(f"    preprocess del .joblib == el embebido en el .js : {mismo_pre}")
    if not (mismo_modelo and mismo_pre):
        aviso("el .js vigente NO proviene de este .joblib. Revise antes de seguir.")
        sys.exit(1)

    # --- Paso 3: referencia corregida ---------------------------------------
    titulo("PASO 3 — Referencia corregida del script 17")
    ref_info = json.loads(REF_JSON.read_text(encoding="utf-8"))
    ref_nueva = [float(x) for x in ref_info["ref_rendimiento"]]
    b = np.asarray(ref_nueva, dtype=float)
    log(f"{REF_JSON.name} — generado {ref_info['generado']} desde "
        f"{ref_info['fuente']}")
    log(f"    ref_rendimiento: n = {len(ref_nueva):,} | media {b.mean():.2f} | "
        f"σ {b.std(ddof=0):.4f}")
    if len(ref_nueva) != ref_info["n_cohorte"]:
        aviso(f"n declarado ({ref_info['n_cohorte']:,}) ≠ longitud real "
              f"({len(ref_nueva):,}).")
        sys.exit(1)
    if ref_nueva != sorted(ref_nueva):
        aviso("la referencia no viene ordenada; `_percentil` hace búsqueda "
              "binaria y la exige ordenada.")
        sys.exit(1)
    log("    ordenada ascendente: sí (requisito de `_percentil`)")

    print(f"\n  {'Referencia':<34}{'n':>8}{'media':>9}{'σ':>10}"
          f"{'p25':>7}{'mediana':>9}{'p75':>7}")
    for nombre, arr in (("embebida hoy (dataset C)", a),
                        ("corregida (dataset B)", b)):
        print(f"  {nombre:<34}{len(arr):>8,}{arr.mean():>9.2f}"
              f"{arr.std(ddof=0):>10.4f}{np.percentile(arr, 25):>7.1f}"
              f"{np.median(arr):>9.1f}{np.percentile(arr, 75):>7.1f}")

    # --- Paso 4: SPEC nuevo --------------------------------------------------
    titulo("PASO 4 — SPEC regenerado")
    dfC = cargar_C()
    registros_C = limpiar(dfC.to_dict("records"))
    log(f"{FUENTE_C.name} — {len(registros_C):,} filas (rangos de engagement)")

    meta = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "generado_por": "notebooks/19_regenerar_deploy_v2.py",
        "n_cohorte": len(ref_nueva),
        "n_entrenamiento": len(registros_C),
        "modelo_puntaje": "Random Forest v2 (optimizado)",
        "version": "v2",
        "ref_rendimiento_fuente": f"outputs/{REF_JSON.name}",
        "ref_rendimiento_cohorte": ref_info["fuente"],
        "ref_rendimiento_sigma": round(float(b.std(ddof=0)), 4),
        "reemplaza": {"artefacto": JS_ACTUAL.name,
                      "n_ref_anterior": len(ref_antes)},
    }
    SPEC_NUEVO = construir_spec(registros_C, PRE, MODEL, ref_nueva, meta)

    # El SPEC nuevo debe diferir del vigente SOLO en `meta` y `ref_rendimiento`.
    claves = sorted(set(SPEC_ACTUAL) | set(SPEC_NUEVO))
    distintas = [k for k in claves
                 if json.dumps(SPEC_ACTUAL.get(k), sort_keys=True, ensure_ascii=False)
                 != json.dumps(SPEC_NUEVO.get(k), sort_keys=True, ensure_ascii=False)]
    print(f"\n  Claves del SPEC: {', '.join(claves)}")
    print(f"  Claves que cambian: {', '.join(distintas)}")
    if set(distintas) != {"meta", "ref_rendimiento"}:
        aviso(f"cambia algo más que `meta` y `ref_rendimiento`: {distintas}")
        sys.exit(1)
    log("OK — `puntaje` (modelo + preprocesamiento), `engagement`, `pesos` y "
        "`categorias` intactos")

    # --- Paso 5: exportación -------------------------------------------------
    titulo("PASO 5 — Exportación del artefacto corregido")
    tam, dif_js = exportar_js(SPEC_NUEVO)
    log(f"escrito → models/deploy/{OUT_JS.name} ({tam / 1024:,.0f} KB)")
    log(f"    {dif_js['lineas_antes']} líneas antes → "
        f"{dif_js['lineas_despues']} después "
        f"(−{dif_js['lineas_quitadas']} / +{dif_js['lineas_puestas']})")
    for l in dif_js["quitadas"]:
        print(f"      − {l}")
    for l in dif_js["puestas"]:
        print(f"      + {l}")
    log("    todo lo demás del cuerpo JS queda byte a byte igual")
    log(f"    {JS_ACTUAL.name} intacto: "
        f"{sha256(JS_ACTUAL) == hash_js_antes}")

    # --- Paso 6: verificación en Python -------------------------------------
    titulo("PASO 6 — Verificación en Python: intérprete de árboles vs sklearn")
    dfB = cargar_B()
    sin_perfil = dfB[dfB[COL_PERFIL].isna()] if COL_PERFIL in dfB.columns \
        else dfB.iloc[0:0]
    muestra_B = limpiar(
        sin_perfil.sample(n=min(N_MUESTRA_B, len(sin_perfil)),
                          random_state=RANDOM_STATE).to_dict("records"))
    muestra = registros_C[:N_MUESTRA_C] + muestra_B
    log(f"muestra: {len(registros_C[:N_MUESTRA_C])} filas con perfil académico "
        f"(ruta v2) + {len(muestra_B)} sin perfil (ruta de imputación) "
        f"= {len(muestra)}")

    y_sk = np.clip(modelo.predict(matriz(muestra, PRE)), 0, 100)
    y_py = np.array([_predecir(features_from_raw(r, PRE), MODEL) for r in muestra])
    d_py = float(np.max(np.abs(y_py - y_sk)))
    log(f"máx |Δ| intérprete Python vs sklearn : {d_py:.3e}")

    # --- Paso 7: verificación cruzada en Node -------------------------------
    titulo("PASO 7 — Verificación en Node: el .js generado vs sklearn")
    nodo = verificar_en_node(muestra)
    d_js = d_idx = d_idx_modo = None
    n_cat = n_exacto = n_redondeo = None
    if nodo is not None:
        log(f"Node {nodo['node']} · {nodo['n_arboles']} árboles · "
            f"ref de {nodo['n_ref']:,} puntajes en el fichero generado")
        y_js = np.array([r["crudo"] for r in nodo["resultados"]])
        d_js = float(np.max(np.abs(y_js - y_sk)))
        log(f"máx |Δ| `_predictPuntaje` del .js vs sklearn : {d_js:.3e}")

        # Índice compuesto completo: el .js intacto contra el predictor Python
        # de models/deploy/ alimentado con el MISMO SPEC corregido.
        pot = cargar_modulo("pot_v1", DEPLOY_DIR / "potencial_stem_predictor.py")
        campos = ("indice_potencial", "componente_rendimiento",
                  "componente_engagement", "componente_resiliencia")
        difs, difs_sin_redondeo, n_cat, n_exacto = [], [], 0, 0
        for r, res in zip(muestra, nodo["resultados"]):
            py = pot.calcular_indice(r, SPEC_NUEVO)
            js = res["indice"]
            difs.append(max(abs(float(py[c]) - float(js[c])) for c in campos))
            # El desacuerdo del último decimal es de MODO de redondeo, no de
            # cálculo: Python usa `round()` (al par) y el JS `Math.round()`
            # (mitad hacia arriba). Al aplicar el criterio del JS a los valores
            # SIN redondear de Python, la diferencia debe desaparecer.
            crudos = componentes_sin_redondear(r, SPEC_NUEVO, pot)
            difs_sin_redondeo.append(
                max(abs(_round2_js(crudos[c]) - float(js[c])) for c in campos))
            if py["categoria"] != js["categoria"]:
                n_cat += 1
            if difs[-1] == 0.0:
                n_exacto += 1
        d_idx = float(max(difs))
        d_idx_modo = float(max(difs_sin_redondeo))
        n_redondeo = sum(1 for d in difs if d > 0.0)
        log(f"máx |Δ| índice compuesto (.js vs predictor Python) : {d_idx:.3e}")
        log(f"    filas idénticas en los 4 componentes: {n_exacto}/{len(muestra)}")
        log(f"    categorías discrepantes             : {n_cat}")
        log(f"    filas que difieren solo en el último decimal: {n_redondeo}")
        log(f"    máx |Δ| aplicando el redondeo del JS a los valores crudos "
            f"de Python: {d_idx_modo:.3e}")
        if d_idx_modo == 0.0:
            log("    → el desacuerdo es 100 % modo de redondeo "
                "(`round()` al par vs `Math.round()`), no cálculo")

    for nombre, d in (("Python vs sklearn", d_py), (".js vs sklearn", d_js)):
        if d is not None and d > 1e-6:
            aviso(f"{nombre}: |Δ| = {d:.3e} por encima de 1e-6.")

    # --- Paso 8: nada más se movió ------------------------------------------
    titulo("PASO 8 — Comprobación de que no cambió nada más")
    inv_despues = inventario([DEPLOY_DIR, MODELS_DIR])
    cambiados = sorted(k for k in set(inv_antes) | set(inv_despues)
                       if inv_antes.get(k) != inv_despues.get(k))
    print(f"  ficheros vigilados (models/ y models/deploy/): {len(inv_antes)}")
    if cambiados:
        aviso(f"cambiaron ficheros que no debían: {cambiados}")
    else:
        log("OK — ningún .joblib ni ningún artefacto previo de models/deploy/ "
            "cambió de hash")
    print(f"  único fichero nuevo: models/deploy/{OUT_JS.name}")
    print("  no se tocó: notebooks/ anteriores · outputs/ previos · Supabase · "
          "Edge Function")

    # --- Resumen -------------------------------------------------------------
    resumen = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "script": "notebooks/19_regenerar_deploy_v2.py",
        "modelo": {
            "joblib": JOBLIB.name,
            "entrenado": bundle["entrenado"],
            "dataset_entrenamiento": bundle["dataset"],
            "n_arboles": len(MODEL["trees"]),
            "n_features": len(bundle["feature_names"]),
            "holdout": mh,
            "identico_al_embebido_en_js_vigente": mismo_modelo,
            "preprocess_identico": mismo_pre,
        },
        "referencia": {
            "antes": {"artefacto": JS_ACTUAL.name, "n": len(ref_antes),
                      "sigma": round(float(a.std(ddof=0)), 4),
                      "cohorte": "dataset_C_perfil.csv (entrenamiento)"},
            "despues": {"artefacto": OUT_JS.name, "n": len(ref_nueva),
                        "sigma": round(float(b.std(ddof=0)), 4),
                        "cohorte": ref_info["fuente"],
                        "fuente": f"outputs/{REF_JSON.name}"},
        },
        "spec_claves_que_cambian": distintas,
        "verificacion": {
            "n_muestra": len(muestra),
            "n_con_perfil": len(registros_C[:N_MUESTRA_C]),
            "n_sin_perfil": len(muestra_B),
            "max_delta_python_vs_sklearn": d_py,
            "max_delta_js_vs_sklearn": d_js,
            "max_delta_indice_js_vs_python": d_idx,
            "filas_con_delta_de_redondeo": n_redondeo,
            "max_delta_indice_con_redondeo_js": d_idx_modo,
            "filas_identicas_en_los_4_componentes": n_exacto,
            "categorias_discrepantes": n_cat,
            "node": None if nodo is None else nodo["node"],
        },
        "artefactos": {
            "nuevo": f"models/deploy/{OUT_JS.name}",
            "bytes": tam,
            "diff_vs_vigente": dif_js,
            "vigente_intacto": sha256(JS_ACTUAL) == hash_js_antes,
            "otros_ficheros_modificados": cambiados,
        },
    }
    OUT_VERIF.write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    log(f"\nresumen → outputs/{OUT_VERIF.name}")

    print("\n" + "=" * 78)
    print(f" COMPLETADO — {OUT_JS.name} listo, con la referencia de "
          f"{len(ref_nueva):,}")
    print(f" {JS_ACTUAL.name} queda intacto para comparación y rollback")
    print("=" * 78)


if __name__ == "__main__":
    main()
