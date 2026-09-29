# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 24: Scores de la cohorte publicada 2026 con la referencia corregida
            (3,050) y la corrección de los 3 documentos duplicados
================================================================================

Decisión declarada (informe 23, §1): la cohorte 2026 ya comunicada CONSERVA el
esquema mixto con el que se publicó —SPEC v2 para quien tiene perfil académico,
SPEC v1 ("v1_fallback") para quien no— y solo cambian dos cosas aprobadas:

    A) `ref_rendimiento` pasa de los 3,072 de F17 a los 3,050 limpios.
    B) Los 3 documentos duplicados cuya fila publicada era la sesión abandonada
       se puntúan con su sesión completada.

El modelo global base NO se aplica a esta cohorte (escenario B del informe 23,
no aprobado).

Método
------
1. CONTROL: se reconstruyen los dos archivos publicados
   (`ml_scores_v2_corrected.csv`, examinados; `ml_scores_sin_examen.csv`, sin
   examen) con la MISMA función `puntuar` del script 18 (idéntica a la de 15 y
   17), sus mismos SPECs y la referencia de F17. Deben coincidir celda a celda
   (salvo `updated_at`). Si no coinciden, el script se detiene.
2. Se repite con la referencia de 3,050 y las filas corregidas.

Privacidad
----------
El archivo de salida identifica estudiantes (`numero_documento`), igual que los
publicados. Por eso NO se escribe en el repo: va a `../_datos_privados/`. En
`outputs/` solo quedan agregados. La lista de descartados solo se usa en memoria
para construir la referencia y NO se marca fila a fila.

Reproducible. Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import datetime, timezone
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

BASE_DIR = Path(__file__).resolve().parent.parent
NB_DIR = BASE_DIR / "notebooks"
DATA_DIR = BASE_DIR / "data"
DEPLOY_DIR = BASE_DIR / "models" / "deploy"
OUTPUTS_DIR = BASE_DIR / "outputs"
PRIVADOS = BASE_DIR.parent / "_datos_privados"

PUB_EXAM = OUTPUTS_DIR / "ml_scores_v2_corrected.csv"    # publicado, solo lectura
PUB_SIN = OUTPUTS_DIR / "ml_scores_sin_examen.csv"       # publicado, solo lectura
EXPORT = DATA_DIR / "copa_stem_dataset_2026-08.csv.csv"
F17_REF = OUTPUTS_DIR / "F17_ref_rendimiento_corregido.json"

OUT_CSV = PRIVADOS / "ml_scores_2026_ref3050.csv"
OUT_RESUMEN = OUTPUTS_DIR / "F24_scores_cohorte_2026_ref3050.json"

TARGET = "puntaje_obtenido"
MUNICIPIOS = ["Bello", "Girardota", "Copacabana"]
ORDEN_CAT = ["Requiere apoyo", "En desarrollo", "Promedio", "Alto potencial",
             "Talento destacado"]


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


s18 = cargar_modulo("s18", NB_DIR / "18_scores_inscritos_sin_examen.py")
s21 = cargar_modulo("s21", NB_DIR / "21_modelo_global_base.py")
pot = cargar_modulo("pot", DEPLOY_DIR / "potencial_stem_predictor.py")
tal = cargar_modulo("tal", DEPLOY_DIR / "talento_oculto_predictor.py")
cond = cargar_modulo("cond", DEPLOY_DIR / "indice_condiciones_predictor.py")
clus = cargar_modulo("clus", DEPLOY_DIR / "clustering_predictor.py")


def specs(ref: list[float]) -> tuple[dict, dict]:
    """Los MISMOS SPECs con los que se publicó (scripts 17 y 18): v1 del
    predictor Python de despliegue y v2 del JS original; solo cambia la ref."""
    v1 = copy.deepcopy(pot.SPEC)
    v2 = s18.spec_desde_js(s18.JS_V2)
    v1["ref_rendimiento"] = ref
    v2["ref_rendimiento"] = ref
    return v1, v2


def perfil(df: pd.DataFrame) -> pd.Series:
    return df["promedio_academico"].notna() if "promedio_academico" in df.columns \
        else pd.Series(False, index=df.index)


def examinados(corregir_dedup: bool) -> tuple[pd.DataFrame, dict]:
    """Los 3,072 examinados publicados. Con `corregir_dedup`, los documentos
    duplicados usan la sesión completada (regla del informe 21) en vez de la
    primera fila (`keep='first'`, la que se publicó)."""
    df = pd.read_csv(EXPORT, encoding="utf-8", dtype={"numero_documento": str})
    df["numero_documento"] = df["numero_documento"].astype(str).str.strip()
    df = df[~df["numero_documento"].isin(s21.DOCS_PRUEBA)]
    dup = df.duplicated("numero_documento", keep=False)
    info = {"documentos_duplicados": int(df.loc[dup, "numero_documento"].nunique())}
    if corregir_dedup:
        t = pd.to_numeric(df["tiempo_usado_segundos"], errors="coerce")
        df = df[~(dup & (t > s21.UMBRAL_SESION_ABANDONADA_S))]
        assert not df.duplicated("numero_documento").any()
    else:
        df = df.drop_duplicates("numero_documento", keep="first")
    df = df.drop(columns=["institucion_educativa"])
    return df.reset_index(drop=True), info


def sin_examen() -> pd.DataFrame:
    """Misma población que el script 18: inscritos limpios sin fila en resultados."""
    ins = pd.read_csv(s18.FUENTES_INSCRITOS[0], encoding="utf-8",
                      dtype={"numero_documento": str})
    ins = s18.limpiar_documentos(ins, "inscritos")
    dfb = pd.read_csv(s18.EXAMINADOS, encoding="utf-8", dtype={"numero_documento": str})
    docs_b = set(dfb["numero_documento"].astype(str).str.strip())
    ins = ins[~ins["numero_documento"].isin(docs_b)].reset_index(drop=True)
    return ins.drop(columns=["institucion_educativa"], errors="ignore")


def puntuar(df: pd.DataFrame, ref: list[float], quitar_nota: bool) -> pd.DataFrame:
    v1, v2 = specs(ref)
    regs = df.to_dict("records")
    if quitar_nota:
        regs = s18.sin_nota(regs)
    filas = s18.puntuar(regs, perfil(df).tolist(), pot, tal, cond, clus, v1, v2)
    return pd.DataFrame(filas)


def comparar(nuevo: pd.DataFrame, publicado: pd.DataFrame) -> dict:
    """Celda a celda, alineado por documento, excluyendo `updated_at`."""
    a = nuevo.set_index("numero_documento").sort_index()
    b = publicado.set_index("numero_documento").sort_index()
    cols = [c for c in b.columns if c != "updated_at"]
    out = {"filas_nuevo": len(a), "filas_publicado": len(b),
           "mismos_documentos": set(a.index) == set(b.index)}
    if not out["mismos_documentos"]:
        return out
    a = a.loc[b.index]
    difs = {}
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
            d = ~np.isclose(pd.to_numeric(x), pd.to_numeric(y), rtol=0, atol=1e-9, equal_nan=True)
        else:
            d = x.astype(str).fillna("") != y.astype(str).fillna("")
        if d.sum():
            difs[c] = int(d.sum())
    out["columnas_con_diferencias"] = difs
    return out


def tabla(munis: pd.Series, antes: pd.Series, despues: pd.Series) -> dict:
    ia, id_ = antes.map(ORDEN_CAT.index), despues.map(ORDEN_CAT.index)
    def fila(k):
        return {"n": int(k.sum()), "cambian": int((ia[k] != id_[k]).sum()),
                "suben": int((id_[k] > ia[k]).sum()), "bajan": int((id_[k] < ia[k]).sum())}
    todos = pd.Series(True, index=antes.index)
    trans = pd.crosstab(antes, despues)
    return {"total": fila(todos),
            "por_municipio": {m: fila(munis == m) for m in MUNICIPIOS if (munis == m).any()},
            "transiciones": {f"{a} -> {b}": int(trans.loc[a, b]) for a in trans.index
                             for b in trans.columns if a != b and trans.loc[a, b] > 0}}


OUT_VERIF = OUTPUTS_DIR / "F24_verificacion_archivo.json"


def verificar_archivo() -> None:
    """`--verificar`: recalcula en memoria y compara con el archivo YA generado,
    sin reescribirlo. Sirve para confirmar que un cambio en otra parte (p. ej.
    el modelo global oficial) no lo afecta. Además cuenta en qué dirección van
    los cambios de `es_talento_oculto` (solo agregados)."""
    import hashlib
    titulo("VERIFICACIÓN del archivo existente (no se reescribe)")
    h_antes = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    existente = pd.read_csv(OUT_CSV, dtype={"numero_documento": str})
    existente["numero_documento"] = existente["numero_documento"].astype(str).str.strip()

    limpio, _ = s21.inventario(s21.cargar_descartados())
    ref_3050 = sorted(float(v) for v in limpio[TARGET])
    del limpio
    ex_cor, _ = examinados(corregir_dedup=True)
    recalc = pd.concat([puntuar(ex_cor, ref_3050, False), puntuar(sin_examen(), ref_3050, True)],
                       ignore_index=True)
    cols = [c for c in recalc.columns if c != "updated_at"]
    cmp = comparar(recalc[cols], existente[cols])

    pub = pd.concat([pd.read_csv(PUB_EXAM, dtype={"numero_documento": str}).assign(poblacion="examinado"),
                     pd.read_csv(PUB_SIN, dtype={"numero_documento": str}).assign(poblacion="sin_examen")])
    pub["numero_documento"] = pub["numero_documento"].astype(str).str.strip()
    m = pub.merge(existente, on=["numero_documento", "poblacion"], suffixes=("_pub", "_nuevo"))
    a = m["es_talento_oculto_pub"].astype(str) == "True"
    b = m["es_talento_oculto_nuevo"].astype(str) == "True"
    talento = {pob: {"entran_a_la_lista": int(((~a) & b & (m["poblacion"] == pob)).sum()),
                     "salen_de_la_lista": int((a & ~b & (m["poblacion"] == pob)).sum()),
                     "en_lista_publicado": int((a & (m["poblacion"] == pob)).sum()),
                     "en_lista_nuevo": int((b & (m["poblacion"] == pob)).sum())}
               for pob in ("examinado", "sin_examen")}
    h_despues = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    res = {"archivo_sha256": h_antes, "archivo_sin_modificar": h_antes == h_despues,
           "recalculo_vs_archivo": cmp,
           "dependencias": ("SPEC v1 del predictor Python de despliegue, SPEC v2 de "
                            "potencial_stem_predictor_v2.js y la referencia de 3,050 "
                            "(datos); NO usa models/modelo_global_base.joblib"),
           "es_talento_oculto_direccion": talento}
    log(f"recálculo vs archivo: {cmp}")
    log(f"talento oculto: {talento}")
    log(f"archivo sin modificar: {res['archivo_sin_modificar']}")
    OUT_VERIF.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"verificación → outputs/{OUT_VERIF.name}")


def main() -> None:
    if "--verificar" in sys.argv:
        verificar_archivo()
        return
    print("=" * 78)
    print(" COPA STEM 2026 — Cohorte publicada con referencia 3,050 · Script 24")
    print("=" * 78)
    f17 = json.loads(F17_REF.read_text(encoding="utf-8"))["ref_rendimiento"]
    limpio, _ = s21.inventario(s21.cargar_descartados())
    ref_3050 = sorted(float(v) for v in limpio[TARGET])
    docs_limpios = set(limpio["numero_documento"])
    del limpio

    pub_exam = pd.read_csv(PUB_EXAM, dtype={"numero_documento": str})
    pub_sin = pd.read_csv(PUB_SIN, dtype={"numero_documento": str})
    for p in (pub_exam, pub_sin):
        p["numero_documento"] = p["numero_documento"].astype(str).str.strip()

    # --- 1. Control: reproducir lo publicado ------------------------------------
    titulo("1 — CONTROL: reconstrucción de los archivos publicados")
    ex_pub, info_dup = examinados(corregir_dedup=False)
    sn = sin_examen()
    ctrl_exam = comparar(puntuar(ex_pub, f17, False), pub_exam)
    ctrl_sin = comparar(puntuar(sn, f17, True), pub_sin)
    log(f"examinados: {ctrl_exam}")
    log(f"sin examen: {ctrl_sin}")
    for c in (ctrl_exam, ctrl_sin):
        if not c["mismos_documentos"] or c["columnas_con_diferencias"]:
            print("\n  ALTO: no se reproduce lo publicado. No se genera nada.\n")
            sys.exit(1)

    # --- 2. Cohorte corregida ----------------------------------------------------
    titulo("2 — Referencia 3,050 + corrección de los 3 documentos duplicados")
    ex_cor, _ = examinados(corregir_dedup=True)
    assert set(ex_cor["numero_documento"]) == set(ex_pub["numero_documento"])
    nuevo_exam = puntuar(ex_cor, ref_3050, False)
    nuevo_sin = puntuar(sn, ref_3050, True)
    ahora = datetime.now(timezone.utc).isoformat()
    for d in (nuevo_exam, nuevo_sin):
        d["updated_at"] = ahora
        d["referencia_rendimiento"] = "3050_limpia (informe 23)"
    nuevo_exam["poblacion"] = "examinado"
    nuevo_sin["poblacion"] = "sin_examen"

    # Cambios de categoría frente a lo publicado.
    def cambios(nuevo, pub, munis_por_doc, subset=None):
        a = pub.set_index("numero_documento")["categoria_potencial"]
        b = nuevo.set_index("numero_documento")["categoria_potencial"].loc[a.index]
        m = munis_por_doc.reindex(a.index)
        if subset is not None:
            k = a.index.isin(list(subset))
            a, b, m = a[k], b[k], m[k]
        return tabla(m, a, b)

    muni_ex = ex_cor.set_index("numero_documento")["municipio"]
    muni_sn = sn.set_index("numero_documento")["municipio"]
    t_pub = ex_pub.set_index("numero_documento")[TARGET]
    t_cor = ex_cor.set_index("numero_documento")[TARGET]
    docs_fila_corregida = set(t_pub.index[(t_pub != t_cor.loc[t_pub.index]).to_numpy()])
    docs_otros = set(pub_exam["numero_documento"]) - docs_limpios  # agregado, no se escribe

    res = {
        "generado": ahora,
        "salida": "../_datos_privados/ml_scores_2026_ref3050.csv (fuera del repo: identifica estudiantes)",
        "esquema": "v2 si tiene perfil / v1_fallback si no — el mismo con el que se publicó",
        "referencia": {"n": len(ref_3050), "media": float(np.mean(ref_3050)),
                       "sigma_ddof0": float(np.std(ref_3050))},
        "control_reproduccion": {"examinados": ctrl_exam, "sin_examen": ctrl_sin},
        "filas": {"examinados": len(nuevo_exam), "sin_examen": len(nuevo_sin),
                  "total": len(nuevo_exam) + len(nuevo_sin)},
        "modelo_version": pd.concat([nuevo_exam, nuevo_sin])["modelo_version"]
                          .value_counts().to_dict(),
        "duplicados": {**info_dup, "filas_corregidas": len(docs_fila_corregida)},
        "cambios_categoria": {
            "examinados_3050_limpios_sin_los_3": cambios(
                nuevo_exam, pub_exam, muni_ex, docs_limpios - docs_fila_corregida),
            "examinados_3_filas_corregidas": cambios(
                nuevo_exam, pub_exam, muni_ex, docs_fila_corregida),
            "examinados_22_resto_publicado": cambios(
                nuevo_exam, pub_exam, muni_ex, docs_otros),
            "examinados_total": cambios(nuevo_exam, pub_exam, muni_ex),
            "sin_examen": cambios(nuevo_sin, pub_sin, muni_sn),
        },
        "otras_columnas_que_cambian": {
            "examinados": comparar(nuevo_exam.drop(columns=["referencia_rendimiento", "poblacion"]),
                                   pub_exam)["columnas_con_diferencias"],
            "sin_examen": comparar(nuevo_sin.drop(columns=["referencia_rendimiento", "poblacion"]),
                                   pub_sin)["columnas_con_diferencias"],
        },
    }
    for k, v in res["cambios_categoria"].items():
        log(f"{k:<36} {v['total']}  {v['transiciones']}")
    log(f"otras columnas que cambian: {res['otras_columnas_que_cambian']}")

    PRIVADOS.mkdir(exist_ok=True)
    pd.concat([nuevo_exam, nuevo_sin], ignore_index=True).to_csv(
        OUT_CSV, index=False, encoding="utf-8")
    OUT_RESUMEN.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"scores → {OUT_CSV} ({res['filas']['total']:,} filas)")
    log(f"resumen → outputs/{OUT_RESUMEN.name}")


if __name__ == "__main__":
    main()
