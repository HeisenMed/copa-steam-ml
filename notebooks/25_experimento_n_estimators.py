# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 25: EXPERIMENTO LATERAL — coste de bajar n_estimators de 300 a 150
================================================================================

NO reentrena ni reemplaza el modelo oficial (`models/modelo_global_base.joblib`)
y NO escribe artefactos de despliegue. Mide, para informar una decisión humana
abierta (informe 23, §7), qué costaría un bosque de 150 árboles frente al de 300:

    · OOF R², MAE y RMSE con el MISMO protocolo del script 21
      (KFold(5, shuffle, random_state=42) sobre las 3,050 filas) y hold-out.
    · Tamaño del SPEC JSON y del JS que produciría, serializados EN MEMORIA con
      el mismo exportador del script 23.

Con random_state fijo, sklearn genera las semillas de los árboles en secuencia:
el bosque de 150 son exactamente los primeros 150 árboles del de 300.

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
NB_DIR = BASE_DIR / "notebooks"
OUT = BASE_DIR / "outputs" / "F25_experimento_n_estimators.json"


def cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s23 = cargar_modulo("s23", NB_DIR / "23_exportar_modelo_global.py")
s21, s12 = s23.s21, s23.s12


def main() -> None:
    print("=" * 78)
    print(" EXPERIMENTO LATERAL — n_estimators 300 vs 150 (no toca el oficial)")
    print("=" * 78)
    df, _ = s21.inventario(s21.cargar_descartados())
    X, num, binr, oh = s12.construir_crudos(df, incluir_nuevas=False)
    y = df[s21.TARGET].astype(float)
    regs = s23.registros(df.drop(columns=["_inst"], errors="ignore"))
    ref = sorted(float(v) for v in y)
    oficial = joblib.load(s23.JOBLIB_GLOBAL)
    cuerpo_js = s23.JS_VIGENTE.read_text(encoding="utf-8")

    cv = KFold(n_splits=5, shuffle=True, random_state=s21.RANDOM_STATE)
    tr, te = train_test_split(X.index, test_size=0.20, random_state=s21.RANDOM_STATE,
                              stratify=s12.estratos(y))
    out = {"etiqueta": "EXPERIMENTO LATERAL — no reemplaza al modelo oficial",
           "protocolo": "igual que el script 21", "resultados": {}}
    for n in (300, 150):
        base = s12.construir_pipeline(num, binr, oh)
        base.set_params(rf__n_estimators=n)
        oof = cross_val_predict(clone(base), X, y, cv=cv, n_jobs=-1)
        m_oof = s21.metricas(y, oof)
        m_oof["r2_ic95_bootstrap"] = s21.ic_bootstrap_r2(y, oof)
        ho = clone(base).fit(X.loc[tr], y.loc[tr])
        m_ho = s21.metricas(y.loc[te], ho.predict(X.loc[te]))
        final = clone(base).fit(X, y)

        bundle = dict(oficial, modelo=final)
        SPEC = s23.construir_spec_global(bundle, regs, ref)
        spec_txt = json.dumps(SPEC, ensure_ascii=False, separators=(",", ":"))
        js_txt = re.sub(r"^const SPEC = .*;$",
                        lambda _: "const SPEC = " + json.dumps(SPEC, ensure_ascii=False) + ";",
                        cuerpo_js, count=1, flags=re.MULTILINE)
        arboles = SPEC["puntaje"]["model"]["trees"]
        out["resultados"][str(n)] = {
            "oof": m_oof, "holdout": m_ho,
            "spec_json_bytes": len(spec_txt.encode("utf-8")),
            "js_bytes": len(js_txt.encode("utf-8")),
            "nodos": sum(len(t) for t in arboles),
        }
        r = out["resultados"][str(n)]
        print(f">>> n={n}: OOF R²={m_oof['r2']:.4f} IC95 {[round(v, 3) for v in m_oof['r2_ic95_bootstrap']]} "
              f"MAE={m_oof['mae']:.3f} RMSE={m_oof['rmse']:.3f} | hold-out R²={m_ho['r2']:.4f} "
              f"MAE={m_ho['mae']:.3f} | SPEC {r['spec_json_bytes']:,} B · JS {r['js_bytes']:,} B · "
              f"nodos {r['nodos']:,}", flush=True)

    a, b = out["resultados"]["300"], out["resultados"]["150"]
    out["diferencia_150_menos_300"] = {
        "oof_r2": b["oof"]["r2"] - a["oof"]["r2"],
        "oof_mae": b["oof"]["mae"] - a["oof"]["mae"],
        "oof_rmse": b["oof"]["rmse"] - a["oof"]["rmse"],
        "spec_json_ratio": b["spec_json_bytes"] / a["spec_json_bytes"],
    }
    # El caso con el n_estimators del modelo oficial vigente debe reproducir su
    # métrica embebida (300 hasta el informe 23 §9; 150 desde esa decisión).
    n_ofi = str(oficial["hiperparametros"]["n_estimators"])
    assert abs(out["resultados"][n_ofi]["oof"]["r2"] - oficial["metricas"]["oof"]["r2"]) < 1e-12, \
        f"el caso de {n_ofi} no reproduce la métrica oficial"
    out["n_estimators_oficial_al_ejecutar"] = int(n_ofi)
    print(f">>> 150 − 300: {out['diferencia_150_menos_300']}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f">>> resultados → outputs/{OUT.name}")


if __name__ == "__main__":
    main()
