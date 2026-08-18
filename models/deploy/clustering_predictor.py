# -*- coding: utf-8 -*-
"""
Predictor PURO de perfil (cluster) — Copa STEM 2026 (K-Means, K=4).
GENERADO por notebooks/07_clustering_perfiles.py — no editar a mano.

    from clustering_predictor import predecir_perfil
    r = predecir_perfil({"puntaje_obtenido": 70, "estrato": 3,
                          "computador_en_casa": "Sí, propio", ...})
    # r = {'cluster_id': .., 'cluster_nombre': '..'}

Asigna al estudiante el centroide más cercano en el espacio estandarizado.
No requiere sklearn ni numpy: solo la librería estándar (json).
"""
import json

SPEC = json.loads(r"""{"meta": {"generado": "2026-07-06T08:02:57", "k": 4, "modelo": "KMeans"}, "features": ["puntaje_obtenido", "estrato", "computador_bin", "internet_bin", "nivel_programacion_ord", "nivel_robotica_ord", "interes_prog_robotica", "n_herramientas", "n_areas_interes"], "medians": [35.0, 2.0, 1.0, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0], "scaler_mean": [41.80571428571429, 2.3834285714285715, 0.7748571428571429, 0.9702857142857143, 0.528, 0.236, 2.866857142857143, 1.8422857142857143, 2.418857142857143], "scaler_scale": [23.105892171937974, 0.5854031229217441, 0.41767637115404094, 0.16979795917140908, 0.6599255369249732, 0.48611109841269823, 1.0902103897310607, 2.142424642067668, 1.3005444385851235], "centroids": [[0.44156452812916513, 0.26594234505338965, 0.4580489390301118, 0.17499789667253576, 1.0259854451558317, 0.7453207080833141, 0.7194076072179135, 1.019965130427435, 0.6858090132836921], [-0.16963557277308375, -0.03485999666028029, 0.5390366146899487, 0.17499789667253596, -0.4806254393568879, -0.36488659352271163, -0.2894493033236519, -0.3999869315194796, -0.35673749987177245], [-0.17098169897605442, -0.18496765424492503, -1.855161547003988, 0.17499789667253576, -0.1679587021330741, -0.10813998591871055, -0.22523667910538722, -0.35724133411011416, -0.03087854348811138], [-0.011566432720452463, -0.6221316353423137, -1.3026542789207725, -5.714354395191654, -0.0424290293878767, 0.14748132253389218, -0.1424667043917976, -0.13283706740730153, 0.04760474619340384]], "nombres": {"1": "Base conectada", "2": "En desarrollo", "3": "Promedio con acceso limitado", "0": "Alto rendimiento tech"}}""")


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
        return 1
    if s.startswith("n"):
        return 0
    return None


def _raw_feature(raw, name):
    """Valor crudo de una feature de clustering (None si falta)."""
    if name == "computador_bin":
        return _bin_si(raw.get("computador_en_casa"))
    if name == "internet_bin":
        return _bin_si(raw.get("internet_en_casa"))
    if name == "nivel_programacion_ord":
        return _ord_level(raw.get("nivel_programacion"))
    if name == "nivel_robotica_ord":
        return _ord_level(raw.get("nivel_robotica"))
    if name == "n_herramientas":
        v = _to_float(raw.get("n_herramientas"))
        return v if v is not None else _parse_count(raw.get("herramientas_conocidas"))
    if name == "n_areas_interes":
        v = _to_float(raw.get("n_areas_interes"))
        return v if v is not None else _parse_count(raw.get("areas_interes"))
    return _to_float(raw.get(name))


def _feature_vector(raw, SPEC):
    """Vector de features imputado (mediana) en el ORDEN de SPEC['features']."""
    vals = []
    for i, name in enumerate(SPEC["features"]):
        v = _raw_feature(raw, name)
        if v is None:
            v = SPEC["medians"][i]
        vals.append(float(v))
    return vals


def predecir_cluster(raw, SPEC):
    """Asigna al estudiante el cluster del CENTROIDE más cercano (espacio estandarizado)."""
    x = _feature_vector(raw, SPEC)
    mean, scale = SPEC["scaler_mean"], SPEC["scaler_scale"]
    z = [(x[i] - mean[i]) / scale[i] for i in range(len(x))]
    best_id, best_d = -1, None
    for cid, c in enumerate(SPEC["centroids"]):
        d = 0.0
        for i in range(len(z)):
            diff = z[i] - c[i]
            d += diff * diff
        if best_d is None or d < best_d:
            best_d, best_id = d, cid
    return {"cluster_id": best_id, "cluster_nombre": SPEC["nombres"][str(best_id)]}



def predecir_perfil(estudiante):
    """Devuelve el cluster_id y el nombre del perfil (centroide más cercano)."""
    return predecir_cluster(estudiante, SPEC)


if __name__ == "__main__":
    ejemplo = {"puntaje_obtenido": 40, "estrato": 3, "computador_en_casa": "Sí, propio", "internet_en_casa": "Sí, estable", "nivel_programacion": "Ninguna", "nivel_robotica": "Ninguna", "interes_prog_robotica": 1.0, "herramientas_conocidas": "[\"Python\",\"JavaScript\",\"Roblox Studio\",\"Minecraft Education\",\"HTML\",\"CSS\",\"Otro\"]", "areas_interes": "[\"Artes\",\"Cultura\",\"Otro\",\"Ciencias Naturales\"]"}
    print(predecir_perfil(ejemplo))
