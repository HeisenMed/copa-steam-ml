/**
 * Predictor PURO de perfil (cluster) — Copa STEM 2026 (K-Means).
 * GENERADO por notebooks/07_clustering_perfiles.py — no editar a mano.
 * Réplica en JavaScript ES6. Sin dependencias.
 *
 *   import { predecirPerfil } from "./clustering_predictor.js";
 *   const r = predecirPerfil({ puntaje_obtenido: 70, estrato: 3,
 *                              computador_en_casa: "Sí, propio" });
 */
const SPEC = {"meta": {"generado": "2026-07-06T08:02:57", "k": 4, "modelo": "KMeans"}, "features": ["puntaje_obtenido", "estrato", "computador_bin", "internet_bin", "nivel_programacion_ord", "nivel_robotica_ord", "interes_prog_robotica", "n_herramientas", "n_areas_interes"], "medians": [35.0, 2.0, 1.0, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0], "scaler_mean": [41.80571428571429, 2.3834285714285715, 0.7748571428571429, 0.9702857142857143, 0.528, 0.236, 2.866857142857143, 1.8422857142857143, 2.418857142857143], "scaler_scale": [23.105892171937974, 0.5854031229217441, 0.41767637115404094, 0.16979795917140908, 0.6599255369249732, 0.48611109841269823, 1.0902103897310607, 2.142424642067668, 1.3005444385851235], "centroids": [[0.44156452812916513, 0.26594234505338965, 0.4580489390301118, 0.17499789667253576, 1.0259854451558317, 0.7453207080833141, 0.7194076072179135, 1.019965130427435, 0.6858090132836921], [-0.16963557277308375, -0.03485999666028029, 0.5390366146899487, 0.17499789667253596, -0.4806254393568879, -0.36488659352271163, -0.2894493033236519, -0.3999869315194796, -0.35673749987177245], [-0.17098169897605442, -0.18496765424492503, -1.855161547003988, 0.17499789667253576, -0.1679587021330741, -0.10813998591871055, -0.22523667910538722, -0.35724133411011416, -0.03087854348811138], [-0.011566432720452463, -0.6221316353423137, -1.3026542789207725, -5.714354395191654, -0.0424290293878767, 0.14748132253389218, -0.1424667043917976, -0.13283706740730153, 0.04760474619340384]], "nombres": {"1": "Base conectada", "2": "En desarrollo", "3": "Promedio con acceso limitado", "0": "Alto rendimiento tech"}};

function _toFloat(v) {
  if (v === null || v === undefined || typeof v === "boolean") return null;
  const f = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isNaN(f) ? null : f;
}

function _parseCount(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  if (s === "" || ["nan", "none", "[]"].includes(s.toLowerCase())) return 0;
  let items = null;
  try { const p = JSON.parse(s); if (Array.isArray(p)) items = p; } catch (e) { items = null; }
  if (items === null) items = s.replace(/[[\]"]/g, "").split(",").filter((x) => x.trim());
  let cnt = 0;
  for (const it of items) {
    const t = String(it).trim().toLowerCase();
    if (t && !["ninguna", "ninguno", "ninguna.", "ninguno."].includes(t)) cnt += 1;
  }
  return cnt;
}

function _ordLevel(v) {
  if (v === null || v === undefined) return null;
  const m = { ninguna: 0, ninguno: 0, "básica": 1, basica: 1, intermedia: 2, avanzada: 3 };
  const s = String(v).trim().toLowerCase();
  return s in m ? m[s] : null;
}

function _binSi(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim().toLowerCase();
  if (["nan", "none", ""].includes(s)) return null;
  if (s.startsWith("s")) return 1;
  if (s.startsWith("n")) return 0;
  return null;
}

function _rawFeature(raw, name) {
  if (name === "computador_bin") return _binSi(raw["computador_en_casa"]);
  if (name === "internet_bin") return _binSi(raw["internet_en_casa"]);
  if (name === "nivel_programacion_ord") return _ordLevel(raw["nivel_programacion"]);
  if (name === "nivel_robotica_ord") return _ordLevel(raw["nivel_robotica"]);
  if (name === "n_herramientas") {
    const v = _toFloat(raw["n_herramientas"]);
    return v !== null ? v : _parseCount(raw["herramientas_conocidas"]);
  }
  if (name === "n_areas_interes") {
    const v = _toFloat(raw["n_areas_interes"]);
    return v !== null ? v : _parseCount(raw["areas_interes"]);
  }
  return _toFloat(raw[name]);
}

function _featureVector(raw, SPEC) {
  const vals = [];
  for (let i = 0; i < SPEC.features.length; i++) {
    let v = _rawFeature(raw, SPEC.features[i]);
    if (v === null) v = SPEC.medians[i];
    vals.push(v);
  }
  return vals;
}

export function predecirPerfil(raw) {
  const x = _featureVector(raw, SPEC);
  const mean = SPEC.scaler_mean, scale = SPEC.scaler_scale;
  const z = x.map((xi, i) => (xi - mean[i]) / scale[i]);
  let bestId = -1, bestD = Infinity;
  for (let cid = 0; cid < SPEC.centroids.length; cid++) {
    const c = SPEC.centroids[cid];
    let d = 0.0;
    for (let i = 0; i < z.length; i++) { const diff = z[i] - c[i]; d += diff * diff; }
    if (d < bestD) { bestD = d; bestId = cid; }
  }
  return { cluster_id: bestId, cluster_nombre: SPEC.nombres[String(bestId)] };
}

if (typeof process !== "undefined" && Array.isArray(process.argv) && process.argv[1]) {
  const _here = decodeURIComponent(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1");
  const _norm = (p) => p.replace(/\\/g, "/").toLowerCase();
  if (_norm(_here) === _norm(process.argv[1])) console.log(predecirPerfil({"puntaje_obtenido": 40, "estrato": 3, "computador_en_casa": "Sí, propio", "internet_en_casa": "Sí, estable", "nivel_programacion": "Ninguna", "nivel_robotica": "Ninguna", "interes_prog_robotica": 1.0, "herramientas_conocidas": "[\"Python\",\"JavaScript\",\"Roblox Studio\",\"Minecraft Education\",\"HTML\",\"CSS\",\"Otro\"]", "areas_interes": "[\"Artes\",\"Cultura\",\"Otro\",\"Ciencias Naturales\"]"}));
}
