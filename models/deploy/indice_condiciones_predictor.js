// Modelo TEÓRICO de condiciones — Copa STEM 2026 (knowledge-driven).
// GENERADO por notebooks/10_modelo_teorico_vs_empirico.py — no editar a mano.
//
// Calcula un indice_condiciones (0–100) a partir SOLO de literatura educativa.
// No usa datos de Copa STEM ni municipio/grado/género/institución/colegio.

const COND_FAVORABLE = 60;
const COND_ADVERSA = 45;

function _txt(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number" && Number.isNaN(v)) return "";
  return String(v).trim().toLowerCase();
}

function _num(v) {
  if (v === null || v === undefined || typeof v === "boolean") return null;
  const f = Number(v);
  return Number.isNaN(f) ? null : f;
}

function _siNo(v, wSi, wNo) {
  const s = _txt(v);
  if (s.startsWith("s")) return wSi;
  if (s.startsWith("n")) return wNo;
  return 0;
}

function _estratoAjuste(v) {
  // Estrato 1-3 (máx real en Copacabana/Girardota/Bello); fuera de rango o NaN -> 0
  const f = _num(v);
  if (f === null) return 0;
  const e = Math.round(f);
  const m = {1: -3, 2: 0, 3: 2};
  return (e in m) ? m[e] : 0;
}

function _nivelAjuste(v, pesos) {
  const s = _txt(v);
  const m = {
    "ninguna": pesos[0], "ninguno": pesos[0], "básica": pesos[1],
    "basica": pesos[1], "intermedia": pesos[2], "avanzada": pesos[3]
  };
  return (s in m) ? m[s] : 0;
}

function _interesAjuste(v) {
  const f = _num(v);
  if (f === null) return 0;
  if (f <= 2) return -2;
  if (f >= 4) return 3;
  return 0;
}

function _nHerramientas(v) {
  const s = _txt(v);
  if (s === "" || s === "nan" || s === "none" || s === "[]") return 0;
  let items = null;
  try {
    const parsed = JSON.parse(String(v));
    if (Array.isArray(parsed)) items = parsed;
  } catch (e) { items = null; }
  if (items === null) {
    items = s.replace(/[\[\]"]/g, "").split(",").filter(x => x.trim());
  }
  let cnt = 0;
  for (const it of items) {
    const t = String(it).trim().toLowerCase();
    if (t && !["ninguna", "ninguno", "ninguna.", "ninguno."].includes(t)) cnt++;
  }
  return cnt;
}

function _herramientasAjuste(v) {
  const n = _nHerramientas(v);
  if (n === 0) return -2;
  if (n >= 3) return 3;
  return 0;
}

export function indiceCondiciones(est) {
  let ajuste = 0.0;
  ajuste += _siNo(est["computador_en_casa"], 3, -3);   // OECD PISA
  ajuste += _siNo(est["internet_en_casa"], 2, -2);     // UNESCO
  ajuste += _estratoAjuste(est["estrato"]);            // recursos hogar
  const conv = _txt(est["con_quien_vive"]);
  if (conv) ajuste += (conv === "ambos padres") ? 1 : -1;
  ajuste += _nivelAjuste(est["nivel_programacion"], [-3, 0, 4, 8]);
  ajuste += _nivelAjuste(est["nivel_robotica"], [-1, 0, 2, 4]);
  ajuste += _siNo(est["participacion_olimpiadas"], 5, 0);
  ajuste += _interesAjuste(est["interes_prog_robotica"]);
  ajuste += _herramientasAjuste(est["herramientas_conocidas"]);
  let val = 50.0 + ajuste;
  if (val < 5.0) val = 5.0;
  if (val > 95.0) val = 95.0;
  return val;
}

export function nivelCondiciones(indice) {
  if (indice > COND_FAVORABLE) return "Favorables";
  if (indice >= COND_ADVERSA) return "Promedio";
  return "Adversas";
}

// Ejemplo:
//   import { indiceCondiciones, nivelCondiciones } from "./indice_condiciones_predictor.js";
//   const ic = indiceCondiciones({ estrato: 1, computador_en_casa: "No", ... });
//   nivelCondiciones(ic); // "Adversas"
