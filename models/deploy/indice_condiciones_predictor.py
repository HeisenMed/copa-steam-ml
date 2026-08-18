# -*- coding: utf-8 -*-
"""
Modelo TEÓRICO de condiciones — Copa STEM 2026 (knowledge-driven).
GENERADO por notebooks/10_modelo_teorico_vs_empirico.py — no editar a mano.

Calcula un `indice_condiciones` (0–100) a partir SOLO de literatura educativa
(OECD PISA, UNESCO, meta-análisis SES). NO usa datos de Copa STEM, ni municipio,
grado, género, tipo de institución o colegio. Mide CONDICIONES, no habilidad.

    from indice_condiciones_predictor import indice_condiciones, nivel_condiciones
    ic = indice_condiciones({"estrato": 1, "computador_en_casa": "No",
                              "nivel_programacion": "Intermedia", ...})
    nivel = nivel_condiciones(ic)   # "Favorables" / "Promedio" / "Adversas"

No requiere librerías externas: solo la librería estándar (json).
"""
import json

COND_FAVORABLE = 60
COND_ADVERSA = 45


def _txt(v):
    """Normaliza a texto en minúsculas sin espacios; None/NaN → ''."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    return str(v).strip().lower()


def _num(v):
    """Convierte a float; None si no es número real."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _si_no(v, w_si, w_no):
    """Sí* → w_si, No* → w_no, desconocido → 0 (neutral)."""
    s = _txt(v)
    if s.startswith("s"):
        return w_si
    if s.startswith("n"):
        return w_no
    return 0


def _estrato_ajuste(v):
    """Estrato 1–3 (máx real en Copacabana/Girardota/Bello). 1=-3, 2=0, 3=+2;
    fuera de rango o NaN → 0."""
    f = _num(v)
    if f is None:
        return 0
    e = int(round(f))
    return {1: -3, 2: 0, 3: 2}.get(e, 0)


def _nivel_ajuste(v, pesos):
    """Mapea Ninguna/Básica/Intermedia/Avanzada según `pesos`; desconocido → 0."""
    s = _txt(v)
    m = {"ninguna": pesos[0], "ninguno": pesos[0], "básica": pesos[1],
         "basica": pesos[1], "intermedia": pesos[2], "avanzada": pesos[3]}
    return m.get(s, 0)


def _interes_ajuste(v):
    """Interés 1–5: bajo(1-2)=-2, medio(3)=0, alto(4-5)=+3; desconocido → 0."""
    f = _num(v)
    if f is None:
        return 0
    if f <= 2:
        return -2
    if f >= 4:
        return 3
    return 0


def _n_herramientas(v):
    """Cuenta herramientas de una lista JSON/CSV, ignorando 'Ninguna/Ninguno'."""
    s = _txt(v)
    if s in ("", "nan", "none", "[]"):
        return 0
    items = None
    try:
        parsed = json.loads(str(v))
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


def _herramientas_ajuste(v):
    """0 herramientas = -2, 1-2 = 0, 3+ = +3."""
    n = _n_herramientas(v)
    if n == 0:
        return -2
    if n >= 3:
        return 3
    return 0


def indice_condiciones(estudiante):
    """Índice de CONDICIONES (0–100) basado SOLO en literatura educativa.

    NO usa municipio, grado, género, tipo de institución ni el colegio (factores
    contaminados en los datos actuales). Mide el contexto socioeconómico y la
    preparación previa, NO la habilidad ni la nota.

        indice = 50 + Σ ajustes,   recortado a [5, 95].
    """
    base = 50.0
    ajuste = 0.0
    ajuste += _si_no(estudiante.get("computador_en_casa"), 3, -3)   # OECD PISA
    ajuste += _si_no(estudiante.get("internet_en_casa"), 2, -2)     # UNESCO
    ajuste += _estrato_ajuste(estudiante.get("estrato"))            # recursos hogar
    # Estabilidad familiar (meta-análisis: efecto pequeño).
    conv = _txt(estudiante.get("con_quien_vive"))
    if conv:
        ajuste += 1 if conv == "ambos padres" else -1
    ajuste += _nivel_ajuste(estudiante.get("nivel_programacion"), (-3, 0, 4, 8))
    ajuste += _nivel_ajuste(estudiante.get("nivel_robotica"), (-1, 0, 2, 4))
    ajuste += _si_no(estudiante.get("participacion_olimpiadas"), 5, 0)
    ajuste += _interes_ajuste(estudiante.get("interes_prog_robotica"))
    ajuste += _herramientas_ajuste(estudiante.get("herramientas_conocidas"))
    val = base + ajuste
    if val < 5.0:
        val = 5.0
    if val > 95.0:
        val = 95.0
    return val


def nivel_condiciones(indice):
    """Etiqueta cualitativa del índice de condiciones."""
    if indice > COND_FAVORABLE:
        return "Favorables"
    if indice >= COND_ADVERSA:
        return "Promedio"
    return "Adversas"



if __name__ == "__main__":
    ejemplo = {"estrato": 1, "computador_en_casa": "No", "internet_en_casa": "Sí",
               "con_quien_vive": "Solo madre", "nivel_programacion": "Ninguna",
               "nivel_robotica": "Ninguna", "participacion_olimpiadas": "No",
               "interes_prog_robotica": 3, "herramientas_conocidas": "[]"}
    ic = indice_condiciones(ejemplo)
    print("indice_condiciones:", round(ic, 2), "->", nivel_condiciones(ic))
