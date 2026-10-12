# -*- coding: utf-8 -*-
"""
================================================================================
 COPA STEM 2026 — Fundación SapienceLab
 Script 00: Generador de datos de ejemplo SINTÉTICOS
================================================================================

Objetivo
--------
Permitir que alguien sin acceso al dataset real (que contiene datos personales
de menores y no se publica) pueda ejecutar y verificar el código del repositorio.

Genera dos archivos en datos_ejemplo/, 300 filas cada uno, con semilla fija:

    A) copa_stem_dataset_SINTETICO.csv
       Mismo esquema que data/copa_stem_dataset.csv (27 columnas), el archivo
       que leen los scripts 01 a 07. Ese export no tiene perfil académico.
    B) copa_stem_dataset_2026-08_SINTETICO.csv
       Mismo esquema que data/copa_stem_dataset_2026-08.csv.csv (31 columnas),
       el export que leen los scripts 11, 21 y 24. Incluye las 5 columnas de
       perfil académico, nulas en la mayoría de las filas como en el real.

Principios de diseño
--------------------
- NO lee nada de data/. El esquema (nombres y orden de columnas, valores
  categóricos, rangos y patrón de nulos) está escrito a mano en este script a
  partir de estadísticas agregadas del dataset real.
- Ningún valor individual del dataset real se copia. Las columnas que
  identifican a una persona se inventan de forma evidente:
    · nombres y apellidos de una lista corta inventada ("Alfa Ejemplo"...)
    · numero_documento de 10 dígitos que empieza por 99; ningún documento del
      dataset real empieza por 99, así que no pueden coincidir
- Los nombres de colegios y municipios son información pública.
- Los valores se generan con distribuciones simples. Las relaciones entre
  variables NO reproducen las reales: cualquier resultado obtenido con estos
  datos carece de significado.

Autor: Equipo de Datos — Fundación SapienceLab
================================================================================
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    print("ERROR: falta una dependencia del entorno virtual.")
    print(f"       Detalle: {exc}")
    print("       pip install -r requirements.txt")
    sys.exit(1)

RANDOM_STATE = 42
N_FILAS = 300

BASE_DIR = Path(__file__).resolve().parent.parent
SALIDA_DIR = BASE_DIR / "datos_ejemplo"
SALIDA_CANONICO = SALIDA_DIR / "copa_stem_dataset_SINTETICO.csv"
SALIDA_2026_08 = SALIDA_DIR / "copa_stem_dataset_2026-08_SINTETICO.csv"


# =============================================================================
# 1. ESQUEMA (escrito a mano; no se lee nada de data/)
# =============================================================================

COLUMNAS_CANONICO = [
    "numero_documento", "nombres", "apellidos", "institucion_educativa",
    "grado_escolar", "puntaje_obtenido", "porcentaje", "tiempo_usado_segundos",
    "cambios_pestana", "intentos_copiar", "intentos_pegar",
    "intentos_click_derecho", "edad_calculada", "genero", "municipio",
    "tipo_institucion", "estrato", "jornada", "con_quien_vive",
    "computador_en_casa", "internet_en_casa", "participacion_olimpiadas",
    "nivel_programacion", "nivel_robotica", "herramientas_conocidas",
    "areas_interes", "interes_prog_robotica",
]

COLUMNAS_PERFIL = [
    "promedio_academico", "horas_estudio_matematicas", "motivacion_participar",
    "clases_extra_matematicas", "gusto_logica",
]

COLUMNAS_2026_08 = (
    [c for c in COLUMNAS_CANONICO if c not in ("nombres", "apellidos")]
    + COLUMNAS_PERFIL + ["created_at"]
)

# Columnas del bloque de inscripción: en el export canónico, ~7 % de las filas
# no tiene inscripción enlazada y todas ellas quedan nulas a la vez.
COLUMNAS_BLOQUE_INSCRIPCION = [
    "edad_calculada", "estrato", "jornada", "con_quien_vive",
    "computador_en_casa", "internet_en_casa", "participacion_olimpiadas",
    "nivel_programacion", "nivel_robotica", "herramientas_conocidas",
    "areas_interes", "interes_prog_robotica",
]

# Colegios por municipio (información pública): (nombre, tipo_institucion).
COLEGIOS = {
    "Copacabana": [
        ("Colegio Cooperativo Juan del Corral", "Privada"),
        ("Colegio La Asunción", "Privada"),
        ("Colegio San Rafael", "Privada"),
        ("I. E. Presbítero Bernardo Montoya Giraldo", "Pública"),
        ("I.E. Gabriela Mistral", "Pública"),
        ("I.E. José Miguel de Restrepo y Puerta", "Pública"),
        ("I.E. San Luis Gonzaga", "Pública"),
    ],
    "Girardota": [
        ("Colegio Neosistemas", "Privada"),
        ("Colegio Nuestra Señora del Rosario", "Privada"),
        ("I.E. Atanasio Girardot", "Pública"),
        ("I.E. Emiliano García", "Pública"),
        ("Instituto Parroquial Nuestra Señora de la Presentación", "Privada"),
    ],
}
# Colegios que solo aparecen en el export 2026-08.
COLEGIOS_SOLO_2026_08 = {
    "Girardota": [
        ("Colegio Forjadores del Mañana", "Privada"),
        ("Colegio Juan Bernardone", "Privada"),
        ("I.E. Manuel José Sierra", "Pública"),
        ("I.E. Rural Colombia", "Pública"),
    ],
    "Bello": [
        ("Colegio Bethlemitas", "Privada"),
        ("Colegio El Rosario", "Privada"),
        ("Colegio La Salle", "Privada"),
        ("Colegio Mano Amiga", "Privada"),
        ("I.E. Abraham Reyes", "Pública"),
        ("I.E. Atanasio Girardot (Bello)", "Pública"),
        ("I.E. Betsabé Espinal", "Pública"),
        ("I.E. Carlos Pérez Mejía", "Pública"),
        ("I.E. Cincuentenario de Fabricato", "Pública"),
        ("I.E. Comercial Antonio Roldán Betancur", "Pública"),
        ("I.E. Federico Sierra Arango", "Pública"),
        ("I.E. Fernando Vélez", "Pública"),
        ("I.E. Gilberto Echeverri Mejía", "Pública"),
        ("I.E. La Milagrosa", "Pública"),
        ("I.E. Nueva Generación", "Pública"),
        ("I.E. Playa Rica", "Pública"),
        ("I.E. Raquel Jaramillo", "Pública"),
    ],
}

# Valores categóricos (exactos) y probabilidades aproximadas.
CATEGORIAS = {
    "genero": (["Masculino", "Femenino", "Prefiero no decirlo", "No binario"],
               [0.49, 0.49, 0.015, 0.005]),
    "jornada": (["Mañana", "Única", "Tarde"], [0.72, 0.278, 0.002]),
    "con_quien_vive": (["Ambos padres", "Solo madre", "Otros familiares",
                        "Solo padre", "Otro", "Acudiente legal"],
                       [0.50, 0.32, 0.10, 0.04, 0.025, 0.015]),
    "computador_en_casa": (["Sí, propio", "No", "Sí, compartido"],
                           [0.52, 0.25, 0.23]),
    "internet_en_casa": (["Sí, estable", "Sí, intermitente", "No"],
                         [0.86, 0.10, 0.04]),
    "participacion_olimpiadas": (["No", "Sí"], [0.85, 0.15]),
    "nivel_programacion": (["Ninguna", "Básica", "Intermedia", "Avanzada"],
                           [0.51, 0.39, 0.095, 0.005]),
    "nivel_robotica": (["Ninguna", "Básica", "Intermedia", "Avanzada"],
                       [0.75, 0.21, 0.037, 0.003]),
    "clases_extra_matematicas": (["No", "Sí"], [0.70, 0.30]),
}

# Listas JSON: vocabulario de opciones del formulario.
HERRAMIENTAS = ["Roblox Studio", "Python", "HTML", "JavaScript",
                "Minecraft Education", "Scratch", "CSS", "Arduino", "micro:bit",
                "C++", "LEGO Mindstorms / Spike", "Otro"]
AREAS = ["Artes", "Inteligencia Artificial", "Programación", "Matemáticas",
         "Cultura", "Ciencias Naturales", "Ingeniería", "Robótica", "Otro"]

# Identificadores inventados, evidentemente falsos.
NOMBRES_FALSOS = ["Alfa", "Beta", "Gamma", "Delta", "Épsilon", "Zeta", "Theta",
                  "Kappa", "Lambda", "Sigma"]
APELLIDOS_FALSOS = ["Ejemplo", "Ficticio", "Sintético", "Inventado", "Prueba",
                    "Demostración"]
DOC_BASE = 9_900_000_000  # 10 dígitos que empiezan por 99


# =============================================================================
# 2. GENERACIÓN
# =============================================================================

def _elegir(rng, columna, n):
    valores, p = CATEGORIAS[columna]
    p = np.array(p) / np.sum(p)
    return rng.choice(valores, size=n, p=p)


def _lista_json(rng, vocabulario, ninguna_p, longitudes, p_longitudes):
    """Lista JSON compacta, como en el export (separador ',' y sin escapar
    acentos). 'Ninguna' siempre va sola."""
    if ninguna_p and rng.random() < ninguna_p:
        return json.dumps(["Ninguna"], ensure_ascii=False, separators=(",", ":"))
    k = int(rng.choice(longitudes, p=np.array(p_longitudes) / sum(p_longitudes)))
    items = list(rng.choice(vocabulario, size=min(k, len(vocabulario)),
                            replace=False))
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _generar_base(rng, n, colegios, p_municipio, p_tiempo_nulo):
    """Columnas comunes a los dos esquemas, sin nulos de bloque."""
    municipios = list(p_municipio)
    muni = rng.choice(municipios, size=n,
                      p=np.array(list(p_municipio.values())))
    inst, tipo = [], []
    for m in muni:
        nombre, t = colegios[m][rng.integers(len(colegios[m]))]
        inst.append(nombre)
        tipo.append(t)
    tipo = np.array(tipo)

    grado = rng.choice([9, 10, 11], size=n, p=[0.35, 0.32, 0.33])
    edad = grado + 5 + rng.choice([0, 1], size=n, p=[0.55, 0.45])
    # Anomalías de captura presentes en el real (edad 0, -1 o 27).
    anom = rng.random(n) < 0.015
    edad = np.where(anom, rng.choice([-1, 0, 27], size=n), edad)

    estrato = np.where(tipo == "Privada",
                       rng.choice([2, 3], size=n, p=[0.35, 0.65]),
                       rng.choice([1, 2, 3], size=n, p=[0.08, 0.55, 0.37]))

    df = pd.DataFrame({
        "institucion_educativa": inst,
        "grado_escolar": grado,
        "edad_calculada": edad,
        "genero": _elegir(rng, "genero", n),
        "municipio": muni,
        "tipo_institucion": tipo,
        "estrato": estrato,
        "jornada": _elegir(rng, "jornada", n),
        "con_quien_vive": _elegir(rng, "con_quien_vive", n),
        "computador_en_casa": _elegir(rng, "computador_en_casa", n),
        "internet_en_casa": _elegir(rng, "internet_en_casa", n),
        "participacion_olimpiadas": _elegir(rng, "participacion_olimpiadas", n),
        "nivel_programacion": _elegir(rng, "nivel_programacion", n),
        "nivel_robotica": _elegir(rng, "nivel_robotica", n),
        "herramientas_conocidas": [
            _lista_json(rng, HERRAMIENTAS, 0.33, range(1, 11),
                        [45, 15, 12, 11, 7, 4, 2, 1, 0.5, 0.5])
            for _ in range(n)],
        "areas_interes": [
            _lista_json(rng, AREAS, 0, range(1, 10),
                        [3, 58, 21, 10, 4, 2, 0.5, 0.5, 0.5])
            for _ in range(n)],
        "interes_prog_robotica": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n,
                                            p=[0.12, 0.25, 0.32, 0.20, 0.11]),
    })

    # Puntaje: múltiplo de 5 entre 0 y 100 (20 preguntas). Relación con las
    # condiciones inventada: solo evita que el modelo vea ruido puro.
    lat = (40 + 4 * (df["estrato"] - 2)
           + 5 * (df["computador_en_casa"] != "No")
           + 4 * (df["nivel_programacion"] != "Ninguna")
           + rng.normal(0, 20, n))
    puntaje = (np.clip(np.round(lat / 5) * 5, 0, 100)).astype(int)
    df["puntaje_obtenido"] = puntaje
    df["porcentaje"] = puntaje

    # Telemetría: solo existe para exámenes en plataforma.
    p_nulo = df["municipio"].map(p_tiempo_nulo).to_numpy()
    en_plataforma = rng.random(n) >= p_nulo
    tiempo = np.clip(np.round(rng.normal(2200, 1100, n)), 51, 5400)
    df["tiempo_usado_segundos"] = np.where(en_plataforma, tiempo, np.nan)
    raro = lambda p: en_plataforma & (rng.random(n) < p)
    df["cambios_pestana"] = np.where(raro(0.06), rng.integers(1, 6, n), 0)
    df["intentos_copiar"] = 0
    df["intentos_pegar"] = np.where(raro(0.004), 1, 0)
    df["intentos_click_derecho"] = np.where(raro(0.05), rng.integers(1, 12, n), 0)
    return df


def generar_canonico(rng, n):
    """Esquema de data/copa_stem_dataset.csv (scripts 01 a 07)."""
    df = _generar_base(rng, n, COLEGIOS,
                       {"Copacabana": 0.74, "Girardota": 0.26},
                       {"Copacabana": 0.22, "Girardota": 0.12})
    df["numero_documento"] = [str(DOC_BASE + i) for i in rng.choice(
        10_000_000, size=n, replace=False)]
    df["nombres"] = [f"{a} {b}" if rng.random() < 0.6 else a for a, b in zip(
        rng.choice(NOMBRES_FALSOS, n), rng.choice(NOMBRES_FALSOS, n))]
    df["apellidos"] = [f"{a} {b}" for a, b in zip(
        rng.choice(APELLIDOS_FALSOS, n), rng.choice(APELLIDOS_FALSOS, n))]

    # Patrón de nulos del export canónico:
    #  - ~7 % sin inscripción enlazada: bloque entero nulo y sin telemetría;
    #    dentro de ese bloque, la mitad sin grado y ~1/5 sin género
    #  - ~2 % adicional solo sin grado escolar
    #  - ~0,1 % sin apellidos
    sin_insc = rng.random(n) < 0.072
    df.loc[sin_insc, COLUMNAS_BLOQUE_INSCRIPCION] = np.nan
    df.loc[sin_insc, "tiempo_usado_segundos"] = np.nan
    df.loc[sin_insc, ["cambios_pestana", "intentos_pegar",
                      "intentos_click_derecho"]] = 0
    df.loc[sin_insc & (rng.random(n) < 0.52), "grado_escolar"] = np.nan
    df.loc[sin_insc & (rng.random(n) < 0.19), "genero"] = np.nan
    df.loc[~sin_insc & (rng.random(n) < 0.02), "grado_escolar"] = np.nan
    df.loc[rng.random(n) < 0.003, "apellidos"] = np.nan
    return df[COLUMNAS_CANONICO]


def generar_2026_08(rng, n):
    """Esquema de data/copa_stem_dataset_2026-08.csv.csv (scripts 11, 21, 24)."""
    colegios = {m: list(COLEGIOS.get(m, [])) + list(COLEGIOS_SOLO_2026_08.get(m, []))
                for m in ("Copacabana", "Girardota", "Bello")}
    df = _generar_base(rng, n, colegios,
                       {"Copacabana": 0.40, "Girardota": 0.36, "Bello": 0.24},
                       {"Copacabana": 0.23, "Girardota": 0.21, "Bello": 0.0})
    df["numero_documento"] = [str(DOC_BASE + i) for i in rng.choice(
        10_000_000, size=n, replace=False)]

    # Perfil académico: se agregó al formulario después. Nulo en todo
    # Copacabana, en ~62 % de Girardota y en ~2 % de Bello. Las 5 columnas son
    # nulas a la vez; entre quienes sí lo tienen hay huecos sueltos.
    p_tiene = df["municipio"].map({"Copacabana": 0.0, "Girardota": 0.38,
                                   "Bello": 0.98}).to_numpy()
    tiene = rng.random(n) < p_tiene
    df["promedio_academico"] = np.clip(np.round(rng.normal(3.8, 0.45, n), 1),
                                       1.0, 5.0)
    df["horas_estudio_matematicas"] = rng.choice(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], size=n,
        p=[0.30, 0.30, 0.20, 0.09, 0.06, 0.02, 0.02, 0.01])
    df["motivacion_participar"] = rng.choice(
        [1.0, 2.0, 3.0, 4.0, 5.0], size=n, p=[0.12, 0.25, 0.33, 0.20, 0.10])
    df["clases_extra_matematicas"] = _elegir(rng, "clases_extra_matematicas", n)
    df["gusto_logica"] = rng.choice(
        [1.0, 2.0, 3.0, 4.0, 5.0], size=n, p=[0.07, 0.22, 0.40, 0.21, 0.10])
    df.loc[~tiene, COLUMNAS_PERFIL] = np.nan
    df.loc[tiene & (rng.random(n) < 0.05), "clases_extra_matematicas"] = np.nan
    df.loc[tiene & (rng.random(n) < 0.015), "promedio_academico"] = np.nan
    df.loc[tiene & (rng.random(n) < 0.015), "horas_estudio_matematicas"] = np.nan

    # created_at: marca de inscripción entre el 22-may y el 30-jul de 2026,
    # con el formato del export ("2026-05-22 14:03:11.123456+00").
    inicio = datetime(2026, 5, 22)
    seg = rng.choice(int(timedelta(days=69).total_seconds()), size=n,
                     replace=False)
    micro = rng.integers(0, 1_000_000, n)
    df["created_at"] = [
        (inicio + timedelta(seconds=int(s), microseconds=int(u)))
        .strftime("%Y-%m-%d %H:%M:%S.%f") + "+00"
        for s, u in zip(seg, micro)]
    return df[COLUMNAS_2026_08]


# =============================================================================
# 3. EJECUCIÓN
# =============================================================================

def _aviso():
    print("=" * 78)
    print("  AVISO: DATOS INVENTADOS")
    print("  Estos archivos son SINTÉTICOS. No describen a ningún estudiante real.")
    print("  Sirven solo para comprobar que el código corre. Cualquier resultado")
    print("  obtenido con ellos (métricas, gráficos, informes) CARECE DE SIGNIFICADO.")
    print("=" * 78)


def main():
    _aviso()
    rng = np.random.default_rng(RANDOM_STATE)
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    canonico = generar_canonico(rng, N_FILAS)
    export = generar_2026_08(rng, N_FILAS)
    # Mismo formato de archivo que los exports reales: UTF-8 sin BOM, CRLF.
    for df, ruta in [(canonico, SALIDA_CANONICO), (export, SALIDA_2026_08)]:
        df.to_csv(ruta, index=False, encoding="utf-8", lineterminator="\r\n")
        print(f">>> {ruta.relative_to(BASE_DIR)}  {df.shape[0]} filas x "
              f"{df.shape[1]} columnas  ({ruta.stat().st_size / 1024:.0f} KB)")
    _aviso()


if __name__ == "__main__":
    main()
