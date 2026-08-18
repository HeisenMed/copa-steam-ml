# Copa STEM 2026 — Modelos de Machine Learning y Ciencia de Datos

Repositorio de análisis de datos y modelos predictivos de **Copa STEM**, la olimpiada de
matemáticas y lógica de la **Fundación SapienceLab** para estudiantes de grados 9°, 10° y 11°
de los municipios de **Copacabana** y **Girardota** (Antioquia, Colombia).

El objetivo es transformar los datos de ~2.000 estudiantes inscritos (variables socioeconómicas,
demográficas, de experiencia previa, resultado del examen y telemetría de comportamiento) en
evidencia accionable para la toma de decisiones pedagógicas y de equidad.

---

## Estructura del proyecto

```
ml-models/
├── data/          # CSVs exportados de Supabase (copa_stem_dataset.csv)
├── notebooks/     # Scripts de análisis numerados (01_, 02_, ...)
├── models/        # Modelos entrenados (.joblib / .pkl)
├── outputs/       # Gráficos PNG, tablas y artefactos de análisis
├── reports/       # Informes documentados en Markdown
├── .venv/         # Entorno virtual de Python (no versionar)
└── README.md      # Este archivo
```

---

## Dataset — `data/copa_stem_dataset.csv`

| Grupo | Variables |
|-------|-----------|
| **Demográficas** | `numero_documento`, `edad_calculada`, `genero`, `municipio`, `grado_escolar`, `tipo_institucion` |
| **Socioeconómicas** | `estrato`, `jornada`, `con_quien_vive`, `computador_en_casa`, `internet_en_casa` |
| **Experiencia previa** | `participacion_olimpiadas`, `nivel_programacion`, `nivel_robotica`, `herramientas_conocidas`, `areas_interes`, `interes_prog_robotica` |
| **Resultado** | `puntaje_obtenido` (0–100, variable objetivo) |
| **Telemetría** | `tiempo_usado_segundos`, `cambios_pestana`, `intentos_copiar`, `intentos_pegar`, `intentos_click_derecho` |

> La telemetría solo existe para exámenes presentados **en plataforma**; los exámenes escritos
> tienen estos campos vacíos.

---

## Configuración del entorno

```powershell
cd "C:\Users\USUARIO\OneDrive\Desktop\Proyecto\SapienceLab\Copa STEM\ml-models"
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn openpyxl plotly scipy statsmodels joblib
```

---

## Cómo ejecutar los análisis

```powershell
.\.venv\Scripts\Activate.ps1
python notebooks\01_analisis_exploratorio.py
```

Cada script es **autocontenido y reproducible** (`random_state=42`). Los gráficos se guardan en
`outputs/` y el informe correspondiente en `reports/`.

---

## Hoja de ruta (fases)

| Fase | Script | Descripción |
|------|--------|-------------|
| 1 | `01_analisis_exploratorio.py` | EDA completo: univariado, bivariado, socioeconómico y telemetría |
| 1 | `02_analisis_brechas.py` | Brechas socioeconómicas y de género |
| 2 | `03_prediccion_puntaje.py` | Regresión para predecir el puntaje |
| 2 | `04_indice_potencial_stem.py` | Índice 0–100 de potencial STEM |
| 2 | `05_talento_oculto.py` | Clasificación de talento oculto (alto potencial + bajo acceso) |
| 3 | `06_clustering_estudiantes.py` | Segmentación de perfiles (K-Means / DBSCAN) |
| 3 | `07_deteccion_anomalias.py` | Detección de posible fraude (telemetría) |
| 3 | `08_vulnerabilidad_tecnologica.py` | Índice de vulnerabilidad tecnológica 0–100 |
| 4 | `09_factor_analisis.py` | Factores que más pesan en el rendimiento |
| 4 | `10_recomendaciones.py` | Recomendación de intervención por estudiante |

---

## Paleta de colores Copa STEM

| Rol | Hex |
|-----|-----|
| Cyan | `#00d4ff` |
| Violeta | `#8b5cf6` |
| Ámbar | `#f59e0b` |
| Verde | `#10b981` |
| Rojo | `#ef4444` |
| Azul | `#0f77ee` |
| Fondo oscuro (marca) | `#050816` |

---

_Fundación SapienceLab · Copa STEM 2026_
