# Copa STEM 2026 — Modelos de Machine Learning y Ciencia de Datos

Repositorio de análisis de datos y modelos predictivos de **Copa STEM**, la olimpiada de
matemáticas y lógica de la **Fundación SapienceLab** para estudiantes de grados 9°, 10° y 11°
de los municipios de **Copacabana**, **Girardota** y **Bello** (Antioquia, Colombia).

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
| **Perfil académico** | `promedio_academico`, `horas_estudio_matematicas`, `motivacion_participar`, `clases_extra_matematicas`, `gusto_logica` |
| **Resultado** | `puntaje_obtenido` (0–100, variable objetivo) |
| **Telemetría** | `tiempo_usado_segundos`, `cambios_pestana`, `intentos_copiar`, `intentos_pegar`, `intentos_click_derecho` |

> La telemetría solo existe para exámenes presentados **en plataforma**; los exámenes escritos
> tienen estos campos vacíos.

> Las 5 columnas de **perfil académico** son opcionales: los primeros ~2.000 estudiantes tienen
> valores NULL. Se agregaron al formulario de inscripción para mejorar el reentrenamiento futuro
> del modelo.

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
| 1 | `02_analisis_brechas.py` | Brechas de equidad: género, socioeconómica y territorial |
| 2 | `03_modelo_predictivo.py` | Regresión para predecir el `puntaje_obtenido` |
| 2 | `04_indice_potencial_stem.py` | Índice 0–100 de potencial STEM (compuesto) |
| 3 | `05_deteccion_trampa.py` | Integridad del examen a partir de la telemetría |
| 3 | `05b_limpiar_dataset.py` | Genera los datasets `completo` y `limpio` (exámenes anulados) |
| 3 | `05c_cruce_computador_familia.py` | Cruce computador × con-quién-vive sobre el puntaje |
| 2/3 | `06_talento_oculto.py` | Talento oculto (alto rendimiento + condiciones adversas) |
| 3 | `07_clustering_perfiles.py` | Segmentación de perfiles de estudiante (K-Means) |
| 4 | `08_framework_validacion.py` | Robustez del modelo: bootstrap, techo teórico y drift |
| 4 | `09_exportar_puntaje_estimado.py` | Exporta `puntaje_estimado` por estudiante para Supabase |
| 4 | `09b_informe_puntaje_estimado.py` | Informe y gráficos del puntaje estimado |
| 4 | `10_modelo_teorico_vs_empirico.py` | Modelo teórico (`indice_condiciones`) vs empírico — auditoría |
| 5 | `11_preparar_experimento_reentrenamiento.py` | Prepara y reporta los datasets A/B/C/C′ del experimento de reentrenamiento (no entrena) |
| 5 | `12_experimento_reentrenamiento.py` | Entrena los cuatro modelos del experimento y compara R², MAE, n y features |
| 5 | `13_analisis_explicabilidad.py` | Importancia MDI y por permutación sobre A y C — qué variables pesan y cuánto |
| 5 | `14_optimizacion_hiperparametros.py` | `RandomizedSearchCV` del modelo v2 + export del predictor JS de despliegue |
| 5 | `15_generar_scores_v2.py` | Ejecución en sombra del vector `ml_scores` con modelo híbrido v2/v1 |

---

## Estado del modelo (Copa STEM 2026)

| Modelo | R² | MAE | Uso en producción |
|--------|----|-----|-------------------|
| Random Forest (script 03) | ~0.238 | ~18 pts | `indice_potencial` (componente rendimiento) |
| XGBoost (script 06) | — | — | `es_talento_oculto` |
| K-Means k=4 (script 07) | — | — | `cluster_nombre` |
| Modelo teórico (script 10) | — | — | `indice_condiciones` (knowledge-driven) |

> **`puntaje_estimado` está en NULL en producción.** La Edge Function lo fija a `null` de forma
> explícita (`index.ts:153`). Es el cambio que activaría el valor del modelo v2 — y el más
> delicado, porque publica una predicción con MAE ≈ 15 puntos.

### Versiones del modelo de puntaje

| | v1 (producción) | v2 (listo para desplegar) |
|---|---|---|
| Dataset | cohorte original (1.748) | `dataset_C_perfil` (1.148, con perfil académico) |
| Features | 18 | 23 (incluye las 5 nuevas) |
| Hiperparámetros | `n_estimators=300, max_depth=10, min_samples_leaf=8` | `n_estimators=200, max_depth=None, min_samples_leaf=8, max_features=0.5` |
| R² (hold-out) | — | **0.1766** |
| MAE (hold-out) | 18.45 * | **15.00** |
| Artefacto JS | `potencial_stem_predictor.js` (1.302 KB) | **`potencial_stem_predictor_v2.js` (816 KB)** |
| Modelo | `mejor_modelo_puntaje.joblib` | `mejor_modelo_puntaje_v2.joblib` |

> \* MAE de v1 medido sobre los 1.148 de C, que están **fuera** de su cohorte de entrenamiento
> (solapamiento verificado = 0). Es la única comparación limpia disponible entre ambos.

**Listo para producción:** `models/deploy/potencial_stem_predictor_v2.js` — verificado en Node v24,
reproduce a sklearn con máx|Δ| = 3.55e-14 y mantiene el mismo contrato de salida que v1.

---

## Experimento de reentrenamiento (Fase 5)

`11_preparar_experimento_reentrenamiento.py` deja listos cuatro datasets en `data/` para medir
por separado el efecto de **más datos** y el de **más variables**:

| Dataset | Filas | Qué aísla |
|---------|-------|-----------|
| `dataset_A_baseline.csv` | 1.735 | Línea base: la cohorte con la que se entrenó el primer modelo |
| `dataset_B_completo.csv` | 3.072 | Efecto de **más datos** (todas las filas con `puntaje_obtenido`) |
| `dataset_C_perfil.csv` | 1.148 | Efecto de **más variables** (filas con perfil académico) |
| `dataset_C_sin_features.csv` | 1.148 | **Control**: las mismas filas de C sin las 5 variables nuevas |

La comparación decisiva es **C vs C′**: al mantener la muestra fija, la diferencia de métricas se
atribuye solo a las 5 variables nuevas y no a un cambio de población. `A vs B` responde si
basta con acumular más inscritos.

La cohorte de A se identifica en cascada: por `created_at` si el export lo trae, si no por el
índice guardado del primer modelo (`models/deploy/puntaje_estimado.csv`), y como último recurso
una muestra aleatoria con `random_state=42`.

```powershell
python notebooks\11_preparar_experimento_reentrenamiento.py
python notebooks\12_experimento_reentrenamiento.py
```

### Resultados (export de 2026-08, 3.077 inscritos)

Mismo Random Forest y mismo protocolo (`KFold(5)` + hold-out 20 % estratificado) en los cuatro.

| Dataset | n | features | CV R² | CV MAE |
|---------|---|----------|-------|--------|
| A_baseline | 1.735 | 18 | +0.086 | 19,01 |
| B_completo | 3.072 | 19 | +0.053 | 17,95 |
| **C_perfil** | 1.148 | 23 | **+0.180** | **14,84** |
| C_sin_features | 1.148 | 18 | +0.098 | 15,72 |

- **Las 5 variables nuevas sí aportan.** Con la muestra fija, el R² casi se duplica
  (+0.098 → +0.180) y el MAE baja 0,88 puntos. Es el contraste limpio del experimento.
- **Acumular más inscritos no ayuda por sí solo.** De A a B el R² incluso baja
  (+0.086 → +0.053) pese a casi duplicar las filas. El MAE no es comparable entre A y B
  porque cambia la población; el R² sí.

> A, B y C′ se entrenan con el **mismo bloque de features base**. A y B arrastran físicamente
> las 5 columnas nuevas (casi todas vacías), así que el script las desactiva de forma explícita:
> si no, el contraste A→B mezclaría «más datos» con «más variables».

---

## Estado actual y tareas pendientes

### Hecho

- [x] Experimento controlado A/B/C/C′ (scripts 11–12) — las 5 variables nuevas casi duplican el R²
- [x] Explicabilidad (script 13) — `promedio_academico` domina con ΔR² = 0.104, 2,7× la segunda
- [x] Optimización de hiperparámetros y export v2 (script 14) — R² 0.1766, JS verificado en Node
- [x] Ejecución en sombra del vector `ml_scores` con modelo híbrido (script 15) — `outputs/ml_scores_v2.csv`

### Pendiente

- [ ] **Subir `outputs/ml_scores_v2.csv` a Supabase** — requiere antes añadir la columna
      `modelo_version` a `ml_scores` y volcar la tabla actual a CSV (el upsert de la Edge
      Function sobrescribe en sitio y **no hay rollback**)
- [ ] **Desplegar `potencial_stem_predictor_v2.js`** a la Edge Function — requiere además
      ampliar `COLS_INSC` en `index.ts` con las 5 columnas de perfil académico, que hoy no lee
- [ ] **Decidir sobre `puntaje_estimado`** — hoy fijado a `null`; es lo que activa el valor de v2
- [ ] **Puntuar a los inscritos SIN examen** — es la población donde el modelo sí alimenta el
      índice; el script 15 solo cubre a los 3.072 que ya presentaron
- [ ] **Dashboard** de seguimiento

> **Advertencia del informe 15.** Desplegar v2 hoy tal cual no mejoraría nada visible: para quien
> ya presentó el examen el índice usa el puntaje **real**, no el modelo, y la única columna que el
> modelo alimenta está en `null`. Lo único que llegaría a producción sería un desplazamiento de
> categorías por el cambio de distribución de referencia — que no es una mejora.

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
