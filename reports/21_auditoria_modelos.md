# Auditoría de modelos — inventario, arms experimentales e hiperparámetros

**Fundación SapienceLab** · Copa STEM 2026 · Auditoría de solo lectura · 2026-09-26

---

## Alcance y método

Este informe **no entrena, no reentrena, no modifica y no borra nada**. Es el
inventario completo de lo que el repositorio ha producido en materia de modelos,
levantado para decidir si un **dashboard de comparación de modelos** es viable.

Qué se hizo, exactamente:

1. Listado de todos los ficheros cargables de `models/`, `models/deploy/` y
   `outputs/` con tamaño y fecha de modificación reales de disco.
2. **Carga efectiva** de los dos `.joblib` en el entorno actual (`.venv`) y
   lectura de `get_params()`, `n_features_in_`, número de árboles y
   profundidades. Se ejecutó `predict()` sobre una matriz sintética para
   comprobar que el objeto es funcional, **no** para producir scores.
3. **Importación efectiva** de los 6 predictores puros de Python y de los 6
   predictores `.js` (Node v24.13.0), con una llamada de ejemplo cada uno.
4. Lectura de los scripts `03`, `06`, `07`, `11`, `12`, `13`, `14`, `15`, `17`,
   `18`, `19` para extraer protocolo, features e hiperparámetros.
5. Reproducción **solo de los splits** (`train_test_split` + `qcut`) de los
   cuatro arms para fijar el n de cada conjunto de evaluación. No se ajustó
   ningún estimador.
6. `grep` exhaustivo de todo el repositorio (excluyendo `.venv/`, `.git/` y
   `__pycache__/`) buscando `GridSearchCV`, `RandomizedSearchCV`, `optuna`,
   `hyperopt`, `BayesSearchCV`, `ParameterGrid`, `ParameterSampler`,
   `param_grid`, `param_distributions` y `validation_curve`.
7. Cruce de toda ruta de artefacto citada en `reports/` y `README.md` contra el
   disco, para detectar referencias sin fichero.

Entorno de verificación: Python 3.14.0, scikit-learn **1.9.0**, numpy 2.5.0,
pandas 3.0.3, joblib 1.5.3, Node v24.13.0.

Este informe se apoya en el informe 20 (reconciliación de métricas de v1) y no lo
contradice en ningún punto.

---

## Resumen ejecutivo

1. **Existen 2 modelos serializados (`.joblib`) y 12 artefactos-predictor
   portables.** Los 14 **cargan y ejecutan** en el entorno actual. Ninguna ruta
   de artefacto citada en los informes falta en disco.
2. **Los cuatro arms del experimento (A, B, C′, C) NO dejaron ningún modelo
   guardado.** El script 12 entrena, mide, escribe tres CSV de métricas y
   descarta los estimadores. No hay `joblib.dump` en el script 12 ni en el 13.
   Para volver a ver esos cuatro modelos hay que reentrenarlos.
3. **Las cuatro cifras de R² citadas (0.086 / 0.053 / 0.098 / 0.180) son todas
   CV R² de 5 folds sobre el 80 % de train, medidas con el mismo protocolo.**
   En ese sentido son homogéneas.
4. **Pero solo una de las dos comparaciones es válida.** C′ vs C es un
   experimento controlado limpio: mismas 1,148 filas, mismo train, mismo
   hold-out (verificado fila a fila en esta auditoría). **A vs B no es
   comparable** y no lo es por tres razones acumuladas, no por una: poblaciones
   disjuntas en el tiempo, espacios de features de distinta dimensión (18 vs 19)
   y **254 de los 347 estudiantes del hold-out de A están dentro del train de
   B**. Lo mismo vale para cualquier comparación A↔C: **A ∩ C = 0 estudiantes**.
5. **Sí hubo una búsqueda de hiperparámetros, y solo una**: `RandomizedSearchCV`
   en el script 14 — 30 candidatos de 240 posibles, `cv=5`, `scoring='r2'`,
   150 ajustes. Ganador: `n_estimators=200`, `max_depth=None`,
   `min_samples_leaf=8`, `max_features=0.5`. Ningún otro script del repositorio
   busca hiperparámetros. El barrido de K=2..10 del script 07 es selección de
   modelo no supervisado, no búsqueda de hiperparámetros de un estimador
   supervisado, y se reporta aparte.
6. **Los cuatro arms son REPRODUCIBLES en esta copia de trabajo, y NINGUNO lo es
   desde el repositorio git.** `.gitignore` excluye el directorio `data/`
   completo: los cuatro datasets del experimento existen en disco pero no están
   versionados. `models/mejor_modelo_puntaje_v2.joblib` tampoco está versionado.
7. **Para un dashboard que re-puntúe a los 3,072 examinados hoy: v1 puede,
   v2 puede con una advertencia grande, y los cuatro arms no pueden sin
   reentrenar.** v1 ya los puntuó a los 3,072 (columna `puntaje_est_v1` de
   `outputs/F15_comparacion_v1_v2.csv`, 3,072 valores no nulos). v2 solo ha
   puntuado a los 1,148 con perfil académico; para los otros 1,924 imputaría por
   mediana las 5 variables que son precisamente su razón de existir.

---

# PARTE 1 — Inventario de artefactos de modelo

## 1.1 Modelos serializados (`.joblib`)

| Ruta | Tamaño | Fecha de modificación | Qué declara ser | ¿Carga hoy? |
| --- | --- | --- | --- | --- |
| `models/mejor_modelo_puntaje.joblib` | 3,330,797 B (3.18 MB) | 2026-07-06 07:51:09 | dict con `modelo` (RandomForestRegressor, 300 árboles, 18 features), `preprocessor`, `feature_names`, `nombre="Random Forest"`, `random_state=42`. Modelo **v1** de producción, generado por el script 03 | **SÍ**, con advertencia de versión: los `DecisionTreeRegressor` se serializaron con **sklearn 1.8.0** y se cargan con **1.9.0**. `predict()` funciona |
| `models/mejor_modelo_puntaje_v2.joblib` | 2,024,418 B (1.93 MB) | 2026-08-30 00:25:48 | dict con `modelo` (RandomForestRegressor, 200 árboles, 23 features), `preprocess`, `feature_names`, `best_params`, `metricas_holdout`, `version="v2"`, `entrenado="2026-08-30T00:25:48"`, `dataset="dataset_C_perfil.csv"`. Modelo **v2** optimizado, script 14 | **SÍ**, sin ninguna advertencia de versión (fue serializado con la sklearn actual). `predict()` funciona |

Detalle estructural verificado al cargar:

| | v1 | v2 |
| --- | --- | --- |
| `n_estimators_` reales | 300 | 200 |
| `n_features_in_` | 18 | 23 |
| Profundidad de árbol (mín / máx / media) | 9 / 10 / 10.0 | 8 / 14 / 10.6 |
| Hojas por árbol (mín / máx / media) | 62 / 86 / 75.4 | 62 / 73 / 68.5 |
| Guarda sus propias métricas | **No** | **Sí** (`r2=0.17663952657605664`, `mae=14.995819066080928`, `n_test=230`) |
| Categorías de `municipio` en el one-hot | `Copacabana`, `Girardota` | `Bello`, `Girardota` |

La asimetría de la última fila importa para el dashboard y se desarrolla en la
Parte 5: **ninguno de los dos modelos vio en entrenamiento las tres alcaldías
que hoy tiene la cohorte completa**.

## 1.2 Predictores portables de despliegue

Todos son código generado (stdlib pura, sin sklearn) con el modelo embebido como
constante `SPEC`/`_SPEC`. Todos fueron **importados y ejecutados** en esta
auditoría.

| Ruta | Tamaño | Modificación | Qué declara ser | ¿Carga hoy? |
| --- | --- | --- | --- | --- |
| `models/predictor.py` | 1,321,242 B | 2026-07-06 07:51:09 | Réplica pura del RF v1: `ensemble`/`mean`, 300 árboles, 18 features, `bias=-1.12e-16` | **SÍ** — `predecir_puntaje(ej)` → 40.899 |
| `models/modelo_coeficientes.json` | 3,506 B | 2026-07-06 07:51:09 | Metadatos de v1: `modelo`, `n_features=18`, `features`, `importancias` (18 valores), `reglas_ejemplo` (árbol 1 de 300 en texto). **No contiene el modelo completo ni métricas** | **SÍ** (JSON válido; no es un modelo ejecutable) |
| `models/deploy/potencial_stem_predictor.py` | 1,335,113 B | 2026-07-06 07:51:13 | Índice de Potencial STEM **v1**. `meta.n_cohorte=1750`, RF de 300 árboles + pesos 50/25/25 + `ref_rendimiento` de 1,750 valores | **SÍ** — devuelve los 5 campos del contrato |
| `models/deploy/potencial_stem_predictor.js` | 1,333,554 B | 2026-07-06 07:51:13 | Igual, para la Edge Function. `export function calcularIndicePotencial` | **SÍ** (Node) |
| `models/deploy/potencial_stem_predictor_v2.js` | 835,796 B | 2026-08-30 00:25:54 | Índice **v2**. `meta.n_cohorte=1148`, RF de 200 árboles, 23 features, `ref_rendimiento` de 1,148 valores. Generado por el script 14 | **SÍ** (Node) |
| `models/deploy/potencial_stem_predictor_v2_corrected.js` | 847,905 B | 2026-09-02 08:24:11 | **v2 con referencia corregida** (script 19). Mismo modelo y mismo preprocesamiento que el anterior; `ref_rendimiento` recalculado sobre los 3,072 de `dataset_B_completo.csv` (σ 22.6583 vs 20.5332). `meta.n_cohorte=3072`, `n_entrenamiento=1148` | **SÍ** (Node) |
| `models/deploy/spec_potencial_v2_corrected.json` | 700,317 B | 2026-09-21 14:19:04 | El `SPEC` del artefacto anterior extraído a JSON. **Verificado en esta auditoría: es idéntico objeto a objeto al `SPEC` embebido en `potencial_stem_predictor_v2_corrected.js`** (839,594 caracteres serializados en ambos casos) | **SÍ** (JSON válido) |
| `models/deploy/talento_oculto_predictor.py` | 103,824 B | 2026-07-06 08:02:50 | Clasificador de talento oculto. `meta.modelo="XGBoost"`, `n_total=1750`, `n_talento=342`; `ensemble`/`sum`, 300 árboles, `bias=-1.4119954839876354`, `proba_transform="sigmoid"`, 26 features | **SÍ** — `detectar_talento_oculto(ej)` → dict de 4 campos |
| `models/deploy/talento_oculto_predictor.js` | 103,085 B | 2026-07-06 08:02:50 | Igual, para la Edge Function | **SÍ** (Node) |
| `models/deploy/clustering_predictor.py` | 6,022 B | 2026-07-06 08:02:57 | K-Means K=4: `scaler_mean`, `scaler_scale`, 4 centroides de 9 dimensiones, `nombres` de los 4 perfiles | **SÍ** — `predecir_perfil(ej)` → cluster 1 "Base conectada" |
| `models/deploy/clustering_predictor.js` | 5,614 B | 2026-07-06 08:02:57 | Igual, para la Edge Function | **SÍ** (Node) |
| `models/deploy/indice_condiciones_predictor.py` | 5,355 B | 2026-07-06 08:18:09 | Modelo **teórico** de condiciones (script 10). Pesos de literatura educativa, **sin entrenamiento**, sin `SPEC` de modelo | **SÍ** — `indice_condiciones(ej)` → 55.0 |
| `models/deploy/indice_condiciones_predictor.js` | 3,461 B | 2026-07-06 08:18:09 | Igual, para la Edge Function | **SÍ** (Node) |
| `models/deploy/validation_framework.py` | 1,330,436 B | 2026-07-06 08:19:03 | Framework de validación del script 08 con el RF v1 embebido; expone `validate_new_data` además de `predecir_puntaje` | **SÍ** |

## 1.3 Artefactos de scores y de métricas (no son modelos, pero un dashboard los consumiría)

| Ruta | Filas | Modificación | Contenido |
| --- | --- | --- | --- |
| `models/scores_potencial_stem.csv` | 1,750 | 2026-07-06 07:51:10 | Índice v1 de la cohorte de julio |
| `models/deploy/scores_potencial_stem.csv` | 1,750 | 2026-07-06 07:51 | Copia de despliegue |
| `models/deploy/puntaje_estimado.csv` | 1,748 | 2026-07-06 08:18 | `puntaje_estimado` vs `puntaje_real` de v1 (la fuente del ~0.241 dentro de muestra del informe 20) |
| `models/deploy/talento_oculto_scores.csv` | 1,750 | 2026-07-06 08:02 | Probabilidades de talento |
| `models/deploy/clustering_perfiles.csv` | 1,750 | 2026-07-06 08:02 | Asignación de cluster |
| `models/deploy/modelo_teorico_scores.csv` | 1,748 | 2026-07-06 08:18 | Índice de condiciones teórico |
| `models/deploy/sospecha_trampa.csv` | 1,805 | 2026-07-06 07:50 | Señales de anomalía del script 05 |
| `outputs/ml_scores_v2.csv` | 3,072 | 2026-08-30 00:35 | Vector completo híbrido (1,148 con v2 + 1,924 con v1 de respaldo) |
| `outputs/ml_scores_v2_corrected.csv` | 3,072 | 2026-08-30 22:00 | Igual, con la referencia corregida del script 17 |
| `outputs/F15_comparacion_v1_v2.csv` | 3,072 | 2026-08-30 00:35 | v1 **y** v2 lado a lado por estudiante |
| `outputs/ml_scores_sin_examen.csv` | 248 | 2026-09-02 00:45 | Inscritos sin examen (script 18) |
| `outputs/F12_experimento_reentrenamiento.csv` | 4 | 2026-08-29 23:45 | **Las métricas de los cuatro arms. Es todo lo que quedó de ellos** |
| `outputs/F12_comparacion_familias.csv` | 3 | 2026-08-29 23:45 | Lineal / RF / XGBoost sobre C |
| `outputs/F12_descomposicion_bloques.csv` | 3 | 2026-08-29 23:45 | Socioeconómicas / académicas / ambas |
| `outputs/F13_importancias.csv` | — | 2026-08-29 23:52 | MDI y permutación de A y C |
| `outputs/F14_optimizacion_v2.json` | — | 2026-08-30 00:25 | Resultado de la búsqueda de hiperparámetros |
| `outputs/F17_ref_rendimiento_corregido.json` | — | 2026-08-30 22:00 | Distribución de referencia n=3,072 |
| `outputs/F19_verificacion_deploy_v2.json` | — | 2026-09-02 08:24 | Verificación de paridad del artefacto v2 corregido |

## 1.4 Artefactos citados en informes que NO existen en disco

Se cruzaron **63 rutas distintas** citadas en `reports/` y `README.md` contra el
disco. **Falta una sola, y no es un artefacto de modelo:**

| Ruta citada | Dónde | Veredicto |
| --- | --- | --- |
| `data/inscripciones_2027.csv` | `reports/08_framework_validacion.md:183` | **No es un artefacto faltante.** Es el argumento de un **ejemplo de uso** de `validate_new_data()` para una convocatoria futura. No se afirma en ninguna parte que exista |

**No hay ningún modelo que exista solo como cita de informe.** Los 14 artefactos
cargables mencionados en la documentación están todos en disco y todos se
ejecutaron en esta auditoría.

## 1.5 Lo que el inventario NO cubre: qué está versionado en git

Distinción crítica para el dashboard, porque no coincide con lo que hay en disco.

| Artefacto | En disco | Versionado en git |
| --- | --- | --- |
| `models/mejor_modelo_puntaje.joblib` (v1) | Sí | **Sí** (quedó rastreado antes de que se añadiera `*.joblib` al `.gitignore`) |
| `models/mejor_modelo_puntaje_v2.joblib` (v2) | Sí | **NO** — ignorado por `*.joblib` |
| Los 12 predictores portables + `spec_..._corrected.json` | Sí | **Sí**, los 13 |
| `data/` (los 11 CSV, incluidos los 4 datasets del experimento) | Sí | **NO** — `.gitignore` excluye el directorio entero |
| `outputs/ml_scores_v2_corrected.csv`, `outputs/ml_scores_sin_examen.csv` | Sí | **NO** — ignorados explícitamente |
| Resto de `outputs/` (99 ficheros, incluidos los F12/F14/F17/F19) | Sí | **Sí** |

Consecuencia: **un clon limpio del repositorio no puede reentrenar ningún arm ni
cargar el modelo v2.** Lo puede hacer esta copia de trabajo. La Parte 4 califica
los arms sobre esta copia de trabajo y lo indica explícitamente.

---

# PARTE 2 — Los cuatro arms experimentales

## 2.1 Quién los produjo

| Paso | Script | Qué hace |
| --- | --- | --- |
| Preparación | `notebooks/11_preparar_experimento_reentrenamiento.py` | Construye los 4 CSV a partir del export `data/copa_stem_dataset_2026-08.csv.csv` (3,077 filas × 31 columnas). **No entrena nada** (lo declara en su docstring y se verificó: no importa ningún estimador) |
| Entrenamiento y medición | `notebooks/12_experimento_reentrenamiento.py` | Entrena los 4 modelos, mide, escribe `outputs/F12_*.csv`. **No guarda ningún modelo** |
| Interpretación | `notebooks/13_analisis_explicabilidad.py` | Reentrena A y C solo para importancias MDI y de permutación. **Tampoco guarda modelos** |
| Informe | `reports/12_experimento_reentrenamiento.md` | Narrativa del experimento |

Las cuatro cifras del enunciado corresponden a la columna `cv_r2` de
`outputs/F12_experimento_reentrenamiento.csv`, con estos valores exactos:

| Arm | `n_samples` | `cv_r2` | `cv_r2_std` | `cv_mae` | `test_r2` | `test_mae` | `n_predictores` | `n_features` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline | 1,735 | **0.08605231405963001** | 0.03018 | 19.0082 | 0.11115 | 19.1825 | 13 | 18 |
| B_completo | 3,072 | **0.05255917551435692** | 0.04008 | 17.9475 | 0.03526 | 17.9423 | 13 | 19 |
| C_sin_features (C′) | 1,148 | **0.09755300473909978** | 0.05036 | 15.7171 | 0.11344 | 15.7801 | 13 | 18 |
| C_perfil (C) | 1,148 | **0.18016358090238777** | 0.02614 | 14.8376 | 0.16232 | 15.0593 | 18 | 23 |

## 2.2 Qué varió cada arm, con precisión

### Filas incluidas

Todos los arms parten del mismo export de agosto de 2026 y pasan por la misma
limpieza en el script 11 (se retiran los documentos de prueba
`{1234, 123456, 123456789, 1234567899}` y los duplicados por
`numero_documento`, conservando el primero). Después:

| Arm | Regla de inclusión | Filas | Ventana de `created_at` | Municipios presentes |
| --- | --- | --- | --- | --- |
| **A** | Las **1,735 inscripciones más antiguas** por `created_at` dentro del universo de B. Es la cascada de identificación del script 11, rama 1 (hay columna de fecha, así que nunca se llega al índice guardado ni al muestreo aleatorio) | 1,735 | 2026-05-22 16:39 → 2026-06-05 16:03 | Copacabana (1,110), Girardota (625) |
| **B** | **Todas** las filas con `puntaje_obtenido` no nulo | 3,072 | 2026-05-22 16:39 → 2026-07-30 11:55 | Copacabana (1,233), Girardota (1,092), Bello (747) |
| **C** | Filas con `promedio_academico` no nulo **y** con `puntaje_obtenido` (el script 12 filtra el target al cargar) | 1,148 | 2026-07-07 11:23 → 2026-07-30 11:55 | Bello (734), Girardota (414) |
| **C′** | Las **mismas 1,148 filas de C**, con las 5 columnas nuevas eliminadas físicamente del CSV | 1,148 | idéntica a C | idéntica a C |

Relaciones de conjuntos verificadas por `numero_documento` en esta auditoría:

- A ⊂ B: **sí**. C ⊂ B: **sí**. C = C′ (mismos 1,148 documentos): **sí**.
- **A ∩ C = 0 estudiantes.** A es la cohorte de mayo-junio; C es la de julio.
  No comparten ni un solo estudiante.
- Media y desviación del target por arm: A 42.52 ± 24.23 · B 41.74 ± 22.66 ·
  C y C′ 41.08 ± 20.54. **La dispersión del target cae un 15 % de A a C**, que es
  por qué el MAE no es comparable entre poblaciones (el propio script 12 lo
  advierte por escrito al final de la sección de contrastes).

### Definición del target

**Idéntica en los cuatro arms:** `puntaje_obtenido`, numérico, 0–100, coercido
con `pd.to_numeric(..., errors="coerce")` y filtrado a no nulo. No hay
transformación, ni recorte, ni binarización, ni cambio de escala. La columna
`porcentaje` —duplicado exacto del target— está explícitamente excluida del
conjunto de features en el script 11 (`NO_FEATURES`) para evitar fuga.

### Conjunto de features, arm por arm

Bloque **base**, común a A, B y C′ (13 predictores declarados):

```
numéricas directas (3) : grado_escolar, estrato, interes_prog_robotica
conteos de lista  (2)  : n_herramientas   <- herramientas_conocidas (JSON/CSV, descarta "Ninguna/Ninguno")
                         n_areas_interes  <- areas_interes           (idem)
ordinales 0-3     (2)  : nivel_programacion_ord <- nivel_programacion
                         nivel_robotica_ord     <- nivel_robotica
                         (mapa: Ninguna/Ninguno=0, Básica=1, Intermedia=2, Avanzada=3)
binarias Sí/No    (3)  : computador_bin <- computador_en_casa
                         internet_bin   <- internet_en_casa
                         olimpiadas_bin <- participacion_olimpiadas
                         (mapa: empieza por "s"=1, por "n"=0, resto NaN)
one-hot           (3)  : genero, municipio, tipo_institucion
```

Bloque **nuevo**, activo **solo en C** (5 predictores más):

```
numéricas nuevas (4) : promedio_academico          (0-5)
                       horas_estudio_matematicas   (horas)
                       motivacion_participar       (Likert 1-5)
                       gusto_logica                (Likert 1-5)
binaria nueva    (1) : clases_extra_bin <- clases_extra_matematicas
```

Dimensión real de entrada al modelo tras el one-hot:

| Arm | Predictores declarados | Dimensión tras one-hot | De dónde sale la diferencia |
| --- | --- | --- | --- |
| A | 13 | **18** | 10 no categóricas + genero(4) + municipio(**2**: Copacabana, Girardota) + tipo_institucion(2) |
| B | 13 | **19** | 10 + genero(4) + municipio(**3**: se suma Bello) + tipo_institucion(2) |
| C′ | 13 | **18** | 10 + genero(4) + municipio(**2**: Bello, Girardota) + tipo_institucion(2) |
| C | 18 | **23** | los 18 de C′ + 4 numéricas nuevas + 1 binaria nueva |

**A y C′ tienen la misma dimensión (18) pero no el mismo espacio de features:**
sus dos columnas de `municipio` significan cosas distintas (Copacabana/Girardota
en A, Bello/Girardota en C′).

Qué se excluyó en los cuatro arms, por decisión explícita del script 12:

- **Toda la telemetría del examen** (`tiempo_usado_segundos`, `cambios_pestana`,
  `intentos_copiar`, `intentos_pegar`, `intentos_click_derecho`): se mide durante
  el examen, no está disponible en el momento de predecir.
- **`porcentaje`**: duplicado exacto del target.
- **`edad_calculada`**: no figura en ninguna de las listas de features del
  script 12, aunque el script 11 sí la coerciona a numérica.

### Un detalle de diseño que el propio informe 12 documenta

`dataset_B_completo.csv` **arrastra físicamente las 5 columnas nuevas** (casi
todas vacías) porque sale del mismo export que C. La inclusión del bloque nuevo
se volvió por eso **explícita por dataset** (el cuarto elemento de la tupla en la
constante `DATASETS` del script 12), no inferida de las columnas del fichero. El
informe 12 registra que antes de esa corrección el R² de B era **+0.092**, y que
tras corregirla cayó a **+0.053**. Es decir: **la cifra de B que aquí se audita es
la corregida**, y la corrección invirtió el signo de la conclusión.

### Preprocesamiento

Idéntico en los cuatro arms, y con una propiedad que vale la pena subrayar:

```python
ColumnTransformer([
  ("num", SimpleImputer(strategy="median"),        numéricas + ordinales),
  ("bin", SimpleImputer(strategy="most_frequent"), binarias),
  ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh",  OneHotEncoder(handle_unknown="ignore"))]), one-hot),
])
```

Todo va **dentro de un `Pipeline` de sklearn**, así que la imputación y el
one-hot se ajustan **solo con el train de cada fold** durante la validación
cruzada. No hay fuga de información del test hacia el imputador en ninguno de
los cuatro arms. Esto es correcto, y **es distinto de lo que hace el script 14**
(ver §3.4).

## 2.3 ¿Se guardó el modelo entrenado?

**No. En ninguno de los cuatro arms.**

- `grep` de `joblib.dump`, `pickle.dump`, `.pkl`, `save_model` y `dump(` sobre
  `notebooks/*.py` devuelve exactamente **cuatro** coincidencias en todo el
  repositorio: `03:873` (`joblib.dump` de v1), `14:484` (`joblib.dump` de v2) y
  dos `get_booster().get_dump()` en `03:658` y `06:619`, que son volcados de
  árboles XGBoost a JSON para la exportación portable, no serialización de un
  modelo.
- El script 12 crea los estimadores dentro de `evaluar()`, los ajusta, calcula
  métricas y sale de la función. Los objetos se destruyen.
- Lo único que sobrevive de los cuatro arms son **tres CSV de métricas**
  (`F12_experimento_reentrenamiento.csv`, `F12_comparacion_familias.csv`,
  `F12_descomposicion_bloques.csv`) y **cuatro PNG de importancias** del script
  13 (`feature_importance_A/C.png`, `permutation_importance_A/C.png`) más
  `F13_importancias.csv`.
- El informe `12_experimento_reentrenamiento.md` **nunca afirma** haber guardado
  un artefacto, y no hay ninguna ruta de artefacto por arm citada en ningún
  informe. No hay, por tanto, ninguna promesa incumplida: hay una ausencia
  consistente.

| Arm | Modelo guardado | Qué quedó |
| --- | --- | --- |
| A | **No** | métricas en `F12_experimento_reentrenamiento.csv` + importancias (MDI y permutación) en `F13_importancias.csv` y 2 PNG |
| B | **No** | solo métricas en `F12_experimento_reentrenamiento.csv` |
| C′ | **No** | solo métricas en `F12_experimento_reentrenamiento.csv` |
| C | **No** | métricas + importancias + 2 PNG. **Sus hiperparámetros, no su modelo, son el punto de partida del script 14** |

Matiz importante sobre C: `models/mejor_modelo_puntaje_v2.joblib` **no es el
modelo del arm C**. Usa el mismo dataset y las mismas 23 features, pero otros
hiperparámetros (los ganadores de la búsqueda) y, sobre todo, está **reajustado
sobre las 1,148 filas completas**, no sobre las 918 de train. Son dos modelos
distintos.

## 2.4 Protocolo de evaluación y n de cada conjunto

**El mismo protocolo en los cuatro arms**, aplicado dentro de cada dataset:

1. Hold-out del **20 %**, `train_test_split(test_size=0.20, random_state=42)`,
   **estratificado por quintil de `puntaje_obtenido`** (`pd.qcut(y, q=5)`).
2. `KFold(n_splits=5, shuffle=True, random_state=42)` **sobre el train**, vía
   `cross_validate` con `scoring={"r2","neg_mean_absolute_error"}`. De ahí salen
   `cv_r2` y `cv_mae`.
3. `modelo.fit(Xtr, ytr)` y evaluación en el hold-out. De ahí `test_r2` y
   `test_mae`.

n exactos, reproducidos en esta auditoría ejecutando **solo los splits**:

| Arm | n total | n train | n hold-out | n por fold de CV (aprox.) | Quintiles efectivos |
| --- | --- | --- | --- | --- | --- |
| A | 1,735 | 1,388 | **347** | ~278 | 5 |
| B | 3,072 | 2,457 | **615** | ~491 | 5 |
| C′ | 1,148 | 918 | **230** | ~184 | 5 |
| C | 1,148 | 918 | **230** | ~184 | 5 |

Naturaleza de cada cifra:

- **Las cuatro cifras del enunciado (0.086 / 0.053 / 0.098 / 0.180) son `cv_r2`:
  media de 5 folds *out-of-fold dentro del train*.** No son in-sample, no son el
  hold-out, y no son un out-of-fold sobre la población completa (el 20 % de
  hold-out queda fuera del cálculo de CV).
- `test_r2` es un **hold-out único** sobre los n de la tabla.
- **Ninguno de los cuatro arms produjo un out-of-fold sobre el total.** El
  0.084 que el informe 20 declara métrica oficial de v1 es un
  `cross_val_predict` del script 09b sobre los 1,748 — otro protocolo, otro
  script, otra cohorte.

## 2.5 ¿Son comparables los arms entre sí? — respuesta directa

**C′ vs C: SÍ, es un experimento controlado limpio.** Verificado en esta
auditoría, no solo leído:

- Mismos 1,148 `numero_documento` (conjuntos idénticos).
- **Mismo conjunto de train (918 documentos) y mismo hold-out (230 documentos),
  comprobados como conjuntos iguales.** El estratificador es el quintil del
  target, que es el mismo vector `y` en ambos, y el `random_state` es 42 en
  ambos: la partición coincide por construcción y así se confirma.
- Mismos 5 folds, mismo modelo, mismo preprocesamiento.
- Única diferencia: 5 columnas → 23 features frente a 18.
- El propio script 12 imprime una verificación de esto en tiempo de ejecución
  (compara `n_samples` de C y C′ y reporta la diferencia en `n_features`).

Conclusión: **el salto 0.098 → 0.180 es atribuible a las 5 variables de perfil
académico y a nada más.** Es el único número del experimento con esa propiedad.

**A vs B: NO son comparables.** Y no por un motivo, sino por tres que se acumulan:

1. **Poblaciones distintas y desplazadas en el tiempo.** A = mayo-junio, dos
   municipios. B = mayo-julio, tres municipios, incluida Bello (747 estudiantes)
   que no existía en A. El informe 12 ya lo reconoce ("A → B no es un
   experimento controlado, y no puede serlo").
2. **Espacios de features de distinta dimensión.** A tiene 18 columnas, B tiene
   19, porque el one-hot de `municipio` gana una categoría. No son el mismo
   modelo sobre más filas: son dos modelos sobre dos matrices distintas.
3. **Los hold-outs se solapan con los trains.** Verificado aquí:
   **254 de los 347 estudiantes del hold-out de A caen dentro del train de B**, y
   solo 93 están también en el hold-out de B. El `test_r2` de B (0.035) se mide
   por tanto sobre un conjunto que incluye estudiantes que el modelo de A nunca
   vio y excluye a otros que sí. Las dos columnas `test_r2` no miden sobre el
   mismo material.

El `cv_r2` mitiga los puntos 1 y 3 en parte —el R² se normaliza por la varianza
de cada muestra, y el MAE no, razón por la cual el script 12 recomienda
explícitamente usar R² y no MAE para este contraste— pero **no los elimina**. La
lectura defendible de A → B es: *entrenar el mismo tipo de modelo con las mismas
variables base sobre la cohorte acumulada no produjo un R² mejor*. La lectura no
defendible es: *añadir datos empeoró el modelo*.

**A vs C, A vs C′, B vs C, B vs C′: NO son comparables.** Peor que A vs B:

- **A ∩ C = 0.** Cero estudiantes en común. Cualquier diferencia entre A y C
  mezcla el efecto de las variables, el de la población, el del municipio y el
  del momento de la convocatoria, sin forma de separarlos con los datos que hay.
- A (18 features) y C′ (18 features) parecen comparables por dimensión, pero sus
  columnas de `municipio` codifican municipios distintos. **A → C′ tampoco es un
  contraste limpio**, aunque su diferencia (0.086 → 0.098) sea pequeña.
- C y C′ son subconjuntos de B, así que B entrenó con ellos: comparar el
  hold-out de C contra el de B es comparar contra un modelo que ya vio a esos
  estudiantes.

**Resumen de comparabilidad — lo que importa más que los números:**

| Contraste | ¿Controlado? | Qué se puede concluir |
| --- | --- | --- |
| **C′ → C** (0.098 → 0.180) | **Sí, limpio** | Las 5 variables de perfil académico casi duplican el R². Conclusión sólida |
| A → B (0.086 → 0.053) | **No** | Solo: "acumular la cohorte con el formulario viejo no mejoró el R²". Nada sobre causalidad |
| A → C / A → C′ / B → C | **No** | Nada. Poblaciones disjuntas o solapadas con el train |

Una fila más de la tabla, del mismo `F12_descomposicion_bloques.csv` y sobre las
mismas 1,148 filas de C con la misma partición, que sí es comparable entre sí:
solo socioeconómicas (18 features) = **0.0976**; solo las 5 académicas
(5 features) = **0.1081**; ambas (23 features) = **0.1802**. Mismo protocolo,
misma muestra, misma partición: comparable.

---

# PARTE 3 — Hiperparámetros

## 3.1 Modelo v1 — `models/mejor_modelo_puntaje.joblib`

`RandomForestRegressor`. Conjunto **completo** leído de `get_params()` del objeto
cargado (no del código):

| Hiperparámetro | Valor | ¿Por defecto en sklearn 1.9? |
| --- | --- | --- |
| `n_estimators` | **300** | No (defecto 100) |
| `criterion` | `'squared_error'` | Sí |
| `max_depth` | **10** | No (defecto `None`) |
| `min_samples_split` | 2 | Sí |
| `min_samples_leaf` | **8** | No (defecto 1) |
| `min_weight_fraction_leaf` | 0.0 | Sí |
| `max_features` | 1.0 | Sí (todas las features) |
| `max_leaf_nodes` | `None` | Sí |
| `min_impurity_decrease` | 0.0 | Sí |
| `bootstrap` | **True** | Sí |
| `oob_score` | False | Sí |
| `n_jobs` | **-1** | No (defecto `None`) — no afecta el resultado |
| `random_state` | **42** | No (defecto `None`) |
| `verbose` | 0 | Sí |
| `warm_start` | False | Sí |
| `ccp_alpha` | 0.0 | Sí |
| `max_samples` | `None` | Sí |
| `monotonic_cst` | `None` | Sí |

**No-defecto que afecta al modelo: `n_estimators=300`, `max_depth=10`,
`min_samples_leaf=8`, `random_state=42`.** Origen:
`notebooks/03_modelo_predictivo.py:506-508`, escritos a mano.

## 3.2 Modelo v2 — `models/mejor_modelo_puntaje_v2.joblib`

`RandomForestRegressor`. Igual lectura, del objeto cargado:

| Hiperparámetro | Valor | ¿Por defecto? |
| --- | --- | --- |
| `n_estimators` | **200** | No |
| `criterion` | `'squared_error'` | Sí |
| `max_depth` | **`None`** | Sí |
| `min_samples_split` | 2 | Sí |
| `min_samples_leaf` | **8** | No |
| `min_weight_fraction_leaf` | 0.0 | Sí |
| `max_features` | **0.5** | No (defecto 1.0) |
| `max_leaf_nodes` | `None` | Sí |
| `min_impurity_decrease` | 0.0 | Sí |
| `bootstrap` | **True** | Sí |
| `oob_score` | False | Sí |
| `n_jobs` | **-1** | No — no afecta el resultado |
| `random_state` | **42** | No |
| `verbose` | 0 | Sí |
| `warm_start` | False | Sí |
| `ccp_alpha` | 0.0 | Sí |
| `max_samples` | `None` | Sí |
| `monotonic_cst` | `None` | Sí |

**No-defecto que afecta al modelo: `n_estimators=200`, `max_features=0.5`,
`min_samples_leaf=8`, `random_state=42`.** Los tres primeros son los **ganadores
de la búsqueda** (§3.4); el bundle los conserva en la clave `best_params`, lo que
permitió cotejar `get_params()` contra `best_params` y confirmar que coinciden.

Los tres artefactos JS/JSON de v2 (`_v2.js`, `_v2_corrected.js`,
`spec_potencial_v2_corrected.json`) embeben **el mismo bosque de 200 árboles** —
el script 19 lo verificó comparando los árboles bit a bit contra el `.joblib`
(`outputs/F19_verificacion_deploy_v2.json`:
`identico_al_embebido_en_js_vigente: true`, `preprocess_identico: true`).

## 3.3 Los otros tres modelos con artefacto

### Detector de talento oculto — `talento_oculto_predictor.py/.js`

`XGBClassifier`. **El artefacto guarda solo los árboles exportados, no los
hiperparámetros**; por eso el conjunto completo se toma del código fuente que lo
generó (`notebooks/06_talento_oculto.py:513-516`):

| Hiperparámetro | Valor | Nota |
| --- | --- | --- |
| `n_estimators` | **300** | Confirmado en el artefacto: 300 árboles |
| `max_depth` | **4** | |
| `learning_rate` | **0.08** | |
| `subsample` | **0.9** | |
| `colsample_bytree` | **0.9** | |
| `eval_metric` | `'logloss'` | |
| `random_state` | **42** | |
| `n_jobs` | -1 | No afecta el resultado |
| `verbosity` | 0 | |
| Resto | defectos de XGBoost | |
| `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`, `bootstrap` | **no aplican** | Son parámetros de árboles de sklearn, no de XGBoost |

Del artefacto se leen además: `combine="sum"`, `op="lt"`,
`bias=-1.4119954839876354` (el logit del `base_score`),
`proba_transform="sigmoid"`, 26 features, `meta.n_total=1750`,
`meta.n_talento=342`.

Advertencia que el propio informe 06 documenta y que un dashboard debe mostrar:
el target `talento_oculto` es **una regla determinista sobre las mismas
variables**, así que el AUC de test = 1.000 es reconstrucción de la regla, no
capacidad predictiva.

### Clustering de perfiles — `clustering_predictor.py/.js`

`KMeans`. De `notebooks/07_clustering_perfiles.py:340`:

| Hiperparámetro | Valor |
| --- | --- |
| `n_clusters` | **4** (elegido por silhouette, ver §3.5) |
| `n_init` | **10** |
| `random_state` | **42** |
| `init`, `max_iter`, `tol`, `algorithm` | defectos de sklearn |

El artefacto guarda `k=4`, 9 features, `scaler_mean` y `scaler_scale` (9 valores
cada uno), 4 centroides de 9 dimensiones y los nombres de los perfiles.
Silhouette reportado: K-Means 0.217 vs GMM 0.168 (`reports/07`).

### Índice de condiciones — `indice_condiciones_predictor.py/.js`

**No tiene hiperparámetros.** Es el modelo teórico del script 10: pesos fijos
tomados de literatura educativa (OECD PISA, UNESCO, meta-análisis de nivel
socioeconómico), sin ajuste con datos de Copa STEM. Umbrales:
`COND_FAVORABLE = 60`, `COND_ADVERSA = 45`. No hay nada que buscar.

### Índice de Potencial STEM (el compuesto)

Los pesos del índice son fijos e idénticos en v1 y v2:
`{rendimiento: 0.50, engagement: 0.25, resiliencia: 0.25}`, con cortes de
categoría `[85 Talento destacado, 70 Alto potencial, 45 Promedio,
25 En desarrollo, 0 Requiere apoyo]`. **No se optimizaron:** vienen del script 04
y el script 14 los reutiliza literalmente. No hay ninguna búsqueda sobre ellos en
el repositorio.

## 3.4 ¿Hubo búsqueda de hiperparámetros? — SÍ, exactamente una

**Sí. `RandomizedSearchCV` en
`notebooks/14_optimizacion_hiperparametros.py:431-434`.** Es la **única** del
repositorio: el `grep` de `GridSearchCV`, `RandomizedSearchCV`,
`HalvingGridSearchCV`, `HalvingRandomSearchCV`, `optuna`, `hyperopt`,
`BayesSearchCV`, `ParameterGrid`, `ParameterSampler`, `param_grid`,
`param_distributions` y `validation_curve` sobre todo el árbol (excluido `.venv/`)
devuelve coincidencias solo en el script 14 y en los informes que lo describen.

### Espacio de búsqueda

```python
GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [5, 10, 15, 20, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features":     ["sqrt", "log2", 0.5],
}
```

4 × 5 × 4 × 3 = **240 combinaciones posibles**.

### Configuración

| Parámetro | Valor |
| --- | --- |
| Estimador base | `RandomForestRegressor(random_state=42, n_jobs=1)` |
| `n_iter` (nº de ensayos) | **30** de 240 (12.5 %) |
| `cv` | **5** — entero, es decir `KFold(n_splits=5, shuffle=False)` |
| `scoring` | `'r2'` |
| `random_state` | 42 |
| `refit` | True |
| **Ajustes totales** | **30 × 5 = 150** |
| Dataset | `dataset_C_perfil.csv`, 1,148 filas |
| Conjunto sobre el que se busca | el **train**: n = 918, 23 features |

### Configuración ganadora

De `outputs/F14_optimizacion_v2.json` y confirmada contra `get_params()` del
artefacto:

| Hiperparámetro | Ganador |
| --- | --- |
| `n_estimators` | **200** |
| `max_depth` | **`None`** |
| `min_samples_leaf` | **8** |
| `max_features` | **0.5** |

Métricas asociadas:

| Métrica | Valor |
| --- | --- |
| `cv_r2` del mejor candidato (`best_score_`) | **0.1748746668603423** |
| R² en el hold-out del 20 % (n=230) | **0.17663952657605664** |
| MAE en el hold-out | **14.995819066080928** |
| R² de referencia con los hiperparámetros del script 03, misma partición | **0.16247738190718541** |
| MAE de referencia | **15.058918118869101** |
| **Ganancia atribuible a la búsqueda** | **+0.0142 de R², −0.063 de MAE** |

### Lo que la búsqueda NO deja trazado

- **Los 30 ensayos individuales no se guardaron.** El script escribe solo
  `best_params_`, `best_score_` y las métricas del hold-out. `cv_results_` se
  descarta. **Cuáles 30 de las 240 combinaciones se probaron, y qué puntuó cada
  una: no determinable desde el repositorio.**
- El modelo de producción **no es el `best_estimator_`**: el script reajusta un
  `RandomForestRegressor(**best_params_)` sobre las **1,148 filas completas**
  (`PRE_full`, `X_full`), y las métricas que publica son las del hold-out del
  `best_estimator_`. Son dos ajustes distintos y el bundle guarda el segundo.

### Dos diferencias de protocolo entre el script 14 y el script 12

Importan para el dashboard, porque hacen que `cv_r2` del script 14 **no sea
directamente comparable** con `cv_r2` del arm C:

1. **`cv=5` entero ⇒ `KFold` sin `shuffle`.** El script 12 usa
   `KFold(5, shuffle=True, random_state=42)`. Y `dataset_C_perfil.csv` está
   **ordenado de forma monótona creciente por `created_at`** (verificado): sin
   `shuffle`, los 5 folds de la búsqueda son **5 bloques temporales**
   consecutivos de la convocatoria, no 5 muestras aleatorias.
2. **La imputación se ajusta fuera de la CV.** El script 14 calcula medianas,
   modas y categorías one-hot con `ajustar_pre(reg_tr)` sobre **todo el train**
   *antes* de lanzar la búsqueda, y le pasa a `RandomizedSearchCV` una matriz
   numérica ya imputada. Cada fold de validación de la búsqueda ve, por tanto,
   estadísticos calculados con su propia participación. El script 12 mete la
   imputación dentro del `Pipeline` y no tiene este problema.

Ni una ni otra invalidan el resultado —la ganancia de +0.014 es pequeña de todos
modos, y el R² del hold-out (0.1766), que sí es limpio, lo confirma— pero
significan que **0.1749 (script 14) y 0.1802 (arm C) no son la misma medida** y no
deben ponerse en la misma columna de un dashboard.

## 3.5 Lo que existe y NO es búsqueda de hiperparámetros (para no confundirlo)

Tres cosas del repositorio se parecen a una búsqueda y no lo son. Se listan para
que nadie las cuente como tal:

| Qué es | Dónde | Por qué no es búsqueda de hiperparámetros |
| --- | --- | --- |
| **Barrido de K = 2..10 por silhouette** | `07_clustering_perfiles.py:305-312` | Es un bucle sobre el único parámetro de `KMeans` (`n_clusters`) con selección por silhouette, en un problema **no supervisado**. Ganador: **K=4** (silhouette 0.217). Es selección de modelo, honesta y reportada; no es el ajuste de un estimador supervisado |
| **Comparación de 4 familias** | `03_modelo_predictivo.py:503-521` | Lineal / RF / XGBoost / LightGBM, cada una con **hiperparámetros escritos a mano y fijos**. Selecciona una familia por R² de test, no explora ningún espacio de parámetros |
| **Comparación de 3 familias sobre C** | `12_experimento_reentrenamiento.py:comparar_familias()` | Lineal / RF / XGBoost con configuraciones heredadas del script 03. El propio docstring lo dice: "esto **NO** es una búsqueda de hiperparámetros ni una selección de modelo por métrica" |

Configuraciones fijas de esas comparaciones, para el registro:

- `LinearRegression()` — todos los defectos.
- `XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
  subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, verbosity=0)`.
- `LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
  subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, verbose=-1)`
  — solo en el script 03.
- `RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
  class_weight="balanced", random_state=42, n_jobs=-1)` — solo en el script 06.
- `LogisticRegression(max_iter=5000, random_state=42)` — solo en el script 06.

## 3.6 Hiperparámetros de los cuatro arms

Los cuatro arms usan **la misma constante**, `MODELO_KWARGS` en
`notebooks/12_experimento_reentrenamiento.py:118`:

```python
MODELO_KWARGS = dict(n_estimators=300, max_depth=10, min_samples_leaf=8,
                     random_state=42, n_jobs=-1)
```

Es decir: **los hiperparámetros del script 03 (los de v1), sin optimizar, en los
cuatro arms.** Todo lo demás son defectos de sklearn:
`criterion='squared_error'`, `min_samples_split=2`, `max_features=1.0`,
`bootstrap=True`, `max_leaf_nodes=None`, `ccp_alpha=0.0`,
`min_impurity_decrease=0.0`, `min_weight_fraction_leaf=0.0`, `max_samples=None`,
`oob_score=False`, `monotonic_cst=None`, `warm_start=False`.

Que los cuatro compartan hiperparámetros es deliberado y es lo que hace legítimo
el contraste C′ vs C. **Ningún arm se benefició de la búsqueda del script 14**,
que se ejecutó después y solo sobre C.

---

# PARTE 4 — Reproducibilidad

Criterios evaluados para cada arm: ¿está el script de entrenamiento? ¿está el
dataset de entrada? ¿está fijada la semilla?

## 4.1 Los tres criterios, arm por arm

| Arm | Script de entrenamiento | Dataset de entrada | Semilla fija | Calificación |
| --- | --- | --- | --- | --- |
| **A** | `notebooks/12_experimento_reentrenamiento.py` — **presente** | `data/dataset_A_baseline.csv` — **presente** (1,735 × 31) | `random_state=42` en el split, en los folds y en el estimador | **REPRODUCIBLE** (en esta copia de trabajo) |
| **B** | ídem — **presente** | `data/dataset_B_completo.csv` — **presente** (3,072 × 31) | ídem | **REPRODUCIBLE** (en esta copia de trabajo) |
| **C′** | ídem — **presente** | `data/dataset_C_sin_features.csv` — **presente** (1,148 × 26) | ídem | **REPRODUCIBLE** (en esta copia de trabajo) |
| **C** | ídem — **presente** | `data/dataset_C_perfil.csv` — **presente** (1,148 × 31) | ídem | **REPRODUCIBLE** (en esta copia de trabajo) |

Los cuatro CSV se verificaron abriéndolos: filas, columnas, presencia de las 5
variables nuevas, cobertura del target y rango de `created_at` coinciden con lo
que declaran los informes 11 y 12. `dataset_C_sin_features.csv` no tiene las 5
columnas nuevas (verificado: las 5 son exactamente la diferencia de columnas
respecto a B), y `dataset_A_baseline.csv` las tiene físicamente pero con 0
valores no nulos, lo que es coherente con el diseño descrito en el script 11.

También está presente el **export origen** del que salen los cuatro
(`data/copa_stem_dataset_2026-08.csv.csv`, 3,077 × 31), así que la cadena es
reproducible **desde el paso anterior**: script 11 → 4 CSV → script 12 →
métricas.

## 4.2 La salvedad que cambia la calificación

Las cuatro calificaciones anteriores valen **para esta copia de trabajo**. Desde
el repositorio git, ninguna vale:

- `.gitignore` contiene `data/` — **el directorio completo**. Ninguno de los
  cuatro datasets del experimento, ni el export origen, está versionado
  (`git ls-files data/` devuelve vacío).
- Los commits `03fa8c0` y `27e8d74` (2026-09-03) dejaron de rastrear los CSV de
  datos de estudiantes por contener información identificable. Es una decisión
  correcta de privacidad, y tiene el efecto colateral de que el experimento no
  se puede reproducir desde el repositorio.
- No hay ningún script de descarga, ninguna consulta SQL versionada y ninguna
  conexión a Supabase en el código: el script 11 documenta explícitamente que
  *"ningún script del repositorio abre conexión a Supabase: todos leen el CSV
  exportado a mano en `data/`"*.

Por tanto, la calificación honesta tiene dos niveles:

| Arm | Desde esta copia de trabajo | Desde un clon limpio del repositorio |
| --- | --- | --- |
| A | **REPRODUCIBLE** | **NO REPRODUCIBLE** — falta `data/dataset_A_baseline.csv` y el export origen |
| B | **REPRODUCIBLE** | **NO REPRODUCIBLE** — ídem con `dataset_B_completo.csv` |
| C′ | **REPRODUCIBLE** | **NO REPRODUCIBLE** — ídem con `dataset_C_sin_features.csv` |
| C | **REPRODUCIBLE** | **NO REPRODUCIBLE** — ídem con `dataset_C_perfil.csv` |

En ambos niveles, el script y la semilla están presentes: **lo único que falta en
el clon son los datos.**

## 4.3 Determinismo: advertencias reales

Aunque `random_state=42` está en el split, en los folds y en el estimador, hay
dos fuentes de variación que un dashboard que prometa cifras idénticas debe
conocer:

1. **`n_jobs=-1` en el `RandomForestRegressor`.** El paralelismo de sklearn con
   semilla fija es determinista en el resultado del bosque (cada árbol recibe su
   propia semilla derivada), así que esto **no** debería mover las métricas. No
   se comprobó ejecutándolo, porque esta auditoría no entrena.
2. **Versión de las librerías.** Los arms se corrieron el 2026-08-29 con la
   sklearn de entonces. El entorno actual tiene sklearn **1.9.0**, y el
   artefacto v1 (julio) muestra que en el repositorio ha convivido al menos
   **1.8.0**. **Con qué versión exacta de sklearn se corrió el script 12: no
   determinable desde el repositorio** — no hay `requirements.txt`,
   `pyproject.toml`, `environment.yml` ni lockfile. Cambios de implementación
   entre versiones pueden mover el tercer o cuarto decimal del R².

## 4.4 Reproducibilidad de los modelos que sí tienen artefacto

Fuera de los cuatro arms, por completitud:

| Modelo | Script | Dataset de entrada | Estado |
| --- | --- | --- | --- |
| **v1** (`mejor_modelo_puntaje.joblib`) | `03_modelo_predictivo.py` — presente | `data/copa_stem_dataset_limpio.csv` — **presente** (1,754 filas). Se verificó que la limpieza del script 03 la reduce a **1,750**, el número que declara el informe 03 | **REPRODUCIBLE** en esta copia · NO REPRODUCIBLE desde el clon |
| **v2** (`mejor_modelo_puntaje_v2.joblib`) | `14_optimizacion_hiperparametros.py` — presente | `data/dataset_C_perfil.csv` — presente | **REPRODUCIBLE** en esta copia · NO REPRODUCIBLE desde el clon (falta el dataset **y** el propio `.joblib` está ignorado) |
| **talento oculto** | `06_talento_oculto.py` — presente | `copa_stem_dataset_limpio.csv` + `models/deploy/scores_potencial_stem.csv` (presente y versionado) | **PARCIALMENTE REPRODUCIBLE** en esta copia: requiere `xgboost`, que **no está instalado en el `.venv` actual**. Los scripts lo tratan como opcional y siguen sin él, lo que cambiaría el modelo seleccionado |
| **clustering** | `07_clustering_perfiles.py` — presente | `copa_stem_dataset_limpio.csv` | **REPRODUCIBLE** en esta copia |
| **índice de condiciones** | `10_modelo_teorico_vs_empirico.py` — presente | no necesita entrenamiento | **REPRODUCIBLE** siempre: son pesos fijos en el código |
| **`spec_potencial_v2_corrected.json`** | **ningún script del repositorio lo escribe** (`grep` sin coincidencias en `.py`) | — | **PARCIALMENTE REPRODUCIBLE**: su contenido se verificó idéntico al `SPEC` de `potencial_stem_predictor_v2_corrected.js`, así que es re-derivable trivialmente de él; **qué script o comando lo produjo: no determinable desde el repositorio** (commit `d0ffaeb`) |

---

# PARTE 5 — Qué necesitaría un dashboard de comparación

## 5.1 Modelos que pueden re-puntuar a los 3,072 examinados HOY

| Modelo | ¿Puede? | Evidencia y vía | Advertencias |
| --- | --- | --- | --- |
| **v1 — puntaje** (`mejor_modelo_puntaje.joblib`, `predictor.py`, `potencial_stem_predictor.py/.js`) | **SÍ, sin trabajo nuevo** | Ya está hecho: `outputs/F15_comparacion_v1_v2.csv`, columna `puntaje_est_v1`, **3,072 valores no nulos**. El script 15 invoca a v1 para **todos** los estudiantes, no solo para los de respaldo | Los **747 estudiantes de Bello** quedan fuera de distribución: el one-hot de `municipio` de v1 solo tiene `Copacabana` y `Girardota`, así que Bello se codifica como vector de ceros, una combinación que el bosque **nunca vio en entrenamiento** (toda fila de train tenía exactamente un 1). En los scores existentes eso se ve: media de `puntaje_est_v1` = 38.64 en Copacabana, 50.87 en Girardota, **45.26 en Bello** — un valor intermedio que es artefacto de la codificación, no una predicción informada |
| **v2 — puntaje** (`mejor_modelo_puntaje_v2.joblib` y sus 3 artefactos JS/JSON) | **SÍ técnicamente, pero solo 1,148 de los 3,072 están bien puntuados** | Verificado en esta auditoría: el artefacto v2 corregido acepta y puntúa un registro **sin** las 5 variables de perfil sin lanzar error (imputa por mediana). Pero eso **no se ha hecho nunca** en el repositorio: la columna `puntaje_est_v2` de `F15` es **híbrida** — v2 para los 1,148 con perfil, v1 para los otros 1,924 (comprobado: para `perfil=False`, `puntaje_est_v1` y `puntaje_est_v2` son idénticos, misma media 42.76, mismo mín/máx) | Dos advertencias, y la primera es grave: **(a)** para los 1,924 sin perfil académico, v2 imputaría por mediana las 4 numéricas nuevas (`promedio_academico`=3.8, `horas_estudio_matematicas`=1.0, `motivacion_participar`=3.0, `gusto_logica`=3.0) y por moda la binaria (`clases_extra_bin`=0.0) — es decir, **imputaría con una constante las 5 variables que son la única razón por la que v2 es mejor que v1**. La predicción resultante tiende a la media y **no es una predicción de v2 en ningún sentido útil**. **(b)** El one-hot de `municipio` de v2 es `Bello`/`Girardota`: ahora son los **1,233 de Copacabana** los que caen en el vector de ceros |
| **Índice de Potencial STEM v1 y v2** (compuesto) | **SÍ** | `outputs/ml_scores_v2.csv` y `ml_scores_v2_corrected.csv`, 3,072 filas cada uno; `F15` trae `indice_v1` e `indice_v2` con 3,072 no nulos | El índice **no invoca el modelo de puntaje** para quien ya tiene nota: el componente de rendimiento usa el **puntaje real** percentilizado contra `ref_rendimiento`. Como los 3,072 presentaron el examen, **el modelo no interviene en el índice de ninguno de ellos**. El script 15 lo dice explícitamente. Lo que cambia entre v1 y v2 en el índice es la **vara de medir** (`ref_rendimiento`: 1,750 valores en v1, 1,148 en v2, **3,072 en el v2 corregido** con σ 22.6583) |
| **Talento oculto** (`talento_oculto_predictor.py/.js`) | **SÍ** | Ya aplicado a los 3,072 en `ml_scores_v2.csv` (`es_talento_oculto`, `probabilidad_talento`) | Necesita `indice_potencial` como input, que a su vez viene del compuesto. Su target es una regla determinista (informe 06): el AUC 1.000 no es capacidad predictiva. Mismo problema de one-hot: `municipio` = `Copacabana`/`Girardota` |
| **Clustering de perfiles** (`clustering_predictor.py/.js`) | **SÍ** | Ya aplicado a los 3,072 (`cluster_id`, `cluster_nombre`) | Usa `puntaje_obtenido` como feature, disponible para los 3,072. Centroides ajustados sobre 1,750 de la cohorte de julio: la asignación de los 747 de Bello es extrapolación |
| **Índice de condiciones** (`indice_condiciones_predictor.py/.js`) | **SÍ, sin ninguna reserva** | Ya aplicado a los 3,072 (`indice_condiciones`, `nivel_condiciones`) | Ninguna: no se entrenó con datos, no tiene one-hot, no tiene cohorte de referencia. Es el único artefacto sin problema de distribución |

## 5.2 Modelos que NO pueden re-puntuar a nadie hoy

| Arm / modelo | Por qué no | Qué falta exactamente |
| --- | --- | --- |
| **A** (n=1,735, CV R² 0.086) | **No existe como artefacto.** El estimador se destruyó al terminar `evaluar()` en el script 12 | Reentrenar: correr el script 12 (o su función `evaluar` sobre `dataset_A_baseline.csv`) y **añadir una serialización**, que hoy no existe. Los ingredientes están todos: script, dataset, semilla |
| **B** (n=3,072, CV R² 0.053) | ídem | ídem con `dataset_B_completo.csv` |
| **C′** (n=1,148, CV R² 0.098) | ídem | ídem con `dataset_C_sin_features.csv` |
| **C** (n=1,148, CV R² 0.180) | ídem. **No confundir con v2**: mismo dataset y mismas 23 features, pero otros hiperparámetros y ajustado sobre 918 filas en vez de 1,148 | ídem con `dataset_C_perfil.csv`. Si lo que se quiere es "el mejor modelo sobre C", **eso ya existe y es v2** |
| **Regresión Lineal sobre C** (CV R² 0.163) | No existe como artefacto; solo su métrica en `F12_comparacion_familias.csv` | Reentrenar. Nótese que está a solo 0.017 de R² del bosque con 1 coeficiente por feature: si el dashboard va a comparar complejidad contra ganancia, este es el candidato más interesante que no tiene artefacto |
| **XGBoost sobre C** (CV R² 0.090) | ídem, y además **`xgboost` no está instalado en el `.venv` actual** | Instalar `xgboost` + reentrenar |
| **XGBoost / LightGBM del script 03** | ídem; solo métricas en `reports/03` | Instalar `xgboost` y `lightgbm` + reentrenar |
| **GMM del script 07** | ídem; solo el silhouette 0.168 en `reports/07` | Reentrenar |

## 5.3 Lo que un dashboard honesto tendría que resolver

Ordenado por lo que más daño hace si se ignora.

1. **No poner en la misma columna métricas de protocolos distintos.** El
   repositorio tiene, para el mismo modelo v1, al menos cuatro números
   defendibles según lo que se mida: **0.084** (out-of-fold sobre 1,748, la
   oficial según el informe 20), **0.115** (hold-out del 20 %, n≈350),
   **0.064 ± 0.024** (CV sobre el train) y **~0.241** (dentro de muestra, jamás
   comparable). El informe 20 ya documenta que mezclarlos hizo que v2 pareciera
   peor que v1. **Todo panel debe llevar el protocolo y el n de evaluación
   pegados al número**, no en una nota al pie.

2. **Etiquetar A vs B como no controlado, en la propia interfaz.** No basta con
   que el informe 12 lo diga. Un panel que muestre "1,735 → 0.086" junto a
   "3,072 → 0.053" invita a la lectura falsa de que más datos empeoraron el
   modelo. Los hechos que hay que mostrar al lado: poblaciones disjuntas en el
   tiempo, 18 vs 19 features, y **254 de los 347 del hold-out de A están en el
   train de B**.

3. **Marcar C′ → C como el único contraste limpio.** Es el resultado que
   sostiene toda la fase 5 y el que justifica la existencia de v2. Merece estar
   destacado, no al mismo nivel visual que los contrastes que no significan nada.

4. **Decidir qué significa "re-puntuar con v2" para los 1,924 sin perfil
   académico.** Hoy el repositorio ha tomado una decisión sensata —usar v1 de
   respaldo— y la ha etiquetado (`modelo_version = "v1_fallback"` en
   `ml_scores_v2.csv`). Si el dashboard quiere una columna "v2 puro" para los
   3,072, tiene que asumir que 1,924 de esos valores salen de imputar con una
   constante las variables que definen a v2. **Esa columna sería engañosa a menos
   que se marque fila a fila.** La columna `modelo_version` ya existe para eso.

5. **Mostrar la cobertura del one-hot de `municipio`.** Ninguno de los dos
   modelos de puntaje cubre las tres alcaldías: v1 no vio Bello (747
   estudiantes), v2 no vio Copacabana (1,233). En ambos casos el municipio no
   visto se codifica como vector de ceros. **1,980 de los 3,072 estudiantes son
   fuera de distribución para al menos uno de los dos modelos.**

6. **Distinguir `ref_rendimiento` del modelo.** Tres artefactos de v2 comparten
   exactamente el mismo bosque de 200 árboles y difieren solo en la distribución
   de referencia: n=1,148 (σ 20.5332) en `_v2.js`, n=3,072 (σ 22.6583) en
   `_v2_corrected.js` y en `spec_potencial_v2_corrected.json`. Un panel que los
   liste como "tres modelos" estará contando mal. Es **un modelo con tres varas
   de medir**, y el script 19 lo dejó verificado
   (`identico_al_embebido_en_js_vigente: true`).

7. **Si el dashboard va a permitir reentrenar, cerrar primero la brecha de
   datos.** Con `data/` fuera de git, cualquier reentrenamiento depende de que
   esta copia de trabajo exista. Lo mínimo para hacerlo sostenible sin volver a
   versionar datos identificables: un `requirements.txt` o lockfile (hoy no hay
   ninguno), y un procedimiento documentado de export desde Supabase (hoy el
   script 11 documenta que el export es manual).

8. **Lo que se necesitaría para que los cuatro arms fueran comparables entre sí,
   si eso es el objetivo.** No es un ajuste de presentación, es un experimento
   nuevo: un único hold-out fijo definido sobre la población completa (los
   3,072), con los cuatro modelos entrenados sobre trains anidados y evaluados
   **sobre ese mismo hold-out**. Es factible con lo que hay en disco —los datos,
   el script y la semilla existen— pero **no es lo que se hizo**, y ninguna
   reorganización de los números actuales lo sustituye.

## 5.4 Veredicto sobre la viabilidad

**Un dashboard de comparación es viable, pero no de "seis modelos".** Lo que el
repositorio puede sostener hoy, sin reentrenar nada, es:

- **6 artefactos ejecutables** (puntaje v1, puntaje v2 con sus 3 envoltorios de
  referencia, talento oculto, clustering, condiciones) que ya han puntuado a los
  3,072 y cuyos resultados están en disco.
- **1 comparación experimental válida** (C′ → C).
- **1 búsqueda de hiperparámetros** con su rejilla, su n de ensayos y su ganador
  documentados, aunque sin los 30 ensayos individuales.
- **4 arms que existen únicamente como cuatro filas de un CSV de métricas**, y
  que un panel solo puede mostrar como métricas históricas anotadas con su
  protocolo — no como modelos consultables.

Si el objetivo es un panel que *re-puntúe* con varios modelos, el conjunto real
disponible es **v1 y v2** (con las advertencias de §5.1) más los tres modelos
auxiliares. Si el objetivo es un panel que *compare métricas*, el trabajo
principal no es de ingeniería: es **etiquetar cada número con su protocolo y su
n**, y marcar visiblemente los contrastes que no son comparables.

---

## Limitaciones de esta auditoría

- **No se entrenó ni reentrenó ningún modelo.** Las métricas de los cuatro arms
  se toman de `outputs/F12_experimento_reentrenamiento.csv` tal como están; no se
  verificaron reejecutando el script 12. Los n de train/hold-out **sí** se
  reprodujeron, ejecutando solo `train_test_split` y `pd.qcut`.
- **No se recalculó ninguna métrica de ningún modelo.** La única ejecución de
  `predict()` fue sobre una matriz sintética de 3 filas, para comprobar que los
  objetos son funcionales.
- **No se puede verificar con qué versión de scikit-learn se corrió cada
  script.** No hay lockfile ni fichero de dependencias en el repositorio. El
  artefacto v1 delata sklearn 1.8.0; el v2 no produce advertencia con 1.9.0.
- **Los 30 ensayos de la búsqueda de hiperparámetros no son recuperables:**
  `cv_results_` no se guardó.
- **`xgboost` y `lightgbm` no están instalados en el `.venv` actual**, así que no
  se pudo verificar la ejecución de los estimadores que dependen de ellos. Los
  artefactos portables de talento oculto **sí** se ejecutaron, porque son código
  puro sin dependencia de XGBoost.
- Esta auditoría **no modificó ningún informe existente**. Es el único fichero
  nuevo.

## Referencias del repositorio

- Experimento y arms: `notebooks/11_*.py`, `notebooks/12_*.py`,
  `notebooks/13_*.py`, `reports/11_preparacion_experimento.md`,
  `reports/12_experimento_reentrenamiento.md`,
  `reports/13_analisis_explicabilidad.md`,
  `outputs/F12_experimento_reentrenamiento.csv`.
- Búsqueda de hiperparámetros y v2: `notebooks/14_*.py`,
  `reports/14_optimizacion_v2.md`, `outputs/F14_optimizacion_v2.json`.
- Modelo v1: `notebooks/03_*.py`, `reports/03_modelo_predictivo.md`.
- Otros modelos con artefacto: `notebooks/06_*.py`, `notebooks/07_*.py`,
  `notebooks/10_*.py` y sus informes.
- Scoring de los 3,072 y corrección de referencia: `notebooks/15_*.py`,
  `notebooks/17_*.py`, `notebooks/19_*.py`,
  `reports/15_scores_v2_comparacion.md`, `reports/17_fix_ref_rendimiento.md`,
  `reports/19_regenerar_deploy_v2.md`, `outputs/F15_comparacion_v1_v2.csv`,
  `outputs/F17_ref_rendimiento_corregido.json`,
  `outputs/F19_verificacion_deploy_v2.json`.
- Reconciliación previa de métricas de v1:
  `reports/20_reconciliacion_metricas_v1.md`.
