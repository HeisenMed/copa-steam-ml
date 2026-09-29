# Informe 23 — Exportación del modelo global base y paridad Python ↔ JS

**Copa STEM 2026 · Fundación SapienceLab**
Scripts: `notebooks/23_exportar_modelo_global.py`, `24_scores_cohorte_2026_ref3050.py`, `25_experimento_n_estimators.py` · Salidas: `models/deploy/spec_potencial_global_base.json`, `models/deploy/potencial_stem_predictor_global_base.js`, `outputs/F23_*.json`, `outputs/F24_*.json`, `outputs/F25_experimento_n_estimators.json`
Fecha: 2026-09-28 · `random_state = 42`

**Numeración:** se usó el **23**. El 22 está ocupado por `reports/22_auditoria_modelos.md`, renombrado desde `21_auditoria_modelos.md` en el índice de git.

---

## 0. Resumen

- **El modelo oficial tiene ahora 150 árboles, no 300** (§9). Bajar a 150 cuesta 0.0011 de R² OOF, unas veinte veces menos que la incertidumbre de la propia métrica (IC 95 % ±0.023), y **reduce a la mitad** los artefactos de despliegue.
  - Se reentrenó `models/modelo_global_base.joblib` (todo lo demás, igual) con sus métricas embebidas actualizadas.
  - Se regeneraron los dos artefactos y **se repitió la verificación de paridad completa** sobre ellos.
- **Métrica oficial: R² OOF = 0.050, IC 95 % [0.027, 0.072], MAE 17.86, RMSE 21.92; n = 3,050, KFold(5).** Hold-out: R² 0.030, MAE 17.66, RMSE 21.83; n = 610.
- **Exportación:** se reutilizaron los exportadores del script 19, sin escribir ningún serializador nuevo. SPEC JSON de **1,120,939 B** y JS de **1,356,087 B**; 150 árboles y 44,734 nodos.
- **API:** el scorer (`scorer.py`) es genérico. `schemas.py` y `config.py` dan por hechas cosas de v2; son cambios de la API, que se gestionan en su propio repo (§3).
- **Paridad (artefactos de 150 árboles):** el scorer de la API, con la semántica de su runtime de producción (Python 3.11), frente al JS: **0 discrepancias en 12,546 comparaciones**, y el puntaje crudo es bit a bit idéntico en 2,091 de 2,091 cargas. Con Python ≥ 3.12 aparecerían 7 diferencias de 0.01, sin cambio de categoría. Es una **fragilidad latente** de `scorer.py` (§5).
- **Veredicto:** los dos artefactos son coherentes entre sí, y los cambios de la API están aplicados y verificados en local. Falta fusionar el pull request y medir el arranque en frío antes de desplegar (§7).
- **Decisiones declaradas sobre la cohorte publicada:**
  - Se aprueba la referencia de 3,050 y la corrección de los 3 duplicados.
  - **No** se aprueba repuntuarla con el SPEC global.
  - Se generó el archivo corregido: **3,320 filas, 25 cambios de categoría, todos hacia arriba**. **Se verificó que el reentrenamiento a 150 no lo afecta** (§8).
- **Listas:** `herramientas_conocidas` y `areas_interes` siguen siendo **opcionales** en la API. Una lista omitida se imputa con la mediana y `[]` cuenta 0 (§5.3).
- **API actualizada** (sin commit, pendiente de PR; §3.3):
  - Usa el SPEC nuevo, acepta estudiantes sin `promedio_academico`, y sus métricas y descripción corresponden al modelo global.
  - `/clustering` y `/talento-oculto` responden **422**, nunca 500, cuando falta `promedio_academico`.
- **Decisión institucional abierta:** qué hacer con el índice publicado de los 22 resultados descartados por sospecha (§8.3).

---

## 1. Decisión declarada: referencia de rendimiento y cohorte publicada

### 1.1 Qué se decidió y por qué

`ref_rendimiento` es la distribución contra la que se calcula el percentil de rendimiento del índice. La política del repo es explícita: **no es un subproducto de los datos de entrenamiento de cada versión**, y el script 19 existe precisamente porque se la estaba tratando así. El cambio a 3,050 es una **decisión de producto declarada**:

- La referencia vigente (F17, n = 3,072) incluye los **22 resultados descartados por sospecha** y las **3 sesiones abandonadas** que `keep="first"` conservó en lugar de la sesión completada (informe 21, §2.1).
- Mientras siga así, los estudiantes honestos se ordenan contra una distribución que es en parte fraudulenta.

| Referencia | n | Media | σ (ddof = 0) |
|---|---:|---:|---:|
| Vigente (F17) | 3,072 | 41.74 | 22.6583 |
| Nueva (conjunto limpio, informe 21) | 3,050 | 41.54 | 22.4948 |

La decisión humana **separa dos cambios**, porque sus perfiles de riesgo son opuestos:

| Cambio | Estado | Motivo |
|---|---|---|
| **A: solo la referencia, 3,072 → 3,050**, manteniendo el esquema de puntuación publicado | **APROBADO** | 20 de 3,050 examinados cambian de categoría y **los 20 suben**. Nadie pierde por quitar de la referencia resultados fraudulentos o abandonados |
| **Corrección de 3 documentos duplicados** cuya fila en producción era la sesión abandonada | **APROBADO, no opcional** | Esos 3 estudiantes se midieron sobre la fila equivocada. Los 3 suben |
| **B: repuntuar la cohorte comunicada con el SPEC global completo** | **NO APROBADO por ahora** | 17 examinados bajarían, 12 de ellos en Copacabana, y unos 60 de los 249 sin examen cambiarían. Con un modelo que explica ~5 % de la varianza, ese movimiento no es una ganancia de exactitud. Además, la lista sin examen es una lista de seguimiento de equidad, y su credibilidad depende de que sea estable |

### 1.2 Qué queda en vigor

- **La cohorte publicada de 2026** se puntuó con el **esquema mixto v2 / v1_fallback**: v2 para quien tiene perfil y el SPEC de v1 para quien no (scripts 17 y 18). La decisión es que conserve ese esquema, con la **referencia corregida de 3,050** y la corrección de los 3 documentos duplicados (§8).
- **El modelo global base, con 150 árboles, se aplica desde el despliegue en adelante**: estudiantes nuevos y API.
- **Coste declarado:** se preserva lo que ya se comunicó y se corrige el rechazo de Copacabana (el modelo global puntúa a quien no tiene perfil sin recurrir a v1). A cambio, **se pierde comparabilidad entre cohortes**. Un índice de la cohorte 2026 y uno calculado después del despliegue salen de modelos distintos y no deben mezclarse en un mismo ranking.

### 1.3 Alcance medido (antes de exportar)

La base de comparación es el esquema de producción reconstruido. Coincide con `outputs/ml_scores_v2_corrected.csv` en **3,047 de 3,050** estudiantes; las 3 diferencias son justamente los documentos duplicados. Solo se reportan agregados. El escenario A no depende del modelo global, así que sus cifras no cambian con los 150 árboles. Las de B se recalcularon con el modelo de 150.

**Examinados (n = 3,050)**

| Escenario | Cambian | Suben | Bajan | Bello (suben/bajan) | Girardota | Copacabana |
|---|---:|---:|---:|---|---|---|
| **A: solo referencia (aprobado)** | **20** | **20** | **0** | 4 / 0 | 11 / 0 | 5 / 0 |
| B: SPEC global de 150 árboles (no aprobado) | 39 | 22 | 17 | 11 / 0 | 10 / 5 | 1 / 12 |

Transiciones en A: Promedio → Alto potencial 14 · Requiere apoyo → En desarrollo 2 · En desarrollo → Promedio 2 · Alto potencial → Talento destacado 2.

**Inscritos sin examen (n = 249):**
- En A cambian 3, los 3 suben (Bello 1, Copacabana 2).
- En B cambiarían 61: 27 suben y 34 bajan (Bello 6/3, Girardota 4/20, Copacabana 17/11). Con el bosque de 300 eran 62.

**Por qué B mueve más que A:** además de la referencia, el SPEC global cambia los rangos de engagement y las imputaciones. Los rangos vigentes en v2 (herramientas 0–11, áreas 2–9) salen de `dataset_C_perfil.csv`, el conjunto de entrenamiento de v2: el mismo defecto de "subproducto del entrenamiento" que tenía la referencia. Con los 3,050 quedan en 0–12 y 1–9. En el SPEC global ya van corregidos, pero **solo se aplican de aquí en adelante**.

---

## 2. Exportadores reutilizados

| Artefacto | Exportador | Qué se reutilizó |
|---|---|---|
| JS para la Edge Function | `notebooks/19_regenerar_deploy_v2.py` (que a su vez replica la mecánica del script 14) | `_extract_rf`, el extractor de árboles; `construir_spec`, la estructura del SPEC; y la mecánica de `exportar_js`: se clona el cuerpo del JS vigente byte a byte y se sustituye solo la línea `const SPEC = …;` y la cabecera |
| SPEC JSON para la API | **Ningún script del repo lo escribía** (commit `d0ffaeb`, y `reports/22_auditoria_modelos.md:809`) | Se deriva como **la serialización compacta del mismo objeto** que construye `construir_spec` del script 19 |

**La cadena de formato del SPEC JSON, verificada.** Antes de escribir, el script comprueba sobre el artefacto vigente que `spec_potencial_v2_corrected.json` es, byte a byte, igual a `json.dumps(SPEC_del_JS, ensure_ascii=False, separators=(",", ":"))`. Resultado: **True**. El formato no se adivinó: es el que ya está desplegado. Después de escribir, se verifica que el JSON nuevo y el `SPEC` incrustado en el JS nuevo son el mismo objeto.

**El cuerpo del JS es genérico:** `_featuresPuntaje` recorre `PRE.numeric`, `PRE.ordinal`, `PRE.binary` y `PRE.onehot_order` sin suponer cuántas features hay.

### 2.1 Adaptador de preprocesamiento

El `PRE` del SPEC se lee del **Pipeline ya entrenado**: medianas y modas de las `statistics_` de sus `SimpleImputer`, y vocabulario one-hot de las `categories_` de su `OneHotEncoder`. El esquema de claves es idéntico al del PRE de v2.

**Verificación, sobre las 3,050 filas de entrenamiento y con el modelo de 150 árboles:**
- El PRE del SPEC reproduce **valor a valor** la matriz del Pipeline: 3,050 × 19, idéntica, 0 filas distintas.
- El intérprete de árboles reproduce `Pipeline.predict` con **máx. |Δ| = 1.4 × 10⁻¹⁴**, que es ruido del orden de suma entre hilos de sklearn.

Valores exportados:
- Medianas: `grado_escolar` 10, `estrato` 2, `interes_prog_robotica` 3, `n_herramientas` 2, `n_areas_interes` 2, niveles 0.
- Modas: computador 1, internet 1, olimpiadas 0.
- Vocabulario: `municipio` = {Bello, **Copacabana**, Girardota}, `genero` (4 categorías), `tipo_institucion` (2). Modas one-hot: Femenino, Copacabana, Pública.

Estos valores no dependen del número de árboles.

---

## 3. Comprobación de la API (lectura, sin modificar nada)

Se leyeron, solo en lectura, `app/models/scorer.py`, `app/schemas.py`, `app/models/predictor.py`, `app/models/loader.py` y `app/core/config.py`. En `../API-mlmodels` no se creó, modificó ni borró nada. Se comprobó que ningún archivo de ese repo cambió de fecha. **Desde este repo no se modifica la API.** Los cambios se gestionan por separado.

### 3.1 `scorer.py`: genérico, pasa la comprobación

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿Da por hechas 23 features? | **No** | `_features_puntaje` recorre `pre["numeric"]`, `pre["ordinal"]`, `pre["binary"]` y `pre["onehot_order"]` del SPEC |
| ¿Da por hecha la presencia de los 5 campos de perfil? | **No** | Ninguno de los 5 aparece en el scorer |
| ¿Tiene un vocabulario fijo de municipios? | **No** | `categorias_onehot()` lee `onehot_cats` del SPEC. Con el SPEC nuevo devuelve `['Bello', 'Copacabana', 'Girardota']` |

`version_modelo()` devuelve `global_base`. Las claves de `meta` que usa `verificar_spec()` están presentes.

### 3.2 Qué necesitaba cada archivo de la API y por qué (aplicado en §3.3, salvo las listas)

| Archivo | Qué da por hecho hoy | Qué necesita | Por qué |
|---|---|---|---|
| `app/schemas.py` | **`promedio_academico` es obligatorio**, uno de los 5 campos de perfil. Las listas son opcionales | (1) Que `promedio_academico` deje de ser obligatorio. (2) Se propuso hacer obligatorias `herramientas_conocidas` y `areas_interes`; **no se aplicó**, siguen opcionales (§5.3) | (1) Tal como estaba, la API **rechazaba con 422** al 97 % de los inscritos de Copacabana y al 63 % de los de Girardota |
| `app/core/config.py` | `model_mae = 15.00` y `model_r2 = 0.1766` fijos: son el hold-out de v2 (n = 230) sobre la subpoblación con perfil | Las métricas del modelo oficial, idealmente leídas de `SPEC.meta.metricas`: **MAE 17.86 y R² 0.050 (OOF, n = 3,050, KFold(5))** | `predictor.py` propaga `model_mae` al intervalo de quien no presentó examen. Con 15.00, el intervalo saldría **más estrecho de lo que es** |
| `app/models/scorer.py` | `sum()` del intérprete para el engagement; ruta por defecto `…/spec_potencial_v2_corrected.json` | (1) Sumar de izquierda a derecha, igual que el `reduce` del JS. (2) Apuntar `SPEC_FILE` al archivo nuevo | (1) Con Python ≥ 3.12, la paridad se rompe sin avisar (§5.2). (2) Operativo |

### 3.3 Cambios aplicados en la API (sin commit, pendientes de pull request)

Aplicados en `../API-mlmodels/copa-stem-api` por instrucción expresa, sin refactorizar ni tocar otros archivos:

| Archivo | Cambio |
|---|---|
| `app/models/spec/spec_potencial_global_base.json` | Nuevo: copia byte a byte (mismo SHA-256) del SPEC de 150 árboles. `spec_potencial_v2_corrected.json` se queda como vía de rollback |
| `app/models/scorer.py` | `SPEC_POR_DEFECTO` y el mensaje de error apuntan al SPEC nuevo. `_engagement` suma de izquierda a derecha con un bucle explícito. Docstrings actualizados (modelo global, 150 árboles, 3,050 examinados) |
| `app/schemas.py` | `promedio_academico` pasa a `Optional[float] = None`, con los mismos límites; el modelo global no lo usa. Docstrings actualizados; `EstudianteInput` ya no dice "consume el modelo v2". Las listas siguen opcionales |
| `app/core/config.py` | `model_r2` 0.1766 → **0.0504** y `model_mae` 15.00 → **17.86**. `model_version` sin cambios, porque `loader.py` lo usa para los nombres de los `.joblib` auxiliares |
| `app/main.py` | La descripción de `/docs` ya no dice "el modelo v2": nombra el modelo global base con la versión que lee del SPEC (`scorer.version_modelo()`) |
| `app/models/predictor.py` | Nueva excepción `DatoRequeridoFaltante`. `talento_oculto` y `clustering` la lanzan si falta `promedio_academico`, **antes** de evaluar la heurística, en lugar de tratar el dato ausente como criterio no cumplido |
| `app/routers/estudiantes.py` | `/talento-oculto` y `/clustering` traducen esa excepción a **422** con el motivo. En `/batch`, un estudiante sin promedio recibe `potencial` e `indice_condiciones`, y `talento_oculto`/`clustering` en `null` con la explicación en `error`. Antes perdía el registro entero, incluido su potencial |

**Verificación local:** Python 3.11.9, las versiones exactas de `requirements.txt` y una clave temporal. No se usó `api_keys.json`.

| Llamada | Resultado |
|---|---|
| `GET /health` | 200 · `scorer_potencial: global_base`, R² 0.0504, MAE 17.86 |
| `POST /potencial`, Copacabana, sin `promedio_academico` | **200** · `model_version: global_base` |
| `POST /talento-oculto` y `POST /clustering`, sin `promedio_academico` (con y sin motivación/gusto) | **422** con el motivo, las 4 llamadas |
| Las tres, con `promedio_academico` | 200 |
| `POST /batch` con un estudiante sin promedio y otro con promedio | 200. El primero trae potencial e índice de condiciones, talento y clustering en `null` y `error` explicativo; el segundo, los cuatro análisis |
| Log del servidor | 0 errores no manejados, 0 respuestas 500 |
| `verificar_spec()` | 150 árboles, referencia de 3,050 |
| Paridad del `scorer.py` editado frente al JS, en Python 3.14 | 0 discrepancias en 12,546 comparaciones: la dependencia de la versión de Python quedó eliminada |

**Configuración muerta, registrada sin corregir:** `low_confidence_r2_threshold = 0.10` en `app/core/config.py` **no se lee en ningún archivo de `app/`**. No activa ninguna advertencia: nadie debe suponer que, con R² = 0.0504, un umbral está añadiendo avisos de confianza reducida. La advertencia que llevan hoy las respuestas sin examen es el texto fijo de `predictor.py` para esa rama, que ahora cita el MAE de 17.86.

**Quedan en la API textos de `/docs` desactualizados** que no entraban en el encargo:
- El docstring de `/v1/estudiantes/potencial` dice que `promedio_academico` es obligatorio, cita "±15 puntos", "200 árboles" y "3.072 examinados".
- El docstring del módulo `schemas.py` dice "modelo v2".
- Un mensaje de log de modo demo en `main.py` menciona "modelo v2".

---

## 4. Artefactos exportados (modelo oficial de 150 árboles)

| | SPEC JSON | JS | Vigente v2 (JSON / JS) |
|---|---:|---:|---:|
| Archivo | `spec_potencial_global_base.json` | `potencial_stem_predictor_global_base.js` | `spec_potencial_v2_corrected.json` / `…_v2_corrected.js` |
| Tamaño | **1,120,939 B** (1.1 MB) | **1,356,087 B** (1.3 MB) | 700,317 B / 847,905 B |
| Árboles | 150 | 150 | 200 |
| Nodos | **44,734** (22,442 hojas) | ídem | 27,212 |
| Columnas de entrada al bosque | 19 (13 predictores, one-hot expandido) | ídem | 23 |

Frente a los vigentes, los artefactos son **1.6 veces más grandes** (antes, con 300 árboles, eran 3.2 veces). Tener menos predictores no reduce el tamaño, porque lo que manda es el número de filas y de árboles. Cada árbol global se entrena con 3,050 filas frente a las 918 de v2, así que con el mismo `min_samples_leaf = 8` tiene más hojas. **No se midió la latencia de arranque en frío**: hay que volver a correr `bench/medir_carga_spec.py` de la API con el archivo nuevo antes de desplegar.

`SPEC.meta` viaja en los dos artefactos y lleva:
- `version = "global_base"`, `n_arboles = 150`, `n_predictores = 13`, `n_entrenamiento = n_cohorte = 3,050`.
- `metricas`: OOF R² 0.050, MAE 17.86, RMSE 21.92, n = 3,050, IC 95 % y partición; hold-out R² 0.030, MAE 17.66, n = 610.
- La SD del target, la `politica_uso` y `ref_rendimiento_decision`.

---

## 5. Paridad (repetida completa sobre los artefactos de 150 árboles)

La verificación anterior validaba **esos archivos concretos**, no la paridad Python ↔ JS en general, así que se repitió entera sobre los artefactos nuevos.

### 5.1 Cargas de prueba

**2,091 cargas.** Cada una produce 6 comparaciones (puntaje crudo a precisión completa, 4 componentes y la categoría), así que son **12,546 comparaciones**.

| Tipo | n | Qué ejercita |
|---|---:|---|
| Sintéticas | 2,000 | Valores tomados del vocabulario del SPEC y de rangos plausibles. La mitad con `puntaje_obtenido` y la mitad sin él |
| One-hot | 18 | Cada categoría de `genero`, `municipio` (**incluida Copacabana**) y `tipo_institucion`, con y sin puntaje |
| Faltantes | 42 | Cada campo, de uno en uno, ausente, en `None` o en `""` |
| Todo ausente | 1 | Carga vacía: todo se imputa del SPEC |
| Listas | 10 | Vacías, `["Ninguna"]`, en el techo del engagement (12 y 9), por encima del techo, CSV plano |
| Fuera de vocabulario | 12 | `Medellín`, `Otro`, `Mixta`, `Experto`, `estrato = "abc"`, `grado = "once"`, `interés = 9`… |
| Tipos | 8 | `" Copacabana "`, `"copacabana"`, números como texto, puntaje −5 y 130 |

Tolerancias: puntaje crudo **1 × 10⁻⁹**; salida pública (2 decimales) **0.01**; categoría **igualdad exacta**.

### 5.2 Resultados

| Comparación | Cargas | Crudo bit a bit idéntico | máx. \|Δ\| crudo | Salida pública idéntica | máx. \|Δ\| salida | Categorías distintas | Discrepancias |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Scorer de la API con suma ingenua** (semántica de su runtime, Python 3.11) ↔ JS | 2,091 | **2,091** | **0** | **2,091** | **0** | **0** | **0** |
| Scorer de la API con la `sum()` de Python 3.14 ↔ JS | 2,091 | 2,091 | 0 | 2,084 | 0.01 | 0 | 7 |
| `models/deploy/potencial_stem_predictor.py` ↔ JS | 2,091 | 2,086 | 3.04 | — | 7.34 | 0 | 87 |
| Ruta de entrenamiento (script 12 + sklearn) ↔ JS, solo el crudo | 2,091 | — | 3.04 | — | — | — | 5 |

**Causas.** Ninguna se corrigió ajustando artefactos.

1. **Las 7 de Python 3.14: la suma compensada de Python ≥ 3.12.** Son empates exactos en el tercer decimal de `componente_engagement` (53.125, 40.625, 59.375, 46.875). Python ≥ 3.12 suma de forma compensada (Neumaier) y el JS de izquierda a derecha; un ulp decide el redondeo. No dependen del modelo: son las mismas 7 cargas con 300 y con 150 árboles.
   - **En producción no aparecen:** el contenedor de la API se fija con `FROM python:3.11-slim` (dato confirmado por el equipo; el `Dockerfile` no se leyó desde aquí), y en 3.11 `sum()` suma de izquierda a derecha.
   - **Fragilidad latente:** la paridad depende de la versión de Python del contenedor. Subir la imagen a 3.12 o posterior la rompería **sin ningún aviso**. La corrección va en `scorer.py` y se gestiona por separado.
2. **82 de las 87 de `potencial_stem_predictor.py`:** redondeo bancario de `round()` frente a `Math.round`, todas ≤ 0.01 y sin cambio de categoría. Ese predictor no es la ruta de la API.
3. **Las otras 5 (y las 5 de la ruta de entrenamiento): listas ausentes.** El script 12 y ese predictor cuentan una lista **ausente** como 0; el JS y la API imputan la mediana (2). Diferencia máxima: 3.04 puntos en el crudo y 7.34 en el índice. En el entrenamiento no hay ninguna lista ausente. Las listas siguen opcionales en la API, así que esta diferencia de convención sigue existiendo para las cargas que omitan el campo (§5.3).

### 5.3 Listas ausentes: estado desplegado

**Lo que quedó desplegado:** `herramientas_conocidas` y `areas_interes` **siguen siendo opcionales** en `EstudianteInput`, como estaban. En una versión anterior de este informe constaba como decisión declarada hacerlas obligatorias. **No se aplicó**: la instrucción para el cambio de la API fue dejarlas opcionales tal cual.

**Consecuencia, sin cambios respecto a antes:** si una lista **no se envía**, el scorer de la API y el JS imputan la **mediana (2)**. Si se envía `[]`, cuentan **0**. El entrenamiento (script 12) contaba una lista ausente como 0, pero el conjunto de entrenamiento tiene **0 listas ausentes**, así que la diferencia solo afecta a cargas de la API que omitan el campo. `herramientas_conocidas` es `NOT NULL` en Supabase, por lo que los datos del formulario siempre la traen. Queda como convención de servicio conocida, no como decisión cerrada.

### 5.4 Qué se hizo ante las discrepancias

No se ajustó ni el SPEC ni el JS. El diagnóstico es reproducible con `python notebooks/23_exportar_modelo_global.py --exportar --scorer-api ../API-mlmodels/copa-stem-api/app/models/scorer.py`. El scorer se lee como texto y se ejecuta en memoria, con `sys.dont_write_bytecode` y con un sustituto de `app.core.config`.

---

## 6. Privacidad

La lista de descartados y `numero_documento` solo existieron en memoria. `outputs/F23_*.json`, `F24_*.json` y `F25_*.json` contienen únicamente agregados. Los artefactos de despliegue no llevan datos de estudiantes. El único archivo con identificadores es el de la cohorte (§8), y está fuera del repo.

---

## 7. ¿Se pueden desplegar juntos los dos artefactos?

**Entre sí, sí.** Con los artefactos de 150 árboles, el JS y el scorer de la API en su runtime de producción dan **0 discrepancias en 12,546 comparaciones**. El puntaje crudo es bit a bit idéntico en las 2,091 cargas y ninguna carga cambia de categoría. Los dos llevan el mismo SPEC y el mismo bosque, y el preprocesamiento reproduce el del modelo entrenado.

**En la API:** los cambios de §3.2 están aplicados y verificados en local (§3.3), sin commit y pendientes de pull request. Antes de desplegar quedan:
1. Revisar y fusionar el pull request.
2. Medir el arranque en frío con los artefactos nuevos (1.6 veces los vigentes).
3. Actualizar los textos de `/docs` desactualizados listados en §3.3.

---

## 8. Cohorte publicada 2026: referencia de 3,050 y corrección de duplicados

**Script:** `notebooks/24_scores_cohorte_2026_ref3050.py`. **Resumen:** `outputs/F24_scores_cohorte_2026_ref3050.json`.
**Archivo:** `../_datos_privados/ml_scores_2026_ref3050.csv`, **fuera del repo**, porque identifica estudiantes. Los archivos publicados (`ml_scores_v2_corrected.csv`, `ml_scores_sin_examen.csv`) no se tocaron.

### 8.1 Cómo se generó y qué contiene

1. **Control:** con la función `puntuar` del script 18 (idéntica a la de los scripts 15 y 17), sus mismos SPECs y la referencia de F17, se reconstruyeron los dos archivos publicados. **Coinciden celda a celda**, salvo `updated_at`: 3,072 de 3,072 y 248 de 248.
2. Se repitió cambiando solo lo aprobado: referencia de 3,050 y sesión completada para los 3 duplicados. **El esquema v2 / v1_fallback no cambia.**

**Contenido:** **3,320 filas** (3,072 examinados y 248 sin examen). La columna **`modelo_version`** indica qué modelo puntuó cada fila: `v2` en 1,239 y `v1_fallback` en 2,081, igual que en lo publicado. Además lleva `referencia_rendimiento` y `poblacion`.

| Grupo | n | Cambian de categoría | Suben | Bajan |
|---|---:|---:|---:|---:|
| Examinados del conjunto limpio (sin los 3 duplicados) | 3,047 | 19 | 19 | 0 |
| Los 3 documentos duplicados corregidos | 3 | 3 | 3 | 0 |
| Los otros 22 examinados publicados | 22 | 0 | 0 | 0 |
| **Examinados, total** | **3,072** | **22** | **22** | **0** |
| **Sin examen** | **248** | **3** | **3** | **0** |

**Cuadre con el escenario A** (20 examinados y 3 sin examen):
- **Examinados, 19 + 1 = 20.** Uno de los 3 duplicados cambia por la corrección de fila y también por la referencia, y en §1.3 se contaba dentro de los 20.
- **Sin examen, 3.** El escenario A usó 249 inscritos; lo publicado son 248, porque el script 18 filtra un documento no válido.

**Talento oculto** (solo agregados):
- `es_talento_oculto` cambia en **3 examinados, y los 3 entran a la lista**; ninguno sale. La lista pasa de 609 a 612.
- En la población sin examen no hay cambios: la lista está vacía en ambas versiones.
- La causa es que el detector recibe el índice como entrada. `probabilidad_talento` cambia en 275 examinados y 1 sin examen.

### 8.2 Verificado: el reentrenamiento a 150 árboles NO afecta este archivo

No se dio por supuesto: se comprobó con `python notebooks/24_scores_cohorte_2026_ref3050.py --verificar`, que recalcula en memoria y compara sin reescribir (`outputs/F24_verificacion_archivo.json`).
- **SHA-256 del archivo:** idéntico antes y después del reentrenamiento (`2A5FC1E2…D946D26C`).
- **Recálculo después del reentrenamiento frente al archivo:** 3,320 de 3,320 filas, los mismos documentos, **0 columnas con diferencias**.
- **Por qué:** el archivo conserva el esquema publicado, que solo usa el SPEC v1 del predictor Python de despliegue, el SPEC v2 de `potencial_stem_predictor_v2.js` y la referencia de 3,050. La referencia sale de los datos, no del modelo. El script 24 no carga `modelo_global_base.joblib`.

### 8.3 Decisión institucional abierta: los 22 resultados descartados por sospecha

Los 22 **se quedan exactamente como se publicaron**, con su índice y su categoría: 0 cambios en el archivo. Se quitaron de la *referencia* porque no deben definir la vara con la que se mide a los demás. Pero **retirar en silencio un índice ya publicado sería aplicar una sanción sin que nadie lo sepa**. Eso requiere una política institucional, no un cambio de datos.

**Decisión pendiente para la fundación.** Depende de:
1. **¿Se informó a los estudiantes o a sus colegios** de que su resultado fue marcado y descartado?
2. **¿Una persona revisó cada marca**, o el descarte fue automático?
3. **¿Existe una vía de apelación**, y cuál es?

Hasta que haya respuesta, ningún script de este repo modifica, marca ni excluye esas 22 filas en archivos de puntuación. Tampoco se escriben identificadores de la lista de descartados.

---

## 9. Decisión: el modelo oficial pasa a 150 árboles

**Decisión tomada:** `n_estimators = 150` (antes 300); el resto de hiperparámetros no cambia (`max_depth = 10`, `min_samples_leaf = 8`, `random_state = 42`).

**Por qué:** los artefactos de 300 árboles eran 3.2 veces los vigentes. Un **experimento lateral** con el mismo protocolo del script 21 (`notebooks/25_experimento_n_estimators.py`, `outputs/F25_experimento_n_estimators.json`) midió el coste:

| | 300 árboles | **150 árboles (oficial)** | Diferencia |
|---|---:|---:|---:|
| R² OOF (n = 3,050, KFold(5)) | 0.0515 [0.028, 0.073] | **0.0504 [0.027, 0.072]** | −0.0011 |
| MAE OOF | 17.849 | **17.858** | +0.009 |
| RMSE OOF | 21.908 | **21.921** | +0.013 |
| R² hold-out (n = 610) | 0.0305 | **0.0296** | −0.0009 |
| MAE hold-out (n = 610) | 17.665 | **17.663** | −0.002 |
| SPEC JSON | 2,219,769 B | **1,120,939 B** | ×0.50 |
| JS | 2,677,927 B | **1,356,087 B** | ×0.51 |
| Nodos | 89,336 | **44,734** | ×0.50 |

La pérdida de R² (0.0011) es unas **veinte veces menor que la amplitud del IC 95 % de la propia métrica (±0.023)**: la diferencia no se distingue del ruido de medición. Los artefactos, en cambio, se reducen a la mitad.

**Qué se hizo:**
1. Se reentrenó `models/modelo_global_base.joblib` con 150 árboles (constante `N_ESTIMADORES` del script 21). Sus métricas embebidas se actualizaron (`hiperparametros.n_estimators = 150`). **El OOF del modelo reentrenado reproduce exactamente el del experimento** (0.0504, 17.858, 21.921). Además, el script 25, vuelto a ejecutar, comprueba que su caso de 150 coincide con la métrica embebida en el joblib.
2. Se regeneraron el SPEC JSON y el JS (§4).
3. Se repitió la verificación de paridad completa (§5).

Las réplicas de los protocolos de los scripts 12 y 13 dentro del script 21 (dispersión del target y reconciliación con F13) conservan sus 300 árboles, porque su propósito es reproducir las cifras de esos informes. El resto de cifras del informe 21 que dependen del modelo oficial se actualizaron en su anexo.

---

**Archivos de este informe** (todos nuevos o regenerados por esta línea de trabajo):
- `notebooks/23_exportar_modelo_global.py`, `24_scores_cohorte_2026_ref3050.py`, `25_experimento_n_estimators.py`
- `models/modelo_global_base.joblib` (150 árboles)
- `models/deploy/spec_potencial_global_base.json`, `potencial_stem_predictor_global_base.js`
- `outputs/F23_impacto_referencia.json`, `F23_paridad_modelo_global.json`, `F24_scores_cohorte_2026_ref3050.json`, `F24_verificacion_archivo.json`, `F25_experimento_n_estimators.json`
- Fuera del repo: `../_datos_privados/ml_scores_2026_ref3050.csv`, sin regenerar.

No se modificó ningún archivo existente del repo. No se tocó la API, no se hizo commit y no se desplegó nada.
