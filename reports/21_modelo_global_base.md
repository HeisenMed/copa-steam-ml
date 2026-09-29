# Informe 21 — Modelo global base (sin perfil académico)

**Copa STEM 2026 · Fundación SapienceLab**
Script: `notebooks/21_modelo_global_base.py` · Artefactos: `models/modelo_global_base.joblib`, `outputs/F21_modelo_global_base.json`
Fecha: 2026-09-28 · `random_state = 42`

---

> **ANEXO (2026-09-28): el modelo oficial tiene ahora 150 árboles.** Por decisión del informe 23, §9, se reentrenó `models/modelo_global_base.joblib` con `n_estimators = 150` en lugar de 300; todo lo demás, igual. La pérdida (R² OOF −0.0011) es unas veinte veces menor que el IC 95 % de la métrica, y los artefactos de despliegue ocupan la mitad. **El cuerpo de este informe describe la versión de 300 árboles.** `outputs/F21_modelo_global_base.json` ya contiene las cifras de 150. Cifras oficiales vigentes:
>
> | Cifra | 300 (cuerpo del informe) | **150 (oficial)** | n · partición |
> |---|---:|---:|---|
> | R² OOF [IC 95 %] | 0.052 [0.028, 0.073] | **0.050 [0.027, 0.072]** | 3,050 · KFold(5) |
> | MAE / RMSE OOF | 17.85 / 21.91 | **17.86 / 21.92** | 3,050 · KFold(5) |
> | R² / MAE / RMSE hold-out | 0.031 / 17.66 / 21.82 | **0.030 / 17.66 / 21.83** | 610 · 20 % estratificado |
> | OOF por municipio (R² / MAE) | Bello 0.080/15.56 · Gir. 0.057/18.91 · Copa. 0.019/18.27 | **Bello 0.079/15.57 · Gir. 0.055/18.92 · Copa. 0.019/18.28** | 729 / 1,088 / 1,233 |
> | Dependencia parcial Gir. / Bello / Copa. | 42.30 / 41.74 / 40.56 | **42.28 / 41.80 / 40.55** | 3,050 |
> | Copacabana − Girardota (DP) | −1.75 | **−1.73** | 3,050 |
> | Ídem, con mes de inscripción (diagnóstico) | −4.20 | **−4.10** | 3,050 |
> | Sensibilidad sin las 14 sesiones > 3 h: R² / MAE / Copa. − Gir. | 0.050 / 17.80 / −1.67 | **0.051 / 17.79 / −1.63** | 3,036 · KFold(5) |
> | Permutación de `municipio` (hold-out, 20 repeticiones) | 0.003 ± 0.009 | **0.005 ± 0.009** | 610 |
> | R² de este modelo sobre la SD de v1 (§3.3) | 0.101 | **0.100** | — |
>
> Ninguna conclusión cambia: el orden Girardota > Bello > Copacabana se mantiene, la contribución propia del municipio sigue entre una cuarta parte y la mitad de la brecha cruda, y la caída de R² frente a v1 sigue explicándose por la varianza del target. Las réplicas de los protocolos de los scripts 12 y 13 (§3.3 dispersión A → B y §4.1 reconciliación con F13) conservan sus 300 árboles, porque su propósito es reproducir esas cifras.

## 0. Resumen

- Se entrenó **un solo modelo** que puntúa a todos los estudiantes de Bello, Girardota y Copacabana con el **mismo conjunto de 13 predictores**. No usa las 5 preguntas de perfil académico. Reemplaza a v1 en producción. v2 no se tocó y queda como artefacto de investigación.
- Se entrenó con **3,050 estudiantes** (Bello 729, Girardota 1,088, Copacabana 1,233). Los conteos locales coinciden exactamente con Supabase, una vez deduplicados y sin los 22 resultados descartados por sospecha.
- **Métrica oficial: R² out-of-fold = 0.052, IC 95 % [0.028, 0.073], MAE 17.85, RMSE 21.91; n = 3,050, partición KFold(5).** Es menor que el R² OOF de v1 (0.084, n = 1,748). La caída **se explica por completo por la menor dispersión del target** (SD 23.10 en la población de v1 frente a 22.49 aquí). El error absoluto no empeoró: RMSE 22.1 → 21.91 (§3.3).
- El resultado **no es sensible** a las 14 sesiones de más de 3 h: sin ellas, R² OOF 0.050 y MAE 17.80 (n = 3,036), con la misma ordenación de municipios (§3.5).
- El modelo sitúa a **Copacabana por debajo de Girardota**, igual que las medias crudas. Pero lo hace por unos **1.7 puntos, frente a 8.02 en las medias crudas**: la contribución propia del municipio es más o menos una cuarta parte de lo que sugiere la comparación cruda. Si se controla el mes de inscripción (solo como diagnóstico), la distancia sube a 4.2 puntos: la magnitud depende de la especificación, el orden no (§4).
- **Reconciliación con F13:** la importancia de `municipio` cae de ΔR² 0.081 (dataset A) a 0.003 (aquí). No es porque `municipio` actuara en A como proxy de la ola de examen. Es porque A solo contenía la cohorte temprana de Girardota. Dentro de Girardota el puntaje medio varía unos 20 puntos entre cohortes de inscripción, y esa variación **es entre instituciones, no dentro de ellas** (§4.1).
- **No hay offset por municipio, ni constante de calibración, ni caso especial para Copacabana.**

---

## 1. Por qué un único modelo sin perfil

La cobertura de las 5 preguntas de perfil académico es muy desigual entre municipios:

| Municipio | Estudiantes puntuados | Con perfil | Cobertura |
|---|---:|---:|---:|
| Bello | 729 | 716 | 98 % |
| Girardota | 1,088 | 412 | 38 % |
| Copacabana | 1,233 | 0 | 0 % |

*(Cobertura calculada sobre los estudiantes puntuados de este conjunto. Las cifras 98/37/3 % del encargo se calcularon sobre todos los inscritos.)*

Un modelo que depende del perfil solo puede servir a una fracción de la población, y esa fracción está sesgada de forma distinta en cada municipio. La alternativa sería enrutar: v2 para quien tiene perfil y otro modelo para el resto. Eso pondría en la misma columna `puntaje_estimado` números de procedencia distinta, y el municipio decidiría qué modelo puntúa a cada estudiante. La decisión se tomó por **comparabilidad entre municipios**: todos los estudiantes se puntúan con el mismo modelo y las mismas variables.

---

## 2. Inventario de datos (paso 1)

**Fuente:** `data/copa_stem_dataset_2026-08.csv.csv` (3,077 filas, 31 columnas). Es el único export local que contiene los tres municipios.

**Verificaciones:**
- `puntaje_obtenido` no tiene nulos y va de 0 a 100. `porcentaje == puntaje_obtenido` en las 3,077 filas, lo que confirma `puntaje_total = 100`.
- No hay documentos de prueba (0 filas retiradas).
- Las filas puntuadas de Copacabana están presentes en local (1,238 crudas).

**PII:** el export trae `institucion_educativa`, que se elimina al cargar. `nombres` y `apellidos` no vienen en este export. `numero_documento` se usa solo en memoria, como clave para deduplicar y excluir, y nunca entra en las features ni en ninguna salida. La lista de descartados se leyó desde `../_datos_privados/descartados.csv` (fuera del repo) y solo se usó en memoria. No se copió ni se escribió en ningún fichero. Las salidas contienen únicamente conteos agregados.

### 2.1 Deduplicación

Hay 5 documentos duplicados, todos de Copacabana. Cada par comparte exactamente el mismo `created_at`.

**Columnas de sesión disponibles en local:** `tiempo_usado_segundos`, `cambios_pestana`, `intentos_copiar`, `intentos_pegar`, `intentos_click_derecho`, `created_at`. **No hay** `estado`, `finalizada_en` ni un conteo de preguntas respondidas.

Los 5 pares, sin identificadores:

| Par | Fila A: puntaje · tiempo | Fila B: puntaje · tiempo | Fila retirada |
|---|---|---|---|
| 1 | 15 · vacío | 0 · 4,876,349 s | B |
| 2 | 0 · 4,661,685 s | 45 · vacío | A |
| 3 | 35 · vacío | 0 · 4,703,849 s | B |
| 4 | 0 · 4,601,279 s | 50 · vacío | A |
| 5 | 5 · 4,412,089 s | 75 · vacío | A |

**Regla usada: se conserva la sesión completada.** En cada par, una fila registra entre 4.4 y 4.9 millones de segundos, es decir, 51 a 56 días. Ningún examen dura eso: el p99 de Copacabana es de unos 5,400 s. Se interpreta como una sesión que el estudiante no cerró y que el sistema cerró semanas después. Se retira la fila con `tiempo_usado_segundos > 10,800` (3 h). La otra fila del par tiene el tiempo vacío, algo normal en esa ola: 280 filas de Copacabana no tienen tiempo registrado y su puntaje medio (39.52) es igual al del municipio. El script comprueba que la regla deja exactamente una fila por documento y **se detiene si no es así**. No recurre a ninguna regla alternativa.

- **Filas retiradas por municipio:** Bello 0 · Girardota 0 · Copacabana 5.
- **Cómo difiere de los scripts 11 y 14:** esos scripts usan `drop_duplicates(keep="first")`, que conserva la primera fila del export. En 3 de los 5 pares, la primera fila es la sesión abandonada, con 0 o 5 puntos. `dataset_B_completo.csv` y cualquier análisis derivado de él arrastran esas 3 filas.
- **Advertencia de transparencia:** la regla no usa el puntaje, pero en estos 5 pares **retira las mismas filas que retiraría "conservar el máximo puntaje"**, porque las 5 sesiones abandonadas son las de 0–5 puntos. El resultado numérico es idéntico. Lo que cambia es el criterio: la regla se sostiene por la duración imposible, no por elegir el mejor de dos puntajes. El impacto es de 5 filas de 3,072, despreciable en las métricas.
- **Observación:** en el conjunto final de 3,050 quedan 14 sesiones **no duplicadas** con más de 3 h (Bello 6, Girardota 4, Copacabana 4). 8 de ellas tienen 0 puntos y las otras 6 entre 10 y 35. En todo el export, 13 de las 19 sesiones de más de 3 h tienen 5 puntos o menos. Siguen en el entrenamiento: la prueba de sensibilidad de §3.5 muestra que retirarlas no cambia el resultado. Si son intentos abandonados, deberían tratarse con una regla explícita en una próxima iteración.

### 2.2 Exclusión de descartados por sospecha

La lista privada contiene 22 documentos. Los 22 tienen fila puntuada en el export.

| Municipio | Excluidos | Con perfil | Sin perfil | Tasa sobre deduplicados |
|---|---:|---:|---:|---:|
| Bello | 18 | 18 | 0 | 2.41 % (18/747) |
| Girardota | 4 | 2 | 2 | 0.37 % (4/1,092) |
| Copacabana | 0 | 0 | 0 | 0.00 % (0/1,233) |
| **Total** | **22** | 20 | 2 | |

Los 2 descartados de Girardota sin perfil explican las 2 filas que no cuadraban en la primera revisión (678 locales frente a 676 en Supabase). Los ~5 casos "marcados" de Copacabana mencionados en el encargo no están en la lista de descartados: aparecen como marcados, pero no como `sospechoso_descartado`.

### 2.3 Conteo final contra Supabase

| Municipio | Crudo local | Tras dedup | Tras excluir | Supabase | Diferencia | Con perfil | Sin perfil |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bello | 747 | 747 | **729** | 729 | 0 | 716 | 13 |
| Girardota | 1,092 | 1,092 | **1,088** | 1,088 | 0 | 412 | 676 |
| Copacabana | 1,238 | 1,233 | **1,233** | 1,233 | 0 | 0 | 1,233 |
| **Total** | 3,077 | 3,072 | **3,050** | ~3,050 | 0 | 1,128 | 1,922 |

Sin discrepancias.

**Brecha de selección dentro de Girardota** en este conjunto: los estudiantes con perfil promedian 35.51 (n = 412) y los que no lo tienen 47.26 (n = 676). La diferencia es de **11.75 puntos** y coincide con la cifra de Supabase.

---

## 3. Modelo y métricas (pasos 2 y 3)

### 3.1 Features y modelo

La ingeniería de features es la de `12_experimento_reentrenamiento.py`, cargada con `importlib` igual que en el script 13: `construir_crudos(df, incluir_nuevas=False)` + `construir_pipeline`. No se reimplementó nada.

- **13 predictores** (bloque del dataset A): `grado_escolar`, `estrato`, `interes_prog_robotica`, `n_herramientas`, `n_areas_interes`, `nivel_programacion_ord`, `nivel_robotica_ord`, `computador_bin`, `internet_bin`, `olimpiadas_bin`, `genero`, `municipio`, `tipo_institucion` (one-hot). Ninguna de las 5 preguntas de perfil entra: el script lo comprueba con un `assert`.
- **Modelo:** `RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=8, random_state=42)`, la configuración del script 12, con imputación dentro del `Pipeline`. No se hizo búsqueda de hiperparámetros.
- **Modelo de producción:** ajustado con las 3,050 filas. Las métricas de abajo salen de validación cruzada y de un hold-out, nunca de ese ajuste.

### 3.2 Métricas

| Métrica | Valor | n | Partición |
|---|---:|---:|---|
| **R² OOF (oficial)** | **0.052** | 3,050 | out-of-fold, KFold(5, shuffle, rs=42), todo el conjunto |
| IC 95 % del R² OOF | [0.028, 0.073] | 3,050 | bootstrap (2,000 remuestreos) sobre las predicciones OOF |
| MAE OOF | 17.85 | 3,050 | ídem |
| RMSE OOF | 21.91 | 3,050 | ídem |
| R² hold-out | 0.031 | 610 | hold-out 20 % estratificado por quintil (rs=42); entrenado con 2,440 |
| MAE hold-out | 17.66 | 610 | ídem |
| RMSE hold-out | 21.82 | 610 | ídem |

El hold-out tiene 149 estudiantes de Bello, 209 de Girardota y 252 de Copacabana. Con un único split de n = 610, el R² de hold-out queda dentro del rango esperable por azar. La cifra de referencia es la OOF.

**OOF por subgrupo** (el mismo vector de predicciones OOF, filtrado por grupo):

| Subgrupo | R² | MAE | n |
|---|---:|---:|---:|
| Bello | 0.080 | 15.56 | 729 |
| Girardota | 0.057 | 18.91 | 1,088 |
| Copacabana | 0.019 | 18.27 | 1,233 |
| Con perfil | 0.046 | 16.12 | 1,128 |
| Sin perfil | 0.052 | 18.86 | 1,922 |

El R² de cada subgrupo se mide contra la varianza *dentro* del subgrupo, así que es más bajo que el global. En Copacabana el modelo apenas explica varianza entre estudiantes.

**Importancia por permutación** (sobre el hold-out, n = 610, 20 repeticiones, ΔR²): `grado_escolar` 0.044 · `n_areas_interes` 0.029 · `olimpiadas_bin` 0.017 · `n_herramientas` 0.017 · `interes_prog_robotica` 0.014 · `municipio` 0.003 (±0.009) · el resto ≤ 0.003.

### 3.3 Comparación con v1

| Modelo | R² OOF | IC 95 % | MAE | RMSE | n | Población de entrenamiento |
|---|---:|---|---:|---:|---:|---|
| v1 (oficial, informe 20) | 0.084 | [0.053, 0.116] | 18.1 | 22.1 | 1,748 | 1,748 estudiantes de **dos** municipios (Copacabana y Girardota) |
| Global base (este) | 0.052 | [0.028, 0.073] | 17.85 | 21.91 | 3,050 | 3,050 estudiantes de **tres** municipios (Bello, Girardota y Copacabana) |

Ambas cifras son OOF con KFold(5), pero sobre poblaciones distintas. El R² de este modelo es menor que el de v1 y los intervalos apenas se solapan (0.053 a 0.073). **No se debe presentar como "el modelo nuevo empeora" ni como "mejora"**: el R² se normaliza por la varianza de *cada* población, y la varianza cambió.

**Dispersión del target** (desviación estándar poblacional de `puntaje_obtenido`, ddof = 0):

| Población | SD | n | Fuente |
|---|---:|---:|---|
| v1 (entrenamiento) | 23.10 | 1,748 | `models/deploy/puntaje_estimado.csv`, columna `puntaje_real` |
| Dataset A (baseline) | 24.22 | 1,735 | `data/dataset_A_baseline.csv` |
| Dataset B (completo) | 22.66 | 3,072 | `data/dataset_B_completo.csv` |
| Este modelo | 22.49 | 3,050 | conjunto de §2.3 |

La SD de v1 que se deduce de sus métricas oficiales, 22.1 / √(1 − 0.084) = 23.09, coincide con la medida (23.10). La identidad se cumple.

**Descomposición de la caída A → B (0.086 → 0.053)** con la identidad RMSE = SD · √(1 − R²). Reproduje el protocolo del script 12 (KFold(5) sobre el train del 80 %): las medias por fold, 0.086 y 0.053, salen idénticas a `F12_experimento_reentrenamiento.csv`. Para que la identidad se cumpla exactamente, la descomposición usa el R² OOF agregado de esos mismos folds:

| | A | B |
|---|---:|---:|
| R² (media de folds, script 12) | 0.086 | 0.053 |
| R² OOF agregado | 0.0895 | 0.0600 |
| RMSE OOF | 23.04 | 22.09 |
| SD del train | 24.14 | 22.78 |
| n train | 1,388 | 2,457 |

- **Con la RMSE de A y la SD de B**, el R² sería 1 − (23.04 / 22.78)² = **−0.022**. Solo el cambio de varianza bajaría el R² 0.112 puntos, **más que la caída entera** (0.0295).
- La parte "de calidad" es −0.082, es decir, favorable: el error absoluto **bajó**, de 23.04 a 22.09 puntos de RMSE.
- En el orden inverso (RMSE de B con la SD de A: R² = 0.163), la varianza explica 0.103 y el error −0.073. La conclusión no depende del orden.

**Con v1 frente a este modelo ocurre lo mismo:** con la RMSE de este modelo (21.91) sobre la SD de v1 (23.10), el R² sería **0.101**, por encima del 0.084 de v1. En el orden inverso, la RMSE de v1 (22.1) sobre la SD de aquí (22.49) da 0.034, por debajo del 0.052 de este modelo.

**Conclusión:** la varianza del target **explica toda la caída de R²**, tanto de A a B como de v1 a este modelo. No hay pérdida de exactitud en puntos: la RMSE bajó. Lo que tampoco se puede afirmar es que el modelo nuevo "explique más": en relación con la dispersión disponible explica algo menos (RMSE/SD 0.974 frente a 0.957 en v1). El R² no sirve para comparar estas poblaciones, y la magnitud que se debe citar es el error en puntos, junto con su n y su partición.

- El MAE también está en puntos y depende de la dispersión de cada población, así que entre poblaciones solo es orientativo.
- Hay un resultado consistente con este: el script 12 ya obtuvo CV R² = 0.053 con las mismas features sobre `dataset_B_completo.csv` (3,072 filas, tres municipios, CV sobre el 80 % de train).

Sea cual sea la comparación, el modelo explica alrededor del **5 % de la varianza** del puntaje. Por eso la política de uso es aún más importante: `puntaje_estimado` **nunca** se usa como cifra puntual para un individuo **ni** para ordenar individuos.

### 3.4 Métricas embebidas en el artefacto

`models/modelo_global_base.joblib` es un `dict` con las claves `modelo` (el `Pipeline`), `metricas` (OOF, hold-out, OOF por subgrupo, población de entrenamiento y comparación con v1, cada cifra con su n y partición), `n_entrenamiento`, `n_por_municipio`, `predictores`, `preprocesamiento_crudo` (la referencia a `construir_crudos` del script 12), `hiperparametros`, `politica_uso`, `sin_ajustes_por_municipio = True`, `sklearn_version` (1.9.0), `version = "global_base"` y la fecha de entrenamiento. Dentro de `metricas` también va `sd_target`, la SD de cada población de §3.3, para que el R² pueda leerse en su contexto. La métrica viaja con el artefacto.

### 3.5 Sensibilidad a las 14 sesiones de más de 3 h

Se repite el mismo pipeline y el mismo protocolo sin las 14 sesiones no duplicadas de más de 3 h (§2.1). El modelo oficial no se modifica.

| | Con las 14 (oficial) | Sin las 14 |
|---|---:|---:|
| n | 3,050 | 3,036 |
| R² OOF | 0.0515 | 0.0500 |
| MAE OOF | 17.85 | 17.80 |
| RMSE OOF | 21.91 | 21.85 |
| DP Bello | 41.74 | 42.07 |
| DP Girardota | 42.30 | 42.39 |
| DP Copacabana | 40.56 | 40.72 |
| Copacabana − Girardota (DP) | −1.75 | −1.67 |

*Partición:* out-of-fold KFold(5, shuffle, rs=42) sobre cada conjunto. La dependencia parcial (DP) se calcula con el modelo ajustado sobre todas las filas de cada conjunto y se promedia sobre esas mismas filas (n = 3,050 y n = 3,036).

**Conclusión:** la diferencia no es material. El R² varía 0.0015, bastante menos que la amplitud de su IC 95 %, el MAE 0.05 puntos, y el orden y la distancia entre municipios se mantienen (−1.75 frente a −1.67). El resultado no depende de esas 14 sesiones, y **el modelo oficial se queda como está**.

---

## 4. Efecto municipio aprendido (paso 4): hallazgo, no objetivo

No se añadió ninguna restricción ni ajuste para forzar un orden.

| Municipio | Media cruda (todos) | Media cruda sin perfil | Dependencia parcial (n = 3,050) | DP sobre filas sin perfil (n = 1,922) | Residuo medio OOF |
|---|---:|---:|---:|---:|---:|
| Bello | 43.26 (n=729) | 40.38 (n=13) | 41.74 | 41.33 | −0.17 |
| Girardota | 42.81 (n=1,088) | 47.26 (n=676) | **42.30** | 41.79 | +0.11 |
| Copacabana | 39.40 (n=1,233) | 39.40 (n=1,233) | **40.56** | 40.28 | −0.23 |

*Dependencia parcial:* se fija `municipio = m` en **todas** las filas, sin tocar el resto de variables, y se promedia la predicción del modelo final. La diferencia entre municipios es el efecto que el modelo atribuye al municipio manteniendo fija la composición de la población.

**Orden aprendido:** Girardota (42.30) > Bello (41.74) > Copacabana (40.56).

**¿El modelo pone a Copacabana por debajo de Girardota, como sugieren las medias crudas? Sí.** El orden coincide, pero la magnitud no:

- **Medias crudas, comparación emparejada sin perfil:** Copacabana 39.24 frente a Girardota 47.26 = **8.02 puntos** por debajo (cifra de Supabase; los 39.24 incluyen las 5 sesiones abandonadas). Sobre el conjunto deduplicado de este informe, Copacabana promedia 39.40 y la brecha es de 7.86 puntos.
- **Dependencia parcial del modelo:** Copacabana 40.56 frente a Girardota 42.30 = **1.75 puntos** por debajo (1.50 si se calcula solo sobre las filas sin perfil).

**La brecha cruda mezcla el municipio con las demás variables**: grado, experiencia previa, áreas de interés, herramientas conocidas, etc. Los estudiantes de cada municipio difieren en esas variables, y la media cruda atribuye todo eso al municipio. Una vez el modelo tiene en cuenta esas variables, **la contribución propia del municipio es aproximadamente una cuarta parte** de lo que indican las medias crudas: 1.75 de 8.02, un 22 %. La importancia por permutación de `municipio` (ΔR² = 0.003 ± 0.009, n = 610) confirma que el municipio pesa poco una vez se conocen las demás variables.

**La magnitud depende de la especificación; el orden, no.** Si se añade al modelo el mes de inscripción (solo como diagnóstico, §4.1), la dependencia parcial pasa a Girardota 43.97, Bello 43.39, Copacabana 39.77, y Copacabana queda **4.20 puntos** por debajo de Girardota (OOF R² 0.102, n = 3,050). Según se controle o no la cohorte de inscripción, la contribución propia del municipio va de una cuarta parte a la mitad de la brecha cruda. En ninguna especificación llega a los 8 puntos, y en todas Copacabana queda por debajo de Girardota.

**Por esto no se codificó ningún offset fijo por municipio.** Un offset tomado de las medias crudas penalizaría a Copacabana entre dos y cuatro veces por encima de lo que los datos respaldan. Además, el propio tamaño del efecto se mueve según qué se controle. Cualquier constante fija quedaría congelada aunque cambie la composición de la población. El modelo aprende el efecto de los propios estudiantes de cada municipio.

**Advertencia sobre la categoría Copacabana:** ver §8. Si los puntajes de Copacabana están inflados, las distancias de arriba subestiman la real.

### 4.1 Reconciliación: por qué `municipio` pesa 27 veces menos aquí que en F13

`outputs/F13_importancias.csv` sitúa a Municipio **en el primer puesto de 13** en el dataset A, con ΔR² = 0.0809 ± 0.0214. Aquí queda en 0.0025 ± 0.0089. Para comparar en igualdad de condiciones, todas las filas de esta tabla usan **el protocolo exacto del script 13**: hold-out del 20 % estratificado por quintil, RF del script 12, permutación de la columna cruda con 30 repeticiones y scoring R².

| Escenario | n filas | n test | ΔR² municipio | ΔR² variable añadida |
|---|---:|---:|---:|---:|
| A: réplica de F13 (Copacabana + Girardota, 22 may – 5 jun) | 1,735 | 347 | **0.0809 ± 0.0214** | — |
| B completo (tres municipios, sin excluir descartados) | 3,072 | 615 | 0.0165 ± 0.0072 | — |
| Este modelo | 3,050 | 610 | **0.0025 ± 0.0089** | — |
| Copacabana + Girardota, **todos los meses** (sin Bello) | 2,321 | 465 | **0.0000 ± 0.0072** | — |
| Copacabana + Girardota, **solo la ventana de A** (≤ 5 jun) | 1,733 | 347 | **0.0629 ± 0.0190** | — |
| Este modelo + mes de inscripción | 3,050 | 610 | 0.0414 ± 0.0146 | mes 0.0798 ± 0.0196 |
| Este modelo + días desde el inicio | 3,050 | 610 | 0.0134 ± 0.0077 | días 0.1478 ± 0.0226 |
| Copacabana + Girardota + mes | 2,321 | 465 | 0.0997 ± 0.0254 | mes 0.1636 ± 0.0221 |

La réplica reproduce F13 dígito a dígito (0.0809 ± 0.0214, n_test = 347), así que la discrepancia no es un error de ninguno de los dos informes.

**Qué la produce:**

1. **Añadir Bello no es la causa.** Sin Bello, pero con todos los meses de Girardota, la importancia ya es 0.000.
2. **La causa es añadir las cohortes posteriores de Girardota.** Si Copacabana + Girardota se restringen a la ventana de A, la importancia vuelve a 0.063. El puntaje medio de Girardota cambia mucho según el mes de inscripción:

   | Cohorte | n | Media | Con perfil |
   |---|---:|---:|---:|
   | Girardota, mayo | 350 | 54.57 | 0 |
   | Girardota, junio | 309 | 40.16 | 0 |
   | Girardota, julio | 429 | 35.12 | 412 |
   | Copacabana, mayo | 943 | 40.08 | 0 |
   | Copacabana, junio | 290 | 37.17 | 0 |
   | Bello, julio | 729 | 43.26 | 716 |

   El dataset A contenía solo la cohorte temprana de Girardota, la de 54.6 puntos. En A, la etiqueta "Girardota" separaba a esas instituciones de Copacabana. Con los tres meses juntos, la categoría Girardota promedia cohortes que difieren unos 20 puntos, su media se acerca a la de Copacabana, y la etiqueta deja de separar. Si se añade el mes al modelo, `municipio` recupera importancia (0.041 en el conjunto global, 0.100 en Copacabana + Girardota).

**La hipótesis de la ola de examen, comprobada:** la hipótesis era que en A `municipio` separaba dos condiciones de medición, porque Copacabana y Girardota estarían en ventanas distintas. **Se descarta en esa forma.** En A los dos municipios comparten la misma ventana de dos semanas (Copacabana del 22 may al 4 jun, Girardota del 25 may al 5 jun), así que `municipio` no pudo actuar allí como proxy de la ola. Los desfases de ventana (Girardota hasta el 24 jul, Bello solo en julio) aparecen al pasar de A al conjunto completo, y el efecto va al revés de lo planteado: no inflan la importancia de `municipio` en A, sino que la diluyen en el conjunto global.

**¿El mes es una condición de medición o una cuestión de qué instituciones se inscribieron en cada momento?** Se probó dentro de Girardota. La institución se usó solo en memoria, como código anónimo, y el informe no la identifica.

- Girardota tiene 9 instituciones. 4 aparecen en más de un mes y suman 575 filas.
- En esas 4, las medias crudas por mes son 51.86 (mayo, n = 274), 37.44 (junio, n = 221) y 38.88 (julio, n = 80).
- **Dentro de cada institución**, la desviación media respecto a la media de su institución es +0.24 (mayo), −0.32 (junio) y +0.08 (julio). Es prácticamente cero.
- Es decir, los estudiantes de una misma institución puntúan igual se inscriban en mayo o en junio. La diferencia de 14 puntos entre meses viene de **qué instituciones se inscribieron en cada mes**, no del momento de la medición.

La telemetría apunta en la misma dirección. En la cohorte de mayo de Girardota, el 0 % de las sesiones registra algún evento (cambio de pestaña o clic derecho), frente al 14.6 % en junio y el 23.1 % en julio. Esa cohorte se midió sin instrumentación efectiva y aun así, dentro de una misma institución, no puntúa más que las cohortes instrumentadas.

**Veredicto:** la reconciliación es **concluyente en lo principal**. La importancia de `municipio` en F13 era una propiedad de la muestra A: su Girardota eran solo las instituciones tempranas y de puntaje alto. No es una propiedad estable del municipio, y la hipótesis de la ola de examen no la explica. La atribución a la composición por instituciones está **bien apoyada pero no es definitiva**: la prueba intra-institución descansa en 4 de 9 instituciones de Girardota y 575 filas, y el mes de inscripción es un proxy de la fecha de examen, no la fecha de examen. El mes no se incorpora al modelo de producción: no generaliza a cohortes futuras y solo codifica qué colegios se inscribieron cuándo.

**Consecuencia para §5.2:** la brecha "de selección" de 11.75 puntos dentro de Girardota se comporta igual. Todo el grupo con perfil de Girardota se inscribió en julio (412 de 412), después de que se añadieran las preguntas al formulario. En las 4 instituciones que tienen estudiantes con y sin perfil (760 filas), las medias crudas son 35.45 y 48.99, pero la desviación intra-institución es +0.54 y −0.60. La brecha es **entre instituciones** (cuáles se inscribieron después del cambio de formulario), no una autoselección de los estudiantes que responden. La conclusión de §5.2 no cambia: la subpoblación con perfil sigue sin ser representativa.

---

## 5. Limitaciones de v2 (paso 5, solo lectura)

v2 no se reentrenó ni se modificó. Se leyeron `notebooks/11_preparar_experimento_reentrenamiento.py`, `notebooks/14_optimizacion_hiperparametros.py`, `outputs/F14_optimizacion_v2.json` y `data/dataset_C_perfil.csv`.

**5.1 v2 se entrenó con resultados descartados por sospecha.** v2 se entrenó con `dataset_C_perfil.csv`: 1,148 filas (918 de train y 230 de hold-out). Ese dataset sale del export de agosto, que **no tiene la columna `sospechoso_descartado`**. Los scripts 11 y 14 solo filtran documentos de prueba, duplicados (`keep="first"`) y puntaje nulo. Al cruzar con la lista de descartados, **20 de las 1,148 filas de v2 son resultados descartados** (Bello 18, Girardota 2, Copacabana 0). Sin ellas quedan 1,128 filas. Esa es exactamente la diferencia entre la n = 1,148 que citan las notas del repo y los 1,128 estudiantes con perfil y puntaje real de Supabase. **Conclusión: el conjunto de entrenamiento de v2 incluyó filas marcadas `sospechoso_descartado`.** Los 20 descartados también pudieron caer en su hold-out de 230, que tampoco los excluía.

**5.2 El MAE de v2 no es comparable con el de v1.** El MAE de 15.00 de v2 (R² 0.177, n = 230, hold-out del 20 % de `dataset_C_perfil.csv`) se midió sobre la **subpoblación que respondió el perfil**. Esa subpoblación no es representativa: dentro de Girardota, quien respondió el perfil promedia 11.75 puntos menos que quien no lo hizo, una brecha mayor que la que separa a los municipios. Según §4.1, esa brecha es entre instituciones: el grupo con perfil son los colegios que se inscribieron en julio. Además, el 64 % de ese conjunto es de Bello (734 de 1,148), cuya dispersión del puntaje es menor (MAE OOF de este modelo: 15.56 en Bello frente a 18.27–18.91 en los otros dos). El 18.45 de v1 (informe 15) y el 18.1 oficial se midieron sobre otra población. **15.00 y 18.45 no están medidos sobre poblaciones comparables y nunca deben presentarse como una comparación limpia** ni como "v2 reduce el error 3.45 puntos".

---

## 6. El coste honesto de esta decisión

Es probable que los estudiantes que **sí** tienen el perfil (principalmente Bello y un 38 % de Girardota) obtuvieran un puntaje más preciso con v2. En la muestra fija del script 12, las 5 preguntas aumentan el R² de CV de 0.098 a 0.180. **La magnitud de esa pérdida en producción es desconocida**, porque el hold-out de v2 no es comparable (§5): es otra población, otra partición e incluye los 20 descartados. Este modelo obtiene un MAE OOF de 16.12 (n = 1,128) en la subpoblación con perfil. Esa cifra **no se puede restar** del 15.00 de v2 para estimar la pérdida, porque ni las filas ni la partición coinciden.

Lo que se gana: todos los estudiantes, de los tres municipios, se puntúan con el mismo instrumento, y un puntaje de Bello y uno de Copacabana tienen la misma procedencia.

---

## 7. Las preguntas de perfil tienen señal real: la acción es recogerlas

Las preguntas de perfil aportan información que este modelo no tiene. Solo `promedio_academico` aporta **ΔR² = 0.104** (importancia por permutación sobre el hold-out del 20 % de `dataset_C_perfil.csv`, n ≈ 230 de un total de 1,148 que incluye los 20 descartados; informe 13, `outputs/F13_importancias.csv`), más que cualquier variable de este modelo. El problema no es de modelado sino de **recolección**: con 0 % de cobertura en Copacabana, ningún modelo puede usarlas de forma comparable.

**Acción recomendada:** hacer **obligatorias las 5 preguntas de perfil en la inscripción** de la próxima cohorte. Con cobertura completa en los tres municipios, un único modelo con perfil sería a la vez más preciso y comparable.

---

## 8. Asimetría antifraude: Copacabana probablemente está inflada

- La ola de examen de Copacabana va del **2026-05-22 al 2026-06-10** (verificado en `created_at` local), antes de que existiera el sistema antifraude.
- Tasa de resultados descartados por sospecha en este conjunto: **Bello 2.41 % (18/747), Girardota 0.37 % (4/1,092), Copacabana 0.00 % (0/1,233)**. En Supabase, Bello marca 18/747 con el sistema activo y Copacabana ~5/1,238. La tasa baja de Copacabana refleja que **faltaba la instrumentación, no que faltara la trampa**.
- Matiz: Copacabana sí registraba telemetría. Hay sesiones con algún evento en el 10.6 % de mayo y el 31.0 % de junio. Lo que no hubo fue revisión y descarte. En Girardota, la cohorte de mayo (0 % de sesiones con eventos) no puntúa más que las posteriores dentro de una misma institución (§4.1). Eso no respalda que la falta de instrumentación infle los puntajes, al menos en Girardota. La inflación de Copacabana es plausible, pero no está demostrada.
- Consecuencia: si Copacabana tuviera la tasa de trampa de Bello, unos 30 resultados de sus 1,233 estarían inflados y siguen en el entrenamiento. Eso **sesga hacia arriba la categoría `municipio = Copacabana`** que aprende el modelo. La distancia real frente a Girardota probablemente es mayor que los 1.75 puntos de §4. No se aplicó ninguna corrección, porque no hay forma de identificar esos resultados con los datos disponibles, y el encargo prohíbe ajustes por municipio.

---

## 9. Artefactos

| Fichero | Contenido |
|---|---|
| `notebooks/21_modelo_global_base.py` | Script reproducible (inventario, entrenamiento, efecto municipio, lectura de v2 y análisis complementarios de §3.3, §3.5 y §4.1) |
| `models/modelo_global_base.joblib` | `Pipeline` + métricas embebidas (incluida la SD del target) + metadatos (§3.4) |
| `outputs/F21_modelo_global_base.json` | Inventario, métricas, importancias, efecto municipio, limitaciones de v2 y bloque `complementos`. Solo agregados, sin documentos, nombres ni instituciones |
| `reports/21_modelo_global_base.md` | Este informe |

No se modificó ningún fichero existente (scripts, modelos v1/v2, informes ni salidas previas). El informe 13 y sus salidas quedan intactos; la reconciliación de §4.1 consta solo aquí.

**Privacidad de la prueba intra-institución:** `institucion_educativa` se convierte al cargar en un código entero anónimo (`pd.factorize`) y el nombre se elimina en ese mismo paso. El código solo existe en memoria. Las salidas contienen únicamente conteos y medias agregadas, sin ninguna etiqueta de institución.

**Política de uso:** toda cifra de R² o MAE se cita con su valor, su n y su partición. `puntaje_estimado` nunca se presenta como cifra puntual para un individuo ni se usa para ordenar individuos.
