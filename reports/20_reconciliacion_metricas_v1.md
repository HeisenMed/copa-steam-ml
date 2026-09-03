# Reconciliación de las métricas del modelo v1 — Copa STEM 2026

**Fundación SapienceLab** · Fase 5 · Auditoría documental · 2026-09-03

---

## Resumen ejecutivo

Circulan en este repositorio **al menos tres cifras distintas de R² para el mismo
modelo v1** (Random Forest del script 03): **0.115** (informe 03), **~0.241**
(informe 14) y **~0.238** (README). No son un error de transcripción de un mismo
número: **0.115 y 0.241 son dos evaluaciones metodológicamente distintas del
mismo bosque**, y ambas conviven en el repositorio sin que ningún documento las
ponga cara a cara.

- **0.115 / MAE 17.44** es el **hold-out del 20 %** (n ≈ 350) del informe 03.
- **0.241 / MAE 16.4** es una **evaluación dentro de muestra** (*resubstitution*)
  sobre los 1,748 examinados, ~80 % de los cuales entrenaron ese mismo bosque.
  **Sí tiene origen trazable**, y se reprodujo exactamente en esta auditoría
  (R² = 0.2412, MAE = 16.4331) a partir de `models/deploy/puntaje_estimado.csv`.
- **~0.238 / ~18 pts** (README, línea 100) **no tiene ninguna fuente trazable**:
  ningún informe, script, artefacto ni commit de la historia produce ese número.

El hallazgo más serio no es la discrepancia en sí, sino **su dirección**: el
informe 14 coloca la cifra *dentro de muestra* de v1 (~0.241) en la misma tabla
que la cifra *hold-out* de v2 (+0.1766), y atribuye su procedencia al **informe
03**, que no la contiene. El efecto es que **v2 aparenta ser peor que v1** cuando
la comparación limpia disponible (informe 15) dice lo contrario.

**Ninguna de las dos cifras en disputa debería ser la métrica oficial de v1.** La
defendible es el **R² out-of-fold = 0.084** (MAE 18.1, RMSE 22.1), con IC 95 %
por bootstrap **[0.053, 0.116]** — el número que los informes 08, 09b y el
INFORME_COMPLETO ya tratan como honesto.

**Este informe no modifica ningún otro documento, no reentrena ningún modelo y no
regenera ningún artefacto.** Es diagnóstico.

---

## Metodología

1. **Barrido exhaustivo** de `reports/`, `outputs/`, `models/`, `notebooks/`,
   `README.md`, `SESION_ACTUAL.md` y `CLAUDE.md` buscando toda ocurrencia de R²,
   MAE y RMSE (`.venv/`, `.git/` y `__pycache__/` excluidos).
2. **Inspección de artefactos serializados**: contenido del diccionario de los
   dos `.joblib`, claves de `models/modelo_coeficientes.json`, bloque `meta` de
   los predictores JS de despliegue y todos los JSON de `outputs/`.
3. **Reproducción de la métrica en disputa** a partir de predicciones ya
   guardadas — `models/deploy/puntaje_estimado.csv`, 1,748 filas con
   `puntaje_estimado` y `puntaje_real`. **No se reentrenó nada**: se recalculó
   `r2_score` / `mean_absolute_error` sobre columnas que ya existían en disco.
4. **Rastreo en la historia de git** (`git log -S`, `git show`) del origen de las
   cifras sin fuente documental.

---

## Hallazgo 1 — Inventario completo de las cifras de v1

Toda ocurrencia encontrada, agrupada por **qué evaluación describe realmente**.

### Grupo A — Hold-out del 20 % (n ≈ 350 de 1,750)

| Fichero:línea | Valor | Población / partición que declara |
| --- | --- | --- |
| `reports/03_modelo_predictivo.md:13-14` | R² test **0.115**, RMSE 21.56, MAE 17.44 | 1,750 estudiantes, split 80/20 estratificado por grado; métrica sobre el test |
| `reports/03_modelo_predictivo.md:37` | fila RF: CV R² 0.064 ± 0.024, CV RMSE 22.31, CV MAE 18.36 · test 0.115 / 21.56 / 17.44 | ídem; la CV es sobre el train (80 %) |
| `reports/03_modelo_predictivo.md:36,38,39` | LR 0.106 · XGBoost 0.053 · LightGBM 0.013 (R² test) | ídem, modelos no seleccionados |
| `reports/03_modelo_predictivo.md:189` | "un R² de 0.115" | ídem, prosa de interpretación |
| `reports/INFORME_COMPLETO_ML_COPA_STEM.md:603-606` | misma tabla: RF 0.064 (CV) / **0.115** (test) / 21.56 / 17.44 | dice "1.754 estudiantes" — **etiqueta errónea**, ver Hallazgo 4 |
| `reports/INFORME_COMPLETO_ML_COPA_STEM.md:609-610` | 0.115 calificado como "estimación optimista de un único split" | contrapone el OOF 0.084 |

### Grupo B — Dentro de muestra sobre los 1,748 (la familia ~0.241)

| Fichero:línea | Valor | Población / partición que declara |
| --- | --- | --- |
| `reports/09_puntaje_estimado.md:56-57` | R² **0.241**, MAE **16.4**, RMSE 20.1 | **explícitamente etiquetado** "Sobre los datos de entrenamiento (in-sample)"; 1,748 examinados |
| `reports/10_modelo_teorico_vs_empirico.md:15` | "Empírico vs real: R² = **0.241**, MAE = **16.4**, r = 0.504" | 1,748 examinados; **sin etiqueta de in-sample** |
| `reports/10_modelo_teorico_vs_empirico.md:97` | tabla: Empírico (Random Forest) \| 0.241 \| 16.4 \| 0.504 | ídem; **sin etiqueta** |
| `reports/14_optimizacion_v2.md:55` | "v1 — script 03 (producción actual) \| cohorte original \| 1,748 \| 18 \| **~0.241 \*** \| **~16.4 \***" | declara "cohorte original", 1,748, 18 features; **atribuido al informe 03** |
| `reports/14_optimizacion_v2.md:59-60` | nota al pie: "Provienen de otro dataset, otra cohorte y otra partición **(informe 03)**" | **atribución incorrecta**, ver Hallazgo 3 |
| `reports/INFORME_COMPLETO_ML_COPA_STEM.md:1018-1019` | R² 0.241 / MAE 16.4 pts / RMSE 20.1 | **correctamente etiquetado** "Dentro de muestra" |
| `reports/INFORME_COMPLETO_ML_COPA_STEM.md:1049` | Empírico (Random Forest) \| 0.241 \| 16.4 \| 0.504 | **sin etiqueta** (hereda el defecto del informe 10) |

### Grupo C — Validación cruzada out-of-fold sobre los 1,748 (la familia 0.084 / 0.085)

| Fichero:línea | Valor | Población / partición que declara |
| --- | --- | --- |
| `reports/09_puntaje_estimado.md:18` | R² OOF **0.084**, MAE **18.1** | 1,748 examinados, 5-fold `cross_val_predict` |
| `reports/09_puntaje_estimado.md:56-57` (col. derecha) | 0.084 / 18.1 / 22.1 | ídem; declarada "la columna honesta" |
| `reports/09_puntaje_estimado.md:65` | "R² = 0.084 … explica alrededor del 8 %" | ídem, prosa |
| `reports/08_framework_validacion.md:10` | R² ≈ **0.085** | out-of-fold sobre los examinados |
| `reports/08_framework_validacion.md:13-14,79-80` | bootstrap 1000×: R² **0.084**, IC 95 % **[0.053, 0.116]** | remuestreo sobre las predicciones OOF |
| `reports/08_framework_validacion.md:52-53` | 0.085 con IC [0.053, 0.116] | ídem |
| `reports/08_framework_validacion.md:64` | R² cae a **−0.044** en el 30 % de inscripciones más recientes | split temporal; *concept drift* leve |
| `reports/08_framework_validacion.md:100-101` | techo teórico R² ≈ **0.137**; el modelo (0.085) ya está cerca | "gemelos estadísticos" sobre los examinados |
| `reports/04_indice_potencial_stem.md:9` | R² ≈ **0.085** | referencia al script 03 |
| `reports/05_deteccion_trampa.md:126-127` | R² (CV) **0.091 ± 0.027** con todos · **0.089 ± 0.008** sin sospechosos "Alto" | CV sobre los examinados, con y sin anulados |
| `reports/INFORME_COMPLETO_ML_COPA_STEM.md:609` | R² ≈ 0.084 (MAE 18.1) | out-of-fold, script 09b |
| `reports/INFORME_COMPLETO_ML_COPA_STEM.md:971-972,983` | bootstrap 0.084 IC [0.053, 0.116]; techo 0.137 vs modelo 0.085 | ídem |

### Grupo D — v1 reevaluado sobre otras poblaciones (fase 5)

| Fichero:línea | Valor | Población / partición que declara |
| --- | --- | --- |
| `reports/12_experimento_reentrenamiento.md:214` | A_baseline: CV R² **+0.086**, CV MAE 19.01, test R² **+0.111**, test MAE 19.18 | 1,735 filas, 18 features — las *variables* de v1 reentrenadas, no el bosque desplegado |
| `reports/14_optimizacion_v2.md:56` | "v1 — hiperparámetros sobre C": R² **+0.1625**, MAE **15.06** | `dataset_C_perfil`, 1,148, 23 features — los *hiperparámetros* de v1, no el modelo v1 |
| `outputs/F14_optimizacion_v2.json` | `baseline_r2` 0.16247…, `baseline_mae` 15.0589… | ídem — es el respaldo de la fila anterior |
| `reports/15_scores_v2_comparacion.md:134` | MAE **18.45** — "Fuera de muestra (verificado)" | el bosque v1 puntuando los 1,148 de C; solapamiento con su entrenamiento verificado = 0 |
| `reports/15_scores_v2_comparacion.md:21,143-148,192` | 18.45 → 15.00, mejora de 3.45 pts (−19 %) | ídem |
| `README.md:117,121` | MAE (hold-out) v1 = **18.45 \*** con nota "medido sobre los 1.148 de C" | ídem |
| `reports/INFORME_COMPLETO_ML_COPA_STEM.md:1255` | MAE 18.45 → 15.00 | ídem |
| `notebooks/16_visualizaciones_comparacion_v2.py:98` | `MAE_V1_OOS = 18.45` (constante en código) | ídem, cifra fijada a mano para las figuras |

### Grupo E — Sin fuente trazable

| Fichero:línea | Valor | Población / partición que declara |
| --- | --- | --- |
| `README.md:100` | "Random Forest (script 03) \| **~0.238** \| **~18 pts**" | declara ser del script 03, "uso en producción: `indice_potencial`" |

### Artefactos que NO contienen métricas de v1

Se inspeccionaron uno por uno; ninguno guarda R² ni MAE del modelo v1:

| Artefacto | Contenido relevante |
| --- | --- |
| `models/mejor_modelo_puntaje.joblib` | dict con `modelo`, `preprocessor`, `feature_names`, `nombre`, `random_state`. **Sin clave de métricas.** |
| `models/modelo_coeficientes.json` | `modelo`, `generado`, `n_features`, `features`, `importancias`, `reglas_ejemplo`, `tipo_export`, `n_arboles`, `combina`, `sesgo`. **Sin clave de métricas.** |
| `models/deploy/potencial_stem_predictor.js` | `meta` = `{generado, n_cohorte: 1750, modelo_puntaje}`. **Sin métricas.** |
| `outputs/F14_optimizacion_v2.json` | solo v2 y el *baseline* de hiperparámetros sobre C. |
| `outputs/F17_ref_rendimiento_corregido.json` | percentiles de `ref_rendimiento`, n=3,072. Sin métricas de modelo. |
| `outputs/F19_verificacion_deploy_v2.json` | hold-out de **v2** (0.17664 / 14.9958 / n=230). Sin métricas de v1. |

**Contraste revelador:** `models/mejor_modelo_puntaje_v2.joblib` **sí** guarda
`metricas_holdout = {r2: 0.17663952657605664, mae: 14.995819066080928, n_test: 230}`.
El bundle de v1 no guarda nada equivalente. **Esa es la causa estructural de todo
este informe:** v1 nunca fijó su métrica en un artefacto, así que cada documento
posterior tuvo que recordarla, y cada uno la recordó distinta.

---

## Hallazgo 2 — El ~0.241 sí tiene origen, y es una evaluación dentro de muestra

Reproducción a partir de predicciones ya en disco (**sin reentrenar**), sobre
`models/deploy/puntaje_estimado.csv` (1,748 filas con estimado y real):

```
n    = 1748
R²   = 0.2412
MAE  = 16.4331
RMSE = 20.1255
r    = 0.5043
```

Coincide dígito a dígito con las cuatro cifras del informe 10 (`0.241`, `16.4`,
`0.504`) y del informe 09 (`0.241`, `16.4`, `20.1`). **El origen queda
establecido sin ambigüedad.**

Qué es ese CSV: las predicciones del bosque v1 **para todos los estudiantes que
presentaron**, incluidos los ~1,400 (80 %) sobre los que ese mismo bosque se
entrenó. Evaluarlo contra el puntaje real es una métrica de *resubstitution*:
mide en gran parte memorización.

El propio informe 09 lo dice, y lo dice bien (líneas 58-62):

> La columna izquierda mide el modelo sobre estudiantes que **ya vio** al
> entrenarse: siempre se ve mejor de lo que es (como un examen con las respuestas
> a la vista). La columna derecha lo mide sobre estudiantes que **no vio**.

**El problema no fue calcularla. Fue que perdió su etiqueta al viajar.** La
cadena de transmisión es:

```
09b (in-sample, ETIQUETADO)  →  10 (sin etiqueta)  →  14 (sin etiqueta y
                                                        reatribuido al informe 03)
```

---

## Hallazgo 3 — El informe 14 atribuye la cifra a una fuente que no la contiene

`reports/14_optimizacion_v2.md:59-60` justifica el asterisco así:

> **\* Las métricas de v1 en producción no son comparables** con las de v2.
> Provienen de otro dataset, otra cohorte y otra partición **(informe 03)**.

**El informe 03 no contiene 0.241 ni 16.4.** Contiene 0.115 / 21.56 / 17.44 en
test y 0.064 / 22.31 / 18.36 en CV. La cifra citada viene de los informes 09b y
10, y no es "otra partición": es **ninguna partición** — es el conjunto completo,
entrenamiento incluido.

Hay que reconocerle al informe 14 que **marcó el problema**: puso el asterisco,
puso la virgulilla y dijo explícitamente que la fila no es comparable. El defecto
es de trazabilidad y de colocación, no de honestidad. Pero el efecto práctico
persiste: en una tabla titulada "Comparación v1 vs v2", un lector ve

```
v1 — script 03 (producción actual)  ~0.241
v2 — optimizado                     +0.1766
```

y concluye que v2 empeoró el modelo en 0.065 de R². **Es exactamente al revés**:
está comparando la nota que v1 se puso con el examen a la vista contra la nota
que v2 sacó a libro cerrado. La única comparación limpia entre ambos modelos que
existe en el repositorio es la del informe 15 — **MAE 18.45 (v1, fuera de
muestra) vs 15.00 (v2, hold-out)** — y esa favorece a v2 en 3.45 puntos.

**Nota a favor del INFORME_COMPLETO:** su §11.1 (líneas 1198-1225), que resume el
script 14, **omitió esa fila**. Reporta el hold-out de v2 y la ganancia de +0.0142
frente al *baseline* de hiperparámetros, sin arrastrar el ~0.241. El error no se
propagó al documento maestro.

---

## Hallazgo 4 — Dos discrepancias menores de población, del mismo origen

Aparecen tres tamaños para la misma cohorte de julio:

| Cifra | Dónde | Qué es en realidad |
| --- | --- | --- |
| **1,754** | `INFORME_COMPLETO:598` ("datos **limpios**: 1.754 estudiantes") | registros **crudos** del CSV, **antes** de limpiar |
| **1,750** | `reports/03:11`, `potencial_stem_predictor.js` (`n_cohorte`) | tras limpieza (se eliminan 4 registros de prueba) — **el número correcto para el informe 03** |
| **1,748** | informes 09, 10, 14 | tras limpieza **y** `drop_duplicates('numero_documento')` en el script 09b (−2 duplicados) |

Verificado ejecutando solo la carga y limpieza del script 03: crudos 1,754 →
limpios 1,750 → deduplicados 1,748.

Es inocuo para las conclusiones (2 estudiantes de 1,750), pero la etiqueta
"1.754 estudiantes **limpios**" del INFORME_COMPLETO es **falsa por definición**:
1,754 es precisamente el número *sucio*.

Además, el informe 03 nunca declara el tamaño de su test. Es el 20 % de 1,750 =
**n ≈ 350**, y ese dato importa para juzgar cuánto pesa el 0.115 (ver Hallazgo 5).

---

## Hallazgo 5 — El ~0.238 del README no existe en ninguna parte

`README.md:100` publica, en la tabla "Estado del modelo (Copa STEM 2026)":

```
| Random Forest (script 03) | ~0.238 | ~18 pts | indice_potencial (componente rendimiento) |
```

Rastreo realizado:

- **No aparece** en ningún fichero de `reports/`, `notebooks/` ni `outputs/`.
- **No aparece** en ningún `.joblib`, `.json` ni predictor JS.
- **No lo produce** ninguna de las evaluaciones documentadas: ni 0.115 (test), ni
  0.241 (in-sample), ni 0.084 (OOF), ni 0.086/0.111 (dataset A), ni 0.1625
  (hiperparámetros sobre C).
- `git log --all -S"0.238"` lo sitúa **desde el primer commit que creó esa tabla**
  (`d5844ef`, "docs: actualizar README con Bello…") y **nunca tuvo otro valor**:
  no es la corrupción progresiva de un número previo.

La hipótesis más económica es que sea el **0.2412 in-sample transcrito de
memoria**, con el `~18 pts` tomado del MAE out-of-fold (18.1) — es decir, **una
fila que mezcla el R² de una evaluación con el MAE de otra**. Pero eso es
inferencia, no evidencia.

**Conclusión explícita, tal como se pidió: el ~0.238 no tiene fuente trazable en
este repositorio.** Debe tratarse como una cifra inventada y retirarse, no
"corregirse" a un decimal cercano.

---

## Conclusiones

### Cuál es la métrica oficial defendible de v1

**El R² out-of-fold = 0.084 (MAE 18.1, RMSE 22.1), con IC 95 % [0.053, 0.116].**
Fuente: `reports/09_puntaje_estimado.md:18,56-57` y
`reports/08_framework_validacion.md:13-14,79-80`.

Razones, todas apoyadas en evidencia del propio repositorio:

1. **Es out-of-fold**: cada predicción viene de un modelo que no vio a ese
   estudiante. Ni 0.241 (que sí los vio) ni 0.115 (que separó bien pero midió una
   sola vez) cumplen eso sobre toda la población.
2. **Cubre a los 1,748 examinados**, no a un test de ~350. Con n = 350 y un R²
   real cerca de 0.08, un único split se mueve varios centésimos por azar: el
   0.115 es un punto de esa distribución, no su centro.
3. **Es el único con intervalo de confianza.** El bootstrap de 1000 remuestreos
   da [0.053, 0.116] — que **no toca el cero** (el poder predictivo es real) y
   que **contiene al 0.115**, confirmando que el hold-out del informe 03 es el
   borde optimista del mismo fenómeno, no un resultado distinto.
4. **Es coherente con todo lo demás:** 0.064 ± 0.024 (CV del informe 03), 0.091 ±
   0.027 (CV del informe 05), 0.086 (dataset A del informe 12). Todo el
   repositorio converge a **0.06–0.09** cuando mide honestamente. El 0.241 es el
   único valor que se sale, y se sale porque no mide lo mismo.
5. **El repositorio ya lo trata así.** `INFORME_COMPLETO:609-610` dice
   literalmente que el OOF 0.084 es "coherente con la CV" y que "el punto 0.115
   del test de Random Forest es una estimación optimista de un único split". La
   recomendación no introduce un criterio nuevo: **hace explícito el que ya se
   venía aplicando**.

### Estatus de cada cifra en disputa

| Cifra | Veredicto |
| --- | --- |
| **0.084 / 18.1** | **Métrica oficial de v1.** Out-of-fold sobre 1,748, con IC 95 %. |
| 0.115 / 17.44 | **Válida y publicable como lo que es**: R² del hold-out del 20 % (n ≈ 350) del informe 03. No es la métrica oficial; es el borde optimista del intervalo. **No debe borrarse del informe 03** — es el resultado real de ese script. |
| ~0.241 / ~16.4 | **Nunca como métrica de v1.** Es dentro de muestra. Legítima solo con la etiqueta puesta, y **jamás en la misma tabla que una métrica hold-out.** |
| ~0.238 / ~18 pts | **Sin fuente. Retirar.** |
| 18.45 (MAE) | **Válida y bien documentada** para comparar v1 con v2: fuera de muestra sobre los 1,148 de C, solapamiento verificado = 0 (informe 15). Es MAE, **no hay R² equivalente**. |

### Por qué pasó esto (y qué lo evita)

v1 **nunca guardó sus métricas en un artefacto**. Su `.joblib` contiene el modelo
y el preprocesador, pero ningún R². v2 sí las guarda (`metricas_holdout`), y por
eso el 0.1766 aparece idéntico en el informe 14, en el JSON F14, en el JSON F19,
en el README y en `SESION_ACTUAL.md` — **cinco sitios, cero divergencias**.

La lección es de ingeniería, no de estadística: **la métrica debe viajar dentro
del artefacto, no en la memoria de quien redacta el informe siguiente.**

---

## Recomendación de ediciones (NO aplicadas)

Este informe es diagnóstico. Las siguientes ediciones dejarían el repositorio
internamente consistente; **ninguna se ha ejecutado**.

**Prioridad alta — cifras que engañan al lector**

1. **`README.md:100`** — retirar `~0.238 | ~18 pts`. Sustituir por
   `0.084 | 18.1 pts` con la nota "R² out-of-fold sobre 1,748 examinados
   (informes 08 y 09b)". Es la única cifra del repositorio sin respaldo.
2. **`reports/14_optimizacion_v2.md:55`** — la fila de v1 muestra una métrica
   dentro de muestra junto a dos hold-out. Opciones, en orden de preferencia:
   (a) sustituir por `0.084 | 18.1` etiquetado "out-of-fold, 1,748 (informe 09b)";
   (b) mantener el ~0.241 pero renombrando la fila a
   "v1 — dentro de muestra (informes 09b/10) — **no comparable**".
3. **`reports/14_optimizacion_v2.md:60`** — corregir la atribución: la cifra
   **no** viene del informe 03. Cambiar "(informe 03)" por "(informes 09b y 10;
   es una evaluación **dentro de muestra**, no otra partición)".

**Prioridad media — etiquetas que faltan**

4. **`reports/10_modelo_teorico_vs_empirico.md:15,97`** — añadir "(dentro de
   muestra)" junto a `R² = 0.241`. Es donde la cifra pierde su etiqueta por
   primera vez y desde donde se propagó.
5. **`reports/INFORME_COMPLETO_ML_COPA_STEM.md:1049`** — misma etiqueta en la
   tabla teórico vs empírico (la de §9.2, línea 1018, ya la tiene).
6. **`reports/03_modelo_predictivo.md`** — añadir una nota tras la tabla de
   comparación: el 0.115 es el test de un único split de n ≈ 350; la métrica
   honesta de referencia es el OOF 0.084 (informes 08 y 09b). **Sin tocar las
   cifras**, que son el resultado real del script.

**Prioridad baja — higiene**

7. **`reports/INFORME_COMPLETO_ML_COPA_STEM.md:598`** — "datos **limpios**
   (1.754 estudiantes)" → 1,750. El 1,754 es el conteo crudo.
8. **`reports/03_modelo_predictivo.md`** — declarar el tamaño del test
   (n ≈ 350). Su ausencia es lo que hace que el 0.115 parezca más sólido de lo
   que es.
9. **Fijar la métrica en el artefacto de v1**: añadir a
   `models/mejor_modelo_puntaje.joblib` (o a `modelo_coeficientes.json`) un
   bloque `metricas` con las tres evaluaciones y su etiqueta, replicando lo que
   v2 ya hace. **Requiere ejecutar código**, así que queda fuera del alcance de
   este informe.

**Regla propuesta para el repositorio:** toda cifra de R² o MAE se publica con
**tres datos inseparables** — valor, población (n) y partición (in-sample /
CV / out-of-fold / hold-out). Las tres cifras en disputa de este informe habrían
sido imposibles bajo esa regla: no se contradicen, describen cosas distintas que
se publicaron como si fueran la misma.

---

## Limitaciones de esta auditoría

- **No se reentrenó ningún modelo ni se regeneró ningún artefacto.** La única
  computación fue recalcular R²/MAE/RMSE sobre predicciones ya guardadas en
  `models/deploy/puntaje_estimado.csv`, y ejecutar la carga y limpieza del script
  03 para verificar los conteos 1,754 / 1,750 / 1,748.
- **No se editó ningún informe existente.** Todas las ediciones son propuestas.
- **El origen del `~0.238` no pudo probarse**, solo descartarse: se demuestra que
  no procede de ninguna evaluación documentada, no de dónde salió. La hipótesis
  de la transcripción de memoria es plausible, no verificada.
- **El 0.115 no se reprodujo.** Hacerlo exigiría reejecutar el split del script
  03, que es reentrenamiento. Se acepta tal como lo reporta el informe 03, cuyas
  cifras son internamente coherentes (la tabla de 4 modelos, el análisis de
  residuos con desviación 21.55 ≈ RMSE 21.56, y la prosa de la línea 189).
- **`.venv/`** se excluyó del barrido tras confirmar que solo aporta falsos
  positivos (código de terceros).

---

## Referencias técnicas

**Ficheros auditados**

`reports/03_modelo_predictivo.md`, `04_indice_potencial_stem.md`,
`05_deteccion_trampa.md`, `08_framework_validacion.md`, `09_puntaje_estimado.md`,
`10_modelo_teorico_vs_empirico.md`, `12_experimento_reentrenamiento.md`,
`13_analisis_explicabilidad.md`, `14_optimizacion_v2.md`,
`15_scores_v2_comparacion.md`, `17_fix_ref_rendimiento.md`,
`18_scores_inscritos_sin_examen.md`, `19_regenerar_deploy_v2.md`,
`INFORME_COMPLETO_ML_COPA_STEM.md`, `README.md`, `SESION_ACTUAL.md`, `CLAUDE.md`,
`notebooks/*.py`, `outputs/F14`, `F17`, `F19` (JSON),
`models/*.joblib`, `models/modelo_coeficientes.json`, `models/deploy/*.js`.

**Definiciones usadas**

- **In-sample (resubstitution):** el modelo se evalúa sobre datos que usó para
  entrenar. Optimista por construcción; mide memorización además de predicción.
- **Hold-out:** partición apartada antes de entrenar y evaluada una sola vez.
  Insesgada, pero de alta varianza cuando n es pequeño.
- **Out-of-fold (OOF):** cada observación se predice con un modelo entrenado sin
  ella (`cross_val_predict`). Insesgada y con la n completa: la mejor de las
  tres para una cifra única de referencia.

**Comando de reproducción del Hallazgo 2** (no reentrena):

```python
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

d = pd.read_csv("models/deploy/puntaje_estimado.csv", encoding="utf-8-sig")
d = d.dropna(subset=["puntaje_real", "puntaje_estimado"])
print(r2_score(d.puntaje_real, d.puntaje_estimado))          # 0.2412
print(mean_absolute_error(d.puntaje_real, d.puntaje_estimado))  # 16.4331
```

---

_Informe de auditoría documental. No modifica informes previos, no reentrena
modelos y no regenera artefactos de despliegue. Toda cifra citada procede de un
fichero del repositorio, identificado por ruta y línea._
