# Scores de los inscritos que NO presentaron el examen — Copa STEM 2026

**Fundación SapienceLab** · Fase 5 · Informe generado: 2026-09-02

---

## Resumen ejecutivo

Los scripts 15 y 17 puntuaron a los 3,072 estudiantes que ya tienen resultado.
En ese grupo el modelo casi no importa: el componente de rendimiento usa la nota
real y el modelo de puntaje ni siquiera se invoca. La población donde el modelo
sí manda el índice es la contraria —los inscritos sin examen— y ninguna corrida
la había cubierto. El script 18 la cubre.

**Son 248 estudiantes.** El resultado está en
`outputs/ml_scores_sin_examen.csv`, con las mismas 18 columnas de
`ml_scores_v2_corrected.csv` y cero solapamiento de documentos con los
examinados.

Cuatro conclusiones, en orden de importancia:

1. **El índice de un estudiante sin examen no es comparable con el de uno
   examinado.** No es un ajuste fino: son dos escalas. σ = 8.97 contra 22.37.
   El 98.8 % de los 248 cae en solo dos categorías.
2. **"Talento destacado" es inalcanzable por construcción.** El techo aritmético
   del índice sin nota es **83.33**, por debajo del umbral de 85. No es que no
   haya salido ninguno: no puede salir. Vale igual para v1 y para v2.
3. **La detección de talento oculto no funciona en este grupo.** Salieron 0
   marcados, con probabilidad máxima de 0.0031. Es un artefacto del método —el
   clasificador usa el puntaje como feature—, no una propiedad de la población.
4. **Esta población está en peores condiciones que la examinada.** 21.4 % en
   condiciones adversas contra 10.6 % de los examinados. Los que no se
   presentaron son desproporcionadamente los que menos recursos tienen. Es el
   hallazgo con más valor de programa de toda la corrida.

Y la advertencia que atraviesa todo: **`puntaje_estimado` para este grupo carga
MAE ~15 puntos** sobre una escala de 0 a 100. Es una banda, no una cifra.

## Qué población es y cómo se identificó

Todos los exports de `data/` son un *inner join* con
`resultados_prueba_copa_stem`: `copa_stem_dataset_2026-08.csv.csv` trae 3,077
filas / 3,072 documentos únicos y **cero** nulos en `puntaje_obtenido`. Los
inscritos que no presentaron no estaban en ninguno de ellos, así que hubo que
exportarlos aparte desde Supabase (`data/inscritos_copa_stem.csv`, 249 filas).

El script hace el anti-join explícitamente:

| Paso | Filas |
| --- | --- |
| Export de `inscripciones_copa_stem` | 249 |
| Menos 1 documento de menos de 5 caracteres | 248 |
| Menos los que ya aparecen en `resultados_prueba_copa_stem` | 248 (ninguno) |
| **Población puntuada** | **248** |

El export ya venía filtrado con `WHERE NOT EXISTS`, por eso el anti-join no
quitó a nadie; se ejecuta igual como verificación. El conjunto "ya presentaron"
se toma **sin filtrar** de `dataset_B_completo.csv`: descartar ahí un documento
de prueba lo devolvería a la población objetivo como si nunca hubiera
presentado, que es lo contrario de lo que se busca.

Quiénes son:

| Corte | Reparto |
| --- | --- |
| Municipio | Copacabana 151 · Girardota 55 · Bello 42 |
| Grado | 9.º 127 · 11.º 63 · 10.º 58 |
| Institución | Pública 201 · Privada 47 |
| Género | Femenino 123 · Masculino 118 · No binario 4 · Prefiere no decirlo 3 |
| Estrato | 2 → 128 · 3 → 87 · 1 → 33 |

## Cómo se calculó el índice

Misma estrategia híbrida de los scripts 15 y 17, sin ninguna variante:

| Caso | SPEC | Estudiantes |
| --- | --- | --- |
| Tiene el bloque de perfil académico (`promedio_academico`) | v2 — 9 numéricas, 4 binarias | **91** (36.7 %) |
| No lo tiene | v1 — 5 numéricas, 3 binarias | **157** (63.3 %) |

86 de los 91 tienen las 5 preguntas de perfil completas. El reparto 37/63 es casi
idéntico al de los examinados (37.4/62.6), así que el fallback pesa lo mismo aquí
que en la corrida del script 17.

La percentilización usa `ref_rendimiento` de **3,072 puntajes**, la referencia
corregida del script 17 (`outputs/F17_ref_rendimiento_corregido.json`, σ = 22.66).

La rama "sin nota" **no se fuerza a mano**. Se le pasa a `calcular_indice` un
registro del que se han borrado `puntaje_obtenido` y la telemetría, y la función
toma sola el camino que ya tenía escrito:

```python
real = _to_float(raw.get("puntaje_obtenido"))
if real is not None:  rend_raw = real
else:                 rend_raw = _predict_puntaje(...)   # ← esta población
```

Es exactamente el código que corre en la Edge Function. Se reutilizan los cuatro
predictores de `models/deploy/` sin tocarlos.

## Distribución de categorías

| Categoría | Sin examen | % | Examinados | % | Δ pp |
| --- | --- | --- | --- | --- | --- |
| Talento destacado | **0** | 0.0 % | 105 | 3.4 % | −3.4 |
| Alto potencial | 3 | 1.2 % | 721 | 23.5 % | −22.3 |
| Promedio | 110 | 44.4 % | 1,006 | 32.7 % | +11.6 |
| En desarrollo | **135** | 54.4 % | 784 | 25.5 % | +28.9 |
| Requiere apoyo | **0** | 0.0 % | 456 | 14.8 % | −14.8 |

245 de 248 (98.8 %) caen en "Promedio" o "En desarrollo". Los dos extremos se
vacían por completo.

Por versión de modelo, la diferencia es despreciable —lo cual confirma que el
efecto no viene del modelo sino de la estructura del índice:

| Versión | n | Índice medio | Estimado medio |
| --- | --- | --- | --- |
| v2 | 91 | 45.41 | 39.43 |
| v1 fallback | 157 | 44.00 | 39.82 |

## Por qué se vacían los extremos

No es una propiedad de estos 248 estudiantes. Es aritmética del índice, y se
puede demostrar de dos maneras.

### Prueba 1: comparar los componentes

| Componente | Sin examen (248) | Examinados (3,072) |
| --- | --- | --- |
| Rendimiento (50 % del índice) | media 51.71 · **σ 12.68** | media 53.25 · **σ 28.08** |
| Engagement (25 %) | media 33.37 · σ 13.67 | media 38.04 · σ 12.64 |
| Resiliencia (25 %) | media 41.27 · **σ 5.05** | media 61.00 · **σ 30.58** |
| **Índice compuesto** | media 44.52 · **σ 8.97** | media 51.39 · **σ 22.37** |

Dos causas se suman.

**El modelo comprime el insumo.** Con R² = 0.1766 predice poco más que "cerca del
promedio". Medido sobre los 3,068 examinados con la nota borrada (modo `--ensayo`
del script):

| Insumo del percentil | Media | σ | p5 | p95 | Rango |
| --- | --- | --- | --- | --- | --- |
| Puntaje real | 41.76 | **22.66** | 10.0 | 85.0 | 0 – 100 |
| Puntaje estimado | 42.17 | **9.20** | 28.4 | 58.5 | 21.5 – 71.1 |

La media acierta; la dispersión no (correlación 0.52). Al percentilizar un
insumo apretado contra una referencia ancha, el rendimiento pierde recorrido.

**La resiliencia se vuelve casi constante.** La fórmula cambia según haya nota:

```python
if presento:  c_resiliencia = min(100.0, c_rendimiento * (1.0 + adv * 0.15))
else:         c_resiliencia = max(0.0, 50.0 - adv * 5.0)
```

Sin nota solo depende del conteo de condiciones adversas, así que puede tomar
**cinco valores y nada más** — 30, 35, 40, 45, 50. Un cuarto del índice queda
clavado en una franja de 20 puntos (σ 5.05 contra 30.58).

### Prueba 2: el techo es inalcanzable

Un bosque promedia hojas, así que el máximo que puede emitir está acotado por la
media de los máximos por árbol:

| | v2 (200 árboles) | v1 (300 árboles) |
| --- | --- | --- |
| Cota superior del estimado | 75.02 | 76.63 |
| Su percentil en la referencia de 3,072 | 91.67 | 91.67 |
| Resiliencia máxima sin nota | 50.00 | 50.00 |
| Engagement máximo teórico | 100.00 | 100.00 |
| **Índice máximo posible** | **83.33** | **83.33** |
| Umbral "Talento destacado" | 85 | 85 |

83.33 < 85. Y la cota es generosísima: exige que un mismo estudiante caiga en la
hoja más alta de los 200 árboles a la vez, con engagement perfecto y cero
condiciones adversas. El máximo real entre los 248 fue **72.3**.

Por el otro extremo el mínimo posible es 14.03, pero requiere engagement 0; el
mínimo observado fue 25.2, apenas por encima del umbral de 25. De ahí los cero
en "Requiere apoyo".

### Qué se sigue de esto

**No se pueden mezclar las dos poblaciones en un mismo ranking ni en un mismo
gráfico de categorías.** Un "Promedio" sin examen y un "Promedio" con examen no
significan lo mismo, y ningún estudiante sin examen podrá aparecer nunca en la
categoría más alta por bueno que sea su perfil. Si el dashboard necesita ordenar
a esta población, tiene que hacerlo dentro del grupo, con su propia escala, y
decir que es una estimación.

## La detección de talento oculto no funciona aquí

| | Sin examen | Examinados |
| --- | --- | --- |
| Marcados como talento oculto | **0** | 609 |
| `probabilidad_talento` media | 0.0004 | — |
| `probabilidad_talento` máxima | 0.0031 | — |

Cero, y con probabilidades tres órdenes de magnitud por debajo de cualquier
umbral útil. Las dos mitades del detector fallan por la misma razón:

- **La regla determinista** exige `puntaje_obtenido >= 60` **o**
  `indice_potencial >= 75`. Sin nota la primera nunca se evalúa, y la segunda
  es inalcanzable en la práctica (máximo observado 72.3).
- **El clasificador** usa `puntaje_obtenido` e `indice_potencial` como features
  de entrada. Sin nota, la primera se imputa con la mediana de la cohorte y la
  segunda llega comprimida, así que el modelo ve a todos como el mismo caso
  promedio.

Esto es grave para el uso previsto: **no se puede usar este CSV para buscar
talento oculto entre los que no se presentaron**, que es justo lo que uno
querría hacer con él. La columna sale en el esquema porque el esquema es fijo,
pero no informa. Y la ironía es que sí hay materia prima: 175 de los 248 (70.6 %)
tienen 3 o más condiciones adversas.

| Condiciones adversas | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Estudiantes | 4 | 15 | 54 | 77 | 55 | 39 | 4 |

## Los que no se presentaron están en peores condiciones

Este es el hallazgo que sí se sostiene solo, porque `indice_condiciones` es un
modelo teórico que **no usa el puntaje** —se calcula solo con estrato, acceso a
computador e internet, convivencia, nivel previo de programación y robótica,
interés y herramientas—. No lo afecta ninguna de las compresiones anteriores.

| Nivel de condiciones | Sin examen | % | Examinados | % |
| --- | --- | --- | --- | --- |
| Adversas | 53 | **21.4 %** | 326 | **10.6 %** |
| Promedio | 166 | 66.9 % | 2,211 | 72.0 % |
| Favorables | 29 | **11.7 %** | 535 | **17.4 %** |

La proporción en condiciones adversas es **el doble**. El perfil por clusters
apunta a lo mismo:

| Perfil | Sin examen | % | Examinados | % |
| --- | --- | --- | --- | --- |
| Alto rendimiento tech | 55 | 22.2 % | 966 | 31.4 % |
| Base conectada | 88 | 35.5 % | 1,372 | 44.7 % |
| En desarrollo | 86 | **34.7 %** | 640 | **20.8 %** |
| Promedio con acceso limitado | 19 | **7.7 %** | 94 | **3.1 %** |

Los dos perfiles de menos recursos concentran el 42.4 % de los no examinados
frente al 23.9 % de los examinados.

**La no presentación no es aleatoria: está correlacionada con la desventaja.**
Eso tiene dos consecuencias. Para el programa, que el seguimiento a estos 248
es una intervención de equidad, no un trámite administrativo. Para el análisis,
que todas las brechas socioeconómicas medidas sobre los examinados
(informe 02) están **subestimadas**, porque los más desfavorecidos se cayeron
de la muestra antes de generar un dato.

## `puntaje_estimado`: MAE ~15, siempre con banda

Para los examinados, `puntaje_estimado` es una columna informativa que nadie
mira porque al lado está la nota real. Para estos 248 **es lo único que hay**, y
hereda todo el error del modelo.

| | Valor |
| --- | --- |
| n | 248 (el 100 % son estimaciones) |
| Media | 39.67 |
| σ | 8.79 |
| p25 · mediana · p75 | 32.2 · 38.8 · 45.3 |
| Rango | 25.6 – 69.5 |
| **MAE del modelo** | **±15.00** puntos (validación, script 14) |

Un estimado de 40 significa **"probablemente entre 25 y 55"**. No significa 40.

Y hay algo peor que el MAE suelto. **El 50 % central de la población cabe en 13
puntos (32.2 a 45.3), una franja más estrecha que la propia barra de error de
±15.** Es decir: para la mitad de estos estudiantes, la diferencia entre uno y
otro es menor que el error de medida de cada uno. Ordenarlos entre sí por esta
columna es ordenar ruido.

Reglas de uso, no negociables:

1. Nunca publicar `puntaje_estimado` como cifra puntual. Siempre con banda, o
   como rango, o como categoría ancha.
2. Nunca mostrarlo en la misma columna que un puntaje real. La columna
   `tiene_puntaje_real` existe para eso y vale `False` en las 248 filas.
3. Nunca usarlo para decisiones individuales de selección, premiación o
   descarte. Sirve para priorizar dónde mirar, no para decidir.
4. **Nunca rankear a esta población por esta columna.** Ver el párrafo anterior.

## Salidas

| Archivo | Filas | Contenido |
| --- | --- | --- |
| `outputs/ml_scores_sin_examen.csv` | 248 | 18 columnas, esquema idéntico a `ml_scores_v2_corrected.csv` |
| `outputs/F18_distribucion_sin_examen.csv` | 5 | Distribución de categorías comparada contra los examinados |

Las 18 columnas son las 17 que la Edge Function escribe en `ml_scores` más
`modelo_version`. Las 3 que no escribe (`nivel_sospecha`,
`n_criterios_sospecha`, `created_at`) las pone el trigger
`after_resultado_insert`, que para esta población nunca se dispara: no hay
resultado que insertar.

## Verificación

- Esquema: 18 columnas, lista idéntica a la de `ml_scores_v2_corrected.csv`.
- `tiene_puntaje_real` = `False` en las 248 filas.
- Solapamiento de documentos con los 3,072 examinados: **0**.
- Referencia de percentil cargada: n = 3,072, σ = 22.66 (la del script 17).
- El export traía las 24 columnas esperadas; ninguna imputación por columna
  ausente.
- El modo `--ensayo` (3,068 examinados con la nota borrada) corre la misma ruta
  de punta a punta y produjo las cifras de compresión de este informe.

## Limitaciones

1. **`puntaje_obtenido` es feature de dos predictores que no son el del índice.**
   El detector de talento oculto y el de clustering lo usan como variable de
   entrada; sin nota, ambos la imputan con la mediana de la cohorte. Es lo que
   haría la Edge Function con un estudiante sin resultado, así que el script lo
   replica tal cual, pero significa que `cluster_id` y `probabilidad_talento`
   son menos informativos aquí. Para `probabilidad_talento` el efecto es total
   (ver la sección correspondiente).
2. **63 % de la población no tiene el bloque de perfil académico**, así que va
   por el fallback v1 y no se beneficia de las 5 variables nuevas del modelo v2.
3. **El conteo de condiciones adversas trata el dato faltante como ausencia de
   adversidad.** Con 63 % de nulos en el bloque de perfil —aunque ese bloque no
   entra en el conteo de adversidad— conviene tener presente el sesgo: donde
   falta información, el índice tiende a verse mejor de lo que es.
4. **No se cubrió `inscripciones_emergencia`.** El export salió solo de
   `inscripciones_copa_stem`. Si esa tabla tiene inscritos sin resultado,
   faltan en estos 248.

## Qué hacer con esto

1. **No subir esta tabla junto a `ml_scores` sin una marca de población.** La
   columna `tiene_puntaje_real` sirve, pero el dashboard tiene que separarlas
   visualmente, no solo filtrarlas.
2. **Revisar la fórmula de resiliencia sin nota.** Cinco valores discretos entre
   30 y 50 para un cuarto del índice es demasiado pobre, y es la mitad de la
   causa del aplanamiento.
3. **Buscar talento oculto en este grupo con otro método.** El detector actual
   no sirve aquí. Con `indice_condiciones` —que no depende del puntaje— más el
   perfil de cluster ya se pueden priorizar los 53 en condiciones adversas sin
   pasar por un modelo que no tiene señal.
4. **Tratar el seguimiento a estos 248 como intervención de equidad.** Están
   sistemáticamente peor que los examinados; recuperarlos corrige una pérdida
   de muestra que sesga todos los análisis de brechas hacia abajo.
