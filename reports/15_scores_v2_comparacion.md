# Scores v2 — Modelo Híbrido y Comparación con v1 — Copa STEM 2026

**Fundación SapienceLab** · Fase 5 · Informe generado: 2026-08-30

---

## Resumen ejecutivo

Se ejecutó **en sombra** el vector completo de `ml_scores` para los 3,072
estudiantes con resultado de examen, con estrategia híbrida: v2 para los 1,148
que declararon perfil académico, v1 para los 1,924 restantes. No se tocó ningún
modelo ni la tabla de producción.

- **`indice_potencial` apenas se mueve**: cambio medio de **−0.61 puntos**
  (rango −3.84 … +3.39). El **89 %** de los estudiantes no cambia de categoría.
- **Y ese pequeño cambio no lo produce el modelo.** El índice compuesto usa el
  puntaje **real** cuando el estudiante presentó el examen, y los 3,072 lo
  presentaron. Lo que cambia es la distribución de referencia con la que se
  percentiliza. **El modelo v2 no interviene en el índice de ningún estudiante
  de esta cohorte.**
- **Donde v2 sí gana es en `puntaje_estimado`**: MAE de **18.45 → 15.00** frente
  al valor real (−19 %), usando la cifra honesta de validación.
- **Pero `puntaje_estimado` está fijado a `null` en la Edge Function.** Desplegar
  v2 tal cual **no cambiaría nada** para quien ya presentó el examen.

## Estrategia híbrida

| Grupo | Criterio | n | Modelo |
| --- | --- | --- | --- |
| v2 | `promedio_academico` no nulo | 1,148 | Random Forest optimizado (script 14) |
| v1_fallback | Sin perfil académico | 1,924 | Modelo de producción actual |

**Verificación del fallback.** En el subgrupo de 1,924 el conteo de talento
oculto es **431 con v1 y 431 con v2**: idéntico. El fallback es un no-op exacto,
como debe ser.

## Hallazgo principal: el modelo no toca el índice

El código de `calcular_indice` en `models/deploy/potencial_stem_predictor.py`:

```python
real = _to_float(raw.get("puntaje_obtenido"))
presento = real is not None
if presento:
    rend_raw = real                                    # el modelo NO se invoca
else:
    rend_raw = _predict_puntaje(_features_puntaje(raw, PRE), MODEL)
c_rendimiento = _percentil(rend_raw, ref)
```

El componente de rendimiento —que pesa el 50 % del índice— sale del puntaje
**real** siempre que exista. Como `dataset_B_completo` se define justamente como
"los que tienen resultado de examen", el modelo de puntaje **no participa en el
cálculo del índice para ninguno de los 3,072**.

Lo único que cambia entre v1 y v2 es `ref_rendimiento`, la distribución contra
la que se percentiliza ese puntaje real:

| SPEC | Cohorte de referencia |
| --- | --- |
| v1 | 1,750 estudiantes |
| v2 | 1,148 estudiantes |

Percentilizar contra una cohorte distinta desplaza ligeramente a todo el mundo.
Ese es el origen del −0.61 medio, y es un efecto **de referencia, no de
predicción**.

**Dónde sí importaría el modelo:** en los estudiantes que **no** han presentado
el examen. Esos no están en `dataset_B_completo`, así que esta ejecución no los
cubre. Si el objetivo es que v2 aporte valor al índice, el dataset a puntuar es
el de inscritos **sin** resultado.

## Comparación de `indice_potencial`

Sobre los 1,148 a los que se aplicó v2:

| Estadístico | Valor |
| --- | --- |
| Cambio medio | −0.61 |
| Mediana | −0.47 |
| Mínimo | −3.84 |
| Máximo | +3.39 |
| Sin cambio (\|Δ\| < 0.01) | 0 |

Todos se mueven un poco —lógico, porque la referencia cambia para todos— pero
ninguno se mueve mucho.

### Cambios de categoría

| | Estudiantes |
| --- | --- |
| Suben de categoría | 58 |
| Bajan de categoría | 71 |
| Sin cambio | 1,019 (88.8 %) |

La distribución, sin embargo, se **desplaza hacia los extremos**:

| Categoría | v1 | v2 | Δ |
| --- | --- | --- | --- |
| Talento destacado | 30 | 51 | **+21** |
| Alto potencial | 247 | 263 | +16 |
| Promedio | 435 | 378 | **−57** |
| En desarrollo | 305 | 274 | −31 |
| Requiere apoyo | 131 | 182 | **+51** |

**Cómo leerlo.** El centro se vacía y los extremos se llenan. Es el efecto
esperable de percentilizar contra una cohorte más pequeña y menos dispersa
(σ = 20.54 en C frente a 22.66 en B): los mismos puntajes reales se reparten
sobre un rango percentil más amplio.

Esto tiene consecuencias prácticas: **21 estudiantes más entrarían en "Talento
destacado" y 51 más en "Requiere apoyo"** sin que su desempeño real haya
cambiado en absoluto. Si alguna decisión del programa depende de esas etiquetas,
el cambio de referencia no es cosmético.

## Talento oculto

| Grupo | v1 | v2 | Δ |
| --- | --- | --- | --- |
| Subgrupo v2 (1,148) | 176 | 179 | +3 |
| Subgrupo fallback (1,924) | 431 | 431 | 0 |
| **Total (3,072)** | **607** | **610** | **+3** |

Movimiento despreciable. Coherente: `es_talento_oculto` depende del
`indice_potencial`, que apenas se movió.

## `puntaje_estimado` — donde v2 sí aporta

Esta es la única columna puramente predictiva, y aquí el modelo sí interviene
siempre. Sobre los 1,148 de v2, comparando contra el puntaje real:

| Modelo | MAE | Nota |
| --- | --- | --- |
| v1 | **18.45** | Fuera de muestra (verificado) |
| v2 — medido aquí | 12.54 | **Dentro de muestra** — no usar |
| **v2 — hold-out (script 14)** | **15.00** | **La cifra honesta** |

> **Por qué 12.54 no vale.** El modelo v2 de producción se reajustó sobre las
> 1,148 filas completas (script 14, paso 4), así que al puntuar a esos mismos
> estudiantes se está evaluando sobre su propio conjunto de entrenamiento. El
> 12.54 mide memorización, no capacidad predictiva.
>
> **Por qué 18.45 sí vale.** Se verificó que el solapamiento entre los 1,148 de
> C y las 1,748 filas de entrenamiento de v1 es **exactamente 0**. Ningún
> estudiante de C participó en el entrenamiento de v1, así que su MAE es
> genuinamente fuera de muestra.

La comparación justa es entonces **18.45 vs 15.00: una mejora de 3.45 puntos
(−19 %)**, no los 5.91 que sugiere la lectura ingenua.

## Fichero generado

`outputs/ml_scores_v2.csv` — 3,072 filas × 18 columnas.

Incluye las **17 columnas que la Edge Function gestiona**, más `modelo_version`:

```
numero_documento, indice_potencial, componente_rendimiento,
componente_engagement, componente_resiliencia, categoria_potencial,
es_talento_oculto, probabilidad_talento, n_condiciones_adversas,
condiciones_detalle, cluster_id, cluster_nombre, indice_condiciones,
nivel_condiciones, tiene_puntaje_real, puntaje_estimado, updated_at,
modelo_version
```

**Tres columnas de `ml_scores` quedan fuera**: `nivel_sospecha`,
`n_criterios_sospecha` y `created_at`. No es una omisión: la propia Edge
Function declara que no las toca — las escribe el trigger
`after_resultado_insert` a partir de la telemetría del examen. Reproducirlas
aquí sería inventar valores.

`puntaje_estimado` sí se rellena con la predicción del modelo, a diferencia de
producción, donde está fijado a `null`. Es el objeto de la comparación.

También se generó `outputs/F15_comparacion_v1_v2.csv` con ambas versiones lado a
lado por estudiante, para auditar caso por caso.

## Conclusión

**Desplegar v2 hoy, tal cual, no mejoraría nada visible.** Tres razones
encadenadas:

1. Para los 3,072 con examen presentado, el índice usa el puntaje real; el
   modelo no interviene.
2. La única columna que el modelo alimenta —`puntaje_estimado`— está fijada a
   `null` en la Edge Function.
3. El único cambio que sí llegaría a producción sería el desplazamiento de
   categorías por el cambio de referencia, que **no es una mejora**: es un
   efecto colateral de haber calculado la referencia sobre una cohorte más
   pequeña.

v2 es un modelo mejor —15.00 frente a 18.45 de MAE, fuera de muestra— pero el
sistema actual no tiene por dónde aprovecharlo.

## Recomendaciones

1. **Decidir primero sobre `puntaje_estimado`.** Es el cambio que activa el
   valor de v2. Requiere aceptar publicar una predicción con MAE ≈ 15 puntos,
   siempre con banda de error y nunca como cifra puntual.
2. **Puntuar a los inscritos SIN examen.** Es la población donde el modelo sí
   alimenta el índice. Esta ejecución no los cubre; conviene repetirla sobre
   ellos antes de decidir.
3. **Tratar `ref_rendimiento` como decisión explícita, no como subproducto.**
   Si se despliega v2, conviene calcular la referencia sobre la cohorte completa
   (3,072) y no sobre las 1,148 de C, para no redistribuir categorías sin
   motivo. Es un cambio de una línea en el script 14.
4. **Añadir `modelo_version` a `ml_scores`** antes de cualquier despliegue
   híbrido. Sin esa columna es imposible saber qué modelo produjo cada fila, y
   el upsert de la Edge Function no guarda historial.
5. **Volcar `ml_scores` a CSV antes de tocar nada.** El upsert sobrescribe en
   sitio y no hay rollback.

## Limitaciones

- **Cohorte parcial.** Solo estudiantes con resultado de examen. La población
  donde v2 más aportaría —los que no han presentado— queda fuera por definición
  del dataset.
- **El MAE de v2 aquí es in-sample** y se reporta solo para dejar constancia de
  por qué no debe usarse. La cifra válida viene del hold-out del script 14.
- **Los otros tres predictores son los de v1.** Talento oculto, clustering y
  condiciones no se reentrenaron (ver informe 14), así que sus diferencias entre
  columnas v1 y v2 provienen únicamente del `indice_potencial` que reciben.
- **No se validó contra la tabla real.** No hay acceso a Supabase desde este
  repositorio, así que no se pudo comparar fila a fila contra los valores hoy
  almacenados en `ml_scores`, solo contra un recálculo de v1.
- **`condiciones_detalle` se serializa como JSON** en el CSV; al cargarlo habrá
  que respetar el tipo de la columna en Postgres.

## Glosario

- **Ejecución en sombra:** *definición* — correr un modelo nuevo en paralelo sin
  que sus salidas afecten a producción. *Analogía* — un piloto en prácticas con
  los mandos desconectados. *Ejemplo* — este CSV en `outputs/`.
- **Distribución de referencia (`ref_rendimiento`):** *definición* — lista
  ordenada de puntajes contra la que se calcula el percentil de cada estudiante.
  *Analogía* — la curva del curso con la que se nota sobre 100. *Ejemplo* —
  1,750 puntajes en v1, 1,148 en v2.
- **In-sample vs out-of-sample:** *definición* — evaluar sobre datos vistos o no
  vistos durante el entrenamiento. *Analogía* — examinarse con los mismos
  ejercicios que se practicaron, o con otros nuevos. *Ejemplo* — 12.54 frente a
  15.00 de MAE.
- **Fallback:** *definición* — modelo de respaldo cuando el principal no es
  aplicable. *Analogía* — la rueda de repuesto. *Ejemplo* — v1 para los 1,924
  sin perfil académico.

---
_Generado a partir de `notebooks/15_generar_scores_v2.py` — Copa STEM 2026._
