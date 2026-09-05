# Corrección de `ref_rendimiento` — población de referencia completa — Copa STEM 2026

**Fundación SapienceLab** · Fase 5 · Informe generado: 2026-08-30

---

## Resumen ejecutivo

El informe 15 encontró que 72 estudiantes cambiaban de categoría de potencial
**sin que su desempeño hubiera cambiado en nada**. La causa no era el modelo:
era la vara de medir. El SPEC v2 percentilizaba los puntajes contra los 1,148
estudiantes de `dataset_C_perfil.csv` en lugar de contra los 3,072 que
presentaron la prueba.

Este script corrige exactamente eso y nada más.

- **La redistribución artificial desaparece.** "Talento destacado" vuelve de
  126 a 105 estudiantes —el mismo número que daba v1— y "Requiere apoyo" baja
  de 478 a 456.
- **El desplazamiento hacia los extremos se reduce de 72 a 29 estudiantes** en
  la cohorte completa, y de 72 a 17 en el subgrupo de v2.
- **Las predicciones del modelo v2 no se tocan.** `puntaje_estimado` sale
  idéntico fila a fila al del script 15: máximo \|Δ\| = 0.000000 sobre 3,072
  filas.
- No se modificó ningún script numerado anterior, ningún artefacto de
  `models/deploy/`, ni Supabase, ni la Edge Function. Todo el cálculo es local.

## Qué se cambió, en una frase

`ref_rendimiento` es la lista ordenada de puntajes contra la que se calcula el
percentil de cada estudiante. Antes se construía sobre 1,148 estudiantes; ahora
se construye sobre los 3,072 de `dataset_B_completo.csv` (tabla
`resultados_prueba_copa_stem`). Nada más cambia.

## Por qué la referencia estaba mal

El componente de rendimiento —el 50 % del índice— no usa el puntaje crudo, usa
su **percentil**. Percentilizar siempre es "contra quién", y ese "quién" debe
ser la población con la que tiene sentido comparar: todos los que presentaron la
prueba.

El script 14 entrenó el modelo v2 sobre el dataset C (los que además
respondieron las 5 preguntas de perfil académico) y, de paso, calculó la
referencia sobre esa misma población. Son dos usos distintos de los datos que
quedaron acoplados sin necesidad: para **entrenar** hace falta C, porque es
donde están las variables nuevas; para **comparar** hace falta B, porque es
donde están todos los examinados.

Las tres distribuciones de referencia:

| Referencia | n | Media | σ | p25 | Mediana | p75 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (cohorte histórica) | 1,750 | 41.81 | 23.11 | 25.0 | 35.0 | 60.0 |
| v2 tal como está (dataset C) | 1,148 | 41.08 | **20.53** | 25.0 | 40.0 | 55.0 |
| **v2 corregida (dataset B)** | **3,072** | 41.74 | **22.66** | 25.0 | 40.0 | 55.0 |

La cohorte C es **más apretada** (σ 20.53 frente a 22.66). Al medir contra una
población menos dispersa, un puntaje que antes caía en el montón central se
despega hacia un extremo: no porque el estudiante haya mejorado o empeorado,
sino porque sus vecinos de comparación se parecen más entre sí.

## Antes y después

### Subgrupo de v2 (1,148 estudiantes)

Esta es la misma tabla del informe 15, con una columna añadida. Las dos primeras
columnas son idénticas a las que ya se publicaron.

| Categoría | v1 (ref 1,750) | v2 antes (ref 1,148) | v2 corregido (ref 3,072) | Δ vs antes | Δ vs v1 |
| --- | --- | --- | --- | --- | --- |
| Talento destacado | 30 | **51** | **28** | −23 | −2 |
| Alto potencial | 247 | 263 | 251 | −12 | +4 |
| Promedio | 435 | **378** | **421** | +43 | −14 |
| En desarrollo | 305 | 274 | 302 | +28 | −3 |
| Requiere apoyo | 131 | **182** | **146** | −36 | +15 |

El patrón se invierte limpiamente: donde antes el centro se vaciaba (−57 y −31)
y los extremos se llenaban (+21 y +51), ahora el centro se rellena (+43 y +28) y
los extremos se desinflan (−23 y −36).

### Cohorte completa (3,072 estudiantes)

| Categoría | v1 | v2 antes | v2 corregido | Δ vs antes | Δ vs v1 |
| --- | --- | --- | --- | --- | --- |
| Talento destacado | 105 | 126 | **105** | −21 | **0** |
| Alto potencial | 699 | 715 | 721 | +6 | +22 |
| Promedio | 1,052 | 995 | 1,006 | +11 | −46 |
| En desarrollo | 789 | 758 | 784 | +26 | −5 |
| Requiere apoyo | 427 | 478 | 456 | −22 | +29 |

"Talento destacado" aterriza en 105, exactamente el conteo de v1.

### Cuántos estudiantes cambian de etiqueta

| Comparación | Cohorte completa | Subgrupo v2 |
| --- | --- | --- |
| v1 → v2 antes | 129 (58 suben / 71 bajan) | 129 (58 / 71) |
| v1 → v2 corregido | 81 (25 / 56) | **33 (3 / 30)** |
| v2 antes → v2 corregido | 150 (66 / 84) | 102 (44 / 58) |
| **Desplazamiento neto a los extremos vs v1** | **72 → 29** | **72 → 17** |

La última fila es la medida directa del artefacto: suma de los desbordes en
"Talento destacado" y "Requiere apoyo" respecto a v1. Era de 72 estudiantes y
queda en 29.

> **Los dos "129" no son casualidad.** En el escenario anterior los 1,924 sin
> perfil académico se puntuaban con el SPEC v1 (fallback), así que sus
> categorías eran las de v1 por construcción. Todo el movimiento v1 → v2 venía
> del subgrupo de 1,148.

### Por qué los 1,924 del fallback también se mueven

De los 150 que cambian entre "antes" y "corregido", **48 pertenecen al grupo
fallback**. No es un efecto secundario indeseado: su referencia era la cohorte
histórica de 1,750 estudiantes, y ahora pasa a ser la de 3,072. Esos 1,750
puntajes están contenidos íntegramente en los 3,072 —se verificó como multiset,
0 valores sin correspondencia—, es decir, la referencia vieja era la misma
población pero desactualizada, antes de que se sumaran los examinados de agosto.

Corregir a los 1,148 y dejar a los 1,924 midiéndose contra una vara antigua
habría creado un segundo artefacto: dos escalas distintas conviviendo en la
misma tabla.

### Magnitud del movimiento del índice

| Grupo | n | Δ medio | Δ mediano | Rango |
| --- | --- | --- | --- | --- |
| Cohorte completa | 3,072 | −0.10 | −0.05 | −2.70 … +2.13 |
| Subgrupo v2 | 1,148 | −0.06 | −0.21 | −2.70 … +2.13 |
| Fallback | 1,924 | −0.13 | −0.05 | −1.01 … +1.02 |

En escala 0–100 esto es ruido. El problema nunca fue el tamaño del movimiento,
sino que cruzaba umbrales de categoría en los sitios equivocados.

### Talento oculto

| Escenario | Talentos ocultos | Δ vs v1 |
| --- | --- | --- |
| v1 | 607 | — |
| v2 antes (ref C) | 610 | +3 |
| **v2 corregido (ref B)** | **609** | **+2** |

Movimiento despreciable, como era de esperar: `es_talento_oculto` depende del
`indice_potencial`, que apenas se mueve.

## Verificación de que el modelo no se tocó

`ref_rendimiento` solo entra en `_percentil`; no aparece en `_predict_puntaje`.
El paso 5 del script lo comprueba fila a fila contra `outputs/ml_scores_v2.csv`:

| Columna | Filas distintas |
| --- | --- |
| `puntaje_estimado` | 0 (máx \|Δ\| = 0.000000) |
| `componente_engagement` | 0 |
| `indice_condiciones` | 0 |
| `cluster_id` | 0 |
| `modelo_version` | 0 |

Cambian únicamente las columnas que dependen del percentil:
`componente_rendimiento`, `componente_resiliencia` (que se deriva del anterior),
`indice_potencial`, `categoria_potencial` y, por arrastre, `es_talento_oculto`.

La estrategia híbrida se conserva intacta: v2 para los 1,148 con
`promedio_academico`, v1 para los 1,924 restantes.

## Ficheros generados

| Fichero | Contenido |
| --- | --- |
| `outputs/ml_scores_v2_corrected.csv` | 3,072 filas × 18 columnas, mismo esquema que `ml_scores_v2.csv` |
| `outputs/F17_comparacion_categorias.csv` | Tabla antes/después por categoría y ámbito |
| `outputs/F17_ref_rendimiento_corregido.json` | La nueva referencia (3,072 puntajes ordenados) lista para inyectar en un SPEC |

El esquema de 18 columnas se mantiene exactamente igual, con las mismas tres
ausencias justificadas (`nivel_sospecha`, `n_criterios_sospecha`, `created_at`,
que escribe el trigger `after_resultado_insert`, no la Edge Function).

## Lo que esto NO arregla

- **Sigue sin cambiar nada para quien ya presentó el examen.** El hallazgo
  central del informe 15 se mantiene: el índice usa el puntaje real de los
  3,072, así que el modelo v2 no interviene. Esta corrección elimina un efecto
  colateral dañino; no añade valor predictivo.
- **La Edge Function sigue sin leer las 5 columnas nuevas.** Un `grep
  promedio_academico` sobre `index.ts` sigue dando cero coincidencias. Ese es el
  próximo paso.
- **`puntaje_estimado` sigue fijado a `null` en producción.**
- **La referencia envejece.** Está calculada sobre los examinados a agosto de
  2026. Cada nueva tanda de resultados la desactualiza; conviene tratarla como
  un artefacto que se regenera, no como una constante.
- **No se escribió ningún artefacto de despliegue.** Para llevar esto a
  producción habría que regenerar `potencial_stem_predictor_v2.js` con la
  referencia de `F17_ref_rendimiento_corregido.json`. Queda fuera de este paso a
  propósito.

## Recomendación

Adoptar `ml_scores_v2_corrected.csv` como la línea base para cualquier decisión
sobre categorías de potencial, y descartar `ml_scores_v2.csv` para ese uso
(sigue siendo válido para leer el efecto del modelo en `puntaje_estimado`).

Si más adelante se despliega v2, la referencia debe declararse explícitamente
como decisión de producto —"comparamos contra todos los examinados"— y no
heredarse del dataset de entrenamiento que toque en cada versión.

## Glosario

- **Distribución de referencia (`ref_rendimiento`):** *definición* — lista
  ordenada de puntajes contra la que se calcula el percentil de cada estudiante.
  *Analogía* — la curva del curso con la que se nota sobre 100. *Ejemplo* —
  1,148 puntajes antes, 3,072 ahora.
- **Artefacto de referencia:** *definición* — cambio en un resultado que viene
  de con quién se compara, no de lo que se mide. *Analogía* — el mismo corredor
  parece más rápido si se cambia de rivales. *Ejemplo* — los 72 estudiantes que
  cambiaban de categoría con el mismo puntaje.

---
_Generado a partir de `notebooks/17_fix_ref_rendimiento.py` — Copa STEM 2026._
