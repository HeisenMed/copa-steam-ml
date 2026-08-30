# Preparación del Experimento de Reentrenamiento — Copa STEM 2026

**Fundación SapienceLab** · Fase 5 · Informe generado: 2026-08-29

---

## Resumen ejecutivo

Se preparó el terreno para un experimento **controlado** que separa dos efectos
que hasta ahora estaban mezclados: el de tener **más datos** y el de tener
**mejores variables**. A partir del export de agosto de 2026
(**3,077 inscritos**, 31 columnas) se construyeron cuatro datasets:

- **A (baseline):** 1,735 filas — la cohorte original del primer modelo.
- **B (completo):** 3,072 filas — todos los que presentaron el examen.
- **C (perfil):** 1,148 filas — los que declararon perfil académico.
- **C′ (control):** las **mismas** 1,148 filas de C, sin las 5 variables nuevas.

La pieza clave es **C′**. Sin un grupo de control con la muestra fija, cualquier
mejora observada en C sería ambigua: no se sabría si mejoró por las variables
nuevas o porque C es una población distinta (más reciente, más motivada, de otros
municipios). C y C′ comparten filas y partición train/test, así que la diferencia
entre ambos es atribuible **solo** al bloque de variables nuevas.

Este script **no entrena nada**: prepara, reporta y deja el experimento listo
para el script 12.

## ¿Por qué un experimento controlado?

El primer modelo (script 03) alcanzó un R² modesto y `puntaje_estimado` quedó en
NULL en producción. Ante ese resultado hay dos hipótesis rivales y opuestas en sus
consecuencias operativas:

- **Hipótesis del volumen:** faltan datos. Con el doble de inscritos el modelo
  aprenderá los patrones que hoy no ve. *Consecuencia:* esperar y recolectar.
- **Hipótesis de las variables:** faltan las variables correctas. Los datos
  socioeconómicos describen el *origen* del estudiante, no su *conducta*.
  *Consecuencia:* cambiar el formulario, no esperar.

Comparar un modelo viejo contra uno nuevo entrenado con más datos **y** más
variables no distingue entre las dos: cambian dos cosas a la vez. El diseño de
cuatro datasets aísla cada factor por separado.

## Los cuatro datasets

| Dataset | Archivo | Filas | Columnas | Qué aísla |
| --- | --- | --- | --- | --- |
| A | `dataset_A_baseline.csv` | 1,735 | 31 | Línea base — punto de partida |
| B | `dataset_B_completo.csv` | 3,072 | 31 | Efecto de **más datos** |
| C | `dataset_C_perfil.csv` | 1,148 | 31 | Efecto de **más variables** |
| C′ | `dataset_C_sin_features.csv` | 1,148 | 26 | **Control** de C |

Los conteos de columnas son los del archivo. En términos de *features* de
modelado —excluyendo `numero_documento`, el target `puntaje_obtenido` y
`porcentaje`— A, B y C tienen 28 y C′ tiene 23.

**`porcentaje` se excluye siempre.** Es un duplicado exacto de
`puntaje_obtenido` en escala 0–100: incluirlo daría un R² cercano a 1 que no
significa nada, porque el modelo estaría leyendo la respuesta. Es el caso de
libro de *fuga de información* (data leakage).

## Identificación de la cohorte A

El dataset A debe reproducir las filas con las que se entrenó el primer modelo.
El script resuelve esto con una **cascada de tres criterios**, de más a menos
fiable, y deja registrado en consola cuál usó:

1. **Por `created_at`** — las 1,735 inscripciones más antiguas.
2. **Por índice guardado** — los `numero_documento` que el primer modelo puntuó,
   leídos de `models/deploy/puntaje_estimado.csv`.
3. **Muestra aleatoria** de 1,735 filas con `random_state=42`.

**En esta corrida se usó el criterio 1 (`created_at`)**, porque el export de
agosto sí trae esa columna. El resultado son exactamente 1,735 filas, con rango
de inscripción del 2026-05-22 al inicio de la ventana.

> **Nota metodológica.** El criterio 2 (emparejar por `numero_documento` contra
> el export original) estaba disponible y **habría dado un resultado distinto**:
> el índice guardado tiene 1,748 documentos, pero **solo 1,622 siguen presentes**
> en el export de agosto, y el solapamiento con las 1,735 más antiguas es de
> **1,454**. Es decir, las dos rutas discrepan en unas 280 filas. Se prefirió
> `created_at` porque define una cohorte temporal íntegra, mientras que el índice
> guardado ya perdió 126 estudiantes (bajas, anulaciones o reinscripciones) y
> produciría una cohorte incompleta.

**Verificación de coherencia.** Ninguna de las 1,735 filas más antiguas tiene
perfil académico (0 de 1,735). Esto es exactamente lo esperado si las 5 preguntas
se añadieron al formulario después de esa cohorte, y confirma que la partición
temporal separa bien los dos regímenes de recolección.

## Caracterización socioeconómica completa

El export de agosto llega **totalmente caracterizado**: de 3,077 filas,
**3,076 tienen los 13 campos socioeconómicos completos** y una sola carece de un
campo (`interes_prog_robotica`). No hay ningún estudiante sin caracterizar.

Esto es consecuencia de que la consulta de exportación en Supabase se resolvió
con un **INNER JOIN** contra la tabla de caracterización, no con un LEFT JOIN.
La decisión ocurre **aguas arriba del script 11**: cuando el CSV llega a
`data/`, los estudiantes sin caracterización ya no están en él.

**Por qué el INNER JOIN es la decisión correcta para entrenar.** Un modelo
supervisado necesita el vector de features completo. Un estudiante inscrito de
urgencia el día del examen —sin estrato, sin jornada, sin datos de acceso a
tecnología— aporta un target pero casi ningún predictor. Incluirlo obligaría a
imputar la mayoría de sus variables con la mediana del grupo, lo que equivale a
**inventar un estudiante promedio** y presentarlo al modelo como observación
real. El efecto es doble y siempre malo:

- **Diluye la señal:** filas que son casi todas medianas empujan al modelo hacia
  predecir la media, achatando el R².
- **Sesga las brechas:** los análisis de equidad (scripts 02, 05c) leerían esas
  medianas imputadas como si fueran estrato o acceso reales.

Para un conteo de participación, esos estudiantes cuentan igual que cualquier
otro y deben aparecer en los reportes de inscripción. Para **entrenar**, no.
Son exclusiones legítimas del dataset de modelado, no bajas del programa.

> **Pendiente de confirmar.** El número exacto de estudiantes que el INNER JOIN
> dejó fuera no es observable desde el CSV —están ausentes por construcción—, así
> que este informe no lo afirma. Para documentarlo hay que contrastar el conteo
> de la tabla de inscripciones contra las 3,077 filas exportadas, del lado de
> Supabase.

## Limpieza aplicada

El reporte de estado se emite sobre el export **crudo**, y solo después se limpia,
de modo que los conteos finales siempre se puedan reconciliar contra el original:

| Paso | Filas retiradas | Resultado |
| --- | --- | --- |
| Export crudo | — | 3,077 |
| Documentos de prueba (`1234`, `123456`, …) | 0 | 3,077 |
| Duplicados por `numero_documento` | 5 | 3,072 |

Los duplicados se resuelven conservando la **primera** aparición, porque
`numero_documento` es la clave primaria en Supabase y las repeticiones son
reinscripciones, no estudiantes distintos.

## Distribución del target

Sobre las 3,077 filas del export crudo:

| Estadístico | Valor |
| --- | --- |
| n | 3,077 |
| Media | 41.72 |
| Desviación estándar | 22.68 |
| Mínimo | 0.00 |
| Q1 (25%) | 25.00 |
| Mediana | 40.00 |
| Q3 (75%) | 55.00 |
| Máximo | 100.00 |

La media por dataset cambia de forma relevante para la interpretación posterior:
A tiene 42.52 (σ = 24.23), B 41.74 (σ = 22.66) y C 41.08 (σ = 20.54). **C es una
población menos dispersa**, lo que hace que su MAE sea naturalmente más bajo sin
que el modelo sea mejor. Es la razón por la que el script 12 compara A contra B
usando R² y no MAE.

## Cobertura de las 5 variables nuevas

| Variable | Con dato | Cobertura |
| --- | --- | --- |
| `promedio_academico` | 1,148 | 37.3 % |
| `horas_estudio_matematicas` | 1,150 | 37.4 % |
| `motivacion_participar` | 1,176 | 38.2 % |
| `clases_extra_matematicas` | 1,109 | 36.0 % |
| `gusto_logica` | 1,177 | 38.3 % |

El dataset C se define por `promedio_academico` no nulo (1,148 filas), que es la
más restrictiva de las cinco. Las 1,148 filas de C tienen **todas** puntaje, así
que las 1,148 son entrenables y no hubo que descartar ninguna.

## Recomendaciones

1. **Renombrar el export.** El archivo quedó guardado como
   `copa_stem_dataset_2026-08.csv.csv` (doble extensión). El script lo resuelve,
   pero conviene corregir el nombre.
2. **Documentar el conteo del INNER JOIN** del lado de Supabase, para dejar
   trazado cuántos inscritos quedan fuera del universo de modelado y por qué.
3. **Subir la cobertura del perfil académico.** Con 37 % de cobertura, C es la
   mitad de B. Si las 5 preguntas se vuelven obligatorias, el dataset de
   modelado crece sin necesidad de esperar una nueva convocatoria.
4. **No versionar los cuatro CSV generados** salvo que se quiera congelar el
   experimento: se regeneran de forma determinista desde el export.

## Limitaciones

- **La cohorte A es una reconstrucción, no un registro.** Se identifica por
  fecha de inscripción, no por un log de qué filas entraron al entrenamiento
  original. Las 280 filas en que las dos rutas de la cascada discrepan son una
  medida directa de esa incertidumbre.
- **C no es una muestra aleatoria de B.** Son los inscritos más recientes, que
  responden a un formulario más largo. Cualquier diferencia entre C y B mezcla el
  efecto de las variables con el de la población — por eso el experimento
  descansa en C vs C′ y no en C vs B.
- **El INNER JOIN se hereda, no se controla.** El script trabaja con lo que llega
  en el CSV; no puede auditar a quién dejó fuera la consulta de exportación.

## Glosario

- **Grupo de control:** *definición* — conjunto idéntico al experimental salvo en
  el factor que se estudia. *Analogía* — dos macetas con la misma tierra, agua y
  luz, donde solo una recibe fertilizante. *Ejemplo* — C′ frente a C.
- **Fuga de información (data leakage):** *definición* — usar como predictor algo
  que incorpora la respuesta. *Analogía* — predecir quién ganó el partido usando
  el marcador final. *Ejemplo* — incluir `porcentaje` para predecir
  `puntaje_obtenido`.
- **INNER JOIN:** *definición* — cruce que conserva solo los registros presentes
  en ambas tablas. *Analogía* — la lista de quienes se inscribieron **y**
  llenaron la encuesta. *Ejemplo* — inscripciones × caracterización.
- **Imputación:** *definición* — rellenar un valor faltante con una estimación.
  *Analogía* — completar una casilla en blanco con el valor típico del grupo.
  *Ejemplo* — poner la mediana del estrato a quien no lo declaró.

---
_Generado a partir de `notebooks/11_preparar_experimento_reentrenamiento.py` — Copa STEM 2026._
