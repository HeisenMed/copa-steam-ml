# Regeneración del artefacto de despliegue v2 con la referencia corregida — Copa STEM 2026

**Fundación SapienceLab** · Fase 5 · Informe generado: 2026-09-02

---

## Resumen ejecutivo

El script 17 corrigió `ref_rendimiento` —la distribución contra la que se
percentiliza el rendimiento— y dejó el resultado en
`outputs/F17_ref_rendimiento_corregido.json`. Esa corrección se quedó en el
lado Python/CSV. El artefacto que consume la Edge Function,
`models/deploy/potencial_stem_predictor_v2.js`, **seguía llevando embebida la
referencia vieja de 1,148**. Desplegarlo tal cual habría reintroducido, por la
puerta de atrás, el problema que el script 17 existió para eliminar.

El script 19 propaga la corrección al artefacto:

- **Antes:** `ref_rendimiento` de **n = 1,148**, σ = **20.5332**, calculada
  sobre `dataset_C_perfil.csv` (el dataset de ENTRENAMIENTO del modelo v2).
- **Después:** `ref_rendimiento` de **n = 3,072**, σ = **22.6583**, calculada
  sobre `dataset_B_completo.csv` (todos los examinados), tomada literalmente de
  `outputs/F17_ref_rendimiento_corregido.json`.
- **Precisión verificada:** el `.js` generado reproduce a `sklearn` con
  **máx \|Δ\| = 2.842 × 10⁻¹⁴** sobre 300 filas — la misma precisión de la
  verificación original del script 14.
- **El fichero viejo no se tocó.** El resultado se escribió como fichero nuevo,
  `models/deploy/potencial_stem_predictor_v2_corrected.js`.

## Qué se cambió, en una frase

Dentro del `.js` hay una única constante, `SPEC`. De sus seis claves solo
cambian dos: `ref_rendimiento` (la lista de 1,148 puntajes pasa a ser la de
3,072) y `meta` (la ficha de procedencia). El modelo, el preprocesamiento, los
rangos de engagement, los pesos y los umbrales de categoría quedan idénticos.

| Clave del `SPEC` | ¿Cambia? | Qué es |
| --- | --- | --- |
| `puntaje.model` | **No** | Los 200 árboles del Random Forest v2 |
| `puntaje.preprocess` | **No** | Medianas, modas y categorías one-hot del entrenamiento |
| `engagement` | **No** | Rangos lo/hi de `n_herramientas` y `n_areas_interes` |
| `pesos` | **No** | 0.50 rendimiento · 0.25 engagement · 0.25 resiliencia |
| `categorias` | **No** | Los 5 umbrales (85 / 70 / 45 / 25 / 0) |
| `ref_rendimiento` | **Sí** | 1,148 puntajes → 3,072 puntajes |
| `meta` | **Sí** | Procedencia: script, fecha, cohorte y σ de la referencia |

El script aborta si esa comprobación encuentra cualquier otra clave modificada.
En la corrida, las claves que cambian fueron exactamente `meta` y
`ref_rendimiento`.

## La referencia: antes y después

| Referencia | n | Media | σ | p25 | Mediana | p75 | Cohorte |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Embebida hasta hoy | 1,148 | 41.08 | **20.5332** | 25.0 | 40.0 | 55.0 | `dataset_C_perfil.csv` (entrenamiento) |
| **Corregida** | **3,072** | 41.74 | **22.6583** | 25.0 | 40.0 | 55.0 | `dataset_B_completo.csv` (examinados) |

Es la misma sustitución del informe 17 y por la misma razón: percentilizar es
siempre "contra quién", y ese quién son los 3,072 que presentaron la prueba, no
el subconjunto que además respondió las 5 preguntas de perfil académico. Que C
sea el dataset de entrenamiento del modelo es irrelevante para esa decisión.

El script comprueba además que la referencia llega ordenada de forma ascendente:
`_percentil` hace búsqueda binaria sobre ella y una lista desordenada daría
percentiles silenciosamente incorrectos.

### El efecto, en el propio artefacto

El `.js` trae al final un estudiante de demostración. Ejecutando los dos
ficheros con `node`, sin tocar nada más:

| Salida | Artefacto vigente (ref 1,148) | Artefacto corregido (ref 3,072) |
| --- | --- | --- |
| `indice_potencial` | 44.66 | **44.99** |
| `componente_rendimiento` | 49.04 | **49.48** |
| `componente_engagement` | 31.53 | 31.53 |
| `componente_resiliencia` | 49.04 | **49.48** |
| `categoria` | En desarrollo | En desarrollo |

El engagement no se mueve —no pasa por el percentil— y el rendimiento sí. Es
exactamente la firma del cambio que se buscaba.

## Cómo se generó

Se reutilizó el proceso de exportación del script 14, con una diferencia: en vez
de clonar el cuerpo del `.js` v1, se clona el del **v2** y se sustituye solo la
línea de la constante `SPEC`. Así el cuerpo del artefacto desplegado se conserva
byte a byte y el contrato con la Edge Function no cambia.

1. Se carga `models/mejor_modelo_puntaje_v2.joblib` (200 árboles, 23 features,
   hold-out R² +0.1766 / MAE 15.00 sobre n = 230) y se serializan sus árboles
   con el mismo extractor del script 14.
2. **Comprobación de procedencia:** los árboles y el preprocesamiento extraídos
   del `.joblib` se comparan bit a bit contra los embebidos en el `.js` vigente.
   Ambos dieron `True`. Si no coincidieran, el `.js` no vendría de ese `.joblib`
   y regenerarlo desde ahí no tendría sentido; el script aborta en ese caso.
3. Se construye el `SPEC` nuevo con `ref_rendimiento` leída del JSON del
   script 17.
4. Se escribe el fichero nuevo y se verifica.

El diff contra el artefacto vigente es de **2 líneas quitadas y 6 puestas**,
sobre 208 → 212 líneas: la línea de la constante `SPEC`, la línea `GENERADO
por…` de la cabecera y cuatro líneas de comentario que dejan escrito dentro del
propio fichero que la referencia es la corregida. Todo lo demás —las funciones
`_featuresPuntaje`, `_predictPuntaje`, `_percentil`, `_engagement`,
`_adversidad`, `calcularIndicePotencial`— queda byte a byte igual.

## Verificación de precisión

La muestra son **300 filas**: 200 de `dataset_C_perfil.csv` (la ruta normal de
v2, con las 5 variables de perfil) más 100 de `dataset_B_completo.csv` **sin**
perfil académico, elegidas con `random_state=42`. Las segundas importan porque
ejercitan la imputación por mediana/moda del `SPEC`, que es justo donde el JS y
`sklearn` divergirían si el preprocesamiento no fuera idéntico. Las mismas filas
—ya normalizadas: `NaN` → `null`— se le pasan a Python y a Node, de modo que la
comparación mida el código y no el paso por JSON.

| Comparación | Máx \|Δ\| | Veredicto |
| --- | --- | --- |
| Intérprete de árboles en Python vs `sklearn.predict` | **2.842 × 10⁻¹⁴** | OK |
| **`_predictPuntaje` del `.js` generado vs `sklearn.predict`** | **2.842 × 10⁻¹⁴** | **OK** |
| Índice compuesto: `.js` generado vs predictor Python de `models/deploy/` | 1.00 × 10⁻² | Ver abajo |

La fila que importa es la segunda: es la misma precisión —del orden de 10⁻¹⁴,
el error de redondeo de un `float64` acumulado sobre 200 árboles— que verificó
el script 14 cuando generó el artefacto original. El `.js` se ejecutó con
**Node v24.13.0** importándolo de dos formas: intacto (para
`calcularIndicePotencial`, que es lo que consume la Edge Function) y con una
línea `export` añadida al final de una copia temporal, para poder leer
`_predictPuntaje` a precisión completa —el artefacto redondea a 2 decimales en
su salida pública—.

### Sobre la diferencia de 0.01 en el índice compuesto

31 de las 300 filas difieren en exactamente 0.01 en el último decimal. **No es
un error de cálculo: es el modo de redondeo.** Python `round()` redondea al par
y `Math.round()` del JS redondea la mitad hacia arriba. Aplicando el criterio
del JS a los valores **sin redondear** de Python, la diferencia máxima cae a
**0.000 × 10⁰** en las 300 filas. Las 269 restantes ya coincidían exactamente en
los 4 componentes, y **la categoría coincidió en las 300**.

Es una propiedad preexistente del par JS/Python —vive en `_round2` frente a
`round()`, no en nada que este script haya tocado— y afecta igual al artefacto
vigente. Se deja anotada, no corregida: corregirla cambiaría el cuerpo del JS,
que es precisamente lo que este script se comprometió a no tocar.

## Qué no se tocó

El script toma el SHA-256 de los 21 ficheros de `models/` y `models/deploy/`
antes y después de ejecutarse. **Ninguno cambió de hash**, incluidos:

- `models/deploy/potencial_stem_predictor_v2.js` — el artefacto vigente, intacto
  y disponible para comparación y rollback. El script aborta si la ruta de
  salida coincidiera con él.
- `models/mejor_modelo_puntaje_v2.joblib` y el resto de modelos: solo se leyeron.
- `potencial_stem_predictor.js` / `.py`, `talento_oculto_predictor.*`,
  `clustering_predictor.*`, `indice_condiciones_predictor.*` y los CSV de
  `models/deploy/`.

Tampoco se modificó ningún script numerado anterior, ningún fichero previo de
`outputs/`, ni Supabase, ni la Edge Function. Todo el cálculo es local.

**El único fichero nuevo en `models/deploy/` es
`potencial_stem_predictor_v2_corrected.js`** (828 KB). El detalle completo de la
verificación queda en `outputs/F19_verificacion_deploy_v2.json`.

## Qué queda pendiente

Este script deja el artefacto listo; **no lo despliega**. Antes de ponerlo en
producción:

1. La Edge Function sigue sin leer las 5 variables de perfil académico
   (`promedio_academico`, `horas_estudio_matematicas`, `motivacion_participar`,
   `clases_extra_matematicas`, `gusto_logica`). Sin ellas, el modelo v2 recibe
   las medianas del `SPEC` en su lugar y se comporta como un v1 caro.
2. Decidir el nombre definitivo del fichero en el despliegue. Mantener el
   sufijo `_corrected` o promoverlo a `potencial_stem_predictor_v2.js` es una
   decisión de operación, no de modelado; mientras tanto el viejo sigue ahí para
   revertir.
3. La recomendación del informe 17 sigue en pie: **`ref_rendimiento` debe
   declararse como decisión de producto** —"comparamos contra todos los
   examinados"— y no heredarse del dataset de entrenamiento que toque en cada
   versión. Este script es la consecuencia de que no lo estuviera.

## Glosario

- **Artefacto de despliegue:** *definición* — el fichero que se copia a
  producción y ejecuta el modelo sin dependencias. *Analogía* — la foto
  revelada, no el negativo. *Ejemplo* — `potencial_stem_predictor_v2.js`, 828 KB
  de JavaScript con los 200 árboles escritos dentro.
- **Propagar una corrección:** *definición* — llevar un arreglo hasta todos los
  sitios donde vive la misma cifra. *Analogía* — corregir la receta y también la
  copia pegada en la nevera. *Ejemplo* — el script 17 arregló el CSV; hasta hoy
  el `.js` seguía con la cifra vieja.
- **Modo de redondeo:** *definición* — la regla para desempatar cuando un valor
  cae justo en la mitad. *Analogía* — 2.5 puede ir a 2 o a 3 según el convenio.
  *Ejemplo* — las 31 filas que difieren en 0.01 entre el `.js` y Python.

---
_Generado a partir de `notebooks/19_regenerar_deploy_v2.py` — Copa STEM 2026._
