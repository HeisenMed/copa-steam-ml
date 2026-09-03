
# Informe Completo de Machine Learning — Copa STEM 2026
### Fundación SapienceLab · Documento de referencia científico-pedagógico

> **Propósito de este documento.** Consolidar todo el trabajo de Data Science y
> Machine Learning realizado para Copa STEM 2026 en un solo documento
> autocontenido. Está escrito para un lector con formación técnica (sabe
> programar) que quiere **entender cada técnica por dentro**: qué es, por qué se
> eligió, cómo funciona y cómo replicarla. Cada afirmación va acompañada de su
> evidencia numérica (p-value, R², N, tamaño de efecto) y de la figura que la
> respalda en `outputs/`.

**Fecha:** 2026-07-06 (dataset v3, estratos corregidos) · **Autor:** Equipo de Datos — Fundación SapienceLab
**Entorno:** Python 3.14 · scikit-learn 1.8 · XGBoost 3.3 · LightGBM 4.6 ·
statsmodels 0.14 · pandas 3.0 · `random_state=42`

> **Corrección de dato (v3, jul-2026).** En Copacabana, Girardota y Bello el
> estrato socioeconómico solo llega a **3**. Los valores 4/5/6 del formulario eran
> **errores de autorreporte** y se reclasificaron a 3 en el dataset. Todo este
> informe y todos los modelos/predictores fueron **re-ejecutados desde cero** con
> los estratos corregidos (distribución: estrato 1 = 95, 2 = 786, 3 = 797, sin
> dato = 131). Las cifras de esta versión reemplazan a las de la anterior.

> **Regla de este documento (vigente desde 2026-09-02).** Este archivo es la
> **narración corrida** del proyecto: **cada script que se termine añade aquí su
> propia sección**, al final, en vez de dejar la historia repartida solo entre
> informes numerados sueltos. El informe individual de cada script sigue
> existiendo en `reports/NN_*.md` con todo el detalle; lo que se añade aquí es el
> relato —qué pregunta respondía ese paso, qué se hizo, qué se encontró— para que
> el documento se pueda leer de principio a fin sin abrir nada más. Las secciones
> anteriores **no se reescriben**: solo se añade al final, y el bloque
> *Estado actual* se actualiza.

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Marco teórico y conceptual](#2-marco-teórico-y-conceptual)
3. [Metodología (pipeline paso a paso)](#3-metodología-pipeline-paso-a-paso)
4. [Resultados por fase](#4-resultados-por-fase)
5. [Recomendaciones para la Fundación](#5-recomendaciones-para-la-fundación)
6. [Guía de replicación](#6-guía-de-replicación)
7. [Limitaciones](#7-limitaciones)
8. [Referencias](#8-referencias)
9. [Fase 4 — Validación, puntaje estimado y modelo teórico (scripts 08–10)](#9-fase-4--validación-puntaje-estimado-y-modelo-teórico-scripts-0810)
10. [Fase 5 — El experimento de las variables (scripts 11–13)](#10-fase-5--el-experimento-de-las-variables-scripts-1113)
11. [Fase 5 — El modelo v2 y su evaluación en sombra (scripts 14–16)](#11-fase-5--el-modelo-v2-y-su-evaluación-en-sombra-scripts-1416)
12. [Fase 5 — La vara de medir y el cierre del ciclo (scripts 17–19)](#12-fase-5--la-vara-de-medir-y-el-cierre-del-ciclo-scripts-1719)
13. [Estado actual](#13-estado-actual)

---

# 1. RESUMEN EJECUTIVO

## Contexto

**Copa STEM** es una olimpiada de matemáticas y lógica para estudiantes de 9.°,
10.° y 11.° de **Copacabana** y **Girardota** (Antioquia), organizada por la
Fundación SapienceLab. El examen tiene **40 preguntas** (16 numéricas de 10 pts +
24 de selección múltiple de 5 pts), dura **90 minutos** y se califica de 0 a 100.

Se dispone de un dataset de **1.809 inscritos** exportado de Supabase, con datos
demográficos, socioeconómicos, de experiencia previa y de telemetría del examen
(tiempo, cambios de pestaña). El objetivo del trabajo fue **cuantificar brechas
de equidad, predecir el rendimiento, detectar talento oculto y trampa, y
segmentar perfiles** para orientar la política de la Fundación.

## Hallazgos principales

- **El rendimiento individual es poco predecible con las variables disponibles.**
  El mejor modelo explica solo el **~9 % de la varianza** del puntaje
  (R² ≈ 0.07–0.09). Esto **no es un fracaso**: es un hallazgo. Significa que la
  nota depende sobre todo de factores **no medidos** (motivación, calidad
  docente, preparación individual), no de la condición socioeconómica.
- **El estrato apenas roza la significancia y su efecto es trivial** (ANOVA,
  p = 0.041, eta² = 0.004). Con los estratos corregidos (1–3) la diferencia cruza
  el umbral estadístico, pero el tamaño de efecto es **prácticamente nulo**: el
  estrato explica ~0.4 % de la varianza. La desigualdad socioeconómica *clásica*
  sigue **sin explicar** el rendimiento en la práctica. (En el ANCOVA que controla
  por colegio, el estrato es no significativo, p = 0.739.)
- **La brecha territorial es la más fuerte del estudio**: Girardota (53.8) supera a
  Copacabana (39.5) con altísima significancia (p ≈ 9×10⁻²⁶).
- **Tener computador aporta +4 puntos reales**, y —contra la intuición— **no es
  un simple proxy del estrato** (correlación r = 0.14; el efecto se mantiene al
  controlar por estrato). Pero el tamaño del efecto es **pequeño** (d = 0.18).
- **342 estudiantes (19.5 %) son "talento oculto"**: alto rendimiento pese a
  condiciones adversas. Son el foco de mayor retorno social.
- **Se detectaron 55 exámenes con fuerte indicio de trampa** (3.0 %), concentrados
  en puntajes altos con tiempos imposiblemente cortos. Curiosamente, **anularlos
  no mejoró el modelo** (la trampa no añadía "ruido" explicable).

## Recomendaciones de política (detalle en §5)

1. **Cerrar la brecha territorial** con acompañamiento diferenciado a los colegios
   de menor promedio.
2. **Programa de dotación tecnológica** focalizado (el computador ayuda, aunque
   poco) **combinado con acompañamiento pedagógico**.
3. **Becas de talento oculto** para los 342 estudiantes identificados.
4. **Exigir el sistema anti-trampa** (telemetría) en todos los exámenes futuros.
5. **Recoger nuevas variables** (horas de estudio, promedio escolar, motivación)
   para que los modelos predigan mejor.

---

# 2. MARCO TEÓRICO Y CONCEPTUAL

## 2.1 ¿Qué es Machine Learning?

**Machine Learning (ML)** es programar una máquina para que **aprenda patrones a
partir de datos**, en lugar de darle reglas explícitas. Si usted sabe programar,
la diferencia es esta:

```python
# Programación clásica: usted escribe la regla.
def aprueba(puntaje):
    return puntaje >= 60

# Machine Learning: usted da EJEMPLOS y la máquina infiere la regla.
modelo.fit(X, y)      # X = características, y = respuesta correcta
modelo.predict(X_nuevo)  # la máquina aplica lo aprendido a datos nuevos
```

En ML clásico no escribimos la fórmula; le mostramos al algoritmo miles de casos
(`X`, `y`) y él **ajusta parámetros internos** para reproducir la relación.

### Supervisado vs. no supervisado

| | **Supervisado** | **No supervisado** |
|---|---|---|
| ¿Hay respuesta correcta (`y`)? | Sí | No |
| Qué aprende | A predecir `y` desde `X` | Estructura oculta en `X` |
| Ejemplo Copa STEM | Predecir el puntaje (script 03) | Agrupar perfiles (script 07) |

### Regresión vs. clasificación (ambas supervisadas)

- **Regresión** → predecir un **número** continuo. *Ej.: predecir el puntaje
  (0–100).* Script 03.
- **Clasificación** → predecir una **categoría**. *Ej.: ¿es talento oculto,
  sí/no?* Script 06.

### Features, target y "entrenar"

- **Features** (`X`): las variables de entrada (grado, estrato, computador…).
- **Target** (`y`): lo que queremos predecir (el puntaje).
- **Entrenar** (`fit`): el proceso de ajustar los parámetros internos del modelo
  para que sus predicciones se parezcan lo más posible a los `y` reales.

### Overfitting y por qué separamos train/test

**Overfitting** (sobreajuste) es cuando el modelo **memoriza** los datos de
entrenamiento en vez de aprender el patrón general. Un modelo sobreajustado
acierta casi perfecto en los datos que ya vio, pero **falla con datos nuevos** —
como un estudiante que memoriza las respuestas del simulacro pero no entiende la
materia.

Para detectarlo, **separamos los datos en dos**:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)   # 80% entrenar, 20% evaluar
```

El modelo aprende **solo con el 80 %** (`train`) y se evalúa con el **20 %
restante** (`test`) que nunca vio. Si rinde bien en `train` pero mal en `test`,
está sobreajustado. Usamos `random_state=42` para que la partición sea
**reproducible** (siempre la misma "moneda al aire").

## 2.2 Métricas de evaluación

### R² — coeficiente de determinación (regresión)

Mide **qué fracción de la variabilidad del target explica el modelo**. Escala:

- **R² = 1.0** → predicción perfecta.
- **R² = 0.0** → el modelo no es mejor que predecir siempre el promedio.
- **R² < 0** → el modelo es *peor* que el promedio (sobreajuste o mal ajuste).

En Copa STEM obtuvimos **R² ≈ 0.07–0.09**: el modelo explica solo el 7–9 % de por
qué unos sacan más que otros. El otro **~91 % depende de factores que no medimos**.

### RMSE y MAE — error en las unidades reales

- **MAE** (error absoluto medio): en promedio, cuántos puntos nos equivocamos.
- **RMSE** (raíz del error cuadrático medio): parecido, pero **penaliza más los
  errores grandes**. Siempre RMSE ≥ MAE.

En el modelo de puntaje: **RMSE ≈ 22, MAE ≈ 18** (puntos, sobre 100). Es decir,
la predicción típica se desvía ~18–22 puntos de la nota real — mucho, coherente
con el R² bajo.

### Accuracy, Precision, Recall, F1 (clasificación)

Con el ejemplo de **talento oculto** (¿es talento, sí/no?). Definimos:

- **VP** (verdadero positivo): era talento y lo detectamos.
- **FP** (falso positivo): NO era talento pero lo marcamos.
- **FN** (falso negativo): era talento y lo dejamos pasar.

| Métrica | Fórmula | Pregunta que responde |
|---|---|---|
| **Accuracy** | (VP+VN)/total | ¿Qué % acerté en total? |
| **Precision** | VP/(VP+FP) | De los que marqué como talento, ¿cuántos lo eran? |
| **Recall** | VP/(VP+FN) | De los talentos reales, ¿cuántos encontré? |
| **F1** | media armónica de precisión y recall | Equilibrio entre ambas |

Para la Fundación, **el recall importa más**: es peor *dejar pasar* un talento
(FN) que *revisar de más* a un no-talento (FP).

### AUC-ROC — y por qué AUC = 1.0 en talento oculto NO es mérito del modelo

La **curva ROC** grafica el recall (verdaderos positivos) contra los falsos
positivos a medida que movemos el umbral de decisión. El **AUC** (área bajo la
curva) resume esto en un número: 0.5 = azar, 1.0 = separación perfecta.

En el clasificador de talento oculto obtuvimos **AUC = 1.000** (ver
`../outputs/F06_roc.png`). **Esto no es un modelo genial: es una advertencia.** El
"talento oculto" se define con una **regla determinista** sobre las mismas
variables que el modelo recibe como features. El modelo simplemente
**reconstruye la regla** — es *fuga de etiqueta* (label leakage). Lo reportamos
con transparencia: el AUC alto mide *consistencia con la regla*, no capacidad
predictiva sobre casos nuevos.

### Cross-validation — por qué un solo split no basta

Un único `train/test` puede salir "con suerte" o "con mala suerte" según qué
estudiantes cayeron en cada lado. La **validación cruzada k-fold** reparte los
datos en *k* bloques y repite el experimento *k* veces, usando cada bloque como
test una vez:

```python
from sklearn.model_selection import cross_validate, KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cross_validate(modelo, X, y, cv=cv, scoring="r2")
```

Con **5-fold** obtenemos 5 estimaciones del R² y reportamos su **media ± desviación**.
Si la desviación es grande, el resultado es inestable. Usamos 5-fold como estándar
(equilibrio entre costo y robustez).

### P-value — cuándo una diferencia es "significativa"

El **p-value** responde: *"si en realidad NO hubiera diferencia entre los grupos,
¿qué tan probable sería observar una diferencia como esta (o mayor) solo por
azar?"*. Convención:

- **p < 0.05** → "estadísticamente significativa" (menos del 5 % de que sea azar).
- **p ≥ 0.05** → no podemos descartar que sea casualidad.

⚠️ **Advertencia crucial** (que aparece en varios hallazgos): con muestras grandes,
diferencias **triviales** pueden dar p < 0.05. Por eso siempre acompañamos el
p-value con el **tamaño del efecto** (Cohen's d, eta²), que mide la *magnitud*
práctica. *Ej.: el computador da p = 0.002 (significativo) pero d = 0.18 (efecto
insignificante).*

## 2.3 Los algoritmos usados

### Regresión Lineal

**Analogía:** trazar la mejor línea recta (o plano, en varias dimensiones) que
pase por la nube de puntos.

**Fórmula:**
```
puntaje = b0 + b1·(grado) + b2·(estrato) + b3·(computador) + …
```
- `b0` (intercepto): el valor base cuando todas las features son 0.
- `b1, b2, …` (coeficientes): cuánto cambia el puntaje por cada unidad de la
  feature. Un `b` positivo grande = variable que **empuja el puntaje hacia
  arriba**. El signo y magnitud de cada `b` son **directamente interpretables**.

El algoritmo elige los `b` que **minimizan la suma de los errores al cuadrado**
(mínimos cuadrados).

- **Ventajas:** simple, rapidísima, totalmente interpretable.
- **Desventajas:** asume relación lineal; sensible a valores atípicos.
- **Cuándo usarla:** como *baseline* y cuando lo que importa es **entender qué
  variables pesan** y en qué dirección.

### Random Forest (Bosque Aleatorio)

**Analogía:** en vez de fiarse de un solo experto, pregunta a **300 expertos**
(árboles) y **promedia** sus opiniones.

**¿Qué es un árbol de decisión?** Una secuencia de preguntas sí/no que termina en
una estimación. Ejemplo real extraído de nuestro modelo
(`reports/03_modelo_predictivo.md`):

```
si [municipio = Copacabana]:
    si [grado_escolar] <= 10.5:
        si [estrato] <= 2.5:  → puntaje estimado ≈ 38
        si no:                 → puntaje estimado ≈ 41
    si no (grado 11):          → puntaje estimado ≈ 45
si no (Girardota):             → puntaje estimado ≈ 50
```

**¿Qué es el "bosque"?** Se entrenan **muchos** árboles, cada uno con:
- una **muestra aleatoria** de estudiantes (bootstrap), y
- un **subconjunto aleatorio** de variables en cada división.

Así cada árbol es distinto. La predicción final es el **promedio** de todos.

**¿Por qué funciona mejor que un solo árbol?** Un árbol solo es inestable
(sobreajusta). Promediar muchos árboles **reduce la varianza** sin aumentar el
sesgo — el ruido de cada uno se cancela.

**Feature importance:** el bosque mide cuánto **reduce el error** cada variable
al usarse para dividir, sumado sobre todos los árboles. En nuestro modelo limpio:

| Variable | Importancia |
|---|---|
| grado_escolar | 0.137 |
| municipio (Copacabana + Girardota) | ≈ 0.18 combinado |
| estrato | 0.060 |

- **Cuándo usarlo:** cuando las relaciones **no son lineales** o hay
  interacciones complejas.

### XGBoost y LightGBM (Gradient Boosting)

**Analogía:** un equipo donde **cada miembro corrige los errores del anterior**.
El primer árbol predice; el segundo se entrena para corregir *lo que el primero
falló*; el tercero corrige lo que quedó… y así sucesivamente (*boosting*).

**Diferencia con Random Forest:**

| | Random Forest | XGBoost / LightGBM |
|---|---|---|
| Estrategia | Árboles **en paralelo**, votan | Árboles **secuenciales**, corrigen |
| Objetivo de cada árbol | Predecir el target | Predecir el **error residual** |
| Riesgo | Bajo sobreajuste | **Mayor** sobreajuste si no se regula |

**¿Por qué ganan Kaggle?** Bien ajustados, capturan patrones muy sutiles y suelen
dar la mejor precisión en datos tabulares. **LightGBM** es una variante más rápida
(crece los árboles por hojas, no por niveles).

**Riesgo de overfitting:** al corregir errores agresivamente, pueden memorizar el
ruido. Se controla con hiperparámetros: `learning_rate` (cuánto corrige cada
árbol), `max_depth` (profundidad), `subsample` (fracción de datos por árbol). En
Copa STEM, con señal tan débil, **XGBoost y LightGBM sobreajustaron** (R² de test
negativo): más potencia no ayuda si no hay señal que capturar.

### K-Means Clustering (no supervisado)

**Analogía:** repartir a los estudiantes en *K* "mesas" de modo que los de cada
mesa **se parezcan entre sí** y difieran de las otras mesas.

**Centroides:** el "estudiante promedio" de cada mesa (el centro del grupo). El
algoritmo:
1. Coloca *K* centroides al azar.
2. Asigna cada estudiante al centroide **más cercano**.
3. Recalcula cada centroide como el promedio de su grupo.
4. Repite 2–3 hasta que se estabiliza.

**¿Cómo se elige K?** Con dos herramientas (ver `../outputs/F07_seleccion_k.png`):
- **Método del codo:** se grafica la *inercia* (dispersión interna) vs. K. Se
  busca el "codo" donde añadir más clusters ya no ayuda mucho.
- **Silhouette score** (−1 a 1): mide qué tan bien separado está cada punto de los
  otros clusters. **+1** = clusters nítidos; **0** = fronteras difusas; **negativo**
  = mal agrupado. Elegimos el K con mayor silhouette.

En Copa STEM, K = 4 con **silhouette 0.217** — moderado, indica que los perfiles
existen pero **se solapan** (no hay fronteras tajantes entre tipos de estudiante).

**Limitaciones:** asume clusters esféricos y de tamaño similar; **es sensible a la
escala** (por eso estandarizamos, §3.2).

### Regresión Logística (clasificación)

**No es regresión, es clasificación** — el nombre es histórico. Predice la
**probabilidad** de pertenecer a una clase.

**Función sigmoide:** convierte cualquier número real en una probabilidad 0–1:

```
p = 1 / (1 + e^(-z))     donde   z = b0 + b1·x1 + b2·x2 + …
```

Cuando `z` es muy negativo, `p → 0`; cuando es muy positivo, `p → 1`; en `z = 0`,
`p = 0.5`. Es decir: calcula una combinación lineal (como la regresión lineal) y
la "aplasta" a un rango de probabilidad.

- **Cuándo usarla:** como **baseline** simple e interpretable para clasificación
  binaria (fue nuestro baseline en talento oculto).

## 2.4 Técnicas estadísticas usadas

- **t-test de Welch:** compara la media de **2 grupos** cuando sus varianzas
  pueden diferir. *Ej.: Copacabana vs. Girardota.* Devuelve un p-value.
- **ANOVA (una vía):** compara **3+ grupos** a la vez. *Ej.: puntaje por grado
  (9.°/10.°/11.°).* Si p < 0.05, al menos un grupo difiere.
- **ANOVA de 2 factores:** mide el efecto de **dos variables y su interacción**.
  *Ej.: computador × con-quién-vive.* La **interacción** responde: *¿el efecto de
  tener computador cambia según con quién vive el estudiante?*
- **Correlación de Pearson:** asociación **lineal** entre dos variables numéricas
  (−1 a 1). **Spearman:** asociación **monótona** (por rangos), robusta a
  outliers.
- **Cohen's d / eta²:** tamaño del efecto. `d`: pequeño 0.2, mediano 0.5, grande
  0.8. Complementa al p-value.
- **Percentiles y normalización:** el percentil 75 (P75) es el valor que deja por
  debajo al 75 % de los datos. *Normalizar* a 0–100 permite comparar variables de
  escalas distintas.

### One-hot encoding

Los modelos necesitan **números**, no texto. El *one-hot* convierte una variable
categórica en columnas binarias (0/1), una por categoría:

| genero | → | genero=Masculino | genero=Femenino | genero=No binario |
|---|---|---|---|---|
| Masculino | | 1 | 0 | 0 |
| Femenino | | 0 | 1 | 0 |

Así "Masculino/Femenino" se vuelve algo que el modelo puede sumar y ponderar, sin
imponer un orden falso (no es que Femenino "valga más" que Masculino).

```python
# En el proyecto lo hacemos manualmente para poder replicarlo sin sklearn:
for cat in ["Masculino", "Femenino", "No binario", "Prefiero no decirlo"]:
    feats.append(1.0 if valor == cat else 0.0)
```

---

# 3. METODOLOGÍA (Pipeline paso a paso)

## 3.1 Obtención de datos

**Origen:** base de datos **Supabase** (PostgreSQL). Los datos provienen de tres
tablas:

- `inscripciones_copa_stem` — inscripción estándar (datos socioeconómicos completos).
- `inscripciones_emergencia` — inscripciones tardías (~7 %, datos parciales).
- `resultados_prueba_copa_stem` — puntaje y telemetría del examen.

**Query de extracción (reconstrucción representativa).** El dataset maestro
unifica inscripciones (estándar + emergencia) con sus resultados. Una consulta
equivalente es:

```sql
-- Dataset maestro: inscripciones (estándar + emergencia) + resultados
SELECT
    i.numero_documento, i.nombres, i.apellidos,
    i.grado_escolar, i.genero, i.municipio, i.tipo_institucion,
    i.institucion_educativa, i.estrato, i.jornada, i.con_quien_vive,
    i.computador_en_casa, i.internet_en_casa,
    i.participacion_olimpiadas, i.nivel_programacion, i.nivel_robotica,
    i.herramientas_conocidas, i.areas_interes, i.interes_prog_robotica,
    i.edad_calculada,
    r.puntaje_obtenido, r.porcentaje, r.tiempo_usado_segundos,
    r.cambios_pestana, r.intentos_copiar, r.intentos_pegar,
    r.intentos_click_derecho
FROM inscripciones_copa_stem i
LEFT JOIN resultados_prueba_copa_stem r
       ON r.numero_documento = i.numero_documento

UNION ALL

SELECT
    e.numero_documento, e.nombres, e.apellidos,
    e.grado_escolar, e.genero, e.municipio, e.tipo_institucion,
    e.institucion_educativa, e.estrato, e.jornada, e.con_quien_vive,
    e.computador_en_casa, e.internet_en_casa,
    e.participacion_olimpiadas, e.nivel_programacion, e.nivel_robotica,
    e.herramientas_conocidas, e.areas_interes, e.interes_prog_robotica,
    e.edad_calculada,
    r.puntaje_obtenido, r.porcentaje, r.tiempo_usado_segundos,
    r.cambios_pestana, r.intentos_copiar, r.intentos_pegar,
    r.intentos_click_derecho
FROM inscripciones_emergencia e
LEFT JOIN resultados_prueba_copa_stem r
       ON r.numero_documento = e.numero_documento;
```

> *Nota: esta consulta es una reconstrucción a partir del esquema de columnas del
> CSV maestro; la extracción real vive en el pipeline de datos y debe ajustarse a
> los nombres exactos de columnas de cada tabla.*

**Limitaciones de la extracción:**
- **Paginación de Supabase:** la API REST devuelve **máximo 1.000 filas** por
  petición; hay que **paginar** (`range`) y concatenar para traer los ~1.800
  registros.
- **Esquemas distintos:** las dos tablas de inscripción no tienen exactamente las
  mismas columnas; las de emergencia traen varios campos socioeconómicos en NULL
  (de ahí los ~131 faltantes que veremos).

## 3.2 Limpieza (aplicada en todos los scripts)

```python
# 1) Eliminar registros de prueba (documentos ficticios)
docs_prueba = ["1234", "123456", "123456789", "1234567899", "0", "00000000"]
df = df[~df["numero_documento"].isin(docs_prueba)]
df = df[df["numero_documento"].str.len() >= 5]

# 2) Normalizar strings vacíos a NaN
df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

# 3) Tipos numéricos robustos
df["puntaje_obtenido"] = pd.to_numeric(df["puntaje_obtenido"], errors="coerce")
```

**Detección y anulación de trampa** (script 05, detalle en §4.4): 55 exámenes con
patrones de trampa se marcan y se retiran del dataset de análisis (script 05b
genera `copa_stem_dataset_limpio.csv`).

**Imputación de faltantes — ¿por qué mediana y no media?** Los ~131 registros de
emergencia carecen de datos socioeconómicos. Los imputamos con la **mediana** (no
la media) porque la mediana es **robusta a valores atípicos y a distribuciones
asimétricas**: un puñado de valores extremos no la desplaza, mientras que sí
arrastraría a la media. Para variables categóricas usamos la **moda** (el valor
más frecuente).

```python
medians = raw.median()          # numéricas/ordinales → mediana
raw_imputado = raw.fillna(medians)
```

**Encoding:** binarias Sí/No → 1/0; ordinales (Ninguna/Básica/Intermedia/Avanzada)
→ 0/1/2/3; nominales (género, municipio) → one-hot (§2.4).

## 3.3 Análisis Exploratorio (scripts 01 y 02)

Antes de modelar, **miramos los datos**. Cinco visualizaciones clave:

1. **Distribución del puntaje** (`../outputs/B01_distribucion_puntaje.png`):
   media ≈ 43, mediana 40, **asimetría positiva** (cola hacia notas altas).
2. **Matriz de correlación** (`../outputs/C03_correlacion_heatmap.png`): **todas
   las correlaciones con el puntaje son débiles** (≤ 0.19). Primera señal de que
   predecir será difícil.
3. **Brechas socioeconómicas** (`../outputs/F02B_brechas_socioeconomicas.png`):
   el estrato **casi no** separa el puntaje (significativo pero de efecto trivial,
   eta² = 0.004); computador **sí**, levemente.
4. **Brechas territoriales** (`../outputs/F02C_brechas_territoriales.png`):
   Girardota > Copacabana, la señal más fuerte.
5. **Talento oculto** (`../outputs/F02E_talento_oculto.png`): existen estudiantes
   de alto puntaje en estratos bajos.

**Lo esperado:** que el estrato y el acceso marcaran fuerte el rendimiento.
**La sorpresa:** aunque el estrato apenas cruza la significancia (p = 0.041), su
efecto es **trivial** (eta² = 0.004), y el efecto del acceso también es pequeño.
El rendimiento no se explica, en la práctica, por la condición socioeconómica medida.

## 3.4 Modelado — decisiones y su porqué

- **¿Por qué split 80/20?** Convención que equilibra: suficientes datos para
  entrenar (80 %) y una muestra de test representativa (20 % ≈ 350 estudiantes).
- **¿Por qué `random_state=42`?** Fija la aleatoriedad (partición, inicialización)
  para que **cualquiera reproduzca exactamente** los mismos resultados. El 42 es
  una convención cultural; cualquier número fijo sirve.
- **¿Por qué CV 5-fold?** Para no depender de un único split afortunado (§2.2).
- **Estratificación:** en el split del puntaje estratificamos por **grado**, y en
  clasificación por el **target**, para que train y test tengan proporciones
  similares.

Patrón de exportación a producción (scripts 03, 04, 06, 07): cada modelo se
serializa **además** como una **función pura** en `.py` y `.js` que **no requiere
sklearn** — solo aritmética y `json`. Esto permite calcular predicciones en el
backend o directamente en el navegador. Cada predictor se **valida numéricamente**
contra el modelo real (diferencia ≈ 10⁻¹⁴, es decir, idéntico).

---

# 4. RESULTADOS POR FASE

## 4.1 Fase 1 — Análisis Exploratorio (scripts 01, 02)

**Distribución del puntaje:** media ≈ **43.1**, mediana **40**, asimetría positiva.
La mayoría se concentra entre 20 y 60 puntos.

**Brechas encontradas** (con su evidencia):

| Brecha | Prueba | p-value | ¿Significativa? | Interpretación |
|---|---|---|---|---|
| **Municipio** | t-test Welch | **9.4×10⁻²⁶** | Sí, fortísima | Girardota (53.8) ≫ Copacabana (39.5) |
| **Grado** | ANOVA | **1.9×10⁻⁸** | Sí | El puntaje sube de 9.° a 11.° |
| **Género** | t-test Welch | **0.0029** | Sí, pero d = 0.142 | Hombres µ 44.9 vs mujeres 41.5 (efecto pequeño) |
| **Jornada** | t-test | 0.032 | Marginal | — |
| **Estrato** | ANOVA | **0.041** | Sí, pero eta² = 0.004 | Cruza el umbral, efecto trivial (~0.4 % de varianza) |
| **Tipo institución** | t-test | 0.236 | NO | Pública ≈ privada |

**Hallazgo 1 — el estrato es estadísticamente significativo pero prácticamente
irrelevante** (p = 0.041, eta² = 0.004). Con los estratos corregidos a 1–3, la
diferencia entre grupos cruza por poco el umbral p < 0.05, **pero el tamaño de
efecto es casi nulo**: el estrato explica apenas ~0.4 % de la varianza del
puntaje. Es el caso de manual de "significancia estadística sin relevancia
práctica" (§2.2): la desigualdad socioeconómica *clásica* sigue **sin explicar**
el rendimiento en términos útiles. Además, al **controlar por colegio** (ANCOVA),
el estrato vuelve a ser no significativo (p = 0.739). Ver
`../outputs/F02B_brechas_socioeconomicas.png`.

**Hallazgo 2 — el computador da +4 puntos reales y no es proxy de estrato**
(detalle en §4.7). Ver `../outputs/F05c_hist_computador.png`.

## 4.2 Fase 2 — Modelo Predictivo del Puntaje (script 03)

Se entrenaron 4 modelos con 18 features (one-hot incluido). Resultados sobre los datos
**limpios** (**1.750 estudiantes**):

| Modelo | R² (CV) | R² (test) | RMSE (test) | MAE (test) |
|---|---|---|---|---|
| Regresión Lineal | 0.083 | 0.106 | 21.66 | 17.53 |
| **Random Forest ⭐** | 0.064 | **0.115** | 21.56 | 17.44 |
| XGBoost | −0.033 | 0.053 | 22.30 | 17.82 |
| LightGBM | −0.124 | 0.013 | 22.77 | 18.15 |

> **Conteos de esta cohorte (corregido el 2026-09-03).** La cadena real es **1.754 crudos
> → 1.750 tras limpieza** (se descartan 4 registros de prueba; es la población que entrena
> el script 03) **→ 1.748 tras deduplicar por documento**
> (`drop_duplicates('numero_documento')` en el script 09b; es la población de los informes
> 09, 10 y 14). Esta sección publicaba antes "datos **limpios** (1.754 estudiantes)": el
> 1.754 es precisamente el conteo **crudo**, anterior a la limpieza. Trazabilidad en el
> informe 20.

Ver `../outputs/F03_comparacion_modelos.png`. La validación honesta out-of-fold
(script 09b) confirma un **R² ≈ 0.084** (MAE 18.1), coherente con la CV: el punto
0.115 del test de Random Forest es una estimación optimista de un único split.

**R² ≈ 0.07–0.09 → las condiciones socioeconómicas explican solo ~9 % de la
varianza del puntaje.** ¿Qué significa? Que **el rendimiento depende sobre todo de
factores que no medimos**: motivación, calidad del docente, preparación
individual, hábitos de estudio. Es un resultado **honesto y valioso**: dice a la
Fundación *dónde NO está la palanca* (no en el estrato ni en variables de perfil)
y **qué datos faltan** recoger.

**Modelo ganador: Random Forest.** (Nota: con los datos *sucios* ganaba la
Regresión Lineal; al limpiar los tramposos, el bosque quedó levemente arriba —
§4.4). Importancia de variables (`../outputs/F03_importancia_1_random.png`):
**grado_escolar** (0.137) es la variable individual más importante; **municipio**
(sumando sus dos columnas one-hot ≈ 0.18) es el factor más fuerte en conjunto;
**estrato** apenas 0.060. Coherente con la EDA: lo territorial manda.

**Exportación a JavaScript.** El modelo se convierte en un recorrido de árboles en
JS puro (nearest-leaf + término de sesgo empírico que absorbe constantes internas
de cada librería), validado contra el modelo real con diferencia < 10⁻⁴. Así la
web puede estimar el puntaje sin backend de ML.

## 4.3 Fase 2 — Índice de Potencial STEM (script 04)

Como el modelo predictivo es débil, **el puntaje crudo no debe ser la única señal
de potencial**. Se diseñó un **índice compuesto** (0–100):

```
índice = 0.50 · rendimiento + 0.25 · engagement + 0.25 · resiliencia
```

- **Rendimiento (50 %):** percentil del puntaje (o del puntaje estimado, si no
  presentó).
- **Engagement (25 %):** promedio de 8 señales de interés/experiencia (niveles de
  programación y robótica, interés, nº de herramientas y áreas, olimpiadas previas,
  acceso a computador e internet).
- **Resiliencia (25 %):** premia rendir bien **a pesar** de condiciones adversas.

**Por qué no usar solo el puntaje:** dos estudiantes con la misma nota pueden
diferir mucho en disposición (engagement) y en mérito contextual (resiliencia). El
índice **captura matices** que la nota sola no ve: la correlación índice-nota es
alta (ρ ≈ 0.99) **pero con dispersión** — una misma nota se abre en un rango de
índices (`../outputs/F04_indice_vs_puntaje.png`).

**Distribución por categorías** (1.750 estudiantes):

| Categoría | Umbral | N |
|---|---|---|
| Talento destacado | ≥ 85 | 65 |
| Alto potencial | 70–84 | 394 |
| Promedio | 45–69 | 582 |
| En desarrollo | 25–44 | 451 |
| Requiere apoyo | < 25 | 258 |

**Caso de estudio:** estudiantes con **nota 100 pero puntaje esperado ~34** (el
modelo no los "veía venir") obtienen índice ~83 (Alto potencial) — el índice
reconoce su logro real. Otros con alta resiliencia (rindieron muy por encima de lo
esperado dado su contexto) suben de categoría. Ver
`../outputs/F04_distribucion_indice.png`.

## 4.4 Fase 3 — Detección de Trampa (script 05)

**Solo los 1.320 exámenes de plataforma** tienen telemetría (tiempo, cambios de
pestaña); los **485 escritos** no (marcados "Sin telemetría", fuera del análisis).

**Criterios de sospecha** (sospechoso si cumple **≥ 2**):

| Criterio | Condición | Justificación estadística |
|---|---|---|
| A | puntaje ≥ 60 y tiempo < 25 min | 40 preguntas bien en <25 min es atípicamente rápido |
| B | puntaje ≥ 80 y tiempo < 35 min | Puntaje muy alto en tiempo corto |
| C | ≥ 5 cambios de pestaña y ≥ 60 | Salir del examen repetidas veces sugiere consulta externa |
| D | velocidad (pts/min) > percentil 95 | Velocidad atípica respecto al grupo |
| E | puntaje = 100 y tiempo < 45 min | Perfecto en <½ prueba: casi imposible sin ayuda |

Ver `../outputs/F05_tiempo_vs_puntaje.png` (el clúster rojo, alto-puntaje/poco-tiempo).

**Resultado:** **55 exámenes recomendados para anulación** (regla: ≥2 criterios y
puntaje ≥ 60) = **3.0 %** del total. El promedio general baja de **43.16 → 41.81**
(−1.36 puntos, `../outputs/F05_impacto_promedio.png`).

**Hallazgo — la trampa NO agregaba ruido al modelo.** Reentrenar el modelo sin los
sospechosos de nivel "Alto" **no mejoró el R²** (0.091 → 0.089, Δ = −0.002). Interpretación: el
cheating **no correlaciona con las variables del estudiante**, así que retirarlo no
ayuda al modelo a explicar el resto. La anulación se justifica por **integridad**,
no por ganancia predictiva. (Lo reportamos así en vez de forzar la narrativa
cómoda de que "los tramposos ensuciaban el modelo".)

## 4.5 Fase 3 — Talento Oculto (script 06)

**Definición operativa** (regla determinista):

```
talento_oculto = alto_rendimiento  Y  (≥ 2 condiciones adversas)

alto_rendimiento : puntaje ≥ P75 (=60)  O  índice_potencial ≥ 75
condiciones adversas (de 6): estrato 1-2 · sin computador · sin internet ·
                             no vive con ambos padres · sin olimpiadas previas ·
                             nivel de programación "Ninguna"
```

**Resultado: 342 talentos ocultos (19.5 %)** de 1.750. Perfil:
- **Por grado:** 9.°: 87 · 10.°: 115 · **11.°: 138** (más en grados superiores).
- **Por género:** Masculino 183 · Femenino 159 (repartido de forma equilibrada).
Ver `../outputs/F06_grado_genero.png` y `../outputs/F06_heatmap_colegio_grado.png`.

**Sobre el clasificador (AUC = 1.0):** como se explicó en §2.2, el target es una
regla sobre las mismas features → el modelo **reconstruye la regla** (fuga de
etiqueta). Su valor real es la **probabilidad continua** para priorizar y la
confirmación de qué condiciones pesan más — **no** es un modelo predictivo
genuino. La lista autoritativa es la **regla**, no la predicción.

**Implicación:** estos 342 estudiantes demuestran alto potencial **pese a**
recursos limitados. Son el **mayor retorno social** de una beca o acompañamiento;
se recomienda contactarlos priorizando por nº de condiciones adversas.

## 4.6 Fase 3 — Clustering de Perfiles (script 07)

Con 9 features estandarizadas, **K = 4** (elegido por silhouette, ver
`../outputs/F07_seleccion_k.png`). Se comparó K-Means (silhouette **0.217**) con
Gaussian Mixture (0.168) y se desplegó **K-Means** por su asignación determinista
"centroide más cercano" (ideal para el predictor portable).

**Los 4 perfiles** (ver radar `../outputs/F07_radar_perfiles.png` y PCA
`../outputs/F07_pca_clusters.png`):

| Perfil | Característica dominante | Intervención sugerida |
|---|---|---|
| **Alto rendimiento tech** | Nota alta + experiencia digital | Rutas STEM avanzadas, mentoría, retos de nivel superior |
| **Base conectada** | Acceso tecnológico bueno, rendimiento por consolidar | Refuerzo académico focalizado |
| **Promedio con acceso limitado** | Rendimiento medio, acceso restringido | Dotación + acompañamiento |
| **En desarrollo** | Rendimiento bajo | Nivelación en matemáticas/lógica, tutoría de base |

El silhouette moderado (0.22) indica **fronteras difusas**: los perfiles son
tendencias, no cajas estancas. Ver distribución por colegio en
`../outputs/F07_clusters_por_colegio.png`.

## 4.7 Cruce Computador × Familia (script 05c)

Análisis específico de si "computador + vivir con ambos padres" es una **ventaja
real** o una **correlación espuria**.

- **(a) ¿Computador = siempre mejor?** Con computador µ = **43.0**, sin µ = **38.9**:
  diferencia **+4.0 pts**, **significativa** (p = 0.002) **pero con efecto
  insignificante** (Cohen's d = **0.18**). Mucho solapamiento entre las dos
  distribuciones (`../outputs/F05c_hist_computador.png`).
- **(b) ¿Ambos padres + computador = combo ganador?** ANOVA de 2 factores:
  **interacción p = 0.638 → NO hay interacción.** El beneficio del computador **no
  cambia** según con quién viva el estudiante; las líneas del gráfico de
  interacción son paralelas (`../outputs/F05c_interaccion.png`).
- **(c) El grupo resiliente:** **84 estudiantes sin computador** sacaron ≥ 60
  (21 % de los sin acceso). El subgrupo "sin computador + Solo madre" (n = 155)
  promedia 39.9. **La falta de acceso no condena el resultado.**
- **(d) ¿Computador es proxy de estrato?** Correlación computador↔estrato = **0.14**
  (débil). Y al **controlar por estrato**, el efecto del computador **no se
  encoge** (+4.0 → +4.2 pts). **Conclusión: NO es un proxy del estrato.** El
  estrato en sí apenas aporta, coherente con §4.1. Ver
  `../outputs/F05c_estrato_computador.png`.

**Conclusión:** la ventaja del computador es **real pero pequeña**, **no explicada
por el estrato** y **sin interacción** con la estructura familiar. La política
eficiente combina **dotación focalizada + acompañamiento pedagógico**.

---

# 5. RECOMENDACIONES PARA LA FUNDACIÓN

**Basadas en la evidencia de cada hallazgo:**

1. **Priorizar la brecha territorial** (la señal más fuerte, p ≈ 10⁻²⁶):
   acompañamiento diferenciado a los colegios de menor promedio de cada municipio.
   *Evidencia: §4.1, `F02C_brechas_territoriales.png`.*
2. **Programa de dotación tecnológica focalizado + tutoría.** El computador ayuda
   (+4 pts) pero poco (d = 0.18) y no por el estrato: la dotación **sola** no basta;
   debe ir con acompañamiento. *Evidencia: §4.7.*
3. **Becas de talento oculto** para los 342 estudiantes identificados, priorizando
   por nº de condiciones adversas. *Evidencia: §4.5, `talento_oculto_scores.csv`.*
4. **Intervención por perfil** (§4.6): cada uno de los 4 clusters necesita una
   estrategia distinta (retos vs. refuerzo vs. dotación vs. nivelación).
5. **Exigir el sistema anti-trampa** (telemetría) en TODOS los exámenes futuros;
   los escritos no son auditables. *Evidencia: §4.4.*

## Qué datos recoger en la próxima Copa STEM

El R² bajo dice que **faltan variables**. Para modelos más predictivos, recoger:

- **Horas de estudio semanales** y **hábitos de preparación**.
- **Promedio escolar (GPA)** del estudiante.
- **Indicadores de motivación** (encuesta corta validada).
- **Calidad/experiencia del docente** de matemáticas.
- **Asistencia a preparación/refuerzo** previo a la Copa.

Estas variables capturan lo que hoy es "el 91 % inexplicado".

---

# 6. GUÍA DE REPLICACIÓN

## Estructura del proyecto

```
ml-models/
├── data/
│   ├── copa_stem_dataset.csv            ← original (Supabase)
│   ├── copa_stem_dataset_completo.csv   ← original + columna 'anulado'  (05b)
│   └── copa_stem_dataset_limpio.csv     ← sin anulados (DEFAULT análisis) (05b)
├── notebooks/
│   ├── 01_analisis_exploratorio.py      ← EDA
│   ├── 02_analisis_brechas.py           ← brechas de equidad
│   ├── 03_modelo_predictivo.py          ← regresión del puntaje
│   ├── 04_indice_potencial_stem.py      ← índice compuesto
│   ├── 05_deteccion_trampa.py           ← integridad del examen
│   ├── 05b_limpiar_dataset.py           ← genera dataset limpio
│   ├── 05c_cruce_computador_familia.py  ← análisis específico
│   ├── 06_talento_oculto.py             ← clasificación de talento
│   ├── 07_clustering_perfiles.py        ← segmentación
│   ├── 08_framework_validacion.py       ← validación (bootstrap, drift, PSI)
│   ├── 09_exportar_puntaje_estimado.py  ← puntaje estimado por estudiante
│   ├── 09b_informe_puntaje_estimado.py  ← informe + R² honesto out-of-fold
│   └── 10_modelo_teorico_vs_empirico.py ← índice teórico (literatura) vs empírico
├── models/
│   ├── mejor_modelo_puntaje.joblib
│   ├── predictor.py                     ← modelo de puntaje puro
│   └── deploy/                          ← CSV + predictores .py/.js para la web
├── outputs/                             ← todos los PNG (dpi=150, paleta Copa STEM)
└── reports/                             ← informes .md (este documento incluido)
```

## Orden de ejecución (desde cero)

Existe una **dependencia**: `05_deteccion_trampa` usa `models/predictor.py` (que
crea `03`), y `03/04` sobre datos limpios necesitan el dataset de `05b`. El orden
que resuelve todo:

```bash
cd "ml-models"
# (activar el entorno)  ;  usar  $env:PYTHONUTF8="1"  en Windows PowerShell

py -3.14 notebooks/01_analisis_exploratorio.py
py -3.14 notebooks/02_analisis_brechas.py
py -3.14 notebooks/03_modelo_predictivo.py     # 1ª pasada → crea models/predictor.py
py -3.14 notebooks/04_indice_potencial_stem.py # 1ª pasada → crea scores
py -3.14 notebooks/05_deteccion_trampa.py      # usa predictor.py → sospecha_trampa.csv
py -3.14 notebooks/05b_limpiar_dataset.py      # crea dataset_limpio.csv
py -3.14 notebooks/03_modelo_predictivo.py     # 2ª pasada → reentrena en limpio
py -3.14 notebooks/04_indice_potencial_stem.py # 2ª pasada → scores limpios
py -3.14 notebooks/06_talento_oculto.py        # usa limpio + scores
py -3.14 notebooks/07_clustering_perfiles.py   # usa limpio
py -3.14 notebooks/05c_cruce_computador_familia.py
```

> **Nota Windows:** exportar `$env:PYTHONUTF8="1"` antes de ejecutar; de lo
> contrario la consola (cp1252) falla al imprimir caracteres como `→` o acentos.
> Los scripts `03` y `04` **prefieren automáticamente** `..._limpio.csv` si existe.

## Actualizar el modelo con datos nuevos

1. Exportar el nuevo CSV de Supabase a `data/copa_stem_dataset.csv`.
2. Ejecutar el pipeline completo en el orden de arriba.
3. Los predictores de `models/deploy/*.py` y `*.js` se **regeneran solos** y se
   **auto-validan** (diferencia ≈ 10⁻¹⁴ vs. el modelo real). Subir los `.csv` de
   `models/deploy/` a Supabase para la web.

## Dependencias y versiones exactas

```
python==3.14
pandas==3.0.1        numpy==2.4.2         scikit-learn==1.8.0
xgboost==3.3.0       lightgbm==4.6.0      statsmodels==0.14.6
matplotlib==3.10.9   seaborn==0.13.2      scipy (con la distribución)
joblib==1.5.3
```

```bash
pip install pandas numpy scikit-learn xgboost lightgbm \
            matplotlib seaborn scipy statsmodels joblib
```

---

# 7. LIMITACIONES

- **R² bajo (~0.07–0.09):** los modelos **no predicen bien el puntaje
  individual**. Sirven para entender *qué pesa* y para el índice/segmentación, no
  para pronosticar la nota de un estudiante concreto.
- **Sesgo de muestra:** solo **2 municipios** (Copacabana, Girardota). Los
  resultados **pueden no generalizar** a otras regiones.
- **Variables no medidas:** motivación, preparación, calidad docente, hábitos de
  estudio — probablemente los verdaderos motores del rendimiento (§5).
- **Estrato limitado a 1–3:** en Copacabana, Girardota y Bello el estrato solo
  llega a 3, así que la variable tiene poca dispersión (95 en estrato 1, 786 en 2,
  797 en 3) — parte de por qué aporta tan poco al modelo. Los antiguos valores
  4/5/6 eran errores de autorreporte, ya reclasificados a 3. Igual precaución con
  géneros minoritarios (submuestras pequeñas).
- **Exámenes escritos sin telemetría (485):** no se puede evaluar trampa en ellos;
  su integridad debe garantizarse por supervisión presencial.
- **Datos autorreportados:** estrato, acceso y niveles los declara el estudiante.
- **Talento oculto y trampa son reglas de política**, no verdades objetivas: sus
  umbrales son revisables y deben combinarse con criterio humano.

---

# 8. REFERENCIAS

**Algoritmos (papers originales):**
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1).
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree*. NeurIPS.
- MacQueen, J. (1967). *Some Methods for Classification and Analysis of
  Multivariate Observations* (K-Means).
- Rousseeuw, P. (1987). *Silhouettes: a graphical aid to cluster analysis*.

**Estadística:**
- Welch, B. L. (1947). *The generalization of "Student's" problem…* (t-test de Welch).
- Fisher, R. A. (1925). *Statistical Methods for Research Workers* (ANOVA).
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
  (Cohen's d, eta²).

**Librerías (documentación):**
- scikit-learn — https://scikit-learn.org/stable/
- XGBoost — https://xgboost.readthedocs.io/
- LightGBM — https://lightgbm.readthedocs.io/
- statsmodels — https://www.statsmodels.org/
- pandas — https://pandas.pydata.org/docs/
- SciPy — Virtanen et al. (2020). *SciPy 1.0*. Nature Methods.

**Marco educativo:**
- OECD (2018). *PISA — Equity in Education*.

---

*Documento generado como referencia del trabajo de Machine Learning para Copa STEM
2026 — Fundación SapienceLab. Todos los resultados son reproducibles con
`random_state=42` siguiendo la §6.*

---

# 9. FASE 4 — VALIDACIÓN, PUNTAJE ESTIMADO Y MODELO TEÓRICO (scripts 08–10)

> **Sobre esta parte del documento.** Las secciones 9 a 13 se añadieron el
> 2026-09-02 para completar la narración hasta el script 19. Todo lo anterior
> (§1–§8) quedó tal como estaba, con fecha 2026-07-06. Cuando una cifra de aquí
> parezca contradecir una de allá, casi siempre es porque cambió la cohorte:
> §1–§8 hablan de los 1,748–1,750 estudiantes del export de julio, y a partir de
> §10 se trabaja con el export de agosto (3,072 examinados). Cada sección dice
> con qué población está hablando.

## 9.1 Script 08 — Framework de validación: ¿el R² bajo es un error o un hallazgo?

**La pregunta.** El script 03 dejó un modelo con R² ≈ 0.085. Antes de sacar
conclusiones de política con él había que responder algo incómodo: ¿ese número
es un fallo del trabajo, o es la verdad sobre estos datos?

**Qué se hizo.** Cuatro pruebas independientes sobre el mismo modelo de
producción, sin cambiarlo: bootstrap de 1,000 remuestreos, un split temporal
simulado (70 % de inscripciones más antiguas → 30 % más recientes), una curva de
calibración de 10 bins y un cálculo de "techo teórico" a partir de estudiantes
**gemelos** —parejas con exactamente las mismas features—.

**Qué se encontró.**

- **El poder predictivo es real, pequeño y estable.** Bootstrap: R² = 0.084 con
  IC 95 % [0.053, 0.116]. El intervalo no toca el cero.
- **El modelo es honesto.** El error medio de calibración es de 2.57 puntos: no
  infla ni subestima de forma sistemática.
- **Pero no se transporta bien a la cohorte más reciente.** En el 30 % de
  inscripciones más nuevas el R² cae a −0.044, por debajo del IC 95 %. Hay
  *concept drift* leve: **hay que re-validar con cada edición**, no asumir que
  el R² se mantiene.
- **El techo no lo pone el algoritmo, lo ponen las variables.** 1,569
  estudiantes (98.9 %) tienen al menos un gemelo, repartidos en 85 grupos, y la
  desviación de puntaje **dentro** de cada grupo es de 21.2 puntos frente a 22.8
  de toda la población. Casi tanta variación adentro como afuera. El **techo
  teórico de R² es ≈ 0.137**, y el modelo (0.085) ya estaba cerca.
- **La varianza se reparte así:** 8.5 % la captura el modelo, 5.2 % es
  alcanzable con mejores modelos o más datos, y **86.3 % es inexplicable con las
  variables actuales**.

**Lo que salió de aquí y marcó todo lo demás.** El informe propuso **cinco
variables nuevas** para el formulario —promedio académico, horas de estudio de
matemáticas, motivación para participar, clases extra de matemáticas y gusto por
la lógica— con una proyección de R² de 0.25–0.40. Los scripts 11 a 14 existen
porque esas preguntas se añadieron de verdad. Se entregó además
`models/deploy/validation_framework.py`, con detección de *drift* por PSI y un
veredicto automático `OK` / `RE-ENTRENAR` que se verificó contra una muestra
perturbada a propósito.

**En una frase:** el modelo no predice bien porque le faltan preguntas, no
porque esté mal hecho — y el informe dijo exactamente qué preguntas añadir.

## 9.2 Scripts 09 y 09b — El puntaje estimado: ¿quién rindió por encima de su contexto?

**La pregunta.** Si el modelo sabe qué esperar de un perfil, la diferencia entre
lo esperado y lo real es una medida de resiliencia. ¿Cuántos estudiantes
superaron su expectativa?

**Qué se hizo.** Se generó un `puntaje_estimado` para cada uno de los **1,748
estudiantes** que presentaron, y se calculó `diferencia = real − estimado`.

**Qué se encontró.** 634 estudiantes (36 %) superaron su expectativa por más de
5 puntos, 305 (17 %) quedaron dentro de ±5 y 809 (46 %) por debajo. La diferencia
media es **+0.06**, señal de que el modelo no tiene sesgo sistemático.

El informe 09b insistió en la lectura honesta de la precisión, y es una
distinción que se arrastra por todo el resto del proyecto:

| Métrica | Dentro de muestra | Validación cruzada (honesta) |
| --- | --- | --- |
| R² | 0.241 | **0.084** |
| MAE | 16.4 pts | **18.1 pts** |
| RMSE | 20.1 pts | **22.1 pts** |

Con un MAE de 18.1 puntos sobre una escala de 0 a 100, si el modelo dice 45 el
estudiante real puede sacar entre 23 y 67. Por eso la diferencia se lee por
tramos amplios y **nunca como un juicio sobre un estudiante concreto**.

**En una frase:** un tercio de los estudiantes rindió por encima de lo que su
contexto hacía esperar, pero el margen de error del modelo es tan ancho que solo
sirve para mirar grupos, no personas.

## 9.3 Script 10 — Modelo teórico vs empírico: ¿los datos se comportan como dice la literatura?

**La pregunta.** El modelo empírico aprendió de los datos reales. Si en esos
datos hay trampa o sesgo, el modelo aprendió también la trampa y el sesgo.
¿Cómo contrastarlo sin usar los mismos datos?

**Qué se hizo.** Se construyó un `indice_condiciones` (0–100) **teórico**, cuyos
pesos salen únicamente de la literatura educativa (OECD PISA, UNESCO,
meta-análisis de nivel socioeconómico) y de ningún resultado de Copa STEM. La
fórmula es `50 + Σ ajustes`, recortada a [5, 95]: computador ±3, internet ±2,
estrato −3/0/+2, estructura familiar ±1, nivel de programación de −3 a +8, nivel
de robótica de −1 a +4, olimpiadas +5, interés −2/0/+3, herramientas −2/0/+3.
Cada peso tiene su fuente escrita al lado. Deliberadamente **no** usa municipio,
grado, género ni institución, por estar potencialmente contaminados.

**Qué se encontró.**

| Modelo | R² vs real | MAE | r (Pearson) |
| --- | --- | --- | --- |
| Empírico (Random Forest) | 0.241 | 16.4 | 0.504 |
| Teórico (literatura) | −0.202 | 21.7 | 0.191 |

El R² del teórico no es comparable en escala —mide *condiciones*, no la nota—,
así que la lectura justa es la correlación. Y el dato que importa es que
**teórico y empírico correlacionan r = 0.466**: dos modelos construidos por
caminos independientes apuntan en la misma dirección, lo que es evidencia de que
los datos de Copa STEM reflejan en buena medida los patrones que la literatura
espera.

El índice también sirvió de detector independiente: marcó **16 casos de alta
sospecha** (resultado demasiado bueno para el contexto *y* examen en menos de 30
minutos), de los cuales **0 eran nuevos** —la telemetría del script 05 ya los
había marcado a todos—. Y, sin mirar la nota, identificó **32 estudiantes en
condiciones adversas (índice < 45) que sacaron ≥ 60**: talento oculto claro.

**En una frase:** se construyó un segundo modelo que no mira los resultados de
Copa STEM, y coincide con el que sí los mira — señal de que los datos son
creíbles.

---

# 10. FASE 5 — EL EXPERIMENTO DE LAS VARIABLES (scripts 11–13)

Aquí empieza el export de agosto de 2026: **3,077 inscritos**, con las cinco
preguntas que había propuesto el script 08 ya incorporadas al formulario. La
pregunta de fase es una sola, y es la que el script 08 dejó abierta.

## 10.1 Script 11 — Preparación: cuatro datasets para separar dos efectos

**La pregunta.** Cuando un modelo mejora tras un año, ¿mejoró porque hay más
datos o porque hay mejores variables? Comparar el modelo viejo contra uno nuevo
no lo distingue: cambian las dos cosas a la vez.

**Qué se hizo.** Se cortaron cuatro datasets del mismo export:

| Dataset | Filas | Qué aísla |
| --- | --- | --- |
| **A** `dataset_A_baseline.csv` | 1,735 | Línea base — la cohorte del primer modelo |
| **B** `dataset_B_completo.csv` | 3,072 | Efecto de **más datos** |
| **C** `dataset_C_perfil.csv` | 1,148 | Efecto de **más variables** |
| **C′** `dataset_C_sin_features.csv` | 1,148 | **Control** de C — las mismas filas, sin las 5 variables |

La pieza clave es **C′**. C y C′ comparten las mismas 1,148 filas y la misma
partición train/test; lo único que las separa son las cinco columnas nuevas. Sin
ese control, cualquier mejora en C sería ambigua: C también es una población más
reciente y de otros municipios.

Se dejó fuera siempre la columna `porcentaje`, duplicado exacto del target en
escala 0–100: incluirla daría un R² cercano a 1 que no significa nada. Es el caso
de libro de *fuga de información*.

**En una frase:** antes de entrenar nada, se montó el experimento de forma que la
respuesta no pudiera salir ambigua.

## 10.2 Script 12 — El experimento: más datos o mejores preguntas

**La pregunta.** ¿Qué mueve el R²: el volumen o la calidad de las variables?

**Qué se hizo.** El mismo Random Forest, el mismo protocolo del script 03
(`n_estimators=300, max_depth=10, min_samples_leaf=8`, KFold 5 + hold-out del
20 % estratificado, imputación ajustada solo con el train), entrenado sobre los
cuatro datasets.

**Qué se encontró.**

| Dataset | Rol | n | Features | CV R² | CV MAE | Test R² | Test MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline | Línea base | 1,735 | 18 | +0.086 | 19.01 | +0.111 | 19.18 |
| B_completo | Más datos | 3,072 | 19 | +0.053 | 17.95 | +0.035 | 17.94 |
| **C_perfil** | **Más variables** | **1,148** | **23** | **+0.180** | **14.84** | **+0.162** | **15.06** |
| C_sin_features | Control de C | 1,148 | 18 | +0.098 | 15.72 | +0.113 | 15.78 |

- **Duplicar los datos no sirvió.** De A a B, con casi el doble de filas, el R²
  **bajó** de +0.086 a +0.053.
- **Las cinco variables nuevas sí sirvieron.** Con la muestra fija en 1,148
  filas, C′ → C lleva el R² de +0.098 a **+0.180**: prácticamente el doble, y el
  MAE baja 0.88 puntos. Es el único contraste que es un experimento controlado en
  sentido estricto.
- **Cinco variables de conducta superan a dieciocho de origen.** Por bloques
  separados: solo socioeconómicas (18 features) → R² +0.098; **solo las 5
  académicas → +0.108**; ambas juntas → +0.180.

Merece registrarse un tropiezo que el propio informe documenta: en una primera
corrida el dataset B arrastró físicamente las 5 columnas nuevas (casi vacías) y
apareció con 24 features en vez de 19, lo que mezclaba los dos efectos que el
experimento pretendía separar. Al corregirlo, **el R² de B cayó de +0.092 a
+0.053**: el número contaminado sugería que más datos ayudaban un poco; el número
limpio muestra lo contrario.

**En una frase:** acumular más inscritos con el formulario viejo no mejora nada;
preguntar mejor duplicó la capacidad de predicción.

## 10.3 Script 13 — Explicabilidad: cuáles variables pesan de verdad

**La pregunta.** El script 12 demostró **que** las variables nuevas ayudan. Falta
saber **cuáles** y **cuánto**.

**Qué se hizo.** Dos medidas de importancia sobre el dataset C: MDI (la que trae
gratis el bosque, sesgada hacia variables con muchos valores distintos) e
**importancia por permutación** —cuánto empeora el R² en datos no vistos al
desordenar una variable—, con 30 repeticiones por variable y calculada sobre las
columnas **crudas**, antes del one-hot, para que `municipio` cuente una sola vez.
Cuando las dos discrepan, manda la permutación.

**Qué se encontró.**

| # | Variable | ΔR² | ± | Bloque |
| --- | --- | --- | --- | --- |
| 1 | **Promedio académico** | **+0.1040** | 0.0291 | Perfil académico |
| 2 | Grado escolar | +0.0380 | 0.0143 | Contexto |
| 3 | Municipio | +0.0246 | 0.0094 | Contexto |
| 4 | Participó en olimpiadas | +0.0187 | 0.0157 | Experiencia previa |
| 5 | Gusto por la lógica | +0.0117 | 0.0148 | Perfil académico |
| 6 | Tipo de institución | +0.0104 | 0.0054 | Contexto |
| 9 | Estrato | +0.0020 | 0.0074 | Socioeconómico |
| 12 | Computador en casa | +0.0001 | 0.0015 | Socioeconómico |

- **`promedio_academico` domina**: desordenarla cuesta 0.104 de R², **2.7 veces
  más** que la segunda variable, y más que las variables 2, 3 y 4 sumadas. Es el
  resultado esperable —el mejor predictor del rendimiento futuro es el
  rendimiento pasado— pero hasta agosto el proyecto no tenía cómo medirlo.
- **El acceso a tecnología no predice la nota.** `computador_en_casa`
  (+0.0001 ± 0.0015) e `internet_en_casa` (−0.0000) son indistinguibles de cero.
  El estrato tampoco (+0.0020 ± 0.0074).
- **El contexto no desaparece, pero deja de encabezar.** En la cohorte A, sin
  perfil académico, el ranking lo lideraba `municipio` (+0.0809) seguido de
  `grado_escolar` (+0.0717). Con el perfil añadido, ambos siguen en los puestos 2
  y 3, pero desplazados.
- **Honestidad sobre las barras de error:** solo los puestos 1, 2, 3 y 6 tienen
  una señal claramente mayor que su propia incertidumbre. Varias variables tienen
  desviación mayor que su importancia, y seis salieron con importancia negativa
  —el modelo predice *mejor* sin ellas—.

**Nota de coherencia con §4.1 y §5.** Que el estrato y el computador no predigan
la *nota* no contradice las brechas del §4: son preguntas distintas. Una cosa es
si un factor **separa grupos** en promedio, y otra si **añade capacidad
predictiva** una vez que el modelo ya conoce el promedio académico del
estudiante. La recomendación de dotación + acompañamiento del §5 sigue en pie por
equidad; lo que este informe dice es que el acceso a tecnología no sirve para
*predecir* la nota individual.

**En una frase:** preguntar por el promedio del colegio predice más que todo el
bloque socioeconómico junto.

---

# 11. FASE 5 — EL MODELO v2 Y SU EVALUACIÓN EN SOMBRA (scripts 14–16)

## 11.1 Script 14 — Optimización de hiperparámetros y nacimiento del modelo v2

**La pregunta.** Ya se sabe qué variables sirven. ¿Cuánto más se puede exprimir
ajustando el estimador?

**Qué se hizo.** `RandomizedSearchCV` sobre `RandomForestRegressor` en el dataset
C: 30 candidatos × 5 folds = **150 ajustes**, `scoring='r2'`. El ganador se
reajustó sobre las 1,148 filas completas y se exportó como artefacto de
producción: `models/mejor_modelo_puntaje_v2.joblib` y
`models/deploy/potencial_stem_predictor_v2.js`.

**Qué se encontró.**

- **Ganadores:** `n_estimators=200`, `max_depth=None`, `min_samples_leaf=8`,
  `max_features=0.5`. R² de CV del mejor candidato: **+0.1749**.
- **Hold-out del 20 %: R² +0.1766, MAE 15.00.**
- **La ganancia de la búsqueda es marginal: +0.0142 de R²** frente a los
  hiperparámetros sin optimizar del script 03 sobre la misma partición
  (+0.1625). El MAE apenas se mueve (−0.06 puntos).
- El predictor JS quedó **verificado en Node**: reproduce a sklearn con
  `máx|Δ| = 3.55e-14`, y mantiene el mismo contrato de salida que v1.

La lectura es coherente con el informe 12: **la elección de hiperparámetros es
secundaria**. Lo que movió el R² fueron las variables (+0.083 al añadir el perfil
académico), no el ajuste del estimador (+0.014).

**Sobre el alcance.** La instrucción original pedía reentrenar los cuatro
predictores con la rejilla optimizada, pero **solo uno de los cuatro es un
`RandomForestRegressor`**: talento oculto es un `XGBClassifier`, clustering es
`KMeans` (su único hiperparámetro es `k`) y condiciones es el modelo teórico del
script 10, cuyos pesos vienen de la literatura y no se entrenan. Reentrenarlos
sobre C habría sido un cambio de semántica disfrazado de optimización, así que
quedaron fuera a propósito.

**En una frase:** afinar el modelo aportó una décima parte de lo que aportó
cambiar las preguntas del formulario.

## 11.2 Script 15 — Scores v2 en sombra: ¿qué cambiaría si se desplegara?

**La pregunta.** Antes de tocar producción: si mañana se despliega v2, ¿qué
cambia para los estudiantes que ya están en la tabla?

**Qué se hizo.** Se calculó **en sombra** el vector completo de `ml_scores` para
los 3,072 examinados, con estrategia híbrida —v2 para los 1,148 con perfil
académico, v1 para los 1,924 restantes— sin tocar ningún modelo ni la tabla de
producción. Resultado en `outputs/ml_scores_v2.csv` (3,072 × 18).

**Qué se encontró, y es el hallazgo más incómodo de la fase.**

- **`indice_potencial` apenas se mueve**: cambio medio de −0.61 puntos (rango
  −3.84 … +3.39), y el **88.8 %** no cambia de categoría.
- **Y ese pequeño cambio no lo produce el modelo.** El índice compuesto usa el
  puntaje **real** cuando el estudiante presentó el examen, y los 3,072 lo
  presentaron. El código lo dice literalmente: `if presento: rend_raw = real`, y
  el modelo no se invoca. **El modelo v2 no interviene en el índice de ningún
  estudiante de esta cohorte.** Lo único que cambia es `ref_rendimiento`, la
  distribución contra la que se percentiliza.
- **Donde v2 sí gana es en `puntaje_estimado`:** MAE de **18.45 → 15.00** frente
  al valor real (−19 %), usando la cifra honesta de validación. El informe es
  explícito en no usar el 12.54 que sale al medir sobre las mismas filas de
  entrenamiento: eso mide memorización.
- **Pero `puntaje_estimado` está fijado a `null` en la Edge Function.** Desplegar
  v2 tal cual **no cambiaría nada visible** para quien ya presentó el examen.

Y una señal de alarma que el informe registró sin resolver: la distribución de
categorías se **desplaza hacia los extremos**. En el subgrupo de v2, "Talento
destacado" pasaba de 30 a 51 (+21) y "Requiere apoyo" de 131 a 182 (+51),
mientras "Promedio" se vaciaba (−57). El informe ya lo atribuyó a percentilizar
contra una cohorte más pequeña y menos dispersa (σ 20.54 en C frente a 22.66 en
B). Ese diagnóstico es el que recogió el script 17.

**En una frase:** el modelo nuevo no cambiaría casi nada para quien ya hizo el
examen, y el poco cambio que se veía venía de la vara de medir, no del modelo.

## 11.3 Script 16 — Las figuras de la comparación v1 vs v2

**La pregunta.** Ninguna nueva: es un script de comunicación.

**Qué se hizo.** Sin entrenar ni recalcular nada, se leyeron los artefactos ya
generados (`outputs/F15_comparacion_v1_v2.csv` y `outputs/F13_importancias.csv`)
y se produjeron cuatro figuras: conteo por categoría v1 vs v2
(`comparacion_categorias_v1_v2.png`), histogramas superpuestos del índice
(`distribucion_potencial_v1_v2.png`), estimado v2 contra real con la diagonal de
predicción perfecta (`puntaje_estimado_vs_real.png`) y top 10 de importancia por
permutación en A vs C (`feature_importance_comparacion.png`).

**Nota honesta.** Es el único script del proyecto **sin informe propio** en
`reports/`; su documentación vive en la cabecera del script y en las propias
figuras. La cabecera deja anotada una advertencia que conviene repetir: el MAE
visible en el scatter (12.54) es **dentro de muestra**, porque el modelo v2 se
reajustó sobre esas mismas 1,148 filas; la cifra honesta es la del hold-out del
script 14 (15.00). Ambas se escriben en la figura para que nadie tome la
optimista por buena.

**En una frase:** cuatro gráficos para ver de un vistazo lo que los informes 13 y
15 ya decían, con la advertencia impresa encima para que no se malinterprete.

---

# 12. FASE 5 — LA VARA DE MEDIR Y EL CIERRE DEL CICLO (scripts 17–19)

## 12.1 Script 17 — Corrección de `ref_rendimiento`: contra quién se compara

**La pregunta.** El informe 15 encontró que 72 estudiantes cambiaban de categoría
de potencial **sin que su desempeño hubiera cambiado en nada**. ¿Por qué?

**Qué se hizo.** El componente de rendimiento —el 50 % del índice— no usa el
puntaje crudo, usa su **percentil**. Y percentilizar siempre es "contra quién".
El script 14 había entrenado el modelo sobre el dataset C y, de paso, había
calculado la referencia sobre esa misma población de 1,148. Son dos usos
distintos de los datos que quedaron acoplados sin necesidad: para **entrenar**
hace falta C, porque es donde están las variables nuevas; para **comparar** hace
falta B, porque es donde están todos los examinados.

El script recalculó `ref_rendimiento` sobre los **3,072** de
`dataset_B_completo.csv`, volvió a puntuar a todos con la misma estrategia
híbrida y los mismos modelos, y guardó la referencia en
`outputs/F17_ref_rendimiento_corregido.json`.

**Qué se encontró.**

| Referencia | n | Media | σ |
| --- | --- | --- | --- |
| v1 (cohorte histórica) | 1,750 | 41.81 | 23.11 |
| v2 tal como estaba (dataset C) | 1,148 | 41.08 | **20.53** |
| **v2 corregida (dataset B)** | **3,072** | 41.74 | **22.66** |

La cohorte C es **más apretada**. Al medir contra una población menos dispersa,
un puntaje que antes caía en el montón central se despega hacia un extremo: no
porque el estudiante haya mejorado o empeorado, sino porque sus vecinos de
comparación se parecen más entre sí.

- **La redistribución artificial desaparece.** En la cohorte completa, "Talento
  destacado" vuelve de 126 a **105** —exactamente el conteo de v1— y "Requiere
  apoyo" baja de 478 a 456.
- **El desplazamiento hacia los extremos cae de 72 a 29 estudiantes** en la
  cohorte completa, y de 72 a 17 en el subgrupo de v2.
- **Las predicciones del modelo no se tocaron.** `ref_rendimiento` solo entra en
  `_percentil`, nunca en `_predict_puntaje`: `puntaje_estimado` salió idéntico
  fila a fila al del script 15, con máximo |Δ| = 0.000000 sobre 3,072 filas.
- El talento oculto se mueve de 607 (v1) a 609: despreciable, como era de
  esperar.

También se corrigió la referencia del **fallback v1**: dejar a los 1,924 sin
perfil midiéndose contra la cohorte histórica de 1,750 habría creado un segundo
artefacto, dos escalas conviviendo en la misma tabla. Se verificó que los 1,750
puntajes viejos están contenidos íntegramente en los 3,072 nuevos.

**En una frase:** 72 estudiantes cambiaban de etiqueta solo porque se les
comparaba con el grupo equivocado, y ahora se les compara con todos los que
presentaron el examen.

## 12.2 Script 18 — Los que no se presentaron: la población donde el modelo sí manda

**La pregunta.** Los scripts 15 y 17 puntuaron a los que ya tienen nota, y ahí el
modelo ni se invoca. ¿Qué pasa con los inscritos que **no** presentaron el
examen, que es justo donde el modelo decide el índice?

**Qué se hizo.** Esa población no estaba en ningún export: todos los `data/*.csv`
son un *inner join* con `resultados_prueba_copa_stem`. Hubo que exportarla aparte
de Supabase (249 filas) y hacer el anti-join explícito. Quedaron **248
estudiantes**, con cero solapamiento con los 3,072 examinados. Se les aplicó la
misma estrategia híbrida y la referencia corregida de 3,072, sin forzar nada: se
le pasa a `calcular_indice` un registro sin `puntaje_obtenido` y la función toma
sola la rama del modelo. Es exactamente el código que corre en la Edge Function.

**Qué se encontró.** Tres cosas, y las tres son advertencias.

**1. El índice sin nota está comprimido y no es comparable con el de un
examinado.** No es un ajuste fino: son dos escalas. σ = 8.97 frente a 22.37.

| Categoría | Sin examen | % | Examinados | % |
| --- | --- | --- | --- | --- |
| Talento destacado | **0** | 0.0 % | 105 | 3.4 % |
| Alto potencial | 3 | 1.2 % | 721 | 23.5 % |
| Promedio | 110 | 44.4 % | 1,006 | 32.7 % |
| En desarrollo | **135** | 54.4 % | 784 | 25.5 % |
| Requiere apoyo | **0** | 0.0 % | 456 | 14.8 % |

245 de 248 (98.8 %) caen en dos categorías y los dos extremos se vacían. Hay dos
causas: el modelo comprime el insumo (σ del estimado 9.20 frente a 22.66 de la
nota real, correlación 0.52) y, sin nota, la resiliencia se calcula como
`max(0, 50 − adversas×5)`, que solo produce cinco valores entre 30 y 50 (σ 5.05
frente a 30.58). Y hay un límite duro: **"Talento destacado" es inalcanzable por
construcción**, porque el techo aritmético del índice sin nota es **83.33**, por
debajo del umbral de 85. No es que no haya salido ninguno: no puede salir. El
máximo real observado fue 72.3.

**2. La detección de talento oculto no funciona en este grupo.** Cero marcados,
probabilidad máxima 0.0031. La regla determinista exige puntaje ≥ 60 o índice
≥ 75, y el clasificador usa el puntaje como feature. Es un artefacto del método,
no una propiedad de la población — que sí tiene materia prima: **175 de 248
(70.6 %) tienen 3 o más condiciones adversas**.

**3. Los que no se presentaron están en peores condiciones.** Este hallazgo se
sostiene porque `indice_condiciones` es teórico y no usa el puntaje: 21.4 % en
condiciones adversas frente al 10.6 % de los examinados, y los dos perfiles de
cluster con menos recursos concentran 42.4 % de los no examinados frente a 23.9 %
de los examinados. La no presentación está correlacionada con la desventaja, así
que **todas las brechas del §4.1 están subestimadas**: los más desfavorecidos se
cayeron de la muestra antes de generar un dato.

Sobre `puntaje_estimado` para este grupo: media 39.67, σ 8.79, rango 25.6–69.5,
con MAE de ±15.00. El 50 % central cabe en 13 puntos (32.2 a 45.3), **una franja
más estrecha que la propia barra de error**. Ordenar a esta población por esa
columna es ordenar ruido.

**En una frase:** los estudiantes que se inscribieron y no se presentaron son
justamente los que están peor, y el índice actual no sabe distinguirlos entre sí.

## 12.3 Script 19 — Propagar la corrección al artefacto que se despliega

**La pregunta.** El script 17 corrigió la referencia en el pipeline de Python y
en los CSV. ¿Llegó esa corrección al fichero que realmente consume la web?

**Qué se hizo.** No había llegado. `models/deploy/potencial_stem_predictor_v2.js`
—el artefacto que lee la Edge Function— seguía llevando embebida, dentro de su
constante `SPEC`, la referencia vieja de **1,148**. Desplegarlo tal cual habría
reintroducido por la puerta de atrás el problema que el script 17 existió para
eliminar.

El script 19 reutiliza el proceso de exportación del script 14 con una
diferencia: clona el cuerpo del `.js` **v2** y sustituye solo la línea de la
constante `SPEC`, de modo que el cuerpo del artefacto se conserva byte a byte y
el contrato con la Edge Function no cambia. Antes de regenerar nada comprueba la
procedencia: los árboles y el preprocesamiento extraídos de
`mejor_modelo_puntaje_v2.joblib` se comparan bit a bit contra los embebidos en el
`.js` vigente (ambos coinciden). El resultado se escribe como **fichero nuevo**,
`potencial_stem_predictor_v2_corrected.js`; el viejo no se toca.

**Qué se encontró / se verificó.**

| Referencia embebida | n | σ | Cohorte |
| --- | --- | --- | --- |
| Antes | 1,148 | 20.5332 | `dataset_C_perfil.csv` (entrenamiento) |
| **Después** | **3,072** | **22.6583** | `dataset_B_completo.csv` (examinados) |

- De las seis claves del `SPEC` **solo cambian dos**: `ref_rendimiento` y `meta`.
  El modelo, el preprocesamiento, los rangos de engagement, los pesos y los
  umbrales quedan idénticos; el script aborta si detectara cualquier otra.
- **El `.js` generado reproduce a sklearn con máx |Δ| = 2.842 × 10⁻¹⁴** sobre 300
  filas (200 con perfil académico + 100 sin perfil, que ejercitan la imputación
  por mediana/moda). Es la misma precisión que verificó el script 14.
- El índice compuesto del `.js` contra el predictor Python difiere en 0.01 en 31
  de las 300 filas: es **modo de redondeo**, no cálculo. Aplicando el criterio de
  `Math.round()` a los valores sin redondear de Python la diferencia cae a cero,
  y la categoría coincidió en las 300.
- El diff contra el artefacto vigente es de 2 líneas quitadas y 6 puestas sobre
  208 → 212. Los hashes SHA-256 de los 21 ficheros de `models/` y
  `models/deploy/` no cambiaron.

El efecto se ve en el propio artefacto: sobre el estudiante de demostración que
el `.js` trae al final, `componente_rendimiento` pasa de 49.04 a 49.48 y el
índice de 44.66 a 44.99, mientras el engagement —que no pasa por el percentil— se
queda clavado en 31.53.

**En una frase:** el arreglo que ya estaba hecho en los cálculos por fin llegó al
archivo que se sube a la web, y se comprobó que sigue dando los mismos números
que el modelo original.

---

# 13. ESTADO ACTUAL

**Actualizado: 2026-09-02** · Pipeline al día hasta el **script 19**.

Este bloque se reescribe cada vez que un script nuevo añade su sección arriba.
Es la única parte de este documento que se actualiza en lugar de solo crecer.

## Pendientes resueltos

| # | Pendiente | Resuelto por | Cómo quedó |
| --- | --- | --- | --- |
| 1 | Saber si el R² bajo era un fallo o un hallazgo | Script 08 | Hallazgo: techo teórico ≈ 0.137 con las variables de entonces |
| 2 | Añadir al formulario las 5 variables de perfil académico | Fundación + export de agosto | Recogidas; 1,148 de 3,072 examinados las declararon |
| 3 | Separar el efecto «más datos» del efecto «mejores variables» | Scripts 11 y 12 | Ganan las variables: R² +0.098 → +0.180 con la muestra fija |
| 4 | Saber **cuáles** variables pesan | Script 13 | `promedio_academico` domina (ΔR² +0.104), 2.7× la segunda |
| 5 | Optimizar hiperparámetros y exportar el modelo v2 | Script 14 | Hold-out R² +0.1766, MAE 15.00; artefacto JS verificado |
| 6 | Saber qué cambiaría al desplegar v2, sin tocar producción | Script 15 | Ejecución en sombra: casi nada cambia para los examinados |
| 7 | Corregir la población de referencia del percentil | Script 17 | `ref_rendimiento` de 1,148 → 3,072 en el pipeline Python/CSV |
| 8 | Puntuar a los inscritos que no presentaron el examen | Script 18 | 248 estudiantes en `outputs/ml_scores_sin_examen.csv` |
| 9 | Propagar la referencia corregida al artefacto de despliegue | Script 19 | `potencial_stem_predictor_v2_corrected.js`, verificado a 1e-14 |
| 10 | **Extender el `SELECT` de la Edge Function a las 5 variables de perfil** | **Repo web (fuera de este repo)** | **Hecho** — ver nota abajo |

> **Sobre el punto 10 — no es un pendiente.** El `COLS_INSC` de la Edge Function
> **ya se extendió** para leer `promedio_academico`,
> `horas_estudio_matematicas`, `motivacion_participar`,
> `clases_extra_matematicas` y `gusto_logica`. Ese cambio se hizo **en el repo
> web (`Recursos Web/sapiencex`), fuera de este repositorio, trabajando en
> Antigravity**, así que no aparece en el historial de `ml-models` ni en ningún
> informe de `reports/`. Se deja escrito aquí para que **no se vuelva a listar
> como pendiente**: informes anteriores de este repo (y `SESION_ACTUAL.md` en su
> versión del 2026-09-02) lo daban por abierto porque `grep promedio_academico`
> sobre `index.ts` no encontraba nada desde este lado.

## Pendientes abiertos

**Despliegue (bloquean el valor del modelo v2)**

1. **Desplegar el artefacto corregido** y decidir su nombre definitivo: mantener
   el sufijo `_corrected` o promoverlo a `potencial_stem_predictor_v2.js`. Es
   decisión de operación, no de modelado; mientras tanto el fichero viejo sigue
   en su sitio para revertir.
2. **Subir `outputs/ml_scores_v2_corrected.csv` a una tabla NUEVA `ml_scores_v2`**
   en Supabase. No sobrescribir `ml_scores`; volcarla a CSV antes de tocar nada.
   `outputs/ml_scores_sin_examen.csv` va aparte, o con marca de población
   explícita.
3. **Añadir la columna `modelo_version` a `ml_scores`** antes de cualquier
   despliegue híbrido: hoy no hay forma de saber qué modelo puntuó cada fila.
4. **Decidir si `puntaje_estimado` deja de estar fijado a `null`.** Mientras siga
   en `null`, desplegar v2 no cambia nada visible para quien ya presentó el
   examen (§11.2). Si se activa, entra con las condiciones del punto 9 de abajo.

**Producto y presentación**

5. **Construir el dashboard académico** en el repo web.
6. **Separar visualmente la población sin examen.** El techo aritmético de 83.33
   (§12.2) hace que "Talento destacado" sea inalcanzable para ella; mostrarla
   junto a los examinados sería engañoso. La marca `tiene_puntaje_real` no basta.

**Método (abiertos por el script 18)**

7. **Revisar la fórmula de resiliencia sin nota.** Cinco valores discretos entre
   30 y 50 para un cuarto del índice es demasiado pobre, y es la mitad de la
   causa del aplanamiento.
8. **Buscar talento oculto en la población sin examen con otro método.** El
   detector actual da 0 por construcción. Con `indice_condiciones` —que no
   depende del puntaje— más el perfil de cluster se pueden priorizar los
   estudiantes en condiciones adversas sin pasar por un modelo sin señal.
9. **Política de `puntaje_estimado`:** nunca como cifra puntual, nunca en la
   misma columna que un puntaje real, nunca para decisiones individuales, nunca
   para rankear a la población sin examen (MAE ±15 sobre 0–100).
10. **Declarar `ref_rendimiento` como decisión de producto**, no como subproducto
    del dataset de entrenamiento de cada versión. Los scripts 17 y 19 existen
    porque no lo estaba.
11. **Re-validar (y quizá re-entrenar) con cada nueva cohorte.** El script 08
    detectó *concept drift* leve: el R² no se transporta solo. El framework
    `models/deploy/validation_framework.py` está para eso.

**Datos**

12. **Revisar si `inscripciones_emergencia` tiene inscritos sin resultado.** El
    export del script 18 salió solo de `inscripciones_copa_stem`; si esa otra
    tabla aporta, faltan estudiantes en los 248.
13. **Tratar el seguimiento a los 248 como intervención de equidad**, no como
    trámite administrativo: son desproporcionadamente los que peor están (§12.2).

**Higiene del repositorio**

14. **`.gitignore` está guardado en UTF-16**, así que git no lo interpreta y sus
    patrones (`.venv/`, `*.joblib`, `__pycache__/`) no se aplican; por eso
    aparecen `.pyc` y `.joblib` en `git status`. Detectado, no corregido.
15. **El script 16 no tiene informe propio** en `reports/`. Su documentación vive
    en la cabecera del script y en las figuras (§11.3).

## Cómo queda el pipeline

```
Fase 1-3  (julio, 1,748-1,750 examinados)   scripts 01-07, 05b, 05c   → §4
Fase 4    (julio, validación y contraste)   scripts 08-10             → §9
Fase 5    (agosto, 3,072 examinados)        scripts 11-13             → §10
          modelo v2 y evaluación en sombra  scripts 14-16             → §11
          referencia corregida y despliegue scripts 17-19             → §12
```

---

_Secciones 9 a 13 añadidas el 2026-09-02 a partir de `reports/08` … `reports/19`
y de los ficheros de `outputs/`. Ninguna cifra de estas secciones es nueva: todas
provienen de un informe o de un artefacto ya generado en este repositorio._

---

# 14. AUDITORÍA DE CIFRAS — RECONCILIACIÓN DEL R² DE v1 (informe 20)

## 14.1 Informe 20 — Tres R² para un solo bosque

**La pregunta.** El §4.2 de este mismo documento publica que v1 alcanzó **R² =
0.115** en test. El §11.1 resume el informe 14, cuya tabla de comparación v1 vs
v2 declara para el mismo modelo **~0.241**. Y el README publica un tercero,
**~0.238**. Un mismo bosque, entrenado una sola vez en julio, con tres notas
distintas. ¿Cuál es la de verdad?

**Qué se hizo.** Un barrido exhaustivo de `reports/`, `outputs/`, `models/`,
`notebooks/`, `README.md` y `SESION_ACTUAL.md` en busca de toda ocurrencia de R²,
MAE y RMSE; la inspección del contenido de los dos `.joblib`, del
`modelo_coeficientes.json`, del bloque `meta` de los predictores JS y de los tres
JSON de `outputs/`; el rastreo en la historia de git de las cifras sin respaldo
documental; y **una única computación**: recalcular R² y MAE sobre las
predicciones ya guardadas en `models/deploy/puntaje_estimado.csv`. No se
reentrenó nada, no se regeneró ningún artefacto y no se editó ningún informe.

**Qué se encontró.**

- **Las tres cifras no son transcripciones de un mismo número: son tres
  evaluaciones distintas del mismo bosque**, y dos de ellas son legítimas por
  separado.
- **El ~0.241 sí tiene origen, y se reprodujo exactamente**: R² = 0.2412,
  MAE = 16.4331, RMSE = 20.1255, r = 0.5043 sobre las 1,748 predicciones
  guardadas. Coincide dígito a dígito con los informes 09b y 10. Es una
  evaluación **dentro de muestra**: el 80 % de esos estudiantes entrenaron el
  bosque que los está puntuando.
- **El informe 09b la etiquetó bien** ("Sobre los datos de entrenamiento"), pero
  **la etiqueta se perdió en el camino**: el informe 10 la publica sin ella, y el
  informe 14 la hereda y además la **atribuye al informe 03**, que no la
  contiene. El informe 03 dice 0.115 / 21.56 / 17.44.
- **El efecto de esa pérdida es que v2 aparenta ser peor que v1.** En la tabla
  del informe 14, `~0.241` (dentro de muestra) queda junto a `+0.1766` (hold-out
  de v2). Es comparar la nota con el examen a la vista contra la nota a libro
  cerrado. La única comparación limpia del repositorio —el §11.2, MAE **18.45**
  de v1 fuera de muestra frente a **15.00** de v2— dice lo contrario: v2 gana por
  3.45 puntos.
- **El ~0.238 del README no existe en ninguna parte.** No lo produce ninguna
  evaluación documentada, no está en ningún artefacto y `git log -S` lo sitúa sin
  cambios desde el commit que creó esa tabla. Es la única cifra del repositorio
  sin fuente trazable.
- **La métrica oficial defendible de v1 es el R² out-of-fold = 0.084** (MAE 18.1,
  RMSE 22.1), con IC 95 % por bootstrap **[0.053, 0.116]**. Es out-of-fold, cubre
  a los 1,748, es la única con intervalo, y ese intervalo **contiene al 0.115**:
  el hold-out del informe 03 no es un resultado distinto, es el borde optimista
  del mismo fenómeno. Todo lo demás converge ahí —0.064 (CV, §4.2), 0.091
  (CV, §4.4), 0.086 (dataset A, §10.2)—; el 0.241 es el único que se sale, y se
  sale porque no mide lo mismo.
- **Tres tamaños para la misma cohorte de julio**, todos del mismo origen:
  **1,754** crudos → **1,750** tras limpieza (informe 03) → **1,748** tras
  deduplicar por documento (informe 09b). El §4.2 de este documento llama
  "datos **limpios** (1.754 estudiantes)" a lo que es precisamente el conteo
  *sucio*.

**La causa estructural.** v1 **nunca guardó sus métricas en un artefacto**: su
`.joblib` contiene el modelo y el preprocesador, ningún R². v2 sí las guarda
(`metricas_holdout`), y por eso su 0.1766 aparece idéntico en cinco sitios
—informe 14, JSON F14, JSON F19, README y `SESION_ACTUAL.md`— sin una sola
divergencia. La lección es de ingeniería, no de estadística: **la métrica tiene
que viajar dentro del artefacto, no en la memoria de quien redacta el informe
siguiente.**

**Qué queda pendiente.** El informe 20 es diagnóstico y **no editó nada**. Las
correcciones que dejarían el repositorio consistente están listadas y priorizadas
en `reports/20_reconciliacion_metricas_v1.md`: retirar el ~0.238 del README,
reetiquetar o sustituir la fila de v1 en el informe 14 y corregir su atribución,
poner la etiqueta "dentro de muestra" en el informe 10 y en el §9.3 de este
documento, anotar en el informe 03 que su 0.115 es un único split de n ≈ 350, y
corregir el 1.754 del §4.2. Ninguna toca un modelo ni un artefacto de despliegue.

**En una frase:** el repositorio no se contradecía, se estaba citando a sí mismo
sin decir de qué examen hablaba cada nota.

---

_Sección 14 añadida el 2026-09-03 a partir de `reports/20_reconciliacion_metricas_v1.md`.
Ninguna cifra de esta sección es nueva: todas provienen de un informe o de un
artefacto ya existente en este repositorio, salvo la reproducción de R² = 0.2412
sobre `models/deploy/puntaje_estimado.csv`, que recalcula una métrica a partir de
predicciones ya guardadas y no reentrena ningún modelo._

---

# 15. RECONCILIACIÓN APLICADA — LA MÉTRICA OFICIAL DE v1 QUEDA FIJADA

## 15.1 De diagnóstico a corrección

**La pregunta.** El §14 cerró con una lista de correcciones pendientes y una frase
incómoda: *"el informe 20 es diagnóstico y **no editó nada**"*. Mientras esa lista
siguiera abierta, el repositorio publicaba en su portada un R² que él mismo ya
había demostrado que **no existe**, y colocaba en una tabla de comparación una
cifra *dentro de muestra* junto a dos *hold-out*. Un lector externo —un evaluador
de la Copa, por ejemplo— se llevaría de ahí una conclusión falsa sin ninguna forma
de detectarlo. La pregunta ya no era *cuál es la cifra buena*, que eso lo resolvió
el informe 20, sino **si el repositorio se corrige a sí mismo o deja el error
publicado**.

**Qué se hizo.** Se aplicaron las correcciones de prioridad alta del informe 20 y
la de higiene del conteo de cohorte. **Solo documentación**: no se tocó ningún
script, ningún modelo, ningún `.joblib`, ningún predictor JS ni ningún CSV o JSON
de `outputs/`, y **no se reentrenó ni se regeneró nada**. Ninguna cifra correcta
se modificó: las tres evaluaciones siguen publicadas con su valor exacto —0.115
*hold-out*, 0.241 *dentro de muestra*, 0.084 *out-of-fold*—; lo que cambió es
**qué etiqueta lleva cada una y a qué informe se atribuye**.

## 15.2 La métrica oficial de v1

**R² out-of-fold = 0.084** (MAE **18.1**, RMSE **22.1**), con IC 95 % por
bootstrap de 1000 remuestreos **[0.053, 0.116]**, sobre los **1.748 estudiantes
examinados**. Fuente: informes 08 y 09b.

Es la oficial por cuatro razones, todas ya establecidas en el §14: es
**out-of-fold** —cada predicción viene de un modelo que no vio a ese estudiante—;
cubre a **los 1.748**, no a un test de ~350; es **la única con intervalo de
confianza**, y ese intervalo no toca el cero (el poder predictivo es real) y
**contiene al 0.115**; y es **coherente con todo lo demás** que este repositorio
mide honestamente: 0.064 (CV, §4.2), 0.091 (CV, §4.4), 0.086 (dataset A, §10.2).
Todo converge a **0.06–0.09**.

**Por qué existen las otras dos cifras**, y por qué ninguna es la oficial:

- **0.115 / MAE 17.44 (informe 03).** Es real y **se queda publicada tal cual**: es
  el *hold-out* del 20 % de aquel script, un único split de **n ≈ 350**. No se
  borra, porque es el resultado que ese script produjo. Pero con esa n un solo
  split se mueve varios centésimos por azar, y el IC del 0.084 lo contiene: es el
  **borde optimista del mismo fenómeno**, no un resultado distinto.
- **~0.241 / MAE 16.4 (informes 09b y 10).** También es real, y se reprodujo dígito
  a dígito —R² = 0.2412, MAE = 16.4331— sobre las 1.748 predicciones ya guardadas
  en `models/deploy/puntaje_estimado.csv`. Es una evaluación **dentro de muestra**:
  el ~80 % de esos estudiantes entrenaron el bosque que los está puntuando, así que
  mide en buena parte memorización. **Legítima solo con su etiqueta puesta, y nunca
  en la misma tabla que una métrica hold-out.**
- **~0.238 / ~18 pts (README).** **No es real.** No procede de ninguna evaluación
  documentada, no aparece en ningún informe, script, artefacto ni commit, y nunca
  tuvo otro valor en la historia de git. **Se retiró, no se "ajustó"**: corregirla a
  un decimal cercano habría sido fabricarle el respaldo que nunca tuvo.

## 15.3 Las ediciones aplicadas

| Fichero | Qué se cambió |
|---|---|
| `README.md` | La tabla *Estado del modelo* publicaba `~0.238 / ~18 pts` para v1. Se sustituye por **0.084 / 18.1 pts** y se añade una nota que declara explícitamente que es **out-of-fold sobre los 1.748 examinados**, con RMSE 22.1 e IC 95 % [0.053, 0.116]. |
| `reports/10_modelo_teorico_vs_empirico.md` | Es el punto donde el ~0.241 **perdió su etiqueta** por primera vez, y desde donde se propagó. Se le devuelve: el resumen ejecutivo y la tabla comparativa marcan ahora esa fila como **dentro de muestra** sobre las 1.748 filas de `models/deploy/puntaje_estimado.csv`, no una estimación de generalización. **Las cifras no se tocaron.** |
| `reports/14_optimizacion_v2.md` | Atribuía el ~0.241 / ~16.4 al **informe 03**, que no contiene esa cifra. Se corrige la atribución a los **informes 09b y 10**, se marca la fila como dentro de muestra y **no comparable** (conservando el asterisco) y se añade la advertencia de que colocar una cifra dentro de muestra junto al hold-out de v2 **hace que v2 parezca artificialmente peor**: la comparación limpia es la del informe 15, **MAE 18.45 (v1, fuera de muestra) vs 15.00 (v2, hold-out)**. |
| Este documento, §4.2 | Llamaba "datos **limpios** (1.754 estudiantes)" a lo que es el conteo crudo. Se corrige a la cadena real: **1.754 crudos → 1.750 tras limpieza → 1.748 tras deduplicar por documento**. |
| Este documento, §15 | Esta sección. |

**Lo que deliberadamente NO se tocó.** Los informes **03, 08, 09b, 12, 15, 17, 18,
19 y 20** quedan intactos: sus cifras son correctas para lo que cada uno declara
medir. Tampoco se tocó ningún script, modelo ni artefacto. Del listado del informe
20 siguen abiertas tres recomendaciones menores: anotar en el informe 03 el tamaño
de su test (n ≈ 350), poner la etiqueta *dentro de muestra* también en la tabla del
§9.3 de este documento, y **fijar las métricas dentro del artefacto de v1** como ya
hace v2 con `metricas_holdout` — esta última exige ejecutar código, así que no
cabía en una corrección solo documental.

**La regla que queda.** Toda cifra de R² o MAE se publica con **tres datos
inseparables**: valor, población (n) y partición (*in-sample* / CV / *out-of-fold*
/ *hold-out*). Las tres cifras en disputa habrían sido imposibles bajo esa regla:
no se contradecían entre sí, describían cosas distintas publicadas como si fueran
la misma.

---

_Sección 15 añadida el 2026-09-03. Registra la **aplicación** de las correcciones
diagnosticadas en `reports/20_reconciliacion_metricas_v1.md`. Ninguna cifra de esta
sección es nueva, y ninguna de las ediciones tocó un script, un modelo o un
artefacto de despliegue._
