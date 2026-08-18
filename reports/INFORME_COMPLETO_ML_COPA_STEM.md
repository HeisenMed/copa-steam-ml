
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

Se entrenaron 4 modelos con 18 features (one-hot incluido). Resultados sobre datos
**limpios** (1.754 estudiantes):

| Modelo | R² (CV) | R² (test) | RMSE (test) | MAE (test) |
|---|---|---|---|---|
| Regresión Lineal | 0.083 | 0.106 | 21.66 | 17.53 |
| **Random Forest ⭐** | 0.064 | **0.115** | 21.56 | 17.44 |
| XGBoost | −0.033 | 0.053 | 22.30 | 17.82 |
| LightGBM | −0.124 | 0.013 | 22.77 | 18.15 |

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
