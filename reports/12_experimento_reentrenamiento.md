# Experimento de Reentrenamiento — Copa STEM 2026

**Fundación SapienceLab** · Fase 5 · Informe generado: 2026-08-29

---

## Resumen ejecutivo

Se entrenó el **mismo** Random Forest, con el **mismo** protocolo de validación,
sobre los cuatro datasets del script 11, para separar el efecto del volumen de
datos del efecto de las variables. El resultado es inequívoco y apunta en una
sola dirección:

- **Duplicar los datos no sirvió.** De A (1,735 filas) a B (3,072 filas) el R²
  **bajó** de +0.086 a +0.053, pese a casi duplicar la muestra.
- **Las 5 variables nuevas sí sirvieron.** Con la muestra fija en 1,148 filas,
  el R² pasó de +0.098 a **+0.180** — prácticamente el doble — y el MAE bajó
  0.88 puntos.
- **Cinco variables académicas superan a dieciocho socioeconómicas.** Por
  separado, el bloque académico explica más (R² = +0.108) que todo el bloque
  socioeconómico (R² = +0.098), usando una tercera parte de las variables.

La conclusión operativa es que **la calidad de las variables pesa más que la
cantidad de datos**. Seguir acumulando inscritos con el formulario viejo no
mejora la predicción; preguntar mejor, sí.

## Diseño experimental

El experimento tiene dos contrastes, cada uno con un solo factor variando:

| Contraste | Qué cambia | Qué se mantiene | Pregunta |
| --- | --- | --- | --- |
| **A → B** | El tamaño de la muestra | El conjunto de variables | ¿Ayuda acumular más inscritos? |
| **C′ → C** | El conjunto de variables | La muestra y la partición | ¿Aportan las 5 variables nuevas? |

**C′ → C es el contraste decisivo.** Es el único de los dos que constituye un
experimento controlado en sentido estricto: mismas 1,148 filas, misma partición
train/test, mismos folds, mismo modelo. Lo único que cambia son las 5 columnas.
Cualquier diferencia en las métricas es atribuible a ellas y a nada más.

**A → B no es un experimento controlado**, y no puede serlo: para tener más
datos hay que cambiar la población. B incluye Bello, que no existía en la
cohorte de A. Por eso su lectura exige más cuidado (ver más abajo).

### Protocolo

Idéntico al del script 03, para que la línea base sea comparable:

- **Modelo:** `RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=8)`.
- **Validación:** `KFold(5, shuffle=True)` sobre el train + hold-out del 20 %
  estratificado por quintil de puntaje.
- **Imputación:** mediana (numéricas y ordinales) y moda (binarias y
  categóricas), ajustadas **solo con el train** dentro del `Pipeline`, para no
  filtrar información del test.
- **Features:** conteos de listas, ordinales 0–3, binarias Sí/No y one-hot.
  **Sin telemetría** —se mide durante el examen, no está disponible al momento
  de predecir— y **sin `porcentaje`**, duplicado exacto del target.
- **Reproducible:** `random_state=42` en todo.

### Un control que hubo que forzar

`dataset_B_completo.csv` **arrastra físicamente las 5 columnas nuevas**, casi
todas vacías, porque sale del mismo export que C. En una primera corrida el
pipeline las incorporó a B sin que se notara, y B apareció con 24 features en vez
de 19. A se salvó por accidente: sus 5 columnas están 100 % vacías y
`SimpleImputer` descarta las columnas sin ningún valor.

Eso rompía el diseño: el contraste A → B habría mezclado «más datos» con «más
variables», que es exactamente lo que el experimento pretende separar. La
inclusión del bloque nuevo se volvió **explícita por dataset** y solo C la
activa. **El efecto de la corrección no fue cosmético: el R² de B cayó de +0.092
a +0.053.** El número contaminado sugería que más datos ayudaban un poco; el
número limpio muestra lo contrario.

## Justificación del modelo: ¿Por qué Random Forest?

Todo el experimento se corrió con un solo algoritmo. Esa decisión se hereda del
script 03 —para que la línea base fuera comparable— pero conviene justificarla,
y sobre todo **someterla a prueba** en vez de defenderla solo con argumentos.

### La naturaleza del problema

Predecir `puntaje_obtenido` es una **regresión tabular con tipos mixtos**: el
target es continuo (0–100) y los predictores combinan numéricas continuas
(`promedio_academico`), conteos (`n_herramientas`), ordinales
(`nivel_programacion` 0–3, escalas Likert 1–5), binarias (`computador_en_casa`)
y categóricas nominales (`municipio`, `genero`). A esto se suman ausencias
estructurales: el 63 % de las filas no tiene perfil académico, y esa ausencia no
es aleatoria —depende de cuándo se inscribió el estudiante—.

No es un problema de percepción (imágenes, texto, audio) ni de series
temporales. Es el escenario clásico de datos tabulares heterogéneos con muestra
pequeña, que es exactamente donde los ensambles de árboles rinden mejor.

### Por qué Random Forest encaja

- **Maneja tipos mixtos sin transformaciones artificiales.** Los árboles parten
  por umbrales, así que una ordinal 0–3 y una continua conviven sin necesidad de
  estandarizar ni de asumir que la distancia entre «Básica» e «Intermedia» es la
  misma que entre «Intermedia» y «Avanzada».
- **Robusto ante valores faltantes.** Combinado con imputación por mediana/moda,
  el promediado sobre 300 árboles diluye el efecto de un valor imputado: ningún
  árbol individual decide el resultado.
- **Importancia de variables nativa e interpretable.** Entrega un ranking
  directo de qué pesa más, que es el insumo del análisis de explicabilidad
  (script 13). Para un proyecto cuyo objetivo es *entender* qué mueve el
  rendimiento, esto no es un extra: es el producto.
- **Estable con muestras de 1,000–3,000 filas.** El bagging reduce la varianza
  sin necesitar el volumen que piden los métodos de gradiente afinados. En la
  comparación de abajo esto se ve en la desviación entre folds.

### La comparación empírica

Tres familias sobre `dataset_C_perfil.csv` (1,148 filas), con el **mismo**
preprocesamiento, la misma partición y los mismos 5 folds. Configuraciones
heredadas del script 03:

| Modelo | CV R² | Desv. entre folds | CV MAE |
| --- | --- | --- | --- |
| Regresión Lineal | +0.163 | 0.062 | 15.06 |
| **Random Forest** | **+0.180** | **0.026** | **14.84** |
| XGBoost | +0.090 | 0.036 | 15.68 |

El resultado **corrige de raíz el argumento teórico habitual**, que dice que se
sacrifica precisión a cambio de interpretabilidad. Aquí no hay tal sacrificio:
Random Forest es a la vez el más preciso y el más interpretable de los tres.

### Por qué no regresión lineal

El argumento teórico es que la relación entre condiciones socioeconómicas y
rendimiento no tiene por qué ser lineal ni aditiva: es plausible que tener
computador importe mucho en estrato 1 y nada en estrato 3, una interacción que
un modelo lineal solo captura si alguien la especifica a mano.

**Los datos matizan ese argumento.** La regresión lineal alcanza R² = +0.163
frente a +0.180 del bosque: la ventaja de modelar no-linealidades existe pero es
**modesta** (+0.017). Lo que sí separa claramente a los dos modelos es la
**estabilidad**: la lineal varía 0.062 entre folds y el bosque 0.026, menos de la
mitad. Con 1,148 filas, la lineal es sensible a qué estudiantes caen en cada
partición; el bosque no.

Conclusión honesta: la regresión lineal es una alternativa **respetable** para
este problema, no un hombre de paja. Se prefiere el bosque por estabilidad y por
manejo nativo de tipos mixtos, no porque la linealidad sea absurda.

### Por qué no XGBoost

El argumento teórico esperado —«XGBoost da un poco más de R² pero es mucho menos
interpretable, mal negocio»— **no se sostiene con estos datos**: XGBoost no da
más R², da bastante menos (+0.090 frente a +0.180). No hay disyuntiva que
resolver, porque el modelo menos interpretable es también el peor.

La razón es el tamaño de la muestra. El *boosting* construye árboles en
secuencia, cada uno corrigiendo el error del anterior; con 918 filas de
entrenamiento y 400 rondas, esa secuencia empieza a ajustar ruido antes de
extraer señal. El *bagging* del Random Forest, que promedia árboles
independientes en vez de encadenarlos, es estructuralmente más resistente en
este régimen.

> **Advertencia.** XGBoost se corrió con la configuración del script 03, **sin
> ajuste de hiperparámetros**. Su mal desempeño aquí refleja parámetros no
> afinados sobre una muestra pequeña, no un techo intrínseco del algoritmo. Con
> búsqueda de hiperparámetros probablemente se acercaría al bosque. Lo que la
> tabla demuestra no es que XGBoost sea inferior, sino que **no hay una ganancia
> gratuita esperándonos** si se cambia de algoritmo sin más trabajo.

### Por qué no redes neuronales

- **Volumen insuficiente.** Las redes necesitan del orden de decenas de miles de
  observaciones para superar a los ensambles de árboles en datos tabulares.
  Aquí hay 1,148 filas: un orden de magnitud por debajo, como mínimo.
- **Sin estructura que explotar.** La ventaja de una red está en aprender
  representaciones de datos con estructura espacial o secuencial. Una tabla de
  23 columnas heterogéneas no tiene esa estructura.
- **Caja negra.** No entregan un ranking de importancia comparable, lo que
  eliminaría el objetivo principal del proyecto.
- **La literatura no las respalda aquí.** En predicción de rendimiento académico
  con datos tabulares, los ensambles de árboles siguen siendo el estándar.

### Limitación honesta

La lectura honesta de la tabla **no** es «renunciamos a precisión por
interpretabilidad» —eso sería falso, el bosque gana en las dos dimensiones—.
Las limitaciones reales son otras tres:

1. **Ningún modelo funciona bien en términos absolutos.** El mejor R² es 0.18:
   más del 80 % de la varianza del puntaje queda sin explicar. La elección de
   algoritmo es secundaria frente a esa brecha; lo que la mueve son las
   variables (hallazgos 1 a 3), no el estimador.
2. **La ventaja sobre la regresión lineal es pequeña frente a su propia
   incertidumbre.** +0.017 de diferencia con desviaciones entre folds de 0.026 y
   0.062 no permite declarar un ganador con firmeza estadística.
3. **La comparación no está afinada.** Ninguna de las tres familias recibió
   búsqueda de hiperparámetros. Es una comparación de configuraciones razonables,
   no de los mejores modelos posibles de cada familia.

Random Forest se mantiene porque es la opción **defendible y estable**, no
porque esté demostrado que sea la óptima.

### Referencia académica

Random Forest (Breiman, 2001) es el modelo más reportado en la literatura de
predicción de rendimiento académico con datos tabulares. Las revisiones
sistemáticas del área —*Educational Data Mining* y *Learning Analytics*— lo
sitúan de forma consistente entre los de mejor desempeño, y su uso extendido
facilita comparar resultados entre estudios. Esa comparabilidad es una razón
adicional para preferirlo en un proyecto que aspira a dialogar con la literatura
educativa, no solo a producir un número.

## Tabla comparativa

| Dataset | Rol | n_samples | n_features | CV R² | CV MAE | test R² | test MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline | Línea base | 1,735 | 18 | +0.086 | 19.01 | +0.111 | 19.18 |
| B_completo | Más datos | 3,072 | 19 | +0.053 | 17.95 | +0.035 | 17.94 |
| **C_perfil** | **Más variables** | **1,148** | **23** | **+0.180** | **14.84** | **+0.162** | **15.06** |
| C_sin_features | Control de C | 1,148 | 18 | +0.098 | 15.72 | +0.113 | 15.78 |

`n_features` es la dimensión de entrada al modelo con el one-hot ya expandido.
CV es la media de los 5 folds sobre el train; test es el hold-out del 20 %. Las
dos columnas coinciden en ordenar igual los cuatro modelos, lo que indica que
los resultados no dependen de una partición afortunada.

B tiene 19 features y no 18 porque su población incluye Bello, un municipio más
en el one-hot.

## Hallazgo 1 — Más datos no mejoraron la predicción

| | A_baseline | B_completo | Δ |
| --- | --- | --- | --- |
| n | 1,735 | 3,072 | +1,337 |
| CV R² | +0.086 | +0.053 | **−0.033** |
| CV MAE | 19.01 | 17.95 | −1.06 |

Casi se duplicó la muestra y la capacidad explicativa **empeoró**.

**El MAE bajó, pero eso no es una mejora.** El MAE se expresa en puntos del
target, así que depende de cuán dispersa sea la población: A tiene σ = 24.23 y B
σ = 22.66. Una población más concentrada produce errores absolutos más pequeños
aunque el modelo no haya aprendido nada nuevo. El R², al normalizar por la
varianza de cada muestra, es la **única métrica comparable entre poblaciones
distintas** — y el R² baja. Leer el descenso del MAE como evidencia de mejora
sería un error de interpretación.

**Por qué ocurre.** Con el formulario viejo, cada estudiante nuevo aporta otra
copia del mismo vector socioeconómico: estrato, jornada, acceso a computador. El
modelo ya había extraído de 1,735 observaciones toda la señal que esas variables
contienen; las 1,337 adicionales no traen información nueva, solo más ruido y una
población más heterogénea (tres municipios en vez de dos). Es un techo de
información, no un problema de tamaño de muestra.

## Hallazgo 2 — Las 5 variables nuevas casi duplicaron el R²

| | C_sin_features | C_perfil | Δ |
| --- | --- | --- | --- |
| n | 1,148 | 1,148 | 0 |
| Features | 18 | 23 | +5 |
| CV R² | +0.098 | **+0.180** | **+0.083** |
| CV MAE | 15.72 | **14.84** | −0.88 |

Mismas filas, misma partición, mismo modelo. Lo único que cambia son
`promedio_academico`, `horas_estudio_matematicas`, `motivacion_participar`,
`clases_extra_matematicas` y `gusto_logica`.

**El R² pasa de +0.098 a +0.180: un aumento del 84 %.** Aquí el MAE sí es
comparable —la población es idéntica— y también mejora: 0.88 puntos menos de
error por estudiante.

Cinco preguntas añadidas al formulario de inscripción lograron lo que 1,337
estudiantes adicionales no lograron.

## Hallazgo 3 — Comportamiento académico frente a origen socioeconómico

Para saber no solo *si* aportan sino *cuánto pesan*, se entrenó cada bloque por
separado sobre las mismas 1,148 filas de C:

| Bloque | Features | CV R² | CV MAE |
| --- | --- | --- | --- |
| Solo socioeconómicas | 18 | +0.098 | 15.72 |
| **Solo académicas** | **5** | **+0.108** | **15.43** |
| Ambas | 23 | +0.180 | 14.84 |

**Cinco variables de comportamiento académico explican más que dieciocho de
origen socioeconómico**, con una tercera parte de la dimensionalidad.

Los dos bloques son además **largamente complementarios**: por separado suman
0.098 + 0.108 = 0.206 y juntos rinden 0.180. El solapamiento es pequeño, lo que
indica que miden cosas distintas — el origen y la conducta no son la misma
información, y el modelo aprovecha ambas.

## Conclusión: la calidad de las variables pesa más que la cantidad de datos

Los dos contrastes apuntan a lo mismo desde ángulos opuestos:

- Multiplicar por 1.8 el número de estudiantes, manteniendo las variables:
  **R² −0.033**.
- Añadir 5 variables, manteniendo los estudiantes: **R² +0.083**.

Cuando un modelo se estanca, el reflejo habitual es pedir más datos. Este
experimento muestra que ese reflejo era erróneo en este caso: el modelo no estaba
limitado por el tamaño de la muestra sino por **qué se le estaba preguntando a
cada estudiante**. Más filas de las mismas columnas no levantan un techo de
información; columnas nuevas sí.

## Implicación académica

El bloque socioeconómico describe el **origen** del estudiante: estrato,
municipio, tipo de institución, con quién vive, si tiene computador. El bloque
nuevo describe su **comportamiento**: qué promedio lleva, cuántas horas estudia
matemáticas, qué tan motivado está, si toma clases extra, si le gusta la lógica.

El experimento muestra que **el rendimiento en Copa STEM se explica mejor por el
comportamiento académico que por el origen socioeconómico**. No es una lectura
interpretativa: es la comparación directa de dos modelos entrenados sobre las
mismas 1,148 filas, donde 5 variables de conducta superan a 18 de contexto.

Esto conversa de forma directa con el resto del proyecto:

- **Refuerza la línea de talento oculto** (script 06). Si el origen predijera el
  desempeño, buscar alto rendimiento en condiciones adversas sería buscar una
  rareza. Que la conducta pese más significa que ese talento es esperable, no
  excepcional.
- **Matiza el modelo teórico** (script 10), construido con pesos de la literatura
  sobre estatus socioeconómico. Su bajo R² directo no era solo un problema de
  escala: le faltaba el bloque que más explica.
- **No contradice la existencia de brechas** (script 02). Las brechas
  socioeconómicas son reales y están documentadas. Lo que dice este experimento
  es que, **a igualdad de condiciones de origen**, lo que el estudiante hace
  discrimina más su resultado que de dónde viene.

La lectura pedagógica es esperanzadora: las variables que más explican el
rendimiento —horas de estudio, motivación, clases extra— son **modificables por
intervención**, mientras que el estrato y el municipio no lo son.

> **Advertencia causal.** Todo lo anterior es predictivo, no causal. Que las
> horas de estudio predigan el puntaje no demuestra que aumentarlas lo suba: es
> igual de compatible con que quien va bien en matemáticas estudie más porque le
> resulta gratificante. Establecer causalidad exigiría un diseño distinto.

## Recomendaciones

1. **Hacer obligatorias las 5 preguntas** en el formulario de inscripción. Es la
   intervención con mejor relación costo-beneficio que muestra el experimento.
2. **Reactivar `puntaje_estimado` solo para quienes tengan perfil académico.**
   Con R² ≈ 0.18 el modelo sigue siendo débil en términos absolutos, pero ya es
   el doble que la alternativa. Debe publicarse con banda de error explícita
   (MAE ≈ 15 puntos) y jamás como cifra puntual.
3. **No esperar a acumular más inscritos** como estrategia de mejora: el
   hallazgo 1 muestra que esa vía está agotada con el formulario actual.
4. **Repetir el experimento cuando C supere las 2,000 filas**, para confirmar
   que la ventaja se sostiene y no es un artefacto del tamaño de C.
5. **Explorar qué preguntas faltan.** Si 5 variables de conducta duplicaron el
   R², conviene medir si otras del mismo tipo (asistencia, hábitos de lectura,
   autoeficacia) siguen aportando.

## Limitaciones

- **R² = 0.18 sigue siendo bajo.** El modelo explica menos de una quinta parte de
  la varianza del puntaje. La mejora es grande en términos relativos, no en
  términos absolutos: no habilita decisiones individuales de alto impacto.
- **C no es una muestra aleatoria.** Son los inscritos más recientes y de
  municipios distintos. La ventaja de las variables nuevas está medida
  limpiamente dentro de C, pero su magnitud podría diferir en otra población.
- **Las variables nuevas son autorreportadas.** `promedio_academico` y
  `horas_estudio_matematicas` no están verificadas contra registros escolares y
  pueden tener sesgo de deseabilidad social.
- **Un solo modelo.** Todo el experimento usa Random Forest con hiperparámetros
  fijos, heredados del script 03 para mantener la comparabilidad. No se exploró
  si otra familia aprovecha mejor las variables nuevas.
- **A es una cohorte reconstruida** por fecha de inscripción, no un registro
  exacto de las filas del entrenamiento original (ver informe 11).

## Glosario

- **R² (coeficiente de determinación):** *definición* — proporción de la varianza
  del target que el modelo explica. *Analogía* — qué parte del rompecabezas
  lograste armar. *Ejemplo* — R² = 0.18 significa que se explica el 18 %.
- **MAE (error absoluto medio):** *definición* — promedio de la distancia entre
  predicción y valor real. *Analogía* — cuántos puntos te equivocas en promedio.
  *Ejemplo* — MAE = 14.84 → el pronóstico se desvía ~15 puntos.
- **Techo de información:** *definición* — límite de precisión impuesto por las
  variables disponibles, insensible a añadir más observaciones. *Analogía* —
  intentar adivinar la estatura de alguien sabiendo solo su ciudad: mil personas
  más no ayudan. *Ejemplo* — el contraste A → B.
- **Variable de conducta vs de origen:** *definición* — lo que el estudiante
  hace frente a las condiciones en que nació. *Analogía* — cuánto entrena un
  atleta frente a en qué país nació. *Ejemplo* — horas de estudio vs estrato.

## Referencias bibliográficas

- Hattie, J. (2009). *Visible Learning*. Síntesis de meta-análisis; los factores
  de conducta y hábitos de estudio muestran tamaños de efecto mayores que los
  de contexto familiar.
- Sirin, S. R. (2005). *Socioeconomic Status and Academic Achievement: A
  Meta-Analytic Review of Research*. Review of Educational Research.
- OECD (2016). *PISA 2015 Results (Volume II): Policies and Practices for
  Successful Schools*. Sobre la fracción de varianza que el nivel
  socioeconómico explica en pruebas estandarizadas.
- Halevy, A., Norvig, P. & Pereira, F. (2009). *The Unreasonable Effectiveness of
  Data*. IEEE Intelligent Systems. Contrapunto útil: el volumen de datos rinde
  cuando el espacio de features es rico, que es justamente la condición que no
  se cumplía en el contraste A → B.
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32. Modelo
  usado en todo el experimento.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*.
  KDD. Familia contrastada en la justificación del modelo.
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). *Why do tree-based models
  still outperform deep learning on typical tabular data?* NeurIPS. Sustenta la
  decisión de no usar redes neuronales en datos tabulares de este tamaño.

---
_Generado a partir de `notebooks/12_experimento_reentrenamiento.py` — Copa STEM 2026._
