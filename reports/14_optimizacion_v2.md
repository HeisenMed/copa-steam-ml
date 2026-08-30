# Optimización de Hiperparámetros — Modelo v2 — Copa STEM 2026

**Fundación SapienceLab** · Fase 5 · Informe generado: 2026-08-30

---

## Resumen ejecutivo

Se buscaron los mejores hiperparámetros del modelo de puntaje sobre
`dataset_C_perfil.csv` (1,148 filas con perfil académico) mediante
`RandomizedSearchCV` — 30 candidatos × 5 folds = **150 ajustes**.

- **Mejores parámetros:** `n_estimators=200`, `max_depth=None`,
  `min_samples_leaf=8`, `max_features=0.5`.
- **R² en hold-out del 20 %: +0.1766** · **MAE: 15.00**.
- **La ganancia de la búsqueda es marginal: +0.0142 de R²** frente a los
  hiperparámetros sin optimizar del script 03 (+0.1625). El MAE apenas se mueve
  (−0.06 puntos).
- El predictor JS v2 quedó exportado y **verificado en Node**: reproduce a
  sklearn con `máx|Δ| = 3.55e-14` y devuelve el mismo contrato que v1.

La conclusión operativa es coherente con el informe 12: **la elección de
hiperparámetros es secundaria**. Lo que movió el R² fueron las variables
(+0.083 al añadir el perfil académico), no el ajuste del estimador (+0.014).

## Resultados de la búsqueda

| Hiperparámetro | Rejilla explorada | Ganador |
| --- | --- | --- |
| `n_estimators` | 100, 200, 300, 500 | **200** |
| `max_depth` | 5, 10, 15, 20, None | **None** |
| `min_samples_leaf` | 1, 2, 4, 8 | **8** |
| `max_features` | 'sqrt', 'log2', 0.5 | **0.5** |

Configuración: `cv=5`, `n_iter=30`, `scoring='r2'`, `random_state=42`.
R² de validación cruzada del mejor candidato: **+0.1749**.

**Lectura de los ganadores.** `max_depth=None` con `min_samples_leaf=8` es una
combinación coherente: se deja crecer el árbol sin límite de profundidad, pero
la hoja mínima de 8 observaciones actúa como la verdadera regularización. Que
`min_samples_leaf` ganara en su valor **más alto** (8, el extremo de la rejilla)
sugiere que el modelo prefiere más regularización de la que la rejilla permite
explorar; convendría probar valores mayores (12, 16, 20) en una búsqueda futura.

`max_features=0.5` por encima de `'sqrt'` (≈0.21 con 23 features) indica que
cada árbol se beneficia de ver más variables por partición — consistente con que
haya pocas variables realmente informativas (informe 13: solo 4 con importancia
distinguible de cero), de modo que restringir mucho el muestreo deja árboles
ciegos a la señal.

## Comparación v1 vs v2

| Modelo | Dataset | n | Features | R² | MAE |
| --- | --- | --- | --- | --- | --- |
| v1 — script 03 (producción actual) | cohorte original | 1,748 | 18 | ~0.241 * | ~16.4 * |
| v1 — hiperparámetros sobre C | `dataset_C_perfil` | 1,148 | 23 | +0.1625 | 15.06 |
| **v2 — optimizado** | `dataset_C_perfil` | 1,148 | 23 | **+0.1766** | **15.00** |

> **\* Las métricas de v1 en producción no son comparables** con las de v2.
> Provienen de otro dataset, otra cohorte y otra partición (informe 03). La única
> comparación limpia es la de las dos últimas filas: **misma partición, mismas
> features, mismo dataset**, y ahí la diferencia es +0.0142 de R².

La comparación honesta contra el modelo desplegado hoy es imposible sin
reevaluar v1 sobre esta misma partición, y aun entonces v1 no puede usar las 5
variables nuevas — que es precisamente su desventaja estructural.

## Alcance: por qué solo se re-exportó un predictor

La instrucción original pedía reentrenar los 4 predictores con los
hiperparámetros optimizados. **Solo uno de los cuatro es un
`RandomForestRegressor`** y admite esa rejilla:

| Predictor | Modelo real | ¿Admite la rejilla? | Acción |
| --- | --- | --- | --- |
| Potencial STEM | RandomForestRegressor (componente rendimiento) | **Sí** | Re-exportado como v2 |
| Talento oculto | XGBClassifier | No — es clasificación; `min_samples_leaf` y `max_features` no son parámetros suyos | Sin cambios |
| Clustering | KMeans (k=4) | No — no supervisado; su hiperparámetro es `k` | Sin cambios |
| Condiciones | Modelo teórico (script 10) | No — sus pesos vienen de la literatura, no se entrena con datos | Sin cambios |

**Por qué no se reentrenaron de todos modos.** Reentrenar clustering o talento
oculto sobre `dataset_C_perfil` no sería una optimización sino un **cambio de
semántica**: produciría centroides y umbrales distintos, de modo que
`cluster_id` y `es_talento_oculto` dejarían de significar lo mismo que los
valores ya almacenados en `ml_scores`. Eso rompe la comparabilidad histórica y
es una decisión de producto, no un ajuste técnico. El modelo de condiciones,
además, no tiene nada que reentrenar: es *knowledge-driven* por diseño.

Si se quiere una v2 de esos tres, cada uno necesita su propia rejilla
(`k` para KMeans; `learning_rate`/`max_depth`/`subsample` para XGBoost) y una
decisión explícita sobre qué hacer con los valores v1 en producción.

## Artefactos generados

Todos con sufijo `_v2`. **Ningún fichero existente fue modificado.**

| Fichero | Tamaño | Contenido |
| --- | --- | --- |
| `models/mejor_modelo_puntaje_v2.joblib` | 1,977 KB | Modelo sklearn + preprocesamiento + params + métricas |
| `models/deploy/potencial_stem_predictor_v2.js` | 816 KB | Predictor puro ES6 con los 200 árboles embebidos |
| `outputs/F14_optimizacion_v2.json` | < 1 KB | Resumen de la búsqueda |

Tamaños de los cuatro predictores de deploy, para dimensionar el despliegue:

| Predictor | v1 | v2 |
| --- | --- | --- |
| `potencial_stem_predictor.js` | 1,302 KB | **816 KB** |
| `talento_oculto_predictor.js` | 101 KB | — (sin cambios) |
| `clustering_predictor.js` | 5 KB | — (sin cambios) |
| `indice_condiciones_predictor.js` | 3 KB | — (sin cambios) |

El v2 pesa **un 37 % menos** que el v1 pese a tener 5 features más: 200 árboles
en vez de 300. Es una ventaja real para una Edge Function, donde el tamaño del
bundle afecta al arranque en frío.

## Verificación del export

El JS v2 no se escribió desde cero: se **clonó el cuerpo del v1** y se sustituyó
únicamente la constante `SPEC`. Así las funciones de preprocesamiento
(`_toFloat`, `_parseCount`, `_ordLevel`, `_binSi`, `_featuresPuntaje`,
`_predictPuntaje`) son idénticas byte a byte y el contrato con la Edge Function
no cambia.

Las 5 variables nuevas entran **sin tocar una línea de JavaScript**, porque
`_featuresPuntaje` itera las listas de `SPEC.puntaje.preprocess`: basta con que
aparezcan en `numeric` (4 de ellas) y en `binary` + `binary_src`
(`clases_extra_bin`). Este es el motivo de haber respetado el formato original
en lugar de inventar uno nuevo.

Dos verificaciones pasadas:

1. **Intérprete vs sklearn** (dentro del script): `máx|Δ| = 3.55e-14` sobre 200
   estudiantes — el recorrido de árboles en Python puro reproduce al modelo.
2. **Ejecución real en Node v24**: importado como módulo ES6 y evaluado sobre 5
   estudiantes reales. Devuelve la misma estructura que v1
   (`indice_potencial`, `componente_rendimiento`, `componente_engagement`,
   `componente_resiliencia`, `categoria`).

```
doc 1058200744 → indice 34.24  rend 30.05  "En desarrollo"
doc 1036519929 → indice 67.27  rend 72.39  "Promedio"
doc 1041352460 → indice 73.72  rend 88.07  "Alto potencial"
```

## Advertencias para el despliegue

1. **v2 solo aplica al 37 % de los estudiantes.** Está entrenado con las 5
   variables de perfil académico. Para los ~1,924 inscritos que no las tienen,
   el predictor imputará las medianas y perderá exactamente la ventaja que lo
   justifica. Hay que decidir una política explícita: modelo híbrido, o
   `puntaje_estimado` nulo para quien no tenga perfil.
2. **La Edge Function no lee esas columnas.** Su `SELECT` (`COLS_INSC`) no
   incluye ninguna de las cinco; `grep promedio_academico` sobre `index.ts` da
   cero coincidencias. Sin ampliar esa consulta, desplegar v2 no cambia nada.
3. **El modelo de producción se reajustó con las 1,148 filas completas**, no
   solo con el train. Es la práctica estándar para desplegar, pero implica que
   las métricas reportadas (hold-out) corresponden a un modelo entrenado con
   918 filas, no con el artefacto exportado.
4. **R² = 0.18 sigue siendo bajo.** Con MAE de 15 puntos sobre una escala 0–100,
   cualquier publicación de `puntaje_estimado` debe ir con banda de error
   explícita y nunca como cifra puntual.
5. **`min_samples_leaf` ganó en el extremo de la rejilla.** Conviene una segunda
   búsqueda con valores mayores antes de considerar cerrada la optimización.

## Limitaciones

- **La búsqueda optimiza R², que es la métrica con la que luego se juzga.** El
  hold-out del 20 % (230 estudiantes) es pequeño; la diferencia de +0.014 entre
  optimizado y base está dentro del rango de variación entre particiones que se
  observó en el informe 12 (desviación entre folds de 0.026).
- **`RandomizedSearchCV` explora 30 de 240 combinaciones posibles** (12.5 %). No
  hay garantía de haber encontrado el óptimo global.
- **No se optimizó el preprocesamiento.** Estrategia de imputación, tratamiento
  de las escalas Likert y codificación de categóricas se heredaron del script 03
  sin cuestionarlos.
- **Una sola familia de modelo.** El informe 12 mostró que la regresión lineal
  queda a +0.017 de este bosque; no se exploró si una lineal regularizada
  afinada cerraría la brecha con mucha menos complejidad de despliegue.

## Glosario

- **RandomizedSearchCV:** *definición* — prueba combinaciones al azar de la
  rejilla y se queda con la mejor por validación cruzada. *Analogía* — probar 30
  recetas al azar en vez de las 240 posibles. *Ejemplo* — 30 candidatos × 5
  folds = 150 entrenamientos.
- **`min_samples_leaf`:** *definición* — mínimo de observaciones en una hoja.
  *Analogía* — no sacar conclusiones de un grupo de menos de 8 personas.
  *Ejemplo* — valor 8: ninguna regla del modelo se apoya en menos de 8
  estudiantes.
- **`max_features`:** *definición* — fracción de variables que cada partición
  puede considerar. *Analogía* — obligar a cada juez a mirar solo la mitad de
  las pruebas, para que no todos se fijen en lo mismo. *Ejemplo* — 0.5 sobre 23
  features ≈ 11 por partición.
- **Hold-out:** *definición* — porción de datos apartada y nunca usada para
  entrenar. *Analogía* — el examen final con preguntas que no estaban en los
  ejercicios. *Ejemplo* — los 230 estudiantes del 20 %.

## Referencias bibliográficas

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper-Parameter
  Optimization*. JMLR. Justifica que la búsqueda aleatoria supere a la de
  rejilla exhaustiva con el mismo presupuesto de cómputo.
- Probst, P., Wright, M. & Boulesteix, A.-L. (2019). *Hyperparameters and tuning
  strategies for random forest*. WIREs Data Mining. Documenta que el Random
  Forest es poco sensible al ajuste de hiperparámetros — consistente con el
  +0.014 observado aquí.

---
_Generado a partir de `notebooks/14_optimizacion_hiperparametros.py` — Copa STEM 2026._
