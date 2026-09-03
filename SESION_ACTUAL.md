# SESIÓN ACTUAL — Copa STEM 2026

Fundación SapienceLab · Última actualización: 2026-09-03

Este archivo se reescribe completo en cada actualización. No se acumula historial.

---

## 1. Qué se hizo en este turno

El turno tuvo dos mitades: una **auditoría de cifras** dentro de este repo y una
**puesta al día del estado real del proyecto**, que hasta ahora estaba repartido
entre este repo, el repo web y Supabase sin que ningún documento lo juntara.

### 1.1 Auditoría y reconciliación de las métricas de v1 (verificado aquí)

**Script 20 — `reports/20_reconciliacion_metricas_v1.md`.** Circulaban tres R²
distintos para el mismo bosque v1: 0.115 (informe 03), ~0.241 (informe 14) y
~0.238 (README). El informe rastreó los tres. No eran errores de transcripción de
un mismo número: son **tres evaluaciones metodológicamente distintas**, y una de
ellas directamente no existe.

* Cifra           * Qué es en realidad                                      * Veredicto
* 0.084 / 18.1    * Out-of-fold sobre los 1,748, IC 95 % [0.053, 0.116]     * **MÉTRICA OFICIAL de v1**
* 0.115 / 17.44   * Hold-out del 20 % del informe 03, un solo split n ≈ 350 * Válida, es el borde optimista
* ~0.241 / ~16.4  * Dentro de muestra sobre los 1,748 (informes 09b y 10)   * Legítima solo con su etiqueta
* ~0.238 / ~18    * No procede de ninguna evaluación documentada            * **Sin fuente. Retirada.**

El ~0.241 se reprodujo dígito a dígito (R² = 0.2412, MAE = 16.4331) recalculando
sobre `models/deploy/puntaje_estimado.csv`, sin reentrenar nada. El ~0.238 no
aparece en ningún informe, script, artefacto ni commit de la historia.

**Commit `a62581c` — correcciones documentales aplicadas.** Excepción autorizada a
la regla de no tocar ficheros existentes: dejar publicada una cifra que el propio
repo ya demostró falsa era peor que editarla.

* Fichero                                     * Qué se corrigió
* README.md                                   * Fila de v1: ~0.238/~18 → **0.084 / 18.1**, con nota OOF, RMSE 22.1 e IC
* reports/10_modelo_teorico_vs_empirico.md    * Devuelta la etiqueta **dentro de muestra** (bullet + tabla). Cifras intactas
* reports/14_optimizacion_v2.md               * Atribución corregida: no es del informe 03 sino de 09b y 10. Advertencia añadida
* reports/INFORME_COMPLETO §4.2               * "1.754 limpios" → cadena real 1.754 crudos → 1.750 limpios → 1.748 deduplicados
* reports/INFORME_COMPLETO §15                * Sección nueva que registra la reconciliación

No se tocó ningún script, modelo, `.joblib`, predictor JS ni artefacto de
`outputs/`. Los informes 03, 08, 09b, 12, 15, 17, 18, 19 y 20 quedaron intactos.

La corrección del informe 14 importa más de lo que parece: su tabla ponía la cifra
*dentro de muestra* de v1 junto al *hold-out* de v2, lo que **hacía parecer que v2
empeoraba el modelo**. Es al revés — la comparación limpia es la del informe 15,
MAE 18.45 (v1, fuera de muestra) vs 15.00 (v2, hold-out).

### 1.2 Trabajo hecho FUERA de este repo (reportado; esta sesión no puede verificarlo)

Todo lo que sigue ocurrió en el repo web y en Supabase. Se registra **como
reportado**, no como verificado desde aquí.

**Edge Function `calcular-ml-scores`:**

- `COLS_INSC` ampliado para leer las 5 columnas de perfil académico. El pendiente
  que decía "`grep promedio_academico` da cero" queda **RESUELTO**.
- Predictor v2 cableado en `index.ts` con **fallback a v1**: enruta según
  `promedio_academico != null` y escribe `modelo_version` (`'v2'` / `'v1_fallback'`).
- Está **commiteado, pero NO verificado en runtime y NO desplegado**.
  **Producción sigue calculando con v1 hoy.**
- `potencial_stem_predictor_v2_corrected.js` copiado al repo web.

**Supabase:**

* Objeto                       * Estado
* `ml_scores.modelo_version`   * Columna añadida
* `ml_scores_backup_20260901`  * Tabla de respaldo creada
* `ml_scores_v2`               * Creada y poblada con los 3,072 scores corregidos
* `ml_scores_sin_examen`       * Creada y poblada con las 248 filas del script 18
* Permisos                     * SELECT anon + política de lectura pública en las dos nuevas

**Repo web:** pestaña **"Estudio ML"** construida en el Modo Monitor, con 8
secciones que cubren el estudio completo. Sus conteos de categoría en vivo se
contrastaron contra los informes 17 y 18 y **coincidieron exactamente** (5/5 y
14/14).

### 1.3 Comprobaciones hechas en este turno (pendiente 8)

Ambas son de solo lectura y ambas dieron resultado **negativo**:

- **El `.gitignore` sigue en UTF-16.** No se aplicó la conversión: está en UTF-16
  LE tanto en el árbol de trabajo como en HEAD. `git check-ignore` no devuelve
  nada para `.venv`, `models/mejor_modelo_puntaje_v2.joblib` ni
  `notebooks/__pycache__` — **ninguno de sus patrones está en vigor**.
- **La búsqueda de CSV en la historia NO salió limpia.** Hay **5 CSV bajo `data/`
  commiteados y presentes en HEAD**, y sus cabeceras incluyen `numero_documento`,
  `nombres`, `apellidos` e `institucion_educativa`: son **datos identificables de
  estudiantes menores de edad**. Ver pendiente 8, que sube de prioridad.

---

## 2. Dónde está cada cosa

**Repos**

* Repo ML  * `Copa STEAM/ml-models` (rama main, scripts 01-20)
* Repo web * `Recursos Web/sapiencex` (repo git independiente: `src/`, `supabase/`, `docs/sql/`)

**Modelos y métricas**

* Modelo v1 (producción) * R² OOF = **0.084**, MAE 18.1, RMSE 22.1, IC 95 % [0.053, 0.116] * `models/mejor_modelo_puntaje.joblib`
* Modelo v2 (listo)      * R² hold-out = **0.1766**, MAE **15.00** (n_test 230)             * `models/mejor_modelo_puntaje_v2.joblib`
* v1 vs v2 limpio        * MAE **18.45** → **15.00** (informe 15, fuera de muestra)         * −3.45 pts, a favor de v2

**Artefactos de despliegue**

* Fichero                                                 * Estado
* `models/deploy/potencial_stem_predictor_v2_corrected.js`* 828 KB, ref n=3,072 — **el bueno**, ya copiado al repo web
* `models/deploy/potencial_stem_predictor_v2.js`          * 816 KB, ref n=1,148 — **NO desplegar**, se conserva para rollback
* `models/deploy/potencial_stem_predictor.js`             * v1, es lo que corre en producción hoy
* `outputs/F17_ref_rendimiento_corregido.json`            * Referencia corregida (n=3,072, media 41.74, σ=22.6583)
* `outputs/F19_verificacion_deploy_v2.json`               * Traza de verificación del artefacto corregido

**Estado del despliegue en una línea:** el código v2 con fallback está escrito y
commiteado en el repo web, pero **producción sigue en v1**. Falta verificarlo en
runtime y desplegarlo (pendiente 1).

**Datos en Supabase**

* Tabla                       * Contenido
* `ml_scores`                 * Producción, calculada con v1; ya tiene `modelo_version`
* `ml_scores_backup_20260901` * Respaldo previo a los cambios
* `ml_scores_v2`              * 3,072 scores v2 con la referencia corregida
* `ml_scores_sin_examen`      * 248 inscritos sin examen (script 18), población separada

---

## 3. Pendientes en orden

1. **Verificar el predictor v2 en runtime dentro de la Edge Function y desplegarlo.**
   Bloqueado: `supabase functions serve` necesita Docker Desktop. Hasta que esto
   pase, **producción sigue calculando con v1** y todo el trabajo de v2 está
   escrito pero inactivo. Es el pendiente que desbloquea el valor de los demás.
2. **Decidir el nombre del fichero en el despliegue:** mantener el sufijo
   `_corrected` o promoverlo a `potencial_stem_predictor_v2.js`. Es decisión de
   operación, no de modelado. Mientras tanto el viejo sigue ahí para revertir.
3. **Actualizar la sección 1 del dashboard:** todavía publica **R² 0.115** para v1.
   Debe pasar a la cifra oficial, **0.084 con su IC 95 % [0.053, 0.116]**,
   declarando que es out-of-fold sobre los 1,748. La edición es del **repo web**;
   este repo ya quedó corregido en el commit `a62581c`.
4. **Revisar si `inscripciones_emergencia` tiene inscritos sin resultado de examen.**
   El export de los 248 salió solo de `inscripciones_copa_stem`; si esa otra tabla
   aporta, faltan personas en esa población.
5. **Revisar la fórmula de resiliencia sin nota.** Cinco valores discretos entre 30
   y 50 para un cuarto del índice es demasiado pobre, y es la mitad de la causa del
   aplanamiento de la distribución sin examen.
6. **Buscar talento oculto en el grupo sin examen con otro método.** El detector
   actual no sirve ahí porque depende del puntaje. Con `indice_condiciones` (que no
   lo usa) más el perfil de cluster se pueden priorizar los 53 en condiciones
   adversas.
7. **Construir la parte práctica del dashboard para docentes y secretarías:** la
   lista de seguimiento de los 248, la lista de talento oculto y el simulador de
   pesos. La pestaña "Estudio ML" ya cubre la parte explicativa; falta la
   accionable. El seguimiento a los 248 es una **intervención de equidad**, no un
   trámite administrativo.
8. **Higiene del repositorio — comprobado este turno, y las dos partes fallan:**
   - El `.gitignore` **sigue en UTF-16** y ninguno de sus patrones está en vigor.
     Convertirlo a UTF-8. (El fichero aparece modificado en el árbol de trabajo
     pero sin commitear y sin cambio de codificación.)
   - Hay **5 CSV bajo `data/` en la historia y en HEAD** con `numero_documento`,
     `nombres`, `apellidos` e `institucion_educativa` de estudiantes menores:
     `copa_stem_dataset.csv`, `_completo`, `_limpio`, `_v3` y el snippet de
     Supabase. **Esto es un asunto de datos personales, no de higiene**, y decidir
     qué hacer (dejar de rastrearlos, reescribir historia, rotar el repo) es una
     decisión que no se toma sola: requiere confirmación explícita antes de actuar.

### Políticas vigentes (no son pendientes; son la regla)

- **`puntaje_estimado`:** nunca como cifra puntual, nunca en la misma columna que
  un puntaje real, nunca para decisiones individuales, nunca para rankear a la
  población sin examen.
- **`ref_rendimiento`** es una decisión de producto declarada, no un subproducto
  del dataset de entrenamiento de cada versión. El script 19 existió porque no lo
  estaba siendo.
- **Población sin examen separada siempre.** El techo de 83.33 obliga a
  presentarla aparte; la marca `tiene_puntaje_real` no basta, la separación tiene
  que ser visual.
- **Toda cifra de R² o MAE se publica con tres datos inseparables:** valor,
  población (n) y partición (in-sample / CV / out-of-fold / hold-out). Las tres
  cifras en disputa del informe 20 habrían sido imposibles bajo esta regla.
- **La métrica debe viajar dentro del artefacto.** v1 nunca guardó su R² en el
  `.joblib` y por eso cada informe la recordó distinta; v2 sí lo hace
  (`metricas_holdout`) y su 0.1766 aparece idéntico en cinco sitios.
