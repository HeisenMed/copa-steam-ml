# SESIÓN ACTUAL — Copa STEM 2026

Fundación SapienceLab · Última actualización: 2026-09-02

Este archivo se reescribe completo en cada actualización. No se acumula historial.

---

## 1. Qué se hizo en este turno

Se creó y se **corrió** el script 19, que regenera el artefacto de despliegue v2
con la referencia corregida de 3,072. Era el pendiente número 1 y el que
bloqueaba todo lo demás. Queda cerrado.

El `.js` que consume la Edge Function llevaba embebida la referencia vieja de
1,148 que el script 17 existió para eliminar. La corrección del 17 se había
quedado en el lado Python/CSV y nunca llegó al artefacto: desplegarlo tal cual
habría reintroducido el problema por la puerta de atrás. Ya no.

No se sobrescribió el `.js` viejo. No se modificó ningún script numerado
anterior, ningún `.joblib`, ningún otro artefacto de `models/deploy/`, ni
Supabase, ni la Edge Function.

### Artefactos

* Archivo                                             * Estado    * Detalle
* notebooks/19_regenerar_deploy_v2.py                 * Terminado * Corrido y verificado
* models/deploy/potencial_stem_predictor_v2_corrected.js * Nuevo   * 828 KB, ref n=3,072
* outputs/F19_verificacion_deploy_v2.json             * Completo  * Traza de verificación
* reports/19_regenerar_deploy_v2.md                   * Completo  * Con los números reales
* models/deploy/potencial_stem_predictor_v2.js        * INTACTO   * Mismo hash, para rollback

### La referencia: antes y después

* Referencia          * n     * Media * σ       * Cohorte
* Embebida hasta hoy  * 1,148 * 41.08 * 20.5332 * dataset_C_perfil.csv (entrenamiento)
* Corregida           * 3,072 * 41.74 * 22.6583 * dataset_B_completo.csv (examinados)

La corregida se tomó literalmente de `outputs/F17_ref_rendimiento_corregido.json`.

### Qué cambia dentro del `.js`

De las 6 claves del `SPEC` solo cambian 2: `ref_rendimiento` y `meta`. El
script aborta si detectara cualquier otra. Quedan idénticos `puntaje.model`
(los 200 árboles), `puntaje.preprocess` (medianas, modas, one-hot),
`engagement`, `pesos` y `categorias`.

El diff contra el artefacto vigente es de 2 líneas quitadas y 6 puestas sobre
208 → 212: la línea de la constante `SPEC`, la línea `GENERADO por…` y cuatro
comentarios que dejan escrito dentro del propio fichero que la referencia es la
corregida. El resto del cuerpo JS queda byte a byte igual.

### Verificación de precisión

Muestra de 300 filas: 200 con perfil académico (ruta v2 normal) + 100 sin perfil
(ejercitan la imputación por mediana/moda), `random_state=42`. Las mismas filas
normalizadas se le pasan a Python y a Node.

* Comparación                                       * Máx \|Δ\|
* Intérprete Python vs sklearn.predict              * 2.842e-14
* `_predictPuntaje` del .js generado vs sklearn     * **2.842e-14**
* Índice compuesto: .js vs predictor Python         * 1.00e-02

La segunda es la que importa: **misma precisión (~1e-14) que la verificación
original del script 14**. Node v24.13.0, importando el fichero generado intacto
y —en copia temporal con una línea `export`— a precisión completa.

La diferencia de 0.01 en 31 de las 300 filas es **modo de redondeo, no cálculo**:
Python `round()` redondea al par y `Math.round()` del JS la mitad hacia arriba.
Aplicando el criterio del JS a los valores sin redondear de Python, la
diferencia máxima cae a 0.000e+00 en las 300 filas. Las categorías coincidieron
en las 300. Es preexistente y afecta igual al artefacto vigente; se anota, no se
corrige, porque corregirla obligaría a tocar el cuerpo del JS.

### Comprobación de procedencia

Los árboles y el preprocesamiento extraídos de `mejor_modelo_puntaje_v2.joblib`
se compararon bit a bit contra los embebidos en el `.js` vigente: ambos `True`.
El `.js` viene de ese `.joblib`, así que regenerarlo desde ahí es legítimo.

### Que no se movió nada más

SHA-256 de los 21 ficheros de `models/` y `models/deploy/` tomados antes y
después: **ninguno cambió**. El único fichero nuevo en `models/deploy/` es el
corregido. El viejo conserva hash y fecha (Aug 30 00:25).

### El efecto, visible en el propio artefacto

Ejecutando los dos ficheros con `node` sobre el estudiante de demostración que
el `.js` trae al final:

* Salida                   * Vigente (ref 1,148) * Corregido (ref 3,072)
* indice_potencial         * 44.66               * 44.99
* componente_rendimiento   * 49.04               * 49.48
* componente_engagement    * 31.53               * 31.53
* componente_resiliencia   * 49.04               * 49.48
* categoria                * En desarrollo       * En desarrollo

El engagement no se mueve —no pasa por el percentil— y el rendimiento sí. Es
exactamente la firma del cambio que se buscaba.

---

## 2. Dónde está cada cosa

* Repo ML: `Copa STEAM/ml-models` (rama main, scripts 01-19)
* Repo web: `Recursos Web/sapiencex` (repo git independiente, tiene `src/`, `supabase/`, `docs/sql/`)
* Modelo v2: R2=0.1766, MAE=15.00, `models/mejor_modelo_puntaje_v2.joblib`
* Desplegable v2 CORREGIDO: `models/deploy/potencial_stem_predictor_v2_corrected.js` (828 KB, ref n=3,072)
* Desplegable v2 viejo: `models/deploy/potencial_stem_predictor_v2.js` (816 KB, ref n=1,148, NO desplegar)
* Referencia corregida: `outputs/F17_ref_rendimiento_corregido.json` (n=3,072, σ=22.6583)
* Traza de verificación: `outputs/F19_verificacion_deploy_v2.json`

---

## 3. Pendientes en orden

1. Actualizar la Edge Function `index.ts` para que su `SELECT` lea
   `promedio_academico`, `horas_estudio_matematicas`, `motivacion_participar`,
   `clases_extra_matematicas` y `gusto_logica`. Hoy `grep promedio_academico`
   sobre `index.ts` sigue dando cero. **Sin esto el v2 recibe las medianas del
   SPEC en lugar de las respuestas reales y se comporta como un v1 caro**, así
   que desplegar el artefacto corregido antes de este paso no aporta nada.
2. Decidir el nombre del fichero en el despliegue: mantener el sufijo
   `_corrected` o promoverlo a `potencial_stem_predictor_v2.js`. Es decisión de
   operación, no de modelado. Mientras tanto el viejo sigue ahí para revertir.
3. Subir `ml_scores_v2_corrected.csv` a una tabla NUEVA `ml_scores_v2` en
   Supabase. No sobrescribir `ml_scores`; volcarla a CSV antes de tocar nada.
   `ml_scores_sin_examen.csv` va aparte o con marca de población explícita.
4. Construir el dashboard académico en el repo web.

Pendientes heredados del script 18 (población sin examen):

- **Decidir cómo se presenta la población sin examen en el dashboard.** El techo
  de 83.33 obliga a separarla; mostrarla junto a los examinados sería engañoso.
  La marca `tiene_puntaje_real` no basta: hay que separarlas visualmente.
- **Revisar la fórmula de resiliencia sin nota.** Cinco valores discretos entre
  30 y 50 para un cuarto del índice es demasiado pobre, y es la mitad de la
  causa del aplanamiento.
- **Buscar talento oculto en ese grupo con otro método.** El detector actual no
  sirve ahí. Con `indice_condiciones` (que no depende del puntaje) más el perfil
  de cluster se pueden priorizar los 53 en condiciones adversas.
- **Tratar el seguimiento a los 248 como intervención de equidad**, no como
  trámite administrativo.
- **Revisar si `inscripciones_emergencia` tiene inscritos sin resultado.** El
  export salió solo de `inscripciones_copa_stem`; si esa otra tabla aporta,
  faltan en los 248.

Pendientes menores heredados:

- Añadir columna `modelo_version` a `ml_scores` antes de cualquier despliegue híbrido.
- Política de `puntaje_estimado`: nunca como cifra puntual, nunca en la misma
  columna que un puntaje real, nunca para decisiones individuales, nunca para
  rankear a la población sin examen.
- Tratar `ref_rendimiento` como decisión de producto declarada, no como
  subproducto del dataset de entrenamiento de cada versión. El script 19 es la
  consecuencia de que no lo estuviera.
- El `.gitignore` está guardado en UTF-16, así que git no lo interpreta y sus
  patrones (`.venv/`, `*.joblib`, `__pycache__/`) no se aplican. Por eso
  aparecen `.pyc` y el `.joblib` en `git status`. Detectado, no tocado.
