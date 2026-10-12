# Informe: datos de ejemplo, dependencias y verificación

Rama `feat-datos-ejemplo` · 2026-10-12 · sin commit ni push.

---

## 1. Versión de Python

**Python 3.14.0**, informado por el propio intérprete de `.venv` (`python --version`) y
coincidente con `.venv/pyvenv.cfg`. Es la que queda en `.python-version`.

Los 10 paquetes fijados tienen versión instalada para 3.14 y funcionan: los 11 scripts de la
verificación (§5) importan y usan todos los de su cadena sin errores. No se comprobó si
existen ruedas de esas versiones para otros sistemas operativos ni para Python 3.13 o anterior.

## 2. Paquetes fijados (`requirements.txt`)

Imports recogidos con `ast` sobre todos los `.py` de `notebooks/` y `models/`, sin contar la
biblioteca estándar ni los módulos locales cargados con `importlib`. Versiones leídas de `.venv`.

| Paquete | Versión | Nombre de import |
|---|---|---|
| joblib | 1.5.3 | `joblib` |
| lightgbm | 4.6.0 | `lightgbm` |
| matplotlib | 3.11.0 | `matplotlib` |
| numpy | 2.5.0 | `numpy` |
| pandas | 3.0.3 | `pandas` |
| scikit-learn | 1.9.0 | `sklearn` |
| scipy | 1.18.0 | `scipy` |
| seaborn | 0.13.2 | `seaborn` |
| statsmodels | 0.14.6 | `statsmodels` |
| xgboost | 3.3.0 | `xgboost` |

No hizo falta fijar ninguna dependencia indirecta. `openpyxl` y `plotly`, que la sección
"Configuración del entorno" del README manda instalar, **no los importa ningún script** y no
se incluyeron. Esa sección no se modificó.

## 3. Columnas clasificadas como identificativas

Solo se documenta su formato; ningún valor se listó ni se copió:

| Columna | Archivos | Formato |
|---|---|---|
| `numero_documento` | ambos | Cadena casi siempre numérica (99,8 % solo dígitos), de 1 a 15 caracteres; el 97 % tiene 10 dígitos. Ninguno empieza por 99 |
| `nombres` | canónico | Texto de 1 a 6 palabras |
| `apellidos` | canónico | Texto de 1 a 4 palabras |

Tratada como **cuasi-identificador**: `created_at` (marca de inscripción). Se documentan solo su
formato y el rango de fechas, no valores individuales.

No se consideraron identificativas, y sí se listan: `institucion_educativa` y `municipio`
(información pública), y `edad_calculada` (solo agregados).

**No existen en el dataset real** columnas de correo, teléfono ni dirección. Por eso no se
generaron: el esquema debía ser exacto.

## 4. Esquema derivado (solo estructura y agregados)

### 4.1 Hay dos esquemas reales, no uno

| Archivo real | Filas × columnas | Lo leen |
|---|---|---|
| `data/copa_stem_dataset.csv` (canónico) | 1.809 × 27 | Scripts 01 a 07. Con nombres; **sin** perfil académico; 2 municipios |
| `data/copa_stem_dataset_2026-08.csv.csv` | 3.077 × 31 | Scripts 11, 21 y 24. Sin nombres; **con** las 5 columnas de perfil académico y `created_at`; 3 municipios |

El encargo pedía a la vez reproducir el esquema exacto, incluir nombres y reproducir la
nulidad mayoritaria de las 5 columnas de perfil. Ningún archivo real cumple las tres cosas. Por
eso **el generador produce dos archivos**, uno por esquema, cada uno exacto.
`copa_stem_dataset_SINTETICO.csv`, el nombre pedido, corresponde al canónico.

Formato de ambos: UTF-8 sin BOM, fin de línea CRLF. Las listas son arrays JSON compactos
(separador `,` sin espacio, acentos sin escapar), y `"Ninguna"` siempre va sola.

### 4.2 Canónico: `copa_stem_dataset.csv` (1.809 filas)

| # | Columna | Tipo pandas | % nulos | Valores / estadísticas |
|---|---|---|---|---|
| 0 | numero_documento | str | 0,0 | IDENTIFICATIVA. Ver §3 |
| 1 | nombres | str | 0,0 | IDENTIFICATIVA. Ver §3 |
| 2 | apellidos | str | 0,1 | IDENTIFICATIVA. Ver §3 |
| 3 | institucion_educativa | str | 0,0 | Copacabana: Colegio Cooperativo Juan del Corral, Colegio La Asunción, Colegio San Rafael, I. E. Presbítero Bernardo Montoya Giraldo, I.E. Gabriela Mistral, I.E. José Miguel de Restrepo y Puerta, I.E. San Luis Gonzaga. Girardota: Colegio Neosistemas, Colegio Nuestra Señora del Rosario, I.E. Atanasio Girardot, I.E. Emiliano García, Instituto Parroquial Nuestra Señora de la Presentación |
| 4 | grado_escolar | float64 | 5,6 | {9, 10, 11} · media 9,955 · desv. 0,808 |
| 5 | puntaje_obtenido | int64 | 0,0 | 0 a 100 en múltiplos de 5 · media 43,129 · desv. 24,074 |
| 6 | porcentaje | int64 | 0,0 | Idéntico a puntaje_obtenido en todas las filas |
| 7 | tiempo_usado_segundos | float64 | 26,8 | mín. 51 · máx. 5.403 · media 2.181,4 · desv. 1.134,5 |
| 8 | cambios_pestana | int64 | 0,0 | mín. 0 · máx. 20 · media 0,151 · desv. 0,861 |
| 9 | intentos_copiar | int64 | 0,0 | Siempre 0 |
| 10 | intentos_pegar | int64 | 0,0 | {0, 1} · media 0,001 · desv. 0,024 |
| 11 | intentos_click_derecho | int64 | 0,0 | mín. 0 · máx. 149 · media 0,273 · desv. 3,814 |
| 12 | edad_calculada | float64 | 7,2 | mín. −1 · máx. 27 · media 15,165 · desv. 2,447 (incluye errores de captura: −1, 0, 27) |
| 13 | genero | str | 1,4 | Masculino, Femenino, Prefiero no decirlo, No binario |
| 14 | municipio | str | 0,0 | Copacabana, Girardota |
| 15 | tipo_institucion | str | 0,0 | Pública, Privada |
| 16 | estrato | float64 | 7,2 | {1, 2, 3} · media 2,418 · desv. 0,597 |
| 17 | jornada | str | 7,2 | Mañana, Única, Tarde |
| 18 | con_quien_vive | str | 7,2 | Ambos padres, Solo madre, Otros familiares, Solo padre, Otro, Acudiente legal |
| 19 | computador_en_casa | str | 7,2 | Sí, propio · No · Sí, compartido |
| 20 | internet_en_casa | str | 7,2 | Sí, estable · Sí, intermitente · No |
| 21 | participacion_olimpiadas | str | 7,2 | No, Sí |
| 22 | nivel_programacion | str | 7,2 | Ninguna, Básica, Intermedia, Avanzada |
| 23 | nivel_robotica | str | 7,2 | Ninguna, Básica, Intermedia, Avanzada |
| 24 | herramientas_conocidas | str (lista JSON) | 7,2 | 433 combinaciones. Opciones: Ninguna, Roblox Studio, HTML, JavaScript, Python, CSS, Minecraft Education, Arduino, Scratch, micro:bit, C++, LEGO Mindstorms / Spike, Otro |
| 25 | areas_interes | str (lista JSON) | 7,2 | 475 combinaciones. Opciones: Artes, Inteligencia Artificial, Programación, Matemáticas, Cultura, Ingeniería, Ciencias Naturales, Robótica, Otro |
| 26 | interes_prog_robotica | float64 | 7,3 | {1, …, 5} · media 2,846 · desv. 1,134 |

**Patrón de nulos:**
- Un 7,2 % de las filas no tiene inscripción enlazada. En ellas son nulas a la vez las columnas 12 y 16 a 26, y también `tiempo_usado_segundos`. Dentro de ese grupo, la mitad tampoco tiene grado y una quinta parte tampoco tiene género.
- Un 2 % adicional solo carece de grado.
- La telemetría falta en los exámenes escritos: 29 % en Copacabana y 20 % en Girardota.

### 4.3 Export 2026-08: `copa_stem_dataset_2026-08.csv.csv` (3.077 filas)

Mismas columnas que el canónico, sin `nombres` ni `apellidos`, y con 5 de perfil y `created_at`
al final. Solo se indica lo que cambia:

| # | Columna | Tipo pandas | % nulos | Valores / estadísticas |
|---|---|---|---|---|
| 1 | institucion_educativa | str | 0,0 | 33 colegios (30 o más: no se listan aquí; los nombres están en el generador) |
| 2 | grado_escolar | int64 | 0,0 | {9, 10, 11} · media 9,980 · desv. 0,815 |
| 3–4 | puntaje_obtenido / porcentaje | int64 | 0,0 | 0 a 100 en múltiplos de 5 · media 41,724 · desv. 22,677 |
| 5 | tiempo_usado_segundos | float64 | 16,6 | mín. 51 · máx. 4.876.349 (valor atípico; el percentil 99 es 5.385) · media 21.295 · desv. 275.245. En Bello no falta nunca |
| 9 | intentos_click_derecho | int64 | 0,0 | mín. 0 · máx. 149 · media 0,377 · desv. 3,365 |
| 10 | edad_calculada | int64 | 0,0 | mín. −86 · máx. 27 · media 15,107 · desv. 3,218 (incluye errores de captura) |
| 12 | municipio | str | 0,0 | Copacabana, Girardota, Bello |
| 14 | estrato | int64 | 0,0 | {1, 2, 3} · media 2,384 · desv. 0,624 |
| 15–23 | socioeconómicas y listas | str | 0,0 | Mismos valores que el canónico |
| 24 | interes_prog_robotica | float64 | 0,0 | {1, …, 5} · media 2,851 · desv. 1,148 |
| 25 | promedio_academico | float64 | 62,7 | mín. 0 · máx. 38 (atípico; el percentil 99 es 4,9) · media 3,836 · desv. 1,658 |
| 26 | horas_estudio_matematicas | float64 | 62,6 | mín. 0 · máx. 30 · media 1,470 · desv. 1,916 |
| 27 | motivacion_participar | float64 | 61,8 | {1, …, 5} · media 2,787 · desv. 1,140 |
| 28 | clases_extra_matematicas | str | 64,0 | No, Sí |
| 29 | gusto_logica | float64 | 61,7 | {1, …, 5} · media 2,949 · desv. 1,053 |
| 30 | created_at | str | 0,0 | CUASI-IDENTIFICADOR. Formato `AAAA-MM-DD hh:mm:ss.ffffff+00`, entre el 2026-05-22 y el 2026-07-30 |

**Patrón de nulos del perfil:**
- Las 5 columnas son nulas a la vez en el 62 % de las filas: en el 100 % de Copacabana, el 62 % de Girardota y el 2 % de Bello.
- Entre quienes sí tienen perfil hay huecos sueltos, sobre todo en `clases_extra_matematicas` (5 %).

## 5. Generador y verificación

`notebooks/00_generar_datos_ejemplo.py`:
- No lee `data/` y usa la semilla 42.
- Escribe dos CSV de 300 filas: 79 KB y 84 KB.
- Imprime el aviso de datos inventados al empezar y al terminar.
- Dos ejecuciones seguidas producen archivos idénticos (mismo MD5).
- Comprobación cruzada, hecha solo en memoria e informando solo conteos: **0 coincidencias** de `numero_documento`, `nombres` o `apellidos` con cualquier archivo de `data/`.

Verificación en una copia temporal fuera del repositorio (carpeta temporal del sistema):
- La copia tenía `notebooks/`, `models/`, `reports/` y `outputs/`, pero **ningún archivo real de `data/`**. Solo los dos sintéticos, con los nombres que esperan los scripts.
- Se usó el intérprete de `.venv`.

| Script | Resultado |
|---|---|
| 01_analisis_exploratorio | **Completó** (código de salida 0). 10 figuras e informe |
| 03_modelo_predictivo | **Completó** (0). Entrena los modelos, exporta el predictor puro con máx\|Δ\| = 1,4e-14 y puntúa a 300 estudiantes |
| 02, 04, 05, 05b, 05c, 06, 07 | Completaron (0) |
| 11_preparar_experimento_reentrenamiento | Completó (0). Leyó el export 2026-08 sintético y construyó A, B, C y C′; C tiene 104 filas con perfil |
| 12_experimento_reentrenamiento | Completó (0) |

Ningún log contiene trazas de error ni avisos de sección omitida por falta de columnas. No hubo
que corregir el generador. La copia temporal se borró al terminar.

**El repositorio real no se tocó:**
- MD5 de los 163 archivos de `outputs/`, `models/` y `reports/`: idénticos antes y después.
- `git status`: idéntico antes y después.
- `data/`: ningún archivo modificado.

## 6. Lo que no se hizo, falló o difiere

- **No se hizo commit.** El paso 8 pedía hacer commit del CSV, pero la instrucción final decía no hacer commit ni push. Prevaleció la final.
- **Tipo de `numero_documento`:** pandas lo infiere como `str` en el real y como `int64` en el sintético. En el real, el `str` viene de unos pocos documentos no numéricos (0,2 %, datos sucios). En el sintético todos son numéricos, como se pidió ("un rango que no pueda coincidir"). Los scripts lo convierten a `str` al cargar, así que no afecta a la ejecución.
- **Valores atípicos no reproducidos:** el tiempo de 4,9 millones de segundos, el promedio de 38, las horas de estudio de 30 y las edades −86 y −6. Sí se reproducen algunas edades erróneas (−1, 0 y 27) y la telemetría rara (pegar, clic derecho, cambios de pestaña).
- **Con 300 filas, A y B del script 11 salen iguales:** el criterio de cohorte toma las 300 inscripciones más antiguas, es decir, todas. No afecta a la verificación, pero el contraste A vs B no significa nada con estos datos (ninguno lo hace).
- **Scripts no probados:** 08, 09, 09b, 10, 13 a 19 y 21 a 25. Varios dependen de artefactos de despliegue, de `inscritos_copa_stem.csv` o de otros archivos que no tienen versión sintética.
- **Aviso por consola:** si la salida se redirige a un archivo o a una tubería en Windows, los acentos del aviso pueden verse corruptos, porque la codificación de esa salida no es UTF-8. No se comprobó en una consola interactiva.
