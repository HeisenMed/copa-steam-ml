# Datos de ejemplo (SINTETICOS)

ESTOS DATOS SON INVENTADOS. No describen a ningun estudiante real. Cualquier
resultado que se obtenga con ellos (metricas, graficos, informes, listas de
talento oculto, clusters) carece de significado.


## Por que existen

El dataset real de Copa STEM no se publica porque contiene datos personales de
menores de edad. Sin el, nadie fuera del equipo podria comprobar que el codigo
de este repositorio corre. Estos archivos tienen el mismo esquema que el real
(mismos nombres de columna, mismo orden, mismos valores categoricos, rangos
numericos plausibles y el mismo patron de valores faltantes) para que los
scripts los lean igual que leerian los datos reales.


## Que contiene

copa_stem_dataset_SINTETICO.csv
  300 filas y 27 columnas. Mismo esquema que data/copa_stem_dataset.csv, el
  archivo que leen los scripts 01 a 07. Ese export no tiene columnas de perfil
  academico. Municipios: Copacabana y Girardota.

copa_stem_dataset_2026-08_SINTETICO.csv
  300 filas y 31 columnas. Mismo esquema que data/copa_stem_dataset_2026-08.csv.csv,
  el export que leen los scripts 11, 21 y 24. Incluye las 5 columnas de perfil
  academico (promedio_academico, horas_estudio_matematicas,
  motivacion_participar, clases_extra_matematicas, gusto_logica), vacias en
  cerca de dos tercios de las filas como en el real: siempre en Copacabana,
  en la mayoria de Girardota y casi nunca en Bello.

Las columnas que identifican a una persona son falsas a proposito:
  nombres y apellidos salen de una lista corta inventada (Alfa, Beta, Gamma;
  Ejemplo, Ficticio, Sintetico...); numero_documento es un numero de 10 digitos
  que empieza por 99, un rango en el que no cae ningun documento real.
Los nombres de colegios y municipios son informacion publica y si son los
reales.

Las relaciones entre variables son inventadas y simples. No reproducen las del
dataset real.


## Como regenerarlos

Desde la raiz del repositorio, con el entorno virtual activo:

    python notebooks\00_generar_datos_ejemplo.py

El script no lee nada de data/: el esquema esta escrito dentro de el. Usa una
semilla fija (42), asi que produce exactamente los mismos archivos cada vez.


## Como usarlos para correr el pipeline

Hagalo en un clon aparte del repositorio. Los scripts sobrescriben archivos
versionados de outputs/, models/ y reports/, y esos resultados no deben subirse.

1. Cree la carpeta data/ en la raiz del clon.
2. Copie los archivos con los nombres que esperan los scripts:

    copy datos_ejemplo\copa_stem_dataset_SINTETICO.csv data\copa_stem_dataset.csv
    copy datos_ejemplo\copa_stem_dataset_2026-08_SINTETICO.csv data\copa_stem_dataset_2026-08.csv.csv

3. Ejecute los scripts en orden, por ejemplo:

    python notebooks\01_analisis_exploratorio.py
    python notebooks\03_modelo_predictivo.py

Verificado el 2026-10-12 con Python 3.14.0: los scripts 01, 02, 03, 04, 05,
05b, 05c, 06, 07, 11 y 12 terminan sin error con estos datos. Los demas no se
probaron.


## Recordatorio

Con 300 filas inventadas, los modelos dan R2 cercano a cero o negativo y las
pruebas estadisticas salen significativas o no por azar. Nada de eso dice algo
sobre los estudiantes de Copa STEM.
