---
name: copa-stem-ml
description: Use this skill for ANY machine learning, data science, statistical analysis, or data exploration task related to Copa STEM or Fundación SapienceLab data. Triggers on mentions of ML, machine learning, modelo, análisis de datos, predicción, clustering, puntaje, rendimiento estudiantil, dataset, pandas, sklearn, visualización de datos, exploración, correlación, regresión, clasificación, or any reference to analyzing student performance, educational outcomes, or STEM competition data. Also use when the user asks to create reports, charts, or insights from Copa STEM data.
---

# Copa STEM — Machine Learning & Data Science Skill

## Contexto del Proyecto

Copa STEM es una olimpiada de matemáticas y lógica para estudiantes de grados 9°, 10° y 11° en Copacabana y Girardota (Antioquia, Colombia), organizada por Fundación SapienceLab. El dataset contiene ~2000 estudiantes con datos socioeconómicos, demográficos y resultados del examen.

## Estructura del Proyecto

```
C:\Users\USUARIO\OneDrive\Desktop\Proyecto\SapienceLab\Copa STEM\ml-models/
  data/               ← CSVs exportados de Supabase
  notebooks/          ← Scripts numerados (01_, 02_, etc.)
  models/             ← Modelos entrenados (.joblib)
  outputs/            ← Gráficos PNG y tablas
  reports/            ← Informes en Markdown
  README.md           ← Descripción general
```

## Dataset Principal: copa_stem_dataset.csv

### Variables disponibles

**Demográficas:**
- `numero_documento` — ID del estudiante (PK)
- `edad_calculada` — Edad en años
- `genero` — Masculino / Femenino / Prefiero no decirlo
- `municipio` — Copacabana / Girardota
- `grado_escolar` — 9, 10, 11
- `tipo_institucion` — Pública / Privada

**Socioeconómicas:**
- `estrato` — 1 a 6 (estratificación socioeconómica colombiana)
- `jornada` — Mañana / Tarde / Única
- `con_quien_vive` — Ambos padres / Solo madre / Solo padre / Abuelos / Otro
- `computador_en_casa` — Sí / No
- `internet_en_casa` — Sí / No

**Experiencia previa:**
- `participacion_olimpiadas` — Sí / No (ha participado antes)
- `nivel_programacion` — Ninguno / Básico / Intermedio / Avanzado
- `nivel_robotica` — Ninguno / Básico / Intermedio / Avanzado
- `herramientas_conocidas` — Texto libre (Scratch, Python, Arduino, etc.)
- `areas_interes` — Texto libre (Programación, Robótica, Matemáticas, etc.)
- `interes_prog_robotica` — Nivel de interés en programación/robótica

**Resultado del examen:**
- `puntaje_obtenido` — 0 a 100 (variable objetivo principal)
- `porcentaje` — Igual a puntaje_obtenido (escala 0-100)

**Telemetría (comportamiento durante el examen en plataforma):**
- `tiempo_usado_segundos` — Duración del examen
- `cambios_pestana` — Veces que cambió de pestaña (posible trampa)
- `intentos_copiar` — Intentos de copiar texto
- `intentos_pegar` — Intentos de pegar texto
- `intentos_click_derecho` — Intentos de clic derecho

**NOTA:** Telemetría solo disponible para exámenes en plataforma. Los exámenes escritos tienen estos campos en NULL.

## Reglas de Desarrollo

### Estilo de código
- Python 3.11+
- Scripts numerados: `01_`, `02_`, `03_`...
- Cada script es autocontenido (se puede correr independientemente)
- `random_state=42` siempre para reproducibilidad
- Comentarios en español explicando decisiones metodológicas
- Imprimir progreso en consola: `print(">>> Paso N: Descripción...")`

### Limpieza de datos (aplicar siempre)
```python
# Eliminar registros de prueba
df = df[~df['numero_documento'].isin(['1234', '123456', '123456789', '1234567899'])]

# Separar presentaron vs no presentaron
df_presentaron = df[df['puntaje_obtenido'].notna()].copy()
df_pendientes = df[df['puntaje_obtenido'].isna()].copy()

# Convertir tipos
df['puntaje_obtenido'] = pd.to_numeric(df['puntaje_obtenido'], errors='coerce')
df['tiempo_usado_segundos'] = pd.to_numeric(df['tiempo_usado_segundos'], errors='coerce')
df['edad_calculada'] = pd.to_numeric(df['edad_calculada'], errors='coerce')
```

### Visualizaciones
- Guardar en `outputs/` como PNG (dpi=150)
- Paleta de colores Copa STEM:
  ```python
  COLORS = {
      'cyan': '#00d4ff',
      'violet': '#8b5cf6', 
      'amber': '#f59e0b',
      'dark': '#050816',
      'green': '#10b981',
      'red': '#ef4444',
      'blue': '#0f77ee',
  }
  PALETTE = ['#00d4ff', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444', '#0f77ee']
  ```
- Fondo blanco para gráficos (imprimir bien)
- Títulos claros en español
- Incluir N (tamaño de muestra) en títulos cuando aplique

### Informes (reports/)
Cada análisis genera un informe markdown en `reports/` con:
1. **Título y fecha**
2. **Resumen ejecutivo** (5-6 líneas con hallazgos clave)
3. **Metodología** (qué técnicas se usaron y por qué)
4. **Hallazgos** (cada uno con: pregunta → método → resultado → interpretación → gráfico)
5. **Conclusiones y recomendaciones**
6. **Limitaciones del estudio**
7. **Referencias técnicas**

### Modelos ML
- Guardar en `models/` con joblib: `joblib.dump(model, 'models/nombre_modelo.joblib')`
- Siempre evaluar con cross-validation (5-fold mínimo)
- Reportar métricas relevantes: accuracy, F1, RMSE, R², MAE según el caso
- Documentar hiperparámetros y justificar elecciones
- Feature importance siempre que el modelo lo soporte

## Fases del Proyecto ML

### Fase 1 — Análisis Exploratorio y Diagnóstico
- `01_analisis_exploratorio.py` — EDA completo
- `02_analisis_brechas.py` — Brechas socioeconómicas y de género

### Fase 2 — Modelos Predictivos
- `03_prediccion_puntaje.py` — Regresión: predecir puntaje
- `04_indice_potencial_stem.py` — Score 0-100 de potencial STEM
- `05_talento_oculto.py` — Clasificar talento oculto (alto potencial + bajo acceso)

### Fase 3 — Segmentación y Análisis Avanzado
- `06_clustering_estudiantes.py` — K-Means / DBSCAN para perfiles
- `07_deteccion_anomalias.py` — Detección de posible trampa (telemetría)
- `08_vulnerabilidad_tecnologica.py` — Índice de vulnerabilidad 0-100

### Fase 4 — Modelos de Impacto
- `09_factor_analisis.py` — ¿Qué factores pesan más en el rendimiento?
- `10_recomendaciones.py` — Sistema de recomendación de intervención por estudiante

## Dependencias
```
pandas numpy scikit-learn xgboost lightgbm 
matplotlib seaborn openpyxl plotly scipy 
statsmodels joblib
```

## Entorno
```bash
cd "C:\Users\USUARIO\OneDrive\Desktop\Proyecto\SapienceLab\Copa STEM\ml-models"
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn openpyxl plotly scipy statsmodels joblib
```
