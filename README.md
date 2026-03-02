# Caso 5: Edificios Inteligentes y Eficiencia Energética

## Contexto del Negocio

En la construcción de edificios sostenibles (como los de certificación LEED), **saber cuánta energía gastarán es la clave del éxito**. No se trata solo de ecología, sino de estrategia:

- **Ahorro:** Menos gasto en recibos de luz y gas a largo plazo.
- **Equipos:** Saber exactamente qué tamaño de aire acondicionado o calefacción comprar.
- **Normas:** Cumplir con las leyes ambientales y reducir la contaminación.

La energía que necesita un edificio para estar a una temperatura agradable depende de su **forma y diseño**: altura, cantidad de ventanas, tamaño del techo, orientación, etc. Predecir esto **antes de construir** permite elegir el mejor diseño y ahorrar dinero.

---

## Objetivos

### Tarea 1 — Regresión
Predecir el valor exacto de **Heating Load (`Y1`)**: cuánta energía consumirá el edificio para mantenerse caliente.

### Tarea 2 — Clasificación
Clasificar cada diseño como **Eficiente (1)** o **No Eficiente (0)** usando como umbral la mediana de `Heating_Load` (18.95 kWh/m²).

---

## Dataset

- **Fuente:** [UCI Machine Learning Repository – Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
- **Archivo:** `ENB2012_data.xlsx`
- **Forma:** 768 filas × 10 columnas
- **Valores nulos:** ninguno

### Variables

| Variable original | Nombre renombrado | Tipo | Descripción |
|---|---|---|---|
| X1 | `Relative_Compactness` | Numérica | Compacidad relativa del edificio |
| X2 | `Surface_Area` | Numérica | Área de superficie (m²) |
| X3 | `Wall_Area` | Numérica | Área de paredes (m²) |
| X4 | `Roof_Area` | Numérica | Área del techo (m²) |
| X5 | `Overall_Height` | Numérica | Altura total (3.5 = 1 piso, 7.0 = 2 pisos) |
| X6 | `Orientation` | Categórica | Orientación: Norte / Este / Sur / Oeste |
| X7 | `Glazing_Area` | Numérica | Ratio de acristalamiento (0–0.4) |
| X8 | `Glazing_Area_Distribution` | Categórica | Distribución del acristalamiento |
| Y1 | `Heating_Load` | Numérica | **Target regresión**: carga de calefacción |
| Y2 | `Cooling_Load` | Numérica | Carga de refrigeración (no usada como target) |
| — | `Eficiente` | Binaria | **Target clasificación**: 1 si Heating_Load ≤ 18.95, 0 si no |

---

## Análisis Exploratorio (EDA)

### Hallazgos clave

- **Sin valores nulos** ni outliers en ninguna variable.
- **Sin sesgos significativos** en las variables numéricas (valor máximo: 0.53).
- Las variables categóricas (`Orientation`, `Glazing_Area_Distribution`) tienen clases balanceadas.
- **Variable objetivo balanceada:** clases 50/50 tras aplicar el umbral de la mediana → no se requiere SMOTE ni `class_weight`.
- **Diferencias de escala** entre variables numéricas → se aplica `StandardScaler`.

### Multicolinealidad detectada

| Par de variables | Correlación | Acción tomada |
|---|---|---|
| `Relative_Compactness` ↔ `Surface_Area` | > 0.8 | Se elimina `Relative_Compactness` |
| `Roof_Area` ↔ `Overall_Height` | > 0.8 | Se elimina `Roof_Area` |

---

## Pipeline de Preprocesamiento

```
ColumnTransformer
├── StandardScaler        → Surface_Area, Wall_Area, Overall_Height, Glazing_Area
└── OneHotEncoder(drop='first') → Orientation, Glazing_Area_Distribution
```

---

## División de Datos

| Conjunto | Tamaño |
|---|---|
| Entrenamiento | 614 filas (80%) |
| Test | 154 filas (20%) |

Semilla: `random_state = 42`

---

## Modelos

### Regresión

| Modelo | Parámetros |
|---|---|
| `LinearRegression` | — |
| `Ridge (L2)` | `alpha=1.0` |
| `Lasso (L1)` | `alpha=1.0` |

**Métricas de evaluación:** RMSE, MAE, R²

### Clasificación

| Modelo | Parámetros |
|---|---|
| `LogisticRegression` (sin regularización) | `penalty=None`, `solver='lbfgs'` |
| `LogisticRegression Ridge (L2)` | `penalty='l2'`, `C=1.0`, `solver='lbfgs'` |
| `LogisticRegression Lasso (L1)` | `penalty='l1'`, `C=1.0`, `solver='liblinear'` |

**Métricas de evaluación:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## Conclusiones de Clasificación

Los tres modelos de Regresión Logística superan ampliamente la clasificación aleatoria (AUC >> 0.5). La regularización **Ridge (L2)** ofrece el mejor equilibrio entre Precision y Recall. El **Lasso (L1)** actúa como selector automático de variables al llevar coeficientes poco informativos a cero.

---

## Requisitos del Entorno

| Requisito | Mínimo |
|---|---|
| Python | 3.12.12 |
| RAM | 12.67 GB |
| CPU | 2 núcleos |
| SO | Linux (recomendado) |

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
Caso-5-Green-Building-Energy-Efficiency/
├── notebooks/
│   ├── Caso5_linear_regression_logistic_regression.ipynb
│   └── datasets/
│       └── energyEfficiency_data/
│           └── ENB2012_data.xlsx
├── requirements.txt
└── README.md
```
