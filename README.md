# Retail Banking Propensity Model

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-Supervised%20Classification-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Predicción de la propensión de un cliente a **suscribir un depósito a plazo**,
> construida sobre el dataset de campañas de marketing bancario (UCI Bank Marketing).

Sistema de ML supervisado que estima la probabilidad de que un cliente de banca
retail contrate un producto (depósito a plazo) durante una campaña de marketing,
a partir de características demográficas y de interacción con la campaña.

---

## Descripción del problema

La banca retail ejecuta campañas de marketing sobre su base de clientes. Llamar
a todos es caro; llamar a los clientes equivocados erosiona la relación y el
presupuesto. El objetivo es **priorizar** — ordenar la base de clientes por
probabilidad de contratación para que el equipo de campaña se centre en los que
tienen más propensión.

El dataset es el clásico **UCI Bank Marketing** (41.188 registros de campañas de
telemarketing de un banco portugués, 20 features). La variable objetivo `y`
indica si el cliente suscribió un depósito a plazo (`1`) o no (`0`). El dataset
está **desbalanceado**: 88.7% de clientes no suscriben frente a 11.3% que sí.

Este es un problema de **propensión a producto / cross-sell**, no de riesgo de
crédito. La métrica que importa en producción es la capacidad de **priorizar la
clase positiva** (los que van a suscribir): por eso la evaluación se centra en
la curva precision-recall, no en accuracy.

---

## Enfoque de modelado

| Decisión | Justificación |
|---|---|
| Random Forest con `class_weight='balanced'` | Modelo interpretable, robusto al desbalanceo sin sobreajustar |
| Split 80/20 **estratificado** y **antes** de ajustar transformadores | Ajustar el scaler/encoders con todo el dataset filtraría información del test set (leakage) |
| ROC-AUC + PR-AUC como métricas principales | PR-AUC mide la capacidad de priorizar la clase minoritaria (la que importa en una campaña) |
| Artefactos (encoders, scaler, columnas) persistidos y reutilizados en inferencia | La predicción en producción aplica exactamente las mismas transformaciones que el entrenamiento |

## Resultados (Random Forest, test set)

| Métrica | Valor |
|---|---|
| ROC-AUC | **0.948** |
| PR-AUC | **0.651** |
| Accuracy | 0.87 |
| Recall (clase positiva) | 0.92 |
| Precision (clase positiva) | 0.46 |

La combinación de **recall alto (0.92)** y **PR-AUC 0.65** frente a una base del
11.3% indica que el modelo captura la gran mayoría de los clientes que sí van a
suscribir, a costa de falsos positivos — el trade-off correcto para priorizar
una lista de llamadas en la que el coste de una llamada errónea es bajo.

Las métricas se persisten automáticamente en `reports/metrics.json` al entrenar.

---

## Estructura del proyecto

```
credito/
├── data/
│   └── raw/            ← credit-train.csv (UCI Bank Marketing)
├── credito/
│   ├── data/           make_dataset.py — carga de datos
│   ├── features/       build_features.py — preprocessing + split sin leakage
│   ├── models/         train_model.py · predict_model.py (métricas persistidas)
│   └── utils/          paths.py — rutas centralizadas
├── models/             RandomForest.joblib + artifacts/ (encoders, scaler, columns)
├── reports/            metrics.json (resultados reales)
├── notebooks/          análisis exploratorio y comparativa de algoritmos
└── main.py             orquestador + interfaz interactiva de predicción
```

## Uso

```bash
# 1. Coloca credit-train.csv en data/raw/
# 2. Ejecuta (entrena si no hay modelo, luego pide datos del cliente)
python main.py
```

El pipeline: carga → preprocesado (split primero, escalado/encoding solo con
train) → entrenamiento → evaluación con métricas persistidas → inferencia
interactiva con las mismas transformaciones del entrenamiento.

## Limitaciones

- El dataset es de una campaña de telemarketing de un banco concreto; los
  coeficientes de features como `euribor3m` o `nr_employed` reflejan ese
  contexto económico y no generalizan a otros mercados sin reentrenar.
- Las variables de interacción (`duration`, `campaign`, `pdays`) se miden
  durante la llamada, así que el modelo es adecuado para **priorizar y
  segmentar**, no para predecir el resultado antes de hacer la llamada.
- `class_weight='balanced'` eleva el recall de la clase positiva a costa de
  precisión; el umbral de decisión final es una decisión de negocio que depende
  del coste de una llamada errónea.

## Autor

Alejandro Cancelas Chapela
