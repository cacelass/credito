# Proyecto de Clasificacion de Credito Bancario

## Descripcion General
Este proyecto implementa un sistema de Machine Learning enfocado en el Aprendizaje Supervisado. El objetivo principal es predecir la probabilidad de que un cliente bancario suscriba un credito basandose en caracteristicas demograficas y datos historicos de campañas de marketing.

El sistema transforma datos crudos en predicciones accionables mediante un pipeline que abarca desde la ingestion de datos hasta una interfaz de inferencia para el usuario final.

## Origen del Desarrollo y Logica
El nucleo analitico de este proyecto nace de una fase de experimentacion detallada. Toda la logica de analisis exploratorio de datos (EDA), la comparativa de algoritmos y las decisiones de preprocesamiento estan documentadas paso a paso en el siguiente archivo:

* Ubicacion: notebooks/Entrega-.ipynb

Este cuaderno contiene la justificacion matematica y visual de las decisiones tomadas, mientras que los archivos .py de este repositorio representan la version refactorizada y optimizada para un entorno de produccion.

## Contexto: Aprendizaje Supervisado
Este repositorio sirve como una exploracion practica de los conceptos fundamentales del aprendizaje supervisado:

1.  Entrenamiento con Etiquetas: El algoritmo aprende a partir de un conjunto de datos historico (credit-train.csv) donde la variable objetivo ('y') es conocida.
2.  Generalizacion: El sistema busca patrones en los datos de entrenamiento para aplicar ese conocimiento a casos nuevos no vistos anteriormente.
3.  Clasificacion Binaria: El problema se modela para distinguir entre dos clases exclusivas: aprobacion o denegacion del credito.

## Estructura del Proyecto
El codigo esta organizado de forma modular siguiendo estandares de ciencia de datos:

* credito/data: Scripts para la carga y lectura de datasets.
* credito/features: Modulos para el preprocesamiento (LabelEncoding, Scaling).
* credito/models: Logica de entrenamiento, persistencia de modelos y evaluacion.
* credito/utils: Configuraciones globales y gestion de rutas.
* main.py: Orquestador principal e interfaz de linea de comandos.
* data/raw: Directorio de entrada para los archivos CSV.
* models/: Almacenamiento de modelos entrenados y artefactos de traduccion.

## Requisitos e Instalacion
Para ejecutar este proyecto, es necesario disponer de Python (version 3.10 o superior) y las librerias especificadas en el archivo de configuracion (pyproject.toml o requirements.txt), principalmente:
* pandas
* scikit-learn
* joblib
* matplotlib / seaborn

Asegurese de instalar dichas dependencias en su entorno virtual antes de ejecutar el programa.

## Guia de Uso

1.  Preparacion de Datos:
    Asegurese de que el archivo 'credit-train.csv' se encuentra dentro de la carpeta 'data/raw/'.

2.  Ejecucion:
    Inicie el programa desde la terminal en la raiz del proyecto:

    python main.py

3.  Funcionamiento:
    * Verificacion: El sistema detecta automaticamente si existe un modelo entrenado.
    * Entrenamiento: Si es la primera ejecucion, el sistema procesara el dataset, entrenara el modelo (Decision Tree por defecto) y guardara los resultados.
    * Prediccion: Se iniciara una interfaz interactiva donde se solicitaran los datos del cliente. El sistema validara que las entradas coincidan con las categorias aprendidas en el notebook original y devolvera una decision (Aprobado/Denegado) junto con un nivel de confianza.

## Modelos Evaluados
Basado en el analisis del notebook, se han considerado los siguientes algoritmos:
* Decision Tree Classifier (Implementado en produccion)
* K-Nearest Neighbors (KNN)
* Logistic Regression

## Autor
Alejandro Cancelas Chapela