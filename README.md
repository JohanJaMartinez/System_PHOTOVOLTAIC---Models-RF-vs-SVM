\# Comparación entre Random Forest y SVM en Sistemas Fotovoltaicos



Proyecto de Inteligencia Artificial enfocado en clasificación y regresión para sistemas fotovoltaicos instrumentados.



\## Objetivos



\- Clasificar condiciones operativas del sistema FV.

\- Predecir potencia generada.

\- Comparar Random Forest vs SVM/SVR.

\- Analizar métricas y variables importantes.



\---



\## Modelos Implementados



\### Clasificación

\- Random Forest Classifier

\- Support Vector Classifier (SVC)



\### Regresión

\- Random Forest Regressor

\- Support Vector Regressor (SVR)



\---



\## Estructura del Proyecto



```text

solar-ai-project/

│

├── data/

├── figures/

├── results/

│

├── data\_simulation.py

├── preprocessing.py

├── models.py

├── evaluation.py

├── visualization.py

├── utils.py

├── main.py

│

├── requirements.txt

├── README.md

└── .gitignore

```



\---



\## Variables del Dataset



\- Hora del día

\- Día del año

\- Irradiancia solar

\- Temperatura ambiente

\- Temperatura del panel

\- Humedad

\- Velocidad del viento

\- Voltaje

\- Corriente

\- Potencia generada



\---



\## Condiciones Operativas



\- Normal

\- Sombreado parcial

\- Paneles sucios

\- Falla de panel

\- Falla de inversor



\---



\## Prevención de Fuga de Información



\### Clasificación

No se utilizó:

\- `condition`

\- `power\_W`



\### Regresión

No se utilizó:

\- `power\_W`

\- `condition`

\- `voltage\_V`

\- `current\_A`



\---



\## Métricas Utilizadas



\### Clasificación

\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Matriz de confusión



\### Regresión

\- MAE

\- RMSE

\- R²



\---



\## Resultados Principales



\### Clasificación

Random Forest obtuvo mejor desempeño:

\- Accuracy = 73.9%

\- F1 Macro = 0.735



\### Regresión

Random Forest obtuvo:

\- RMSE = 207.25 W

\- R² = 0.765



\---



\## Ejecución



Instalar dependencias:



```bash

pip install -r requirements.txt

```



Ejecutar proyecto:



```bash

python main.py

```



\---



\## Tecnologías Utilizadas



\- Python

\- Scikit-learn

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn



\---



\## Autor(es)



Johan Javier Martínez M.

Reinaldo Torres Pastrana

Juan Diego García G.

