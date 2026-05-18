# AI-Based Photovoltaic System Analysis

### Random Forest vs Support Vector Machines

Artificial Intelligence project focused on supervised learning techniques applied to photovoltaic systems under different operational conditions.

The project compares Random Forest and Support Vector Machines (SVM) models for both classification and regression tasks using a synthetic dataset generated from realistic physical behaviors.

---

# Citation and Academic Use

This project was developed exclusively for academic and educational purposes.

If this repository, codebase, methodology, visualizations, or any part of the implementation is used, modified, referenced, or adapted in other academic projects, reports, publications, repositories, or research works, proper citation and acknowledgment of the original authors is mandatory.

Unauthorized copying or redistribution without attribution is not permitted.

---

# Project Workflow

```text
Data Simulation
        ↓
Preprocessing
        ↓
Model Training
        ↓
Evaluation
        ↓
Visualization & Analysis
```

---

# Engineering Problem

Modern photovoltaic systems require intelligent monitoring mechanisms capable of identifying operational conditions and estimating generated power using environmental and electrical variables.

This project addresses two machine learning problems:

### Classification

Prediction of photovoltaic operating conditions:

* Normal operation
* Partial shading
* Dirty panels
* Panel failure
* Inverter failure

### Regression

Prediction of generated electrical power using environmental variables.

---

# Dataset Simulation

The synthetic dataset was generated using simplified physical models related to:

* Solar irradiance
* Ambient temperature
* Panel temperature
* Humidity
* Wind speed
* Voltage
* Current
* Generated power

The simulation incorporates realistic noise and operational faults to emulate real photovoltaic behavior.

---

# Information Leakage Prevention

To ensure a valid machine learning workflow, information leakage was explicitly prevented.

### Classification Inputs

Excluded variables:

* `condition`
* `power_W`

### Regression Inputs

Excluded variables:

* `power_W`
* `condition`
* `voltage_V`
* `current_A`

---

# Implemented Models

## Classification

* Random Forest Classifier
* Support Vector Classifier (SVC)

## Regression

* Random Forest Regressor
* Support Vector Regressor (SVR)

---

# Evaluation Metrics

## Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## Regression Metrics

* MAE
* RMSE
* R² Score

---

# Main Results

## Classification

Random Forest achieved the best overall performance:

* Accuracy = 73.9%
* F1 Macro = 0.735

## Regression

Random Forest obtained:

* RMSE = 207.25 W
* R² = 0.765

The results demonstrate that Random Forest models handled nonlinear relationships and class imbalance more effectively than SVM-based approaches.

---

# Project Structure

```text
Examen2_IA/
│
├── data/
├── figures/
├── results/
│
├── data_simulation.py
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

---

# Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run complete pipeline:

```bash
python main.py
```

---

# Technologies

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

# Authors

* Johan Javier Martínez M.
* Reinaldo Torres Pastrana
* Juan Diego García G.
