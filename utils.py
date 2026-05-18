"""
utils.py — Configuración global, rutas y utilidades compartidas.

Este módulo centraliza todos los parámetros de configuración para garantizar
reproducibilidad y mantenibilidad del proyecto.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# REPRODUCIBILIDAD
# ─────────────────────────────────────────────
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# RUTAS DEL PROYECTO
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"
SRC_DIR = ROOT_DIR / "src"

# Archivos principales
DATASET_PATH = RAW_DIR / "synthetic_pv_dataset.csv"
RESULTS_CLASSIFICATION_PATH = RESULTS_DIR / "classification_results.json"
RESULTS_REGRESSION_PATH = RESULTS_DIR / "regression_results.json"
METRICS_TABLE_PATH = RESULTS_DIR / "metrics_comparison.csv"

# ─────────────────────────────────────────────
# PARÁMETROS DE SIMULACIÓN
# ─────────────────────────────────────────────
SIMULATION_CONFIG = {
    "n_samples": 5000,           # Número de muestras totales
    "random_state": RANDOM_STATE,
    # Parámetros del panel fotovoltaico
    "panel_efficiency": 0.18,    # η: eficiencia del panel (18%)
    "panel_area": 10.0,          # A: área del arreglo en m²
    "noct": 45.0,                # NOCT: temperatura nominal de operación (°C)
    "temp_coeff": -0.004,        # γ: coeficiente de temperatura (/°C)
    # Distribución de clases (%)
    "class_distribution": {
        "normal": 0.50,
        "partial_shading": 0.20,
        "dirty_panels": 0.15,
        "panel_failure": 0.10,
        "inverter_failure": 0.05,
    },
    # Factor de reducción por condición (f_falla)
    "fault_factors": {
        "normal": 1.00,
        "partial_shading": 0.65,
        "dirty_panels": 0.80,
        "panel_failure": 0.40,
        "inverter_failure": 0.10,
    },
    # Ruido gaussiano (desviación estándar como fracción)
    "noise_irradiance": 15.0,    # W/m²
    "noise_temp": 1.5,           # °C
    "noise_power": 10.0,         # W
    "noise_voltage": 1.0,        # V
    "noise_current": 0.05,       # A
    "noise_humidity": 3.0,       # %
    "noise_wind": 0.5,           # m/s
}

# ─────────────────────────────────────────────
# PARÁMETROS DE PREPROCESAMIENTO
# ─────────────────────────────────────────────
PREPROCESSING_CONFIG = {
    "test_size": 0.20,
    "random_state": RANDOM_STATE,
    "stratify": True,
    # Variables de entrada para clasificación (excluye 'condition' y 'power_W')
    "classification_features": [
        "hour",
        "day_of_year",
        "irradiance_W_m2",
        "ambient_temp_C",
        "panel_temp_C",
        "humidity_percent",
        "wind_speed_m_s",
        "voltage_V",
        "current_A",
    ],
    "classification_target": "condition",
    # Variables de entrada para regresión (SOLO ambientales)
    # Excluye: power_W (target), condition (fuga), voltage_V, current_A (P≈V·I)
    "regression_features": [
        "hour",
        "day_of_year",
        "irradiance_W_m2",
        "ambient_temp_C",
        "panel_temp_C",
        "humidity_percent",
        "wind_speed_m_s",
    ],
    "regression_target": "power_W",
}

# ─────────────────────────────────────────────
# PARÁMETROS DE MODELOS (GridSearchCV)
# ─────────────────────────────────────────────
MODEL_CONFIG = {
    "cv_folds": 5,
    "scoring_classification": "f1_weighted",
    "scoring_regression": "neg_root_mean_squared_error",
    "n_jobs": -1,
    "verbose": 1,
    # Random Forest Classifier
    "rf_classifier_params": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    },
    # SVC
    "svc_params": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "poly"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced"],
    },
    # Random Forest Regressor
    "rf_regressor_params": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    # SVR
    "svr_params": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf"],
        "gamma": ["scale", "auto"],
        "epsilon": [0.1, 0.5, 1.0],
    },
}

# ─────────────────────────────────────────────
# CLASES Y ETIQUETAS
# ─────────────────────────────────────────────
CLASS_NAMES = ["normal", "partial_shading", "dirty_panels", "panel_failure", "inverter_failure"]
CLASS_LABELS_ES = {
    "normal": "Normal",
    "partial_shading": "Sombreado Parcial",
    "dirty_panels": "Paneles Sucios",
    "panel_failure": "Falla de Panel",
    "inverter_failure": "Falla de Inversor",
}

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────
def setup_logging(level=logging.INFO):
    """Configura el sistema de logging del proyecto."""
    log_format = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────
# FUNCIONES DE UTILIDAD
# ─────────────────────────────────────────────
def ensure_dirs():
    """Crea todos los directorios necesarios si no existen."""
    for d in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path):
    """Guarda un diccionario como archivo JSON con formato legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def load_json(path: Path) -> dict:
    """Carga un archivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp() -> str:
    """Retorna un timestamp formateado para nombrar archivos."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


logger = setup_logging()
