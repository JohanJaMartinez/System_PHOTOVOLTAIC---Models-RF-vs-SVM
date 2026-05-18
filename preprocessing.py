"""
preprocessing.py — Preprocesamiento del dataset fotovoltaico.

Implementa:
    - Separación de variables según el problema (clasificación / regresión)
    - Prevención explícita de fuga de información (data leakage)
    - División estratificada train/test
    - Codificación de etiquetas para clasificación
    - Escalamiento (StandardScaler) OBLIGATORIO para SVM/SVR
    - Pipelines reproducibles con random_state fijo

NOTA SOBRE PREVENCIÓN DE FUGA DE INFORMACIÓN:
    Clasificación — Features permitidas (9):
        hour, day_of_year, irradiance_W_m2, ambient_temp_C, panel_temp_C,
        humidity_percent, wind_speed_m_s, voltage_V, current_A
    Excluidas: condition (target), power_W (recomendado excluir)

    Regresión — Features permitidas (7, SOLO ambientales):
        hour, day_of_year, irradiance_W_m2, ambient_temp_C, panel_temp_C,
        humidity_percent, wind_speed_m_s
    Excluidas: power_W (target), condition (fuga directa),
               voltage_V y current_A (P ≈ V·I → fuga trivial)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from utils import PREPROCESSING_CONFIG, PROCESSED_DIR, logger, ensure_dirs
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import PREPROCESSING_CONFIG, PROCESSED_DIR, logger, ensure_dirs


# ─────────────────────────────────────────────
# ESTRUCTURAS DE RETORNO
# ─────────────────────────────────────────────

class ClassificationData:
    """Contenedor de datos preprocesados para clasificación."""

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # Versiones escaladas (para SVM/SVR)
        self.X_train_scaled = None
        self.X_test_scaled = None
        # Objetos de transformación
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None

    def __repr__(self):
        return (
            f"ClassificationData("
            f"train={self.X_train.shape}, test={self.X_test.shape}, "
            f"classes={self.class_names})"
        )


class RegressionData:
    """Contenedor de datos preprocesados para regresión."""

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # Versiones escaladas (para SVR)
        self.X_train_scaled = None
        self.X_test_scaled = None
        # Objetos de transformación
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None

    def __repr__(self):
        return (
            f"RegressionData("
            f"train={self.X_train.shape}, test={self.X_test.shape})"
        )


# ─────────────────────────────────────────────
# PREPROCESAMIENTO PARA CLASIFICACIÓN
# ─────────────────────────────────────────────

def preprocess_classification(df: pd.DataFrame, config: dict = None) -> ClassificationData:
    """
    Preprocesa el dataset para el problema de clasificación.

    Prevención de fuga de información:
        - 'condition' se usa solo como target (y), NUNCA como feature
        - 'power_W' se excluye de features (recomendado por el examen)

    Args:
        df: DataFrame completo del dataset
        config: Configuración de preprocesamiento

    Returns:
        ClassificationData con splits escalados y no escalados
    """
    if config is None:
        config = PREPROCESSING_CONFIG

    logger.info("=" * 50)
    logger.info("PREPROCESAMIENTO PARA CLASIFICACIÓN")
    logger.info("=" * 50)

    # ── Selección de features (prevención de fuga) ──────────────────────────
    features = config["classification_features"]
    target = config["classification_target"]

    logger.info(f"Features de entrada ({len(features)}): {features}")
    logger.info(f"Target: '{target}'")
    logger.info("Variables EXCLUIDAS explícitamente:")
    logger.info("  - 'condition': es el target (fuga directa)")
    logger.info("  - 'power_W': recomendado excluir (correlación con condition)")

    X = df[features].copy()
    y_raw = df[target].copy()

    # ── Codificación de etiquetas ───────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    logger.info(f"Clases codificadas: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── División stratificada train/test ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,  # Mantiene proporciones de clases en ambos splits
    )

    logger.info(f"División: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # ── Escalamiento (StandardScaler) — FIT SOLO EN TRAIN ──────────────────
    # CRÍTICO: el scaler se ajusta ÚNICAMENTE en datos de entrenamiento
    # para evitar data leakage del conjunto de prueba.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Solo transform, NO fit

    logger.info("StandardScaler ajustado en train, aplicado a test (sin re-fit)")

    # ── Empaquetar resultados ───────────────────────────────────────────────
    data = ClassificationData()
    data.X_train = X_train
    data.X_test = X_test
    data.y_train = y_train
    data.y_test = y_test
    data.X_train_scaled = X_train_scaled
    data.X_test_scaled = X_test_scaled
    data.scaler = scaler
    data.label_encoder = le
    data.feature_names = features
    data.class_names = list(le.classes_)

    return data


# ─────────────────────────────────────────────
# PREPROCESAMIENTO PARA REGRESIÓN
# ─────────────────────────────────────────────

def preprocess_regression(df: pd.DataFrame, config: dict = None) -> RegressionData:
    """
    Preprocesa el dataset para el problema de regresión.

    Prevención de fuga de información (CRÍTICO):
        - 'power_W' se usa solo como target, NUNCA como feature
        - 'condition' se excluye (información sobre el estado del sistema)
        - 'voltage_V' y 'current_A' se excluyen porque P ≈ V · I
          (incluirlas haría la predicción trivialmente perfecta)

    Solo se usan variables ambientales para predecir potencia.

    Args:
        df: DataFrame completo del dataset
        config: Configuración de preprocesamiento

    Returns:
        RegressionData con splits escalados y no escalados
    """
    if config is None:
        config = PREPROCESSING_CONFIG

    logger.info("=" * 50)
    logger.info("PREPROCESAMIENTO PARA REGRESIÓN")
    logger.info("=" * 50)

    # ── Selección de features (prevención de fuga) ──────────────────────────
    features = config["regression_features"]
    target = config["regression_target"]

    logger.info(f"Features de entrada ({len(features)}, solo ambientales): {features}")
    logger.info(f"Target: '{target}'")
    logger.info("Variables EXCLUIDAS explícitamente:")
    logger.info("  - 'power_W': es el target (fuga directa)")
    logger.info("  - 'condition': etiqueta del sistema (fuga de información)")
    logger.info("  - 'voltage_V': P ≈ V·I → predicción trivial")
    logger.info("  - 'current_A': P ≈ V·I → predicción trivial")

    X = df[features].copy()
    y = df[target].copy()

    # ── División train/test ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        # No stratify para regresión (variable continua)
    )

    logger.info(f"División: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # ── Escalamiento para SVR — FIT SOLO EN TRAIN ───────────────────────────
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # También escalamos y para SVR (mejora convergencia)
    scaler_y = StandardScaler()
    y_train_arr = y_train.values.reshape(-1, 1)
    y_test_arr = y_test.values.reshape(-1, 1)
    scaler_y.fit(y_train_arr)  # Solo fit en train

    logger.info("StandardScaler para X e y ajustado en train únicamente")

    # ── Empaquetar resultados ───────────────────────────────────────────────
    data = RegressionData()
    data.X_train = X_train
    data.X_test = X_test
    data.y_train = y_train
    data.y_test = y_test
    data.X_train_scaled = X_train_scaled
    data.X_test_scaled = X_test_scaled
    data.scaler_X = scaler_X
    data.scaler_y = scaler_y
    data.feature_names = features

    return data


# ─────────────────────────────────────────────
# FUNCIÓN DE GUARDADO (OPCIONAL)
# ─────────────────────────────────────────────

def save_splits(clf_data: ClassificationData, reg_data: RegressionData):
    """Guarda los splits en disco para reproducibilidad."""
    ensure_dirs()

    # Clasificación
    clf_train = pd.DataFrame(clf_data.X_train, columns=clf_data.feature_names)
    clf_train["condition_encoded"] = clf_data.y_train
    clf_train.to_csv(PROCESSED_DIR / "clf_train.csv", index=False)

    clf_test = pd.DataFrame(clf_data.X_test, columns=clf_data.feature_names)
    clf_test["condition_encoded"] = clf_data.y_test
    clf_test.to_csv(PROCESSED_DIR / "clf_test.csv", index=False)

    # Regresión
    reg_train = pd.DataFrame(reg_data.X_train, columns=reg_data.feature_names)
    reg_train["power_W"] = reg_data.y_train.values
    reg_train.to_csv(PROCESSED_DIR / "reg_train.csv", index=False)

    reg_test = pd.DataFrame(reg_data.X_test, columns=reg_data.feature_names)
    reg_test["power_W"] = reg_data.y_test.values
    reg_test.to_csv(PROCESSED_DIR / "reg_test.csv", index=False)

    logger.info(f"Splits guardados en: {PROCESSED_DIR}")


if __name__ == "__main__":
    from data_simulation import load_or_generate_dataset
    df = load_or_generate_dataset()

    clf_data = preprocess_classification(df)
    reg_data = preprocess_regression(df)

    print(clf_data)
    print(reg_data)
