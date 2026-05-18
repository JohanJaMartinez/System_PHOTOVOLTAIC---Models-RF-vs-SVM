"""
models.py — Entrenamiento y configuración de modelos de ML.

Implementa cuatro modelos organizados en dos problemas:

CLASIFICACIÓN:
    1. RandomForestClassifier — Ensemble de árboles de decisión
    2. SVC (Support Vector Classifier) — Kernel RBF (principal) + Poly

REGRESIÓN:
    3. RandomForestRegressor — Ensemble para regresión
    4. SVR (Support Vector Regressor) — Kernel RBF

Todos los modelos usan:
    - GridSearchCV con validación cruzada estratificada (k=5)
    - random_state=42 para reproducibilidad
    - Escalamiento obligatorio para SVM/SVR (recibe datos pre-escalados)
    - Justificación explícita de hiperparámetros
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score

try:
    from utils import MODEL_CONFIG, RESULTS_DIR, RANDOM_STATE, logger, ensure_dirs
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import MODEL_CONFIG, RESULTS_DIR, RANDOM_STATE, logger, ensure_dirs


# ─────────────────────────────────────────────
# CONTENEDOR DE RESULTADOS DE MODELOS
# ─────────────────────────────────────────────

class TrainedModel:
    """Contenedor para un modelo entrenado y sus metadatos."""

    def __init__(self, name: str, model, best_params: dict, cv_score: float):
        self.name = name
        self.model = model  # Mejor estimador después de GridSearch
        self.best_params = best_params
        self.cv_score = cv_score

    def predict(self, X):
        return self.model.predict(X)

    def __repr__(self):
        return f"TrainedModel(name='{self.name}', cv_score={self.cv_score:.4f})"


# ─────────────────────────────────────────────
# MODELOS DE CLASIFICACIÓN
# ─────────────────────────────────────────────

def train_random_forest_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict = None,
) -> TrainedModel:
    """
    Entrena un Random Forest Classifier con búsqueda de hiperparámetros.

    Justificación de hiperparámetros buscados:
        - n_estimators: más árboles reduce varianza; 100-200 es suficiente para 5000 muestras
        - max_depth: controla complejidad y overfitting; None permite crecimiento completo
        - min_samples_split: regularización mínima del árbol
        - class_weight='balanced': compensa desbalance de clases

    Args:
        X_train: Features de entrenamiento (sin escalar, RF no lo requiere)
        y_train: Etiquetas codificadas
        config: Configuración del modelo

    Returns:
        TrainedModel con el mejor modelo encontrado
    """
    if config is None:
        config = MODEL_CONFIG

    logger.info("Entrenando Random Forest Classifier con GridSearchCV...")

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=config["n_jobs"],
    )

    param_grid = config["rf_classifier_params"]

    cv = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=config["scoring_classification"],
        n_jobs=config["n_jobs"],
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_

    logger.info(f"RF Classifier — Mejores parámetros: {best_params}")
    logger.info(f"RF Classifier — CV F1-weighted: {cv_score:.4f}")

    return TrainedModel(
        name="Random Forest Classifier",
        model=best,
        best_params=best_params,
        cv_score=cv_score,
    )


def train_svc(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    config: dict = None,
) -> TrainedModel:
    """
    Entrena un Support Vector Classifier con búsqueda de hiperparámetros.

    IMPORTANTE: Recibe datos PRE-ESCALADOS (StandardScaler aplicado).
    SVM es sensible a la escala de las variables; sin escalamiento su
    rendimiento degrada significativamente.

    Justificación del kernel RBF:
        - RBF (Radial Basis Function) funciona bien cuando la relación entre
          clases no es lineal (caso típico en sistemas fotovoltaicos con fallas)
        - Poly se incluye como comparación secundaria
        - El parámetro C controla el trade-off entre margen y errores de clasificación
        - gamma controla el radio de influencia de cada punto de soporte

    Args:
        X_train_scaled: Features ESCALADAS de entrenamiento
        y_train: Etiquetas codificadas
        config: Configuración del modelo

    Returns:
        TrainedModel con el mejor SVC encontrado
    """
    if config is None:
        config = MODEL_CONFIG

    logger.info("Entrenando SVC con GridSearchCV (datos pre-escalados)...")

    base_model = SVC(
        random_state=RANDOM_STATE,
        probability=True,  # Necesario para predict_proba si se requiere
    )

    param_grid = config["svc_params"]

    cv = StratifiedKFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=config["scoring_classification"],
        n_jobs=config["n_jobs"],
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train_scaled, y_train)

    best = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_

    logger.info(f"SVC — Mejores parámetros: {best_params}")
    logger.info(f"SVC — CV F1-weighted: {cv_score:.4f}")

    return TrainedModel(
        name="Support Vector Classifier",
        model=best,
        best_params=best_params,
        cv_score=cv_score,
    )


# ─────────────────────────────────────────────
# MODELOS DE REGRESIÓN
# ─────────────────────────────────────────────

def train_random_forest_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict = None,
) -> TrainedModel:
    """
    Entrena un Random Forest Regressor con búsqueda de hiperparámetros.

    Justificación:
        - RF Regressor promedia predicciones de múltiples árboles reduciendo varianza
        - n_estimators=100-200: suficiente para estabilizar predicciones
        - max_depth controla complejidad; None permite ajuste completo al train set
        - No requiere escalamiento (invariante a escala)

    Args:
        X_train: Features de entrenamiento (sin escalar)
        y_train: Potencia generada en W
        config: Configuración del modelo

    Returns:
        TrainedModel con el mejor RF Regressor
    """
    if config is None:
        config = MODEL_CONFIG

    logger.info("Entrenando Random Forest Regressor con GridSearchCV...")

    base_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=config["n_jobs"],
    )

    param_grid = config["rf_regressor_params"]

    cv = KFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=config["scoring_regression"],
        n_jobs=config["n_jobs"],
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = -grid_search.best_score_  # neg_rmse → positivo

    logger.info(f"RF Regressor — Mejores parámetros: {best_params}")
    logger.info(f"RF Regressor — CV RMSE: {cv_score:.2f} W")

    return TrainedModel(
        name="Random Forest Regressor",
        model=best,
        best_params=best_params,
        cv_score=cv_score,
    )


def train_svr(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    config: dict = None,
) -> TrainedModel:
    """
    Entrena un Support Vector Regressor con búsqueda de hiperparámetros.

    IMPORTANTE: Recibe datos PRE-ESCALADOS (StandardScaler).
    SVR es especialmente sensible a la escala; sin escalamiento los resultados
    son significativamente peores.

    Justificación del kernel RBF:
        - La relación irradiancia → potencia no es estrictamente lineal
          (efectos de temperatura, fallas, etc. introducen no-linealidades)
        - RBF captura estas no-linealidades mejor que un kernel lineal
        - epsilon: zona de no-penalización alrededor de las predicciones
        - C: regularización; valores altos = menos margen, más ajuste a datos

    Args:
        X_train_scaled: Features ESCALADAS de entrenamiento
        y_train: Potencia generada en W (sin escalar; SVR internamente la maneja)
        config: Configuración del modelo

    Returns:
        TrainedModel con el mejor SVR
    """
    if config is None:
        config = MODEL_CONFIG

    logger.info("Entrenando SVR con GridSearchCV (datos pre-escalados)...")

    base_model = SVR()

    param_grid = config["svr_params"]

    cv = KFold(
        n_splits=config["cv_folds"],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=config["scoring_regression"],
        n_jobs=config["n_jobs"],
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train_scaled, y_train)

    best = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = -grid_search.best_score_

    logger.info(f"SVR — Mejores parámetros: {best_params}")
    logger.info(f"SVR — CV RMSE: {cv_score:.2f} W")

    return TrainedModel(
        name="Support Vector Regressor",
        model=best,
        best_params=best_params,
        cv_score=cv_score,
    )


# ─────────────────────────────────────────────
# ENTRENAMIENTO COMPLETO
# ─────────────────────────────────────────────

def train_all_models(clf_data, reg_data, config: dict = None) -> dict:
    """
    Entrena los cuatro modelos del examen.

    Args:
        clf_data: ClassificationData de preprocessing.py
        reg_data: RegressionData de preprocessing.py
        config: Configuración de modelos

    Returns:
        Dict con los cuatro modelos entrenados
    """
    logger.info("\n" + "=" * 60)
    logger.info("ENTRENANDO MODELOS")
    logger.info("=" * 60)

    models = {}

    # ── Clasificación ───────────────────────────────────────────────────────
    logger.info("\n[1/4] Random Forest Classifier")
    models["rf_classifier"] = train_random_forest_classifier(
        clf_data.X_train, clf_data.y_train, config
    )

    logger.info("\n[2/4] Support Vector Classifier")
    models["svc"] = train_svc(
        clf_data.X_train_scaled, clf_data.y_train, config
    )

    # ── Regresión ───────────────────────────────────────────────────────────
    logger.info("\n[3/4] Random Forest Regressor")
    models["rf_regressor"] = train_random_forest_regressor(
        reg_data.X_train, reg_data.y_train, config
    )

    logger.info("\n[4/4] Support Vector Regressor")
    models["svr"] = train_svr(
        reg_data.X_train_scaled, reg_data.y_train, config
    )

    logger.info("\nTodos los modelos entrenados exitosamente.")
    return models


def save_models(models: dict, output_dir: Path = None):
    """Guarda los modelos entrenados en disco."""
    if output_dir is None:
        output_dir = RESULTS_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, trained_model in models.items():
        path = output_dir / f"{key}.joblib"
        joblib.dump(trained_model, path)
        logger.info(f"Modelo guardado: {path}")
