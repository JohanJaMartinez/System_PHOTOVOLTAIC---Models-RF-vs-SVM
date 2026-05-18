"""
evaluation.py — Cálculo de métricas de evaluación para clasificación y regresión.

Métricas implementadas:
    CLASIFICACIÓN:
        - Accuracy, Precision (macro/weighted), Recall (macro/weighted), F1-score
        - Matriz de confusión
        - Reporte detallado por clase (classification_report)

    REGRESIÓN:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R² (Coeficiente de determinación)
        - MAPE (Mean Absolute Percentage Error) — adicional

Todas las métricas se guardan en formato JSON y CSV para el informe.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from utils import (
        RESULTS_DIR,
        RESULTS_CLASSIFICATION_PATH,
        RESULTS_REGRESSION_PATH,
        METRICS_TABLE_PATH,
        logger,
        ensure_dirs,
        save_json,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import (
        RESULTS_DIR,
        RESULTS_CLASSIFICATION_PATH,
        RESULTS_REGRESSION_PATH,
        METRICS_TABLE_PATH,
        logger,
        ensure_dirs,
        save_json,
    )


# ─────────────────────────────────────────────
# MÉTRICAS DE CLASIFICACIÓN
# ─────────────────────────────────────────────

def evaluate_classifier(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    best_params: dict = None,
    cv_score: float = None,
) -> dict:
    """
    Calcula todas las métricas de clasificación requeridas.

    Args:
        model_name: Nombre del modelo
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Nombres de las clases
        best_params: Mejores hiperparámetros del GridSearch
        cv_score: Score de validación cruzada

    Returns:
        Diccionario con todas las métricas
    """
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec_w, 4),
        "precision_macro": round(prec_m, 4),
        "recall_weighted": round(rec_w, 4),
        "recall_macro": round(rec_m, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "best_params": best_params or {},
        "cv_f1_weighted": round(cv_score, 4) if cv_score is not None else None,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTADOS: {model_name}")
    logger.info(f"{'='*50}")
    logger.info(f"  Accuracy:           {acc:.4f}")
    logger.info(f"  F1-score (weighted): {f1_w:.4f}")
    logger.info(f"  F1-score (macro):    {f1_m:.4f}")
    logger.info(f"  Precision (weighted): {prec_w:.4f}")
    logger.info(f"  Recall (weighted):    {rec_w:.4f}")

    return metrics


# ─────────────────────────────────────────────
# MÉTRICAS DE REGRESIÓN
# ─────────────────────────────────────────────

def evaluate_regressor(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_params: dict = None,
    cv_score: float = None,
) -> dict:
    """
    Calcula todas las métricas de regresión requeridas.

    Args:
        model_name: Nombre del modelo
        y_true: Valores verdaderos de potencia (W)
        y_pred: Valores predichos de potencia (W)
        best_params: Mejores hiperparámetros del GridSearch
        cv_score: Score de validación cruzada (RMSE)

    Returns:
        Diccionario con todas las métricas
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE — solo donde y_true > 0 para evitar división por cero
    mask = y_true > 1.0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")

    # Estadísticas de residuos
    residuals = y_true - y_pred
    residual_std = float(np.std(residuals))
    residual_mean = float(np.mean(residuals))

    metrics = {
        "model": model_name,
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "mape_percent": round(float(mape), 4) if not np.isnan(mape) else None,
        "residual_mean": round(residual_mean, 4),
        "residual_std": round(residual_std, 4),
        "best_params": best_params or {},
        "cv_rmse": round(cv_score, 4) if cv_score is not None else None,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTADOS: {model_name}")
    logger.info(f"{'='*50}")
    logger.info(f"  MAE:   {mae:.4f} W")
    logger.info(f"  RMSE:  {rmse:.4f} W")
    logger.info(f"  R²:    {r2:.4f}")
    if not np.isnan(mape):
        logger.info(f"  MAPE:  {mape:.2f}%")

    return metrics


# ─────────────────────────────────────────────
# EVALUACIÓN COMPLETA
# ─────────────────────────────────────────────

def evaluate_all_models(models: dict, clf_data, reg_data) -> dict:
    """
    Evalúa todos los modelos y guarda resultados.

    Args:
        models: Diccionario de TrainedModel
        clf_data: ClassificationData
        reg_data: RegressionData

    Returns:
        Diccionario con todos los resultados
    """
    ensure_dirs()
    results = {"classification": {}, "regression": {}}

    logger.info("\n" + "=" * 60)
    logger.info("EVALUACIÓN DE MODELOS")
    logger.info("=" * 60)

    # ── Clasificación ───────────────────────────────────────────────────────
    rf_clf = models["rf_classifier"]
    y_pred_rf = rf_clf.predict(clf_data.X_test)
    results["classification"]["random_forest"] = evaluate_classifier(
        model_name=rf_clf.name,
        y_true=clf_data.y_test,
        y_pred=y_pred_rf,
        class_names=clf_data.class_names,
        best_params=rf_clf.best_params,
        cv_score=rf_clf.cv_score,
    )

    svc_m = models["svc"]
    y_pred_svc = svc_m.predict(clf_data.X_test_scaled)
    results["classification"]["svc"] = evaluate_classifier(
        model_name=svc_m.name,
        y_true=clf_data.y_test,
        y_pred=y_pred_svc,
        class_names=clf_data.class_names,
        best_params=svc_m.best_params,
        cv_score=svc_m.cv_score,
    )

    # ── Regresión ───────────────────────────────────────────────────────────
    rf_reg = models["rf_regressor"]
    y_pred_rf_reg = rf_reg.predict(reg_data.X_test)
    results["regression"]["random_forest"] = evaluate_regressor(
        model_name=rf_reg.name,
        y_true=reg_data.y_test.values,
        y_pred=y_pred_rf_reg,
        best_params=rf_reg.best_params,
        cv_score=rf_reg.cv_score,
    )

    svr_m = models["svr"]
    y_pred_svr = svr_m.predict(reg_data.X_test_scaled)
    results["regression"]["svr"] = evaluate_regressor(
        model_name=svr_m.name,
        y_true=reg_data.y_test.values,
        y_pred=y_pred_svr,
        best_params=svr_m.best_params,
        cv_score=svr_m.cv_score,
    )

    # ── Guardar resultados ──────────────────────────────────────────────────
    save_json(results["classification"], RESULTS_CLASSIFICATION_PATH)
    save_json(results["regression"], RESULTS_REGRESSION_PATH)

    # Tabla comparativa CSV
    _save_comparison_table(results)

    # Agregar predicciones para visualización
    results["predictions"] = {
        "rf_clf": y_pred_rf,
        "svc": y_pred_svc,
        "rf_reg": y_pred_rf_reg,
        "svr": y_pred_svr,
    }

    logger.info(f"\nResultados guardados en: {RESULTS_DIR}")
    return results


def _save_comparison_table(results: dict):
    """Genera tabla CSV comparativa de métricas."""
    rows = []

    # Clasificación
    for key, r in results["classification"].items():
        rows.append({
            "Problema": "Clasificación",
            "Modelo": r["model"],
            "Accuracy": r["accuracy"],
            "F1 (weighted)": r["f1_weighted"],
            "F1 (macro)": r["f1_macro"],
            "Precision": r["precision_weighted"],
            "Recall": r["recall_weighted"],
            "MAE": "-",
            "RMSE": "-",
            "R²": "-",
        })

    # Regresión
    for key, r in results["regression"].items():
        rows.append({
            "Problema": "Regresión",
            "Modelo": r["model"],
            "Accuracy": "-",
            "F1 (weighted)": "-",
            "F1 (macro)": "-",
            "Precision": "-",
            "Recall": "-",
            "MAE": r["mae"],
            "RMSE": r["rmse"],
            "R²": r["r2"],
        })

    df_table = pd.DataFrame(rows)
    df_table.to_csv(METRICS_TABLE_PATH, index=False)
    logger.info(f"Tabla comparativa guardada en: {METRICS_TABLE_PATH}")

    # Mostrar tabla en consola
    logger.info("\n" + "=" * 60)
    logger.info("TABLA COMPARATIVA DE MÉTRICAS")
    logger.info("=" * 60)
    logger.info("\n" + df_table.to_string(index=False))
