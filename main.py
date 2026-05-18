"""
main.py — Pipeline principal del proyecto Examen 2 IA.

Ejecuta el flujo completo:
    1. Generación del dataset sintético fotovoltaico
    2. Preprocesamiento con prevención de fuga de información
    3. Entrenamiento de 4 modelos (RF + SVM/SVR) con GridSearchCV
    4. Evaluación con métricas completas
    5. Generación de figuras profesionales
    6. Guardado de resultados

Uso:
    python src/main.py

Requisitos:
    pip install -r requirements.txt
"""

import sys
import time
import traceback
import pandas as pd
from pathlib import Path


# Agregar src/ al path para importaciones relativas
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from utils import (
    logger,
    ensure_dirs,
    RANDOM_STATE,
    DATASET_PATH,
    RESULTS_DIR,
)
from data_simulation import load_or_generate_dataset
from preprocessing import preprocess_classification, preprocess_regression, save_splits
from models import train_all_models, save_models
from evaluation import evaluate_all_models
from visualization import generate_all_figures


def main(force_regenerate: bool = False):
    """
    Pipeline principal del proyecto.

    Args:
        force_regenerate: Si True, regenera el dataset aunque ya exista.
    """
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("EXAMEN 2 — INTELIGENCIA ARTIFICIAL PARA INGENIERÍA")
    logger.info("Comparación Random Forest vs SVM/SVR en Sistema Fotovoltaico")
    logger.info(f"Random State Global: {RANDOM_STATE}")
    logger.info("=" * 70)

    # ── Paso 1: Crear directorios ───────────────────────────────────────────
    logger.info("\n[PASO 1/6] Inicializando estructura de directorios...")
    ensure_dirs()

    # ── Paso 2: Generar / cargar dataset ───────────────────────────────────
    logger.info("\n[PASO 2/6] Generando dataset fotovoltaico sintético...")
    df = load_or_generate_dataset(force_regenerate=force_regenerate)
    logger.info(f"Dataset listo: {df.shape[0]:,} muestras × {df.shape[1]} variables")

    # ── Paso 3: Preprocesamiento ────────────────────────────────────────────
    logger.info("\n[PASO 3/6] Preprocesando datos...")
    clf_data = preprocess_classification(df)
    reg_data = preprocess_regression(df)
    save_splits(clf_data, reg_data)

    logger.info(f"\nCLASIFICACIÓN: {clf_data}")
    logger.info(f"REGRESIÓN:     {reg_data}")

    # ── Paso 4: Entrenamiento de modelos ────────────────────────────────────
    logger.info("\n[PASO 4/6] Entrenando modelos con GridSearchCV...")
    models = train_all_models(clf_data, reg_data)
    save_models(models)

    # ── Paso 5: Evaluación ──────────────────────────────────────────────────
    logger.info("\n[PASO 5/6] Evaluando modelos en conjunto de prueba...")
    results = evaluate_all_models(models, clf_data, reg_data)

    # ── Paso 6: Visualizaciones ─────────────────────────────────────────────
    logger.info("\n[PASO 6/6] Generando figuras...")
    generate_all_figures(df, models, results, clf_data, reg_data)

    # ── Resumen final ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETADO en {elapsed:.1f}s")
    logger.info("=" * 70)

    logger.info("\n📊 RESULTADOS DE CLASIFICACIÓN:")
    for key, r in results["classification"].items():
        logger.info(f"  {r['model']:35s} | Accuracy={r['accuracy']:.4f} | F1={r['f1_weighted']:.4f}")

    logger.info("\n📈 RESULTADOS DE REGRESIÓN:")
    for key, r in results["regression"].items():
        logger.info(f"  {r['model']:35s} | R²={r['r2']:.4f} | RMSE={r['rmse']:.2f}W | MAE={r['mae']:.2f}W")

    logger.info(f"\n📁 Resultados en: {RESULTS_DIR}")
    from utils import FIGURES_DIR
    logger.info(f"🖼️  Figuras en:    {FIGURES_DIR}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Examen 2 IA — Random Forest vs SVM/SVR"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerar el dataset aunque ya exista en disco",
    )
    args = parser.parse_args()

    try:
        results = main(force_regenerate=args.force_regenerate)
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
