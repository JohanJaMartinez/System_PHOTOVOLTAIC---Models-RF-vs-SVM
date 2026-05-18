"""
visualization.py — Generación de gráficas profesionales para el informe.

Genera automáticamente todas las figuras requeridas por el examen:
    1.  class_distribution.png      — Distribución de clases
    2.  correlation_matrix.png      — Mapa de calor de correlaciones
    3.  irradiance_daily_cycle.png  — Ciclo diario de irradiancia
    4.  power_distribution.png      — Distribución de potencia por condición
    5.  confusion_matrix_rf.png     — Matriz de confusión RF Classifier
    6.  confusion_matrix_svc.png    — Matriz de confusión SVC
    7.  feature_importance_clf.png  — Importancia de variables (clasificación)
    8.  feature_importance_reg.png  — Importancia de variables (regresión)
    9.  pred_vs_real_rf_reg.png     — Predicción vs realidad RF Regressor
    10. pred_vs_real_svr.png        — Predicción vs realidad SVR
    11. residuals_rf_reg.png        — Residuos RF Regressor
    12. residuals_svr.png           — Residuos SVR
    13. metrics_comparison_clf.png  — Comparación métricas clasificación
    14. metrics_comparison_reg.png  — Comparación métricas regresión

Todas las figuras se guardan en: figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica (compatibilidad servidor)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

try:
    from utils import FIGURES_DIR, CLASS_NAMES, CLASS_LABELS_ES, logger, ensure_dirs
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import FIGURES_DIR, CLASS_NAMES, CLASS_LABELS_ES, logger, ensure_dirs


# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DE ESTILO
# ─────────────────────────────────────────────

# Paleta de colores institucional
PALETTE_CLASSES = {
    "normal": "#2ecc71",
    "partial_shading": "#f39c12",
    "dirty_panels": "#e67e22",
    "panel_failure": "#e74c3c",
    "inverter_failure": "#8e44ad",
}
COLOR_RF = "#2980b9"
COLOR_SVM = "#e74c3c"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})


def _save_fig(fig: plt.Figure, filename: str):
    """Guarda una figura en la carpeta figures/."""
    ensure_dirs()
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Figura guardada: {path}")
    return path


# ─────────────────────────────────────────────
# 1. DISTRIBUCIÓN DE CLASES
# ─────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame):
    """Gráfica de barras con la distribución de condiciones operativas."""
    counts = df["condition"].value_counts().reindex(CLASS_NAMES)
    labels_es = [CLASS_LABELS_ES.get(c, c) for c in counts.index]
    colors = [PALETTE_CLASSES[c] for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels_es, counts.values, color=colors, edgecolor="white", linewidth=0.8)

    # Etiquetas de valor encima de cada barra
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val:,}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title("Distribución de Condiciones Operativas\nSistema Fotovoltaico Sintético")
    ax.set_ylabel("Número de muestras")
    ax.set_xlabel("Condición operativa")
    ax.set_ylim(0, counts.max() * 1.20)
    plt.xticks(rotation=20, ha="right")

    return _save_fig(fig, "class_distribution.png")


# ─────────────────────────────────────────────
# 2. MATRIZ DE CORRELACIONES
# ─────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame):
    """Heatmap de correlaciones entre variables numéricas."""
    numeric_cols = [
        "irradiance_W_m2", "ambient_temp_C", "panel_temp_C",
        "humidity_percent", "wind_speed_m_s", "voltage_V",
        "current_A", "power_W",
    ]
    corr = df[numeric_cols].corr()

    labels = [
        "Irradiancia", "T. Ambiente", "T. Panel",
        "Humedad", "Viento", "Voltaje", "Corriente", "Potencia"
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 9},
    )
    ax.set_title("Matriz de Correlaciones — Variables del Sistema FV")

    return _save_fig(fig, "correlation_matrix.png")


# ─────────────────────────────────────────────
# 3. CICLO DIARIO DE IRRADIANCIA
# ─────────────────────────────────────────────

def plot_irradiance_daily_cycle(df: pd.DataFrame):
    """Perfil diario de irradiancia promedio por hora."""
    hourly = df.groupby("hour")["irradiance_W_m2"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hourly.index, hourly["mean"], color=COLOR_RF, linewidth=2.5, label="Irradiancia media")
    ax.fill_between(
        hourly.index,
        hourly["mean"] - hourly["std"],
        hourly["mean"] + hourly["std"],
        alpha=0.25, color=COLOR_RF, label="±1 desv. estándar"
    )
    ax.set_title("Ciclo Diario de Irradiancia Solar (Promedio Anual)")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Irradiancia (W/m²)")
    ax.set_xlim(0, 23)
    ax.set_ylim(0)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    return _save_fig(fig, "irradiance_daily_cycle.png")


# ─────────────────────────────────────────────
# 4. DISTRIBUCIÓN DE POTENCIA POR CONDICIÓN
# ─────────────────────────────────────────────

def plot_power_by_condition(df: pd.DataFrame):
    """Box plot de potencia generada según condición operativa."""
    fig, ax = plt.subplots(figsize=(9, 5))

    data_by_class = [df[df["condition"] == c]["power_W"].values for c in CLASS_NAMES]
    colors = [PALETTE_CLASSES[c] for c in CLASS_NAMES]
    labels = [CLASS_LABELS_ES[c] for c in CLASS_NAMES]

    bp = ax.boxplot(
        data_by_class, tick_labels=labels, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title("Distribución de Potencia Generada por Condición Operativa")
    ax.set_ylabel("Potencia (W)")
    ax.set_xlabel("Condición")
    plt.xticks(rotation=20, ha="right")

    return _save_fig(fig, "power_by_condition.png")


# ─────────────────────────────────────────────
# 5-6. MATRICES DE CONFUSIÓN
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray, class_names: list, model_name: str, filename: str
):
    """Heatmap de matriz de confusión con porcentajes."""
    labels_es = [CLASS_LABELS_ES.get(c, c) for c in class_names]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=labels_es, yticklabels=labels_es,
        ax=ax, linewidths=0.5, vmin=0, vmax=100,
        annot_kws={"size": 10},
    )

    # Añadir conteos absolutos
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j + 0.5, i + 0.72,
                f"(n={cm[i,j]})",
                ha="center", va="center",
                fontsize=7, color="grey",
            )

    ax.set_title(f"Matriz de Confusión\n{model_name}")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.xticks(rotation=30, ha="right")

    return _save_fig(fig, filename)


# ─────────────────────────────────────────────
# 7-8. IMPORTANCIA DE VARIABLES
# ─────────────────────────────────────────────

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    model_name: str,
    filename: str,
):
    """Gráfica horizontal de importancia de variables (Random Forest)."""
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    labels_map = {
        "hour": "Hora del día",
        "day_of_year": "Día del año",
        "irradiance_W_m2": "Irradiancia (W/m²)",
        "ambient_temp_C": "Temp. Ambiente (°C)",
        "panel_temp_C": "Temp. Panel (°C)",
        "humidity_percent": "Humedad (%)",
        "wind_speed_m_s": "Velocidad Viento (m/s)",
        "voltage_V": "Voltaje (V)",
        "current_A": "Corriente (A)",
    }
    df_imp["label"] = df_imp["feature"].map(labels_map).fillna(df_imp["feature"])

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.6)))
    colors = [COLOR_RF if v >= df_imp["importance"].median() else "#95a5a6" for v in df_imp["importance"]]
    bars = ax.barh(df_imp["label"], df_imp["importance"], color=colors, edgecolor="white")

    for bar, val in zip(bars, df_imp["importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_title(f"Importancia de Variables\n{model_name}")
    ax.set_xlabel("Importancia (Gini Impurity Reduction)")
    ax.set_xlim(0, df_imp["importance"].max() * 1.18)

    return _save_fig(fig, filename)


# ─────────────────────────────────────────────
# 9-10. PREDICCIÓN VS REALIDAD
# ─────────────────────────────────────────────

def plot_pred_vs_real(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str,
    metrics: dict = None,
):
    """Scatter plot de predicción vs valor real para regresión."""
    # Muestrear para no sobrecargar la figura
    n_plot = min(1000, len(y_true))
    idx = np.random.default_rng(42).choice(len(y_true), n_plot, replace=False)
    yt = np.array(y_true)[idx]
    yp = np.array(y_pred)[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(yt, yp, alpha=0.35, s=20, color=COLOR_RF, edgecolors="none", label="Muestras")

    # Línea de predicción perfecta
    lims = [min(yt.min(), yp.min()) - 10, max(yt.max(), yp.max()) + 10]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")

    if metrics:
        info = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.2f} W\nMAE = {metrics['mae']:.2f} W"
        ax.text(
            0.05, 0.95, info,
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.8),
            fontsize=10, family="monospace",
        )

    ax.set_title(f"Predicción vs. Valor Real\n{model_name}")
    ax.set_xlabel("Potencia Real (W)")
    ax.set_ylabel("Potencia Predicha (W)")
    ax.legend(loc="lower right")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    return _save_fig(fig, filename)


# ─────────────────────────────────────────────
# 11-12. RESIDUOS
# ─────────────────────────────────────────────

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str,
):
    """Gráfica de residuos: residuos vs predicción + histograma."""
    residuals = np.array(y_true) - np.array(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Residuos vs predicción
    ax1.scatter(y_pred, residuals, alpha=0.3, s=15, color=COLOR_RF, edgecolors="none")
    ax1.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax1.set_title(f"Residuos vs. Predicción\n{model_name}")
    ax1.set_xlabel("Potencia Predicha (W)")
    ax1.set_ylabel("Residuo (W)")

    # Histograma de residuos
    ax2.hist(residuals, bins=50, color=COLOR_RF, alpha=0.75, edgecolor="white", density=True)

    # Curva normal superpuesta
    from scipy import stats
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 200)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label=f"Normal(μ={mu:.1f}, σ={sigma:.1f})")
    ax2.set_title(f"Distribución de Residuos\n{model_name}")
    ax2.set_xlabel("Residuo (W)")
    ax2.set_ylabel("Densidad")
    ax2.legend()

    return _save_fig(fig, filename)


# ─────────────────────────────────────────────
# 13. COMPARACIÓN MÉTRICAS CLASIFICACIÓN
# ─────────────────────────────────────────────

def plot_metrics_comparison_classification(results_clf: dict):
    """Gráfica comparativa de métricas de clasificación."""
    metrics_to_plot = ["accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted"]
    labels = ["Accuracy", "F1 (weighted)", "F1 (macro)", "Precisión", "Recall"]

    rf_vals = [results_clf["random_forest"][m] for m in metrics_to_plot]
    svc_vals = [results_clf["svc"][m] for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_rf = ax.bar(x - width/2, rf_vals, width, label="Random Forest", color=COLOR_RF, alpha=0.85)
    bars_svc = ax.bar(x + width/2, svc_vals, width, label="SVC", color=COLOR_SVM, alpha=0.85)

    for bar in bars_rf:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", fontsize=8.5, fontweight="bold")
    for bar in bars_svc:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_title("Comparación de Métricas — Clasificación")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Métrica")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    return _save_fig(fig, "metrics_comparison_clf.png")


# ─────────────────────────────────────────────
# 14. COMPARACIÓN MÉTRICAS REGRESIÓN
# ─────────────────────────────────────────────

def plot_metrics_comparison_regression(results_reg: dict):
    """Gráfica comparativa de métricas de regresión."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    metrics = [
        ("mae", "MAE (W)", False),
        ("rmse", "RMSE (W)", False),
        ("r2", "R²", True),
    ]

    colors = {"random_forest": COLOR_RF, "svr": COLOR_SVM}
    names = {"random_forest": "Random Forest", "svr": "SVR"}

    for ax, (metric, label, higher_better) in zip(axes, metrics):
        vals = {k: results_reg[k][metric] for k in ["random_forest", "svr"]}
        bar_colors = [colors[k] for k in vals]
        bar_labels = [names[k] for k in vals]
        bars = ax.bar(bar_labels, list(vals.values()), color=bar_colors, alpha=0.85, edgecolor="white")

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.02,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_title(label)
        ax.set_ylabel(label)
        if higher_better:
            ax.set_ylim(0, 1.1)
        else:
            ax.set_ylim(0, max(vals.values()) * 1.25)

    fig.suptitle("Comparación de Métricas — Regresión", fontsize=14, fontweight="bold")

    return _save_fig(fig, "metrics_comparison_reg.png")


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def generate_all_figures(df: pd.DataFrame, models: dict, results: dict, clf_data, reg_data):
    """
    Genera y guarda todas las figuras del proyecto.

    Args:
        df: Dataset completo
        models: Diccionario de modelos entrenados
        results: Resultados de evaluación
        clf_data: ClassificationData
        reg_data: RegressionData
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERANDO FIGURAS")
    logger.info("=" * 60)

    preds = results["predictions"]

    # ── Dataset (EDA) ─────────────────────────────────────────────────────
    plot_class_distribution(df)
    plot_correlation_matrix(df)
    plot_irradiance_daily_cycle(df)
    plot_power_by_condition(df)

    # ── Matrices de confusión ─────────────────────────────────────────────
    cm_rf = np.array(results["classification"]["random_forest"]["confusion_matrix"])
    plot_confusion_matrix(
        cm_rf, clf_data.class_names,
        "Random Forest Classifier", "confusion_matrix_rf.png"
    )

    cm_svc = np.array(results["classification"]["svc"]["confusion_matrix"])
    plot_confusion_matrix(
        cm_svc, clf_data.class_names,
        "Support Vector Classifier", "confusion_matrix_svc.png"
    )

    # ── Importancia de variables (RF) ─────────────────────────────────────
    rf_clf_model = models["rf_classifier"].model
    plot_feature_importance(
        rf_clf_model.feature_importances_,
        clf_data.feature_names,
        "Random Forest Classifier",
        "feature_importance_clf.png",
    )

    rf_reg_model = models["rf_regressor"].model
    plot_feature_importance(
        rf_reg_model.feature_importances_,
        reg_data.feature_names,
        "Random Forest Regressor",
        "feature_importance_reg.png",
    )

    # ── Predicción vs realidad ────────────────────────────────────────────
    plot_pred_vs_real(
        reg_data.y_test, preds["rf_reg"],
        "Random Forest Regressor", "pred_vs_real_rf_reg.png",
        results["regression"]["random_forest"],
    )
    plot_pred_vs_real(
        reg_data.y_test, preds["svr"],
        "Support Vector Regressor", "pred_vs_real_svr.png",
        results["regression"]["svr"],
    )

    # ── Residuos ──────────────────────────────────────────────────────────
    plot_residuals(
        reg_data.y_test, preds["rf_reg"],
        "Random Forest Regressor", "residuals_rf_reg.png"
    )
    plot_residuals(
        reg_data.y_test, preds["svr"],
        "Support Vector Regressor", "residuals_svr.png"
    )

    # ── Comparación de métricas ───────────────────────────────────────────
    plot_metrics_comparison_classification(results["classification"])
    plot_metrics_comparison_regression(results["regression"])

    logger.info(f"\nTodas las figuras guardadas en: {FIGURES_DIR}")
