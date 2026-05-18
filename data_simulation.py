"""
data_simulation.py — Generación del dataset sintético para sistema fotovoltaico.

Implementa un modelo físico simplificado pero coherente que simula el comportamiento
de un sistema fotovoltaico bajo diferentes condiciones operativas.

Fórmulas físicas implementadas:
    T_panel = T_amb + (NOCT - 20) / 800 * G + ε_T
    P = η · A · G_eff · [1 + γ · (T_panel - 25)] · f_falla + ε_P
    I ≈ P / V  (relación eléctrica simplificada)

Clases simuladas:
    - normal:            Operación estándar
    - partial_shading:   Sombreado parcial (reduce irradiancia efectiva)
    - dirty_panels:      Suciedad (reduce eficiencia)
    - panel_failure:     Falla de panel (reducción severa de potencia)
    - inverter_failure:  Falla de inversor (potencia casi nula)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Importación robusta — funciona tanto al ejecutar el módulo directamente
# como cuando es importado desde main.py
try:
    from utils import SIMULATION_CONFIG, CLASS_NAMES, DATASET_PATH, ensure_dirs, logger
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import SIMULATION_CONFIG, CLASS_NAMES, DATASET_PATH, ensure_dirs, logger


# ─────────────────────────────────────────────
# FUNCIONES DE CICLO SOLAR
# ─────────────────────────────────────────────

def _solar_irradiance(hour: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """
    Calcula la irradiancia solar base siguiendo un perfil gaussiano diario
    y una modulación anual estacional.

    Args:
        hour: Hora del día (0-23)
        day_of_year: Día del año (1-365)

    Returns:
        Irradiancia base en W/m²
    """
    # Perfil gaussiano centrado en el mediodía solar (~12h)
    daily_profile = np.exp(-0.5 * ((hour - 12.0) / 3.5) ** 2)

    # Modulación estacional: máximo en verano (día ~172), mínimo en invierno (día ~355)
    seasonal_factor = 0.75 + 0.25 * np.cos(2 * np.pi * (day_of_year - 172) / 365)

    # Irradiancia máxima típica en Colombia (~1000 W/m²)
    G_max = 1050.0
    irradiance = G_max * daily_profile * seasonal_factor

    # Solo hay irradiancia entre 06:00 y 18:00
    irradiance[hour < 6] = 0.0
    irradiance[hour > 18] = 0.0

    return np.clip(irradiance, 0, G_max)


def _ambient_temperature(hour: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """
    Temperatura ambiente con variación diaria y estacional.

    Args:
        hour: Hora del día
        day_of_year: Día del año

    Returns:
        Temperatura ambiente en °C
    """
    # Temperatura media anual ~22°C con variación estacional de ±5°C
    T_mean = 22.0 + 5.0 * np.cos(2 * np.pi * (day_of_year - 172) / 365 + np.pi)

    # Variación diaria: mínimo a las 05:00, máximo a las 14:00
    T_daily = 6.0 * np.sin(np.pi * (hour - 5) / 14)
    T_daily = np.where(hour < 5, -3.0, T_daily)

    return T_mean + T_daily


def _humidity(hour: np.ndarray, irradiance: np.ndarray) -> np.ndarray:
    """
    Humedad relativa: inversamente correlacionada con irradiancia y temperatura.

    Args:
        hour: Hora del día
        irradiance: Irradiancia solar (W/m²)

    Returns:
        Humedad en %
    """
    # Humedad base más alta en la madrugada, baja al mediodía
    base_humidity = 80.0 - 35.0 * np.sin(np.pi * (hour - 6) / 12)
    base_humidity = np.clip(base_humidity, 30, 95)

    # Reducción leve por alta irradiancia
    irr_effect = -0.015 * irradiance
    return np.clip(base_humidity + irr_effect, 25, 98)


def _wind_speed(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Velocidad del viento con distribución Weibull (físicamente realista).

    Args:
        n: Número de muestras
        rng: Generador de números aleatorios

    Returns:
        Velocidad del viento en m/s
    """
    # Distribución Weibull típica para viento (k=2, lambda=3 m/s)
    return rng.weibull(2, size=n) * 3.0


# ─────────────────────────────────────────────
# GENERADOR PRINCIPAL DEL DATASET
# ─────────────────────────────────────────────

def simulate_photovoltaic_dataset(config: dict = None) -> pd.DataFrame:
    """
    Genera un dataset sintético de sistema fotovoltaico con relaciones físicas
    coherentes y cinco clases de condición operativa.

    Args:
        config: Diccionario de configuración. Si es None usa SIMULATION_CONFIG.

    Returns:
        DataFrame con todas las variables del sistema fotovoltaico.
    """
    if config is None:
        config = SIMULATION_CONFIG

    cfg = config
    n = cfg["n_samples"]
    rng = np.random.default_rng(cfg["random_state"])

    logger.info(f"Iniciando simulación de {n} muestras con seed={cfg['random_state']}")

    # ── 1. Asignar clases según distribución especificada ──────────────────
    class_dist = cfg["class_distribution"]
    class_counts = {}
    remaining = n
    class_list = list(class_dist.keys())
    for i, cls in enumerate(class_list[:-1]):
        count = int(round(class_dist[cls] * n))
        class_counts[cls] = count
        remaining -= count
    class_counts[class_list[-1]] = remaining

    conditions = np.concatenate([
        np.full(count, cls) for cls, count in class_counts.items()
    ])

    # Barajar para mezclar clases
    shuffle_idx = rng.permutation(n)
    conditions = conditions[shuffle_idx]

    logger.info(f"Distribución de clases: {class_counts}")

    # ── 2. Variables temporales ─────────────────────────────────────────────
    hour = rng.integers(0, 24, size=n).astype(float)
    day_of_year = rng.integers(1, 366, size=n).astype(float)

    # ── 3. Irradiancia base + ruido gaussiano ──────────────────────────────
    G_base = _solar_irradiance(hour, day_of_year)
    noise_G = rng.normal(0, cfg["noise_irradiance"], n)
    irradiance = np.clip(G_base + noise_G, 0, 1200)

    # ── 4. Temperatura ambiente ─────────────────────────────────────────────
    T_amb_base = _ambient_temperature(hour, day_of_year)
    noise_T = rng.normal(0, cfg["noise_temp"], n)
    ambient_temp = T_amb_base + noise_T

    # ── 5. Temperatura del panel (fórmula física del examen) ────────────────
    # T_panel = T_amb + (NOCT - 20) / 800 * G + ε_T
    noct = cfg["noct"]
    T_panel = ambient_temp + (noct - 20.0) / 800.0 * irradiance
    T_panel += rng.normal(0, cfg["noise_temp"] * 0.5, n)

    # ── 6. Humedad y viento ─────────────────────────────────────────────────
    humidity = _humidity(hour, irradiance)
    humidity += rng.normal(0, cfg["noise_humidity"], n)
    humidity = np.clip(humidity, 10, 100)

    wind_speed = _wind_speed(n, rng)

    # ── 7. Irradiancia efectiva según condición ─────────────────────────────
    fault_factors = cfg["fault_factors"]
    f_falla = np.array([fault_factors[c] for c in conditions])

    # Sombreado parcial: reduce irradiancia efectiva directamente
    G_eff = irradiance.copy()
    partial_shade_mask = conditions == "partial_shading"
    G_eff[partial_shade_mask] *= 0.65  # 35% bloqueado

    # Paneles sucios: reduce irradiancia efectiva levemente
    dirty_mask = conditions == "dirty_panels"
    G_eff[dirty_mask] *= 0.85

    # ── 8. Potencia generada (fórmula física del examen) ───────────────────
    # P = η · A · G_eff · [1 + γ · (T_panel - 25)] · f_falla + ε_P
    eta = cfg["panel_efficiency"]
    A = cfg["panel_area"]
    gamma = cfg["temp_coeff"]

    P_base = eta * A * G_eff * (1 + gamma * (T_panel - 25)) * f_falla
    noise_P = rng.normal(0, cfg["noise_power"], n)
    power = np.clip(P_base + noise_P, 0, None)

    # ── 9. Voltaje y corriente (para clasificación, no para regresión) ──────
    # Modelo simplificado: V_oc base con corrección por temperatura e irradiancia
    V_oc_base = 36.0  # Voltaje de circuito abierto típico
    V_mpp_factor = 0.80  # Fracción del V_oc en punto de máxima potencia

    voltage = V_oc_base * V_mpp_factor + (irradiance / 1000.0) * 2.0
    voltage += rng.normal(0, cfg["noise_voltage"], n)

    # Fallas que afectan voltaje
    panel_fail_mask = conditions == "panel_failure"
    inverter_fail_mask = conditions == "inverter_failure"
    voltage[panel_fail_mask] *= 0.60 + rng.normal(0, 0.05, panel_fail_mask.sum())
    voltage[inverter_fail_mask] *= 0.15 + rng.normal(0, 0.03, inverter_fail_mask.sum())
    voltage = np.clip(voltage, 0, 50)

    # Corriente: I = P / V (con protección ante división por cero)
    current = np.where(voltage > 0.5, power / voltage, 0.0)
    current += rng.normal(0, cfg["noise_current"], n)
    current = np.clip(current, 0, 15)

    # ── 10. Agregar overlap realista entre clases ───────────────────────────
    # Añadir ruido adicional controlado para que las clases no sean perfectamente
    # separables (overlap moderado requerido por el examen)
    overlap_noise_scale = 0.03  # 3% de ruido extra en irradiancia
    irradiance += rng.normal(0, irradiance.mean() * overlap_noise_scale, n)
    irradiance = np.clip(irradiance, 0, 1200)

    power += rng.normal(0, power.std() * 0.02, n)
    power = np.clip(power, 0, None)

    # ── 11. Construir DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame({
        "hour": hour.astype(int),
        "day_of_year": day_of_year.astype(int),
        "irradiance_W_m2": np.round(irradiance, 2),
        "ambient_temp_C": np.round(ambient_temp, 2),
        "panel_temp_C": np.round(T_panel, 2),
        "humidity_percent": np.round(humidity, 2),
        "wind_speed_m_s": np.round(wind_speed, 3),
        "voltage_V": np.round(voltage, 3),
        "current_A": np.round(current, 4),
        "power_W": np.round(power, 2),
        "condition": conditions,
    })

    logger.info(f"Dataset generado: {df.shape[0]} filas × {df.shape[1]} columnas")
    logger.info(f"Distribución final:\n{df['condition'].value_counts()}")
    logger.info(f"Estadísticas de power_W: mean={df['power_W'].mean():.1f}, std={df['power_W'].std():.1f}")

    return df


def save_dataset(df: pd.DataFrame, path: Path = None) -> Path:
    """
    Guarda el dataset generado en disco.

    Args:
        df: DataFrame del dataset simulado.
        path: Ruta de destino. Por defecto usa DATASET_PATH de utils.

    Returns:
        Ruta donde fue guardado el archivo.
    """
    if path is None:
        path = DATASET_PATH

    ensure_dirs()
    df.to_csv(path, index=False)
    logger.info(f"Dataset guardado en: {path}")
    return path


def load_or_generate_dataset(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Carga el dataset desde disco si existe, o lo genera si no existe.

    Args:
        force_regenerate: Si True, regenera el dataset aunque exista.

    Returns:
        DataFrame con el dataset.
    """
    if not force_regenerate and DATASET_PATH.exists():
        logger.info(f"Cargando dataset existente desde: {DATASET_PATH}")
        return pd.read_csv(DATASET_PATH)

    logger.info("Generando nuevo dataset...")
    df = simulate_photovoltaic_dataset()
    save_dataset(df)
    return df


if __name__ == "__main__":
    df = simulate_photovoltaic_dataset()
    save_dataset(df)
    print("\n=== RESUMEN DEL DATASET ===")
    print(df.describe())
    print("\nDistribución de clases:")
    print(df["condition"].value_counts())
