"""
Atmospheric Thermodynamic Equations (NumPy & PyTorch Compatible).
"""

from typing import Union, Tuple, Dict
import numpy as np
import torch
from skyguard.config import PHYSICS


def saturation_vapor_pressure(
    temperature_k: Union[float, np.ndarray, torch.Tensor]
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes saturation vapor pressure e_s(T) in hPa using the Magnus-Tetens formula.
    Input temperature must be in Kelvin.
    """
    if isinstance(temperature_k, torch.Tensor):
        t_c = temperature_k - PHYSICS.KELVIN_OFFSET
        return PHYSICS.TETENS_A * torch.exp((PHYSICS.TETENS_B * t_c) / (t_c + PHYSICS.TETENS_C))
    elif isinstance(temperature_k, np.ndarray):
        t_c = temperature_k - PHYSICS.KELVIN_OFFSET
        return PHYSICS.TETENS_A * np.exp((PHYSICS.TETENS_B * t_c) / (t_c + PHYSICS.TETENS_C))
    else:
        t_c = float(temperature_k) - PHYSICS.KELVIN_OFFSET
        return float(PHYSICS.TETENS_A * np.exp((PHYSICS.TETENS_B * t_c) / (t_c + PHYSICS.TETENS_C)))


def actual_vapor_pressure(
    humidity_percent: Union[float, np.ndarray, torch.Tensor],
    temperature_k: Union[float, np.ndarray, torch.Tensor]
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes actual water vapor partial pressure e in hPa.
    e = (RH / 100) * e_s(T)
    """
    e_s = saturation_vapor_pressure(temperature_k)
    return (humidity_percent / 100.0) * e_s


def virtual_temperature(
    temperature_k: Union[float, np.ndarray, torch.Tensor],
    pressure_hpa: Union[float, np.ndarray, torch.Tensor],
    humidity_percent: Union[float, np.ndarray, torch.Tensor]
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes virtual temperature T_v in Kelvin:
    T_v = T_K / [ 1 - (1 - ε_a) * (e / P) ]
    where ε_a ≈ 0.622, and e is actual vapor pressure.
    """
    e = actual_vapor_pressure(humidity_percent, temperature_k)
    ratio = (e / pressure_hpa) * (1.0 - PHYSICS.EPSILON_A)
    
    if isinstance(temperature_k, torch.Tensor):
        # Clip denominator away from zero for numerical stability
        denom = torch.clamp(1.0 - ratio, min=0.01)
        return temperature_k / denom
    elif isinstance(temperature_k, np.ndarray):
        denom = np.clip(1.0 - ratio, 0.01, None)
        return temperature_k / denom
    else:
        denom = max(0.01, 1.0 - ratio)
        return float(temperature_k / denom)


def atmospheric_refractive_index(
    temperature_k: Union[float, np.ndarray, torch.Tensor],
    pressure_hpa: Union[float, np.ndarray, torch.Tensor],
    humidity_percent: Union[float, np.ndarray, torch.Tensor]
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes radio/optical refractive index n:
    n = 1 + 7.76e-5 * (P / T_K) + 3.73e-1 * (e / T_K^2)
    """
    e = actual_vapor_pressure(humidity_percent, temperature_k)
    term1 = PHYSICS.REFRACTIVE_K1 * (pressure_hpa / temperature_k)
    term2 = PHYSICS.REFRACTIVE_K2 * (e / (temperature_k ** 2))
    return 1.0 + term1 + term2


def compute_thermodynamic_state(
    t_celsius: Union[float, np.ndarray, torch.Tensor],
    p_hpa: Union[float, np.ndarray, torch.Tensor],
    rh_percent: Union[float, np.ndarray, torch.Tensor]
) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
    """
    Convenience function returning full physical state dictionary.
    """
    t_k = t_celsius + PHYSICS.KELVIN_OFFSET
    e_s = saturation_vapor_pressure(t_k)
    e = actual_vapor_pressure(rh_percent, t_k)
    t_v = virtual_temperature(t_k, p_hpa, rh_percent)
    n = atmospheric_refractive_index(t_k, p_hpa, rh_percent)

    return {
        "temperature_k": t_k,
        "sat_vapor_press_hpa": e_s,
        "actual_vapor_press_hpa": e,
        "virtual_temp_k": t_v,
        "refractive_index": n,
    }
