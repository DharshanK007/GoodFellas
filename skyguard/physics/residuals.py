"""
Thermodynamic Residuals and Consistency Evidence.
"""

from typing import Dict, Union
import numpy as np
import torch
from skyguard.physics.thermodynamics import (
    virtual_temperature,
    atmospheric_refractive_index,
    actual_vapor_pressure,
)
from skyguard.config import PHYSICS


class PhysicsResidualCalculator:
    """
    Computes thermodynamic residuals between observed state X = [T, P, RH]
    and reconstructed/expected state X_hat = [T_hat, P_hat, RH_hat].
    """

    # Baseline scaling factors for normalization
    SCALE_TV = 5.0      # Typical virtual temp deviation scale in K
    SCALE_N = 2.0e-5    # Typical refractive index deviation scale
    SCALE_E = 4.0       # Typical vapor pressure deviation scale in hPa

    @classmethod
    def compute_residuals(
        cls,
        obs_t_c: Union[float, np.ndarray],
        obs_p_hpa: Union[float, np.ndarray],
        obs_rh: Union[float, np.ndarray],
        rec_t_c: Union[float, np.ndarray],
        rec_p_hpa: Union[float, np.ndarray],
        rec_rh: Union[float, np.ndarray],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculates raw and normalized thermodynamic residuals.
        """
        t_k_obs = obs_t_c + PHYSICS.KELVIN_OFFSET
        t_k_rec = rec_t_c + PHYSICS.KELVIN_OFFSET

        # 1. Virtual Temperature residual
        tv_obs = virtual_temperature(t_k_obs, obs_p_hpa, obs_rh)
        tv_rec = virtual_temperature(t_k_rec, rec_p_hpa, rec_rh)
        r_tv_raw = np.abs(tv_obs - tv_rec)
        r_tv_norm = r_tv_raw / cls.SCALE_TV

        # 2. Refractive Index residual
        n_obs = atmospheric_refractive_index(t_k_obs, obs_p_hpa, obs_rh)
        n_rec = atmospheric_refractive_index(t_k_rec, rec_p_hpa, rec_rh)
        r_n_raw = np.abs(n_obs - n_rec)
        r_n_norm = r_n_raw / cls.SCALE_N

        # 3. Vapor Pressure residual
        e_obs = actual_vapor_pressure(obs_rh, t_k_obs)
        e_rec = actual_vapor_pressure(rec_rh, t_k_rec)
        r_e_raw = np.abs(e_obs - e_rec)
        r_e_norm = r_e_raw / cls.SCALE_E

        # Aggregate composite physics inconsistency score
        composite_physics_score = (r_tv_norm + r_n_norm + r_e_norm) / 3.0

        return {
            "r_virtual_temp_k": r_tv_raw,
            "r_refractive_index": r_n_raw,
            "r_vapor_pressure_hpa": r_e_raw,
            "norm_r_tv": r_tv_norm,
            "norm_r_n": r_n_norm,
            "norm_r_e": r_e_norm,
            "composite_physics_inconsistency": composite_physics_score,
        }
