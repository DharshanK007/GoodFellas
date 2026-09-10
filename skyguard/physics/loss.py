"""
Differentiable PyTorch Physics Consistency Loss Layer.
"""

from typing import Optional
import torch
import torch.nn as nn
from skyguard.physics.thermodynamics import (
    virtual_temperature,
    atmospheric_refractive_index,
    actual_vapor_pressure,
)
from skyguard.config import PHYSICS, MODEL_CONFIG


class PhysicsConsistencyLoss(nn.Module):
    """
    Computes end-to-end differentiable thermodynamic loss during model training.
    Enforces that reconstructed state [T_hat, P_hat, RH_hat] satisfies the physical
    equations derived from the true atmospheric state [T, P, RH].
    """

    def __init__(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        lambda_tv: float = MODEL_CONFIG.LAMBDA_TV,
        lambda_n: float = MODEL_CONFIG.LAMBDA_N,
        lambda_e: float = MODEL_CONFIG.LAMBDA_E,
    ):
        super().__init__()
        # Register buffers for normalization parameters
        default_mean = torch.tensor([20.0, 1013.25, 60.0], dtype=torch.float32)
        default_std = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float32)

        self.register_buffer("mean", mean if mean is not None else default_mean)
        self.register_buffer("std", std if std is not None else default_std)

        self.lambda_tv = lambda_tv
        self.lambda_n = lambda_n
        self.lambda_e = lambda_e

        # Normalization scale factors
        self.scale_tv = 5.0
        self.scale_n = 2.0e-5
        self.scale_e = 4.0

    def update_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean.copy_(mean)
        self.std.copy_(std)

    def denormalize(self, norm_tensor: torch.Tensor) -> torch.Tensor:
        """norm_tensor: (batch_size, 3) -> [T_c, P_hpa, RH]"""
        return norm_tensor * self.std + self.mean

    def forward(
        self, x_norm_target: torch.Tensor, x_norm_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates normalized physics consistency loss.
        """
        x_target = self.denormalize(x_norm_target)
        x_pred = self.denormalize(x_norm_pred)

        # Extract channels: 0 -> T (°C), 1 -> P (hPa), 2 -> RH (%)
        t_c_target, p_target, rh_target = x_target[:, 0], x_target[:, 1], x_target[:, 2]
        t_c_pred, p_pred, rh_pred = x_pred[:, 0], x_pred[:, 1], x_pred[:, 2]

        # Convert to Kelvin
        t_k_target = t_c_target + PHYSICS.KELVIN_OFFSET
        t_k_pred = t_c_pred + PHYSICS.KELVIN_OFFSET

        # Clamp RH and P for gradient stability during early training
        rh_target_clamped = torch.clamp(rh_target, min=1.0, max=100.0)
        rh_pred_clamped = torch.clamp(rh_pred, min=1.0, max=100.0)
        p_target_clamped = torch.clamp(p_target, min=800.0, max=1100.0)
        p_pred_clamped = torch.clamp(p_pred, min=800.0, max=1100.0)

        # 1. Virtual Temperature Consistency Loss
        tv_target = virtual_temperature(t_k_target, p_target_clamped, rh_target_clamped)
        tv_pred = virtual_temperature(t_k_pred, p_pred_clamped, rh_pred_clamped)
        loss_tv = torch.mean(torch.abs(tv_target - tv_pred)) / self.scale_tv

        # 2. Refractive Index Consistency Loss
        n_target = atmospheric_refractive_index(t_k_target, p_target_clamped, rh_target_clamped)
        n_pred = atmospheric_refractive_index(t_k_pred, p_pred_clamped, rh_pred_clamped)
        loss_n = torch.mean(torch.abs(n_target - n_pred)) / self.scale_n

        # 3. Vapor Pressure Consistency Loss
        e_target = actual_vapor_pressure(rh_target_clamped, t_k_target)
        e_pred = actual_vapor_pressure(rh_pred_clamped, t_k_pred)
        loss_e = torch.mean(torch.abs(e_target - e_pred)) / self.scale_e

        total_physics_loss = (
            self.lambda_tv * loss_tv +
            self.lambda_n * loss_n +
            self.lambda_e * loss_e
        )

        return total_physics_loss
