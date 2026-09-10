"""
Unit Tests for Atmospheric Thermodynamics and Differentiable Physics Loss.
"""

import unittest
import numpy as np
import torch

from skyguard.physics.thermodynamics import (
    saturation_vapor_pressure,
    actual_vapor_pressure,
    virtual_temperature,
    atmospheric_refractive_index,
)
from skyguard.physics.loss import PhysicsConsistencyLoss
from skyguard.config import PHYSICS


class TestAtmosphericThermodynamics(unittest.TestCase):

    def test_saturation_vapor_pressure(self):
        # At 0°C (273.15 K), e_s should be ~6.11 hPa
        e_s_0 = saturation_vapor_pressure(273.15)
        self.assertAlmostEqual(e_s_0, 6.112, places=2)

        # At 20°C (293.15 K), e_s should be ~23.38 hPa
        e_s_20 = saturation_vapor_pressure(293.15)
        self.assertTrue(22.0 < e_s_20 < 25.0)

    def test_actual_vapor_pressure(self):
        # At 20°C, 50% RH -> e ≈ 0.5 * 23.38 ≈ 11.69 hPa
        e_20 = actual_vapor_pressure(50.0, 293.15)
        self.assertTrue(11.0 < e_20 < 12.5)

    def test_virtual_temperature(self):
        # Moist air is less dense than dry air, so T_v > T
        t_k = 300.0  # ~26.85°C
        p = 1013.25
        rh = 80.0
        t_v = virtual_temperature(t_k, p, rh)
        self.assertTrue(t_v > t_k, f"Expected T_v ({t_v}) > T_K ({t_k})")
        self.assertTrue(t_v < t_k + 10.0, f"Virtual temperature difference is within realistic atmospheric bounds")

    def test_atmospheric_refractive_index(self):
        # Standard radio refractive index n is typically ~1.000300 to 1.000400
        t_k = 293.15
        p = 1013.25
        rh = 60.0
        n = atmospheric_refractive_index(t_k, p, rh)
        self.assertTrue(1.00025 < n < 1.00050)

    def test_differentiable_physics_loss(self):
        # Test backpropagation gradients in PyTorch
        loss_fn = PhysicsConsistencyLoss()
        
        # Batch of 4 snapshots: [T_norm, P_norm, RH_norm]
        x_target = torch.randn(4, 3, requires_grad=False)
        x_pred = torch.randn(4, 3, requires_grad=True)

        loss = loss_fn(x_target, x_pred)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.item() >= 0.0)

        # Backward pass
        loss.backward()
        self.assertIsNotNone(x_pred.grad)
        self.assertTrue(torch.all(torch.isfinite(x_pred.grad)))
        self.assertFalse(torch.all(x_pred.grad == 0.0))


if __name__ == "__main__":
    unittest.main()
