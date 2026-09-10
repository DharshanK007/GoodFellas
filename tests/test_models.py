"""
Unit Tests for Channel 1 & Channel 2 PyTorch Autoencoders.
"""

import unittest
import torch
import numpy as np

from skyguard.models.temporal_channel import TemporalSequenceAE
from skyguard.models.physics_channel import PhysicsInformedAE
from skyguard.models.dual_channel import DualChannelSkyGuardModel


class TestModels(unittest.TestCase):

    def test_temporal_channel_forward(self):
        batch_size = 8
        seq_len = 10
        input_dim = 3
        model = TemporalSequenceAE(seq_len=seq_len)
        x = torch.randn(batch_size, seq_len, input_dim)

        recon_seq, z_temp = model(x)
        self.assertEqual(recon_seq.shape, (batch_size, seq_len, input_dim))
        self.assertEqual(z_temp.shape[0], batch_size)

        scores = model.compute_anomaly_score(x)
        self.assertEqual(scores["total_score"].shape, (batch_size,))
        self.assertEqual(scores["error_temp"].shape, (batch_size,))

    def test_physics_channel_forward_and_loss(self):
        batch_size = 8
        model = PhysicsInformedAE()
        x = torch.randn(batch_size, 3)

        recon, z_r = model(x)
        self.assertEqual(recon.shape, (batch_size, 3))
        self.assertEqual(z_r.shape[0], batch_size)

        loss, loss_dict = model.compute_loss(x)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_rec", loss_dict)
        self.assertIn("loss_phys", loss_dict)

    def test_dual_channel_model(self):
        dual = DualChannelSkyGuardModel()
        seq = torch.randn(1, 10, 3)
        mean = np.array([20.0, 1013.25, 60.0])
        std = np.array([8.0, 12.0, 18.0])

        evidence = dual.compute_local_anomaly(seq, mean, std)
        self.assertIn("local_anomaly_score", evidence)
        self.assertIn("temporal_score", evidence)
        self.assertIn("physics_score", evidence)
        self.assertIn("residuals", evidence)
        self.assertIn("expected_temperature", evidence)


if __name__ == "__main__":
    unittest.main()
