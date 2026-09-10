"""
Dual-Channel SkyGuard Model: Channel 1 (Temporal) + Channel 2 (Physics-Informed).
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from skyguard.models.temporal_channel import TemporalSequenceAE
from skyguard.models.physics_channel import PhysicsInformedAE
from skyguard.physics.residuals import PhysicsResidualCalculator


class DualChannelSkyGuardModel(nn.Module):
    """
    Unified dual-channel architecture orchestrating:
    - Channel 1: Temporal dynamics over sequence window [t-k ... t]
    - Channel 2: Atmospheric thermodynamics on latest snapshot X_t
    """

    def __init__(
        self,
        temporal_model: Optional[TemporalSequenceAE] = None,
        physics_model: Optional[PhysicsInformedAE] = None,
        weight_temp: float = 0.5,
        weight_phys: float = 0.5,
    ):
        super().__init__()
        self.temporal_channel = temporal_model or TemporalSequenceAE()
        self.physics_channel = physics_model or PhysicsInformedAE()
        self.weight_temp = weight_temp
        self.weight_phys = weight_phys

    def compute_local_anomaly(
        self,
        sequence_norm: torch.Tensor,
        denorm_mean: np.ndarray,
        denorm_std: np.ndarray,
    ) -> Dict[str, Any]:
        """
        sequence_norm: (batch_size, seq_len, 3) or (1, seq_len, 3)
        returns dictionary of combined local anomaly evidence.
        """
        self.eval()
        with torch.no_grad():
            # 1. Channel 1: Temporal Evaluation
            ch1_out = self.temporal_channel.compute_anomaly_score(sequence_norm)
            score_temp = ch1_out["total_score"]

            # 2. Channel 2: Physics-Informed Snapshot Evaluation on latest step t
            latest_snapshot_norm = sequence_norm[:, -1, :]
            mean_tensor = torch.tensor(denorm_mean, dtype=torch.float32, device=sequence_norm.device)
            std_tensor = torch.tensor(denorm_std, dtype=torch.float32, device=sequence_norm.device)
            ch2_out = self.physics_channel.compute_anomaly_score(latest_snapshot_norm, mean_tensor, std_tensor)
            score_phys = ch2_out["ch2_total_score"]

            # 3. Composite Local Anomaly Evidence
            local_anomaly_score = self.weight_temp * score_temp + self.weight_phys * score_phys

            # 4. Variable-level attribution (Mean of Ch1 and Ch2 errors)
            var_err_temp = 0.5 * (ch1_out["error_temp"] + ch2_out["error_temp"])
            var_err_press = 0.5 * (ch1_out["error_press"] + ch2_out["error_press"])
            var_err_rh = 0.5 * (ch1_out["error_rh"] + ch2_out["error_rh"])

            # 5. Physical values and residuals
            obs_phys = np.array(latest_snapshot_norm.cpu().tolist()) * denorm_std + denorm_mean
            rec_phys = np.array(ch2_out["recon_physical"].cpu().tolist())

            residuals = PhysicsResidualCalculator.compute_residuals(
                obs_t_c=obs_phys[:, 0],
                obs_p_hpa=obs_phys[:, 1],
                obs_rh=obs_phys[:, 2],
                rec_t_c=rec_phys[:, 0],
                rec_p_hpa=rec_phys[:, 1],
                rec_rh=rec_phys[:, 2],
            )

            return {
                "local_anomaly_score": float(local_anomaly_score[0]),
                "temporal_score": float(score_temp[0]),
                "physics_score": float(score_phys[0]),
                "var_error_temperature": float(var_err_temp[0]),
                "var_error_pressure": float(var_err_press[0]),
                "var_error_humidity": float(var_err_rh[0]),
                "expected_temperature": float(rec_phys[0, 0]),
                "expected_pressure": float(rec_phys[0, 1]),
                "expected_humidity": float(rec_phys[0, 2]),
                "residuals": {k: float(v[0]) if hasattr(v, '__len__') else float(v) for k, v in residuals.items()},
            }
