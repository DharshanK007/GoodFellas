"""
Channel 2 — Physics-Informed Atmospheric Relationship Autoencoder.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from skyguard.config import MODEL_CONFIG
from skyguard.physics.loss import PhysicsConsistencyLoss
from skyguard.physics.residuals import PhysicsResidualCalculator


class PhysicsInformedEncoder(nn.Module):
    """Encodes atmospheric snapshot [T, P, RH] into atmospheric relationship latent z_r."""
    def __init__(
        self,
        input_dim: int = MODEL_CONFIG.INPUT_DIM,
        hidden_dim: int = MODEL_CONFIG.HIDDEN_DIM,
        latent_dim: int = MODEL_CONFIG.LATENT_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsInformedDecoder(nn.Module):
    """Reconstructs atmospheric snapshot [T_hat, P_hat, RH_hat] from latent z_r."""
    def __init__(
        self,
        latent_dim: int = MODEL_CONFIG.LATENT_DIM,
        hidden_dim: int = MODEL_CONFIG.HIDDEN_DIM,
        output_dim: int = MODEL_CONFIG.INPUT_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z_r: torch.Tensor) -> torch.Tensor:
        return self.net(z_r)


class PhysicsInformedAE(nn.Module):
    """
    Channel 2 Model: Atmospheric Relationship Autoencoder constrained by
    thermodynamic relationships (virtual temperature, refractive index, vapor pressure).
    """
    def __init__(
        self,
        input_dim: int = MODEL_CONFIG.INPUT_DIM,
        hidden_dim: int = MODEL_CONFIG.HIDDEN_DIM,
        latent_dim: int = MODEL_CONFIG.LATENT_DIM,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = PhysicsInformedEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = PhysicsInformedDecoder(latent_dim, hidden_dim, input_dim)
        self.physics_loss_fn = PhysicsConsistencyLoss(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, 3) -> normalized [T, P, RH]
        returns: (x_recon, z_r)
        """
        z_r = self.encoder(x)
        x_recon = self.decoder(z_r)
        return x_recon, z_r

    def compute_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Total training loss = λ_rec * L_rec + λ_phys * L_phys + λ_latent * L_latent
        """
        x_recon, z_r = self.forward(x)

        # 1. Standard Reconstruction Loss (MSE)
        l_rec = nn.functional.mse_loss(x_recon, x)

        # 2. Physics Consistency Loss (Differentiable thermodynamic calculations)
        l_phys = self.physics_loss_fn(x, x_recon)

        # 3. Latent Regularization
        l_latent = torch.mean(torch.pow(z_r, 2.0))

        total_loss = (
            MODEL_CONFIG.LAMBDA_REC * l_rec +
            l_phys +
            MODEL_CONFIG.LAMBDA_LATENT * l_latent
        )

        return total_loss, {
            "total_loss": total_loss,
            "loss_rec": l_rec,
            "loss_phys": l_phys,
            "loss_latent": l_latent,
        }

    def compute_anomaly_score(
        self, x_norm: torch.Tensor, denorm_mean: torch.Tensor, denorm_std: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-time anomaly evaluation:
        Computes reconstruction errors per variable and thermodynamic residuals.
        """
        self.eval()
        with torch.no_grad():
            x_recon, z_r = self.forward(x_norm)

            sq_err = torch.pow(x_norm - x_recon, 2.0)
            err_temp = sq_err[:, 0]
            err_press = sq_err[:, 1]
            err_rh = sq_err[:, 2]
            rec_score = torch.mean(sq_err, dim=1)

            # Denormalize to physical units
            x_phys_obs = x_norm * denorm_std + denorm_mean
            x_phys_rec = x_recon * denorm_std + denorm_mean

            # Physics residuals
            phys_loss = self.physics_loss_fn(x_norm, x_recon)

            # Combined Channel 2 Anomaly Evidence
            ch2_score = 0.6 * rec_score + 0.4 * phys_loss

            return {
                "ch2_total_score": ch2_score,
                "reconstruction_score": rec_score,
                "physics_loss": phys_loss,
                "error_temp": err_temp,
                "error_press": err_press,
                "error_rh": err_rh,
                "recon_norm": x_recon,
                "recon_physical": x_phys_rec,
            }
