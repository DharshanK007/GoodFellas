"""
Baseline Models for Benchmark Comparison & Ablation Studies.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from skyguard.data.schema import AWSObservation


class ThresholdQCBaseline:
    """Conventional threshold and rate-of-change Quality Control rules."""

    def __init__(
        self,
        t_range: tuple = (-40.0, 50.0),
        p_range: tuple = (870.0, 1085.0),
        rh_range: tuple = (0.0, 100.0),
        max_rate_t_per_hour: float = 8.0,
        max_rate_p_per_hour: float = 6.0,
    ):
        self.t_min, self.t_max = t_range
        self.p_min, self.p_max = p_range
        self.rh_min, self.rh_max = rh_range
        self.max_rate_t = max_rate_t_per_hour
        self.max_rate_p = max_rate_p_per_hour

    def evaluate(self, current: AWSObservation, previous: Optional[AWSObservation] = None) -> Dict[str, Any]:
        anomalous = False
        reasons = []

        if not (self.t_min <= current.temperature <= self.t_max):
            anomalous = True
            reasons.append("Temperature out of bounds")

        if not (self.p_min <= current.pressure <= self.p_max):
            anomalous = True
            reasons.append("Pressure out of bounds")

        if not (self.rh_min <= current.humidity <= self.rh_max):
            anomalous = True
            reasons.append("Humidity out of bounds")

        if previous is not None:
            dt_hours = (current.timestamp - previous.timestamp).total_seconds() / 3600.0
            if 0.0 < dt_hours <= 2.0:
                rate_t = abs(current.temperature - previous.temperature) / dt_hours
                rate_p = abs(current.pressure - previous.pressure) / dt_hours
                if rate_t > self.max_rate_t:
                    anomalous = True
                    reasons.append(f"Temperature step change ({rate_t:.1f}°C/hr) exceeded")
                if rate_p > self.max_rate_p:
                    anomalous = True
                    reasons.append(f"Pressure step change ({rate_p:.1f}hPa/hr) exceeded")

        return {
            "is_anomaly": anomalous,
            "anomaly_score": 1.0 if anomalous else 0.0,
            "reasons": reasons,
        }


class StatisticalIsolationForestBaseline:
    """Standard statistical unsupervised anomaly detector (Isolation Forest)."""

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        """X: (N, 3) representing [T, P, RH]"""
        self.model.fit(X)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly scores: higher = more anomalous."""
        if not self.fitted:
            raise ValueError("Isolation Forest is not fitted.")
        # decision_function: lower values mean more abnormal
        scores = -self.model.decision_function(X)
        return scores


class StandardMultivariateAE(nn.Module):
    """
    Standard Unconstrained Autoencoder (Ablation baseline without thermodynamic physics loss).
    Learns purely statistical correlations from [T, P, RH].
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Pure MSE reconstruction loss without physics."""
        x_recon = self.forward(x)
        return nn.functional.mse_loss(x_recon, x)
