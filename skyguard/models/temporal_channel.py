"""
Channel 1 — Temporal Behavior Sequence Autoencoder.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from skyguard.config import MODEL_CONFIG


class TemporalEncoder(nn.Module):
    """Encodes a time sequence [t-k ... t] into a temporal latent vector z_temp."""
    def __init__(
        self,
        input_dim: int = MODEL_CONFIG.INPUT_DIM,
        hidden_dim: int = MODEL_CONFIG.HIDDEN_DIM,
        latent_dim: int = MODEL_CONFIG.LATENT_DIM,
        num_layers: int = 2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_latent = nn.Linear(hidden_dim * 2, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        out, h_n = self.gru(x)
        # Pool or take final step
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1) # (batch_size, hidden_dim * 2)
        z_temp = self.fc_latent(self.relu(last_hidden))
        return z_temp


class TemporalDecoder(nn.Module):
    """Reconstructs the time sequence from latent vector z_temp."""
    def __init__(
        self,
        latent_dim: int = MODEL_CONFIG.LATENT_DIM,
        hidden_dim: int = MODEL_CONFIG.HIDDEN_DIM,
        output_dim: int = MODEL_CONFIG.INPUT_DIM,
        seq_len: int = MODEL_CONFIG.TEMPORAL_WINDOW_SIZE,
        num_layers: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_temp: torch.Tensor) -> torch.Tensor:
        # z_temp: (batch_size, latent_dim)
        hidden_init = torch.relu(self.fc_init(z_temp)) # (batch_size, hidden_dim)
        # Repeat across sequence length
        repeated = hidden_init.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq_len, hidden_dim)
        out, _ = self.gru(repeated)
        recon_seq = self.fc_out(out) # (batch, seq_len, output_dim)
        return recon_seq


class TemporalSequenceAE(nn.Module):
    """
    Channel 1 Model: Learns normal temporal dynamics, diurnal patterns,
    rates of change, and temporal continuity.
    """
    def __init__(
        self,
        input_dim: int = MODEL_CONFIG.INPUT_DIM,
        hidden_dim: int = MODEL_CONFIG.HIDDEN_DIM,
        latent_dim: int = MODEL_CONFIG.LATENT_DIM,
        seq_len: int = MODEL_CONFIG.TEMPORAL_WINDOW_SIZE,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TemporalDecoder(latent_dim, hidden_dim, input_dim, seq_len)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, seq_len, 3)
        returns: (reconstructed_seq, z_temp)
        """
        z_temp = self.encoder(x)
        x_recon = self.decoder(z_temp)
        return x_recon, z_temp

    def compute_anomaly_score(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluates temporal reconstruction error for input sequence.
        Focuses error evaluation on the latest timestep t (or whole window).
        """
        self.eval()
        with torch.no_grad():
            x_recon, z_temp = self.forward(x)
            
            # Element-wise squared error across sequence
            sq_err_seq = torch.pow(x - x_recon, 2.0)
            
            # Latest timestep error
            latest_err = sq_err_seq[:, -1, :] # (batch, 3)
            err_temp = latest_err[:, 0]
            err_press = latest_err[:, 1]
            err_rh = latest_err[:, 2]
            
            total_score = torch.mean(latest_err, dim=1) # (batch,)

            return {
                "total_score": total_score,
                "error_temp": err_temp,
                "error_press": err_press,
                "error_rh": err_rh,
                "recon_latest": x_recon[:, -1, :],
                "recon_seq": x_recon,
            }
