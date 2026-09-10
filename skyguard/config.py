"""
Configuration and Physical Constants for SkyGuard AI.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PhysicsConstants:
    """Standard atmospheric thermodynamic constants."""
    EPSILON_A: float = 0.622          # Ratio of molecular weight of water vapor to dry air (Rd / Rv)
    KELVIN_OFFSET: float = 273.15      # 0°C in Kelvin
    
    # Magnus-Tetens saturation vapor pressure constants over water (0 to 50°C standard)
    TETENS_A: float = 6.112           # hPa
    TETENS_B: float = 17.67           # dimensionless
    TETENS_C: float = 243.5           # °C
    
    # Atmospheric Refractive index constants
    REFRACTIVE_K1: float = 7.76e-5    # K / hPa
    REFRACTIVE_K2: float = 3.73e-1    # K^2 / hPa


@dataclass
class DACMConfig:
    """Parameters for Dynamic Advective Coupling and Verification Mechanism."""
    SPATIAL_DECAY_GAMMA: float = 40.0   # km (baseline distance decay scale)
    WIND_ADVECTION_ALPHA: float = 0.35  # s/m (wind speed sensitivity)
    CALM_WIND_THRESHOLD: float = 0.5    # m/s (below this, wind direction is calm/unreliable)
    STABILITY_WINDOW_SIZE: int = 6      # Timesteps for circular wind direction stability

    # --- Proximity-First Station Selection (Stage 1) ---
    # Only stations within this radius are eligible as DACM candidates.
    # Wind direction plays NO role in this gate — it is purely geographic.
    # Per master spec §62: "Do not allow a favorable wind direction alone to make
    # an extremely distant station a high-confidence verification partner."
    PROXIMITY_RADIUS_KM: float = 80.0   # Hard geographic gate in km

    # After proximity filtering, retain at most this many nearest neighbors.
    # This prevents dilution from many stations all just inside the radius boundary.
    TOP_K_NEIGHBORS: int = 6

    # --- Wind-Enhanced Coupling (Stage 2, within the proximity pool) ---
    # Minimum alignment to label a link as "is_downstream" in the coupling result.
    MIN_ALIGNMENT_DOWNSTREAM: float = 0.30

    # Outer fallback guard — absolute maximum even for Stage 1
    MAX_CANDIDATE_DISTANCE_KM: float = 150.0

    PROPAGATION_UNCERTAINTY_FRAC: float = 0.35 # Relative uncertainty on travel time τ_AB
    MIN_UNCERTAINTY_MINUTES: float = 5.0 # Absolute minimum delay uncertainty window


@dataclass
class ModelConfig:
    """Neural Network hyperparameters for Channel 1 and Channel 2."""
    TEMPORAL_WINDOW_SIZE: int = 10     # Number of consecutive timesteps per sequence
    INPUT_DIM: int = 3                 # [T, P, RH]
    LATENT_DIM: int = 16               # Latent space dimension
    HIDDEN_DIM: int = 32               # Hidden layer size
    
    # Physics Loss Weights (Normalized)
    LAMBDA_REC: float = 1.0            # Reconstruction loss weight
    LAMBDA_TV: float = 0.5             # Virtual temperature consistency weight
    LAMBDA_N: float = 0.3              # Refractive index consistency weight
    LAMBDA_E: float = 0.4              # Vapor pressure consistency weight
    LAMBDA_LATENT: float = 0.01        # Latent regularization weight
    
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 50
    BATCH_SIZE: int = 32


@dataclass
class FusionThresholds:
    """Thresholds for evidence fusion and classification."""
    LOCAL_ANOMALY_PERCENTILE: float = 95.0   # Baseline calibrated percentile
    HIGH_LOCAL_ANOMALY_THRESHOLD: float = 0.65
    HIGH_DACM_COUPLING_THRESHOLD: float = 0.40
    HIGH_PROPAGATION_THRESHOLD: float = 0.50
    SENSOR_FAULT_PROP_MAX: float = 0.25
    CALM_COUPLING_THRESHOLD: float = 0.15


# Global instances
PHYSICS = PhysicsConstants()
DACM_CONFIG = DACMConfig()
MODEL_CONFIG = ModelConfig()
THRESHOLDS = FusionThresholds()
