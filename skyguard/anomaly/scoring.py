"""
Anomaly Score Normalization and Threshold Calibration.
"""

from typing import List, Optional
import numpy as np


class AnomalyScoreCalibrator:
    """
    Fits empirical percentiles on clean validation anomaly scores
    to map raw reconstruction errors to calibrated [0, 1] probabilities.
    """

    def __init__(self, p90: float = 0.5, p95: float = 1.0, p99: float = 2.5):
        self.p90 = p90
        self.p95 = p95
        self.p99 = p99
        self.fitted = False

    def fit(self, val_scores: np.ndarray) -> None:
        """Computes 90th, 95th, 99th percentiles from validation scores."""
        clean = val_scores[np.isfinite(val_scores)]
        if len(clean) < 10:
            return
        self.p90 = float(np.percentile(clean, 90))
        self.p95 = float(np.percentile(clean, 95))
        self.p99 = float(np.percentile(clean, 99))
        self.fitted = True

    def calibrate(self, raw_score: float) -> float:
        """
        Maps a raw score to a normalized [0, 1] probability using smooth sigmoid-like interpolation.
        0.0 -> score at median
        0.5 -> score at p95
        1.0 -> score >= p99 * 2
        """
        if raw_score <= 0.0:
            return 0.0
        
        # Softmax / Sigmoid scaling centered at p95
        scale = max(0.1, (self.p99 - self.p90) / 2.0)
        norm = 1.0 / (1.0 + np.exp(-(raw_score - self.p95) / scale))
        return float(np.clip(norm, 0.0, 1.0))
