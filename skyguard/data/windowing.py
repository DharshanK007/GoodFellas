"""
Sliding Window Sequence Generator for Temporal ML Channels.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from skyguard.data.schema import AWSObservation, CanonicalDataset
from skyguard.data.preprocessing import AWSPreprocessor


class TemporalWindowGenerator:
    """
    Constructs sliding temporal windows of [T, P, RH] for sequence autoencoders
    while respecting station boundaries and temporal continuity.
    """

    def __init__(self, window_size: int = 10, preprocessor: Optional[AWSPreprocessor] = None):
        self.window_size = window_size
        self.preprocessor = preprocessor or AWSPreprocessor()

    def generate_station_windows(
        self, series: List[AWSObservation]
    ) -> Tuple[np.ndarray, List[AWSObservation]]:
        """
        Creates sequences of shape (num_windows, window_size, 3) for a single station's series.
        Also returns the reference target observation (the latest timestep t in the window).
        """
        if len(series) < self.window_size:
            return np.empty((0, self.window_size, 3), dtype=np.float32), []

        # Ensure series is sorted
        sorted_series = sorted(series, key=lambda x: x.timestamp)
        raw_vectors = np.array([obs.vector_3d for obs in sorted_series], dtype=np.float32)

        # Normalize features
        norm_vectors = self.preprocessor.normalize(raw_vectors)

        windows: List[np.ndarray] = []
        target_observations: List[AWSObservation] = []

        for i in range(len(sorted_series) - self.window_size + 1):
            window = norm_vectors[i : i + self.window_size]
            target_obs = sorted_series[i + self.window_size - 1]
            windows.append(window)
            target_observations.append(target_obs)

        return np.array(windows, dtype=np.float32), target_observations

    def generate_dataset_windows(
        self, dataset: CanonicalDataset
    ) -> Tuple[torch.Tensor, List[AWSObservation]]:
        """
        Processes entire dataset across all stations and collates into a PyTorch Tensor.
        """
        all_windows: List[np.ndarray] = []
        all_targets: List[AWSObservation] = []

        for stn_id in dataset.all_station_ids():
            stn_series = dataset.get_station_series(stn_id)
            windows, targets = self.generate_station_windows(stn_series)
            if len(windows) > 0:
                all_windows.append(windows)
                all_targets.extend(targets)

        if not all_windows:
            return torch.empty((0, self.window_size, 3), dtype=torch.float32), []

        combined = np.vstack(all_windows)
        tensor = torch.tensor(combined, dtype=torch.float32)
        return tensor, all_targets
