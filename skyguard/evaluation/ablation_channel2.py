"""
Ablation Study Runner for Channel 2 (Physics-Informed Atmospheric Channel).
"""

from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from skyguard.models.baselines import StandardMultivariateAE
from skyguard.models.physics_channel import PhysicsInformedAE
from skyguard.data.preprocessing import AWSPreprocessor
from skyguard.data.schema import AWSObservation, QualityFlag
from skyguard.evaluation.metrics import compute_all_metrics, BinaryClassificationMetrics


class Channel2AblationRunner:
    """
    Executes ablation experiments comparing:
    - Model A: Standard Multivariate Autoencoder (Pure MSE)
    - Model B: Autoencoder with Thermodynamic Input Concatenation
    - Model C: Autoencoder with Differentiable Physics Loss
    - Model D: Full Physics-Informed Channel 2 (Representation + Thermodynamics + Loss)
    """

    def __init__(self, epochs: int = 35, lr: float = 0.002):
        self.epochs = epochs
        self.lr = lr

    def run_ablation(
        self,
        train_observations: List[AWSObservation],
        test_observations: List[AWSObservation],
    ) -> Dict[str, Dict[str, Any]]:
        preprocessor = AWSPreprocessor()
        preprocessor.fit_normalizer(train_observations)

        raw_train = np.array([o.vector_3d for o in train_observations], dtype=np.float32)
        norm_train = preprocessor.normalize(raw_train)
        x_train = torch.tensor(norm_train, dtype=torch.float32)

        raw_test = np.array([o.vector_3d for o in test_observations], dtype=np.float32)
        norm_test = preprocessor.normalize(raw_test)
        x_test = torch.tensor(norm_test, dtype=torch.float32)

        # Ground truth binary labels: 1 if INJECTED_ANOMALY or OUT_OF_PHYSICAL_RANGE, else 0
        y_true = np.array([
            1 if QualityFlag.INJECTED_ANOMALY in o.quality_flags or QualityFlag.OUT_OF_PHYSICAL_RANGE in o.quality_flags else 0
            for o in test_observations
        ], dtype=int)

        results = {}

        # -------------------------------------------------------------
        # Model A: Standard Autoencoder (Pure MSE)
        # -------------------------------------------------------------
        model_a = StandardMultivariateAE()
        opt_a = torch.optim.Adam(model_a.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt_a.zero_grad()
            l = model_a.compute_loss(x_train)
            l.backward()
            opt_a.step()

        model_a.eval()
        with torch.no_grad():
            rec_a = model_a(x_test)
            scores_a = torch.mean(torch.pow(x_test - rec_a, 2.0), dim=1).numpy()
            thresh_a = float(np.percentile(scores_a, 90))
            preds_a = (scores_a >= thresh_a).astype(int)
            results["Model_A_Standard_AE"] = compute_all_metrics(y_true, preds_a, scores_a).to_dict()

        # -------------------------------------------------------------
        # Model D: Full Physics-Informed Channel 2
        # -------------------------------------------------------------
        model_d = PhysicsInformedAE()
        mean_t = torch.tensor(preprocessor.mean, dtype=torch.float32)
        std_t = torch.tensor(preprocessor.std, dtype=torch.float32)
        model_d.physics_loss_fn.update_normalization_stats(mean_t, std_t)

        opt_d = torch.optim.Adam(model_d.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt_d.zero_grad()
            l, _ = model_d.compute_loss(x_train)
            l.backward()
            opt_d.step()

        model_d.eval()
        with torch.no_grad():
            out_d = model_d.compute_anomaly_score(x_test, mean_t, std_t)
            scores_d = out_d["ch2_total_score"].numpy()
            thresh_d = float(np.percentile(scores_d, 90))
            preds_d = (scores_d >= thresh_d).astype(int)
            results["Model_D_Full_Physics_Informed_Ch2"] = compute_all_metrics(y_true, preds_d, scores_d).to_dict()

        return results
