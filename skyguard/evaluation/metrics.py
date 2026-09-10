"""
Evaluation Metrics for Anomaly Detection and Event-vs-Fault Discrimination.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class BinaryClassificationMetrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    false_alarm_rate: float
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "false_alarm_rate": round(self.false_alarm_rate, 4),
        }


def compute_all_metrics(
    y_true: np.ndarray, y_pred_binary: np.ndarray, y_scores: np.ndarray
) -> BinaryClassificationMetrics:
    """
    Computes standard classification and ranking metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred_binary, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_scores))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = float(fp / max(1, fp + tn))

    return BinaryClassificationMetrics(
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
        pr_auc=pr_auc,
        false_alarm_rate=far,
        confusion_matrix=cm.tolist(),
    )
