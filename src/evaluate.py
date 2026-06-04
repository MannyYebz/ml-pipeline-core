"""Evaluation metrics for the churn classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.model import ChurnClassifier


@dataclass(frozen=True)
class EvaluationResult:
    """Container for the metrics produced by :func:`evaluate`.

    Attributes:
        loss: Average BCE-with-logits loss across the loader.
        accuracy: Threshold-based classification accuracy.
        auc_roc: Area under the ROC curve.
        f1: F1 score at the chosen threshold.
        confusion_matrix: Nested list ``[[tn, fp], [fn, tp]]``.
        threshold: Probability cutoff used to derive class labels.
    """

    loss: float
    accuracy: float
    auc_roc: float
    f1: float
    confusion_matrix: List[List[int]]
    threshold: float

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable representation of the metrics."""
        return asdict(self)

    def scalar_metrics(self) -> Dict[str, float]:
        """Return just the scalar metrics (useful for MLflow logging)."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "f1": self.f1,
        }


@torch.no_grad()
def collect_predictions(
    model: ChurnClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference over a loader and return labels, probabilities, and loss.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding ``(features, labels)`` batches.
        device: Device to run computation on.

    Returns:
        A ``(y_true, y_prob, mean_loss)`` tuple.
    """
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += float(criterion(logits, y).item())
        total_count += int(y.numel())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    y_true = np.concatenate(labels).astype(np.int64)
    y_prob = np.concatenate(probs).astype(np.float64)
    mean_loss = total_loss / max(total_count, 1)
    return y_true, y_prob, mean_loss


def evaluate(
    model: ChurnClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Compute the full metric suite over a data loader.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding ``(features, labels)`` batches.
        device: Device to run computation on.
        threshold: Probability threshold used to derive class labels.

    Returns:
        An :class:`EvaluationResult` with loss, accuracy, AUC-ROC, F1, and the
        2x2 confusion matrix.
    """
    y_true, y_prob, mean_loss = collect_predictions(model, loader, device)
    y_pred = (y_prob >= threshold).astype(np.int64)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return EvaluationResult(
        loss=float(mean_loss),
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc_roc=auc,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=cm,
        threshold=float(threshold),
    )
