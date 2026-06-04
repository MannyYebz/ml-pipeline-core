"""PyTorch ``Dataset`` and ``DataLoader`` helpers for the churn pipeline."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ChurnDataset(Dataset):
    """In-memory ``Dataset`` wrapping a preprocessed feature matrix.

    The features and labels are converted to tensors once at construction so
    repeated indexing during training is allocation-free.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Initialize from numpy arrays produced by the preprocessor.

        Args:
            features: 2-D ``float32``-compatible array of shape
                ``(n_samples, n_features)``.
            labels: 1-D integer array of binary labels.

        Raises:
            ValueError: If ``features`` and ``labels`` have mismatched lengths.
        """
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels length mismatch: {len(features)} vs {len(labels)}"
            )
        self._features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self._labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return int(self._features.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(features, label)`` tensor pair at ``idx``."""
        return self._features[idx], self._labels[idx]

    @property
    def input_dim(self) -> int:
        """Number of feature columns in the wrapped matrix."""
        return int(self._features.shape[1])


def build_dataloader(
    dataset: ChurnDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a ``DataLoader`` with the pipeline's standard settings.

    Args:
        dataset: The :class:`ChurnDataset` to wrap.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle each epoch (set ``True`` for training only).
        num_workers: Worker processes for data loading.

    Returns:
        A configured ``DataLoader``.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
