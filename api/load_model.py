"""Loads the production model and preprocessor for the API service.

The :class:`ChurnPredictor` is constructed once on FastAPI startup and reused
for every request. It resolves the model from the MLflow Model Registry
(falling back to the local checkpoint when MLflow is unavailable, which is
useful for tests and local smoke runs) and reads the preprocessor artifact
that was saved during training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Tuple

import mlflow
import numpy as np
import torch

from src.config import CONFIG, Config
from src.model import ChurnClassifier
from src.preprocess import Preprocessor
from src.registry import get_latest_version, model_uri_for_stage

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Thread-safe wrapper around the preprocessor and the loaded model."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        model: ChurnClassifier,
        version: Optional[str],
        threshold: float = 0.5,
    ) -> None:
        """Initialize with already-loaded components.

        Args:
            preprocessor: Fitted preprocessor from training.
            model: Loaded ``ChurnClassifier`` (eval mode is enforced).
            version: Model registry version identifier (or ``None`` if loaded
                from a local checkpoint).
            threshold: Probability threshold for the boolean output.
        """
        self._preprocessor = preprocessor
        self._model = model.eval()
        self._device = next(model.parameters()).device
        self._version = version
        self._threshold = float(threshold)

    @property
    def version(self) -> Optional[str]:
        """Model registry version (or ``None`` when loaded from a checkpoint)."""
        return self._version

    @property
    def threshold(self) -> float:
        """Probability threshold used for the boolean output."""
        return self._threshold

    def predict(self, record: Mapping[str, object]) -> Tuple[bool, float]:
        """Run inference for a single request payload.

        Args:
            record: Raw feature mapping (matches ``ChurnRequest.model_dump()``).

        Returns:
            A ``(churn, probability)`` tuple.
        """
        features = self._preprocessor.transform_record(record)
        tensor = torch.from_numpy(features).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            prob = float(torch.sigmoid(logits).item())
        return bool(prob >= self._threshold), prob


def _load_preprocessor(config: Config) -> Preprocessor:
    """Load the saved preprocessor artifact from disk."""
    path = config.data.artifacts_dir / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {path}. Train the model first."
        )
    return Preprocessor.load(path)


def _load_model_from_registry(
    config: Config,
) -> Tuple[Optional[ChurnClassifier], Optional[str]]:
    """Try to load the production model from the MLflow registry.

    Args:
        config: Pipeline configuration.

    Returns:
        ``(model, version)`` on success, ``(None, None)`` on failure.
    """
    try:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        version = get_latest_version(config.api.model_stage)
        if version is None:
            version = get_latest_version(stage=None)
        if version is None:
            return None, None
        uri = model_uri_for_stage(config.api.model_stage)
        try:
            model = mlflow.pytorch.load_model(uri)
        except Exception:  # noqa: BLE001 - try by-version URI if stage alias fails
            model = mlflow.pytorch.load_model(
                f"models:/{config.mlflow.registered_model_name}/{version.version}"
            )
        if not isinstance(model, ChurnClassifier):
            logger.warning(
                "Loaded model is %s, not ChurnClassifier; using anyway", type(model)
            )
        return model.eval(), str(version.version)
    except Exception as exc:  # noqa: BLE001 - registry is best-effort
        logger.warning("Could not load model from MLflow registry: %s", exc)
        return None, None


def _load_model_from_checkpoint(config: Config) -> ChurnClassifier:
    """Reconstruct the model from the local PyTorch checkpoint."""
    path: Path = config.data.artifacts_dir / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No registry model and no local checkpoint at {path}. Train first."
        )
    ckpt = torch.load(path, map_location="cpu")
    model = ChurnClassifier.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def build_predictor(config: Config = CONFIG) -> ChurnPredictor:
    """Construct the API's :class:`ChurnPredictor`.

    Loading strategy:
        1. MLflow Model Registry — preferred, gives version traceability.
        2. Local PyTorch checkpoint — fallback for local development.

    Args:
        config: Pipeline configuration.

    Returns:
        A ready-to-serve predictor.
    """
    preprocessor = _load_preprocessor(config)
    model, version = _load_model_from_registry(config)
    if model is None:
        logger.info("Falling back to local checkpoint for model weights")
        model = _load_model_from_checkpoint(config)
        version = None
    logger.info("Predictor ready (version=%s)", version)
    return ChurnPredictor(preprocessor=preprocessor, model=model, version=version)
