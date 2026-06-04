"""Centralized configuration for the ML pipeline.

All tunable values — paths, hyperparameters, MLflow settings — live here so the
rest of the codebase imports configuration rather than hard-coding constants.
Values can be overridden through environment variables for deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    """Read an integer-valued environment variable with a fallback."""
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    """Read a float-valued environment variable with a fallback."""
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _env_str(name: str, default: str) -> str:
    """Read a string-valued environment variable with a fallback."""
    return os.environ.get(name, default)


@dataclass(frozen=True)
class DataConfig:
    """Filesystem paths and split ratios for the dataset."""

    raw_path: Path = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    target_column: str = "Churn"
    id_column: str = "customerID"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the feedforward churn classifier."""

    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    use_batch_norm: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Optimization hyperparameters for the training loop."""

    learning_rate: float = _env_float("LEARNING_RATE", 1e-3)
    weight_decay: float = _env_float("WEIGHT_DECAY", 1e-5)
    batch_size: int = _env_int("BATCH_SIZE", 64)
    epochs: int = _env_int("EPOCHS", 30)
    patience: int = _env_int("PATIENCE", 5)
    num_workers: int = _env_int("NUM_WORKERS", 0)
    device: str = _env_str("TRAIN_DEVICE", "auto")


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow tracking server and registry settings."""

    tracking_uri: str = _env_str(
        "MLFLOW_TRACKING_URI",
        f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}",
    )
    experiment_name: str = _env_str("MLFLOW_EXPERIMENT", "telco-churn")
    registered_model_name: str = _env_str(
        "MLFLOW_REGISTERED_MODEL", "telco-churn-classifier"
    )
    production_stage: str = "Production"
    staging_stage: str = "Staging"


@dataclass(frozen=True)
class APIConfig:
    """Runtime settings for the FastAPI service."""

    host: str = _env_str("API_HOST", "0.0.0.0")
    port: int = _env_int("API_PORT", 8000)
    model_stage: str = _env_str("API_MODEL_STAGE", "Production")


@dataclass(frozen=True)
class Config:
    """Top-level container aggregating the per-component configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    api: APIConfig = field(default_factory=APIConfig)


CONFIG = Config()
