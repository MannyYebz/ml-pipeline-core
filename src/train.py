"""Training loop with MLflow logging, checkpointing, and registry promotion."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CONFIG, Config
from src.dataset import ChurnDataset, build_dataloader
from src.evaluate import EvaluationResult, evaluate
from src.model import ChurnClassifier
from src.preprocess import class_weights, prepare_splits
from src.registry import promote, register_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Filesystem locations of artifacts produced by :func:`train`.

    Attributes:
        checkpoint_path: Best-by-val-AUC PyTorch checkpoint.
        preprocessor_path: Saved ``Preprocessor`` joblib file.
        metrics_path: JSON file with the final test-set metrics.
        run_id: MLflow run ID for the training run.
        registered_version: Version number of the registered model.
    """

    checkpoint_path: Path
    preprocessor_path: Path
    metrics_path: Path
    run_id: str
    registered_version: str


def resolve_device(preference: str) -> torch.device:
    """Resolve a config ``device`` string to a concrete ``torch.device``.

    Args:
        preference: ``"auto"``, ``"cpu"``, ``"cuda"``, or a ``cuda:N`` string.

    Returns:
        The selected device.
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(
    model: ChurnClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean batch loss.

    Args:
        model: Model in training mode.
        loader: Training data loader.
        optimizer: Optimizer driving the parameter updates.
        criterion: Loss function (expects logits).
        device: Compute device.

    Returns:
        The mean per-sample training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * int(y.numel())
        running_count += int(y.numel())
    return running_loss / max(running_count, 1)


def _pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute the ``pos_weight`` tensor for BCE-with-logits on imbalanced data."""
    weights = class_weights(y_train)
    ratio = weights[1] / weights[0] if weights[0] > 0 else 1.0
    return torch.tensor([ratio], device=device, dtype=torch.float32)


def train(config: Config = CONFIG) -> TrainingArtifacts:
    """Run the end-to-end training pipeline.

    Steps:
        1. Load and preprocess the dataset.
        2. Build a model and optimizer.
        3. Train with early stopping on validation AUC.
        4. Log loss and AUC-ROC to MLflow every epoch.
        5. Save the best checkpoint and evaluate it on the test set.
        6. Register the model and promote it to ``Staging``.

    Args:
        config: Top-level configuration to use.

    Returns:
        Paths and IDs for the artifacts produced.
    """
    _set_seeds(config.data.random_state)
    device = resolve_device(config.train.device)
    logger.info("Training on device: %s", device)

    artifacts_dir = config.data.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = artifacts_dir / "preprocessor.joblib"
    checkpoint_path = artifacts_dir / "best_model.pt"
    metrics_path = artifacts_dir / "test_metrics.json"

    splits, preprocessor = prepare_splits(
        csv_path=config.data.raw_path,
        save_preprocessor_to=preprocessor_path,
    )
    logger.info(
        "Splits: train=%d, val=%d, test=%d, features=%d",
        len(splits.y_train),
        len(splits.y_val),
        len(splits.y_test),
        splits.input_dim,
    )

    train_loader = build_dataloader(
        ChurnDataset(splits.X_train, splits.y_train),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    val_loader = build_dataloader(
        ChurnDataset(splits.X_val, splits.y_val),
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    test_loader = build_dataloader(
        ChurnDataset(splits.X_test, splits.y_test),
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    model = ChurnClassifier(input_dim=splits.input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(splits.y_train, device))

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    best_auc = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "batch_size": config.train.batch_size,
                "epochs": config.train.epochs,
                "patience": config.train.patience,
                "hidden_dims": json.dumps(list(model.hidden_dims)),
                "dropout": model.dropout,
                "use_batch_norm": model.use_batch_norm,
                "input_dim": model.input_dim,
            }
        )

        for epoch in range(1, config.train.epochs + 1):
            train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_result: EvaluationResult = evaluate(model, val_loader, device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_result.loss,
                    "val_auc_roc": val_result.auc_roc,
                    "val_accuracy": val_result.accuracy,
                    "val_f1": val_result.f1,
                },
                step=epoch,
            )
            logger.info(
                "epoch=%03d train_loss=%.4f val_loss=%.4f val_auc=%.4f val_f1=%.4f",
                epoch,
                train_loss,
                val_result.loss,
                val_result.auc_roc,
                val_result.f1,
            )

            improved = val_result.auc_roc > best_auc
            if improved:
                best_auc = val_result.auc_roc
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": model.config_dict(),
                        "epoch": epoch,
                        "val_auc": best_auc,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.train.patience:
                logger.info(
                    "Early stopping at epoch %d (best epoch=%d, best AUC=%.4f)",
                    epoch,
                    best_epoch,
                    best_auc,
                )
                break

        ckpt = torch.load(checkpoint_path, map_location=device)
        model = ChurnClassifier.from_config(ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_result = evaluate(model, test_loader, device)
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_result.scalar_metrics().items()}
        )
        metrics_path.write_text(json.dumps(test_result.to_dict(), indent=2))
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessing")
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

        model_artifact_path = "model"
        mlflow.pytorch.log_model(
            pytorch_model=model.cpu(),
            artifact_path=model_artifact_path,
        )
        model.to(device)

        version = register_model(run_id=run_id, artifact_path=model_artifact_path)
        promote(version=version.version, stage=config.mlflow.staging_stage)

    logger.info(
        "Best val AUC=%.4f at epoch %d. Test metrics: %s",
        best_auc,
        best_epoch,
        test_result.scalar_metrics(),
    )

    return TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
        metrics_path=metrics_path,
        run_id=run_id,
        registered_version=str(version.version),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``python -m src.train``."""
    parser = argparse.ArgumentParser(description="Train the churn classifier.")
    parser.add_argument(
        "--promote-to-production",
        action="store_true",
        help="After training and Staging registration, also promote to Production.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    artifacts = train()
    if args.promote_to_production:
        promote(version=artifacts.registered_version, stage=CONFIG.mlflow.production_stage)
        logger.info("Promoted version %s to Production", artifacts.registered_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
