"""MLflow Model Registry helpers.

Wraps the common workflows — registering a new version, promoting between
stages, and resolving the URI of the current production model — behind a small,
typed API.
"""

from __future__ import annotations

import logging
from typing import Optional

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from src.config import CONFIG, MLflowConfig

logger = logging.getLogger(__name__)


def _client(cfg: MLflowConfig = CONFIG.mlflow) -> MlflowClient:
    """Return an ``MlflowClient`` pointed at the configured tracking URI."""
    mlflow.set_tracking_uri(cfg.tracking_uri)
    return MlflowClient(tracking_uri=cfg.tracking_uri)


def register_model(
    run_id: str,
    artifact_path: str,
    cfg: MLflowConfig = CONFIG.mlflow,
) -> ModelVersion:
    """Register a logged model as a new version of the registered model.

    Args:
        run_id: The MLflow run ID that produced the model artifact.
        artifact_path: Path within the run's artifact tree where the model
            was logged (the ``artifact_path=`` argument to ``mlflow.pytorch.log_model``).
        cfg: MLflow configuration.

    Returns:
        The created :class:`ModelVersion`.
    """
    client = _client(cfg)
    try:
        client.create_registered_model(cfg.registered_model_name)
        logger.info("Created registered model %s", cfg.registered_model_name)
    except mlflow.exceptions.RestException:
        logger.debug("Registered model %s already exists", cfg.registered_model_name)
    except Exception as exc:  # noqa: BLE001 - tolerate file-store quirks
        logger.debug("create_registered_model raised: %s", exc)

    source = f"runs:/{run_id}/{artifact_path}"
    version = client.create_model_version(
        name=cfg.registered_model_name,
        source=source,
        run_id=run_id,
    )
    logger.info(
        "Registered %s version %s from %s",
        cfg.registered_model_name,
        version.version,
        source,
    )
    return version


def promote(
    version: str,
    stage: str,
    cfg: MLflowConfig = CONFIG.mlflow,
    archive_existing: bool = True,
) -> ModelVersion:
    """Promote a model version to ``Staging`` or ``Production``.

    Args:
        version: The version number to transition (as returned by
            :func:`register_model`).
        stage: Target stage, typically ``"Staging"`` or ``"Production"``.
        cfg: MLflow configuration.
        archive_existing: If ``True``, demote any current occupants of ``stage``
            to ``Archived``.

    Returns:
        The updated :class:`ModelVersion`.
    """
    client = _client(cfg)
    updated = client.transition_model_version_stage(
        name=cfg.registered_model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info(
        "Transitioned %s v%s to %s", cfg.registered_model_name, version, stage
    )
    return updated


def get_latest_version(
    stage: Optional[str] = None,
    cfg: MLflowConfig = CONFIG.mlflow,
) -> Optional[ModelVersion]:
    """Return the most recent model version, optionally filtered by stage.

    Args:
        stage: If provided, restrict to versions currently in this stage.
        cfg: MLflow configuration.

    Returns:
        The matching :class:`ModelVersion`, or ``None`` if none exists.
    """
    client = _client(cfg)
    stages = [stage] if stage else None
    try:
        versions = client.get_latest_versions(cfg.registered_model_name, stages=stages)
    except mlflow.exceptions.RestException:
        return None
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def production_model_uri(cfg: MLflowConfig = CONFIG.mlflow) -> str:
    """Return the ``models:/`` URI for the current production model.

    Falls back to the latest available version if no version is in the
    production stage yet.

    Args:
        cfg: MLflow configuration.

    Returns:
        A URI suitable for ``mlflow.pyfunc.load_model`` / ``mlflow.pytorch.load_model``.

    Raises:
        RuntimeError: If no versions of the registered model exist.
    """
    version = get_latest_version(cfg.production_stage, cfg=cfg)
    if version is None:
        version = get_latest_version(stage=None, cfg=cfg)
    if version is None:
        raise RuntimeError(
            f"No versions found for registered model '{cfg.registered_model_name}'"
        )
    return f"models:/{cfg.registered_model_name}/{version.version}"


def model_uri_for_stage(stage: str, cfg: MLflowConfig = CONFIG.mlflow) -> str:
    """Return the alias URI for a stage (e.g. ``models:/foo/Production``).

    Args:
        stage: Stage name (``"Production"``, ``"Staging"``, etc.).
        cfg: MLflow configuration.

    Returns:
        A ``models:/`` URI string.
    """
    return f"models:/{cfg.registered_model_name}/{stage}"
