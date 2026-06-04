"""Data fetcher for the Telco Customer Churn dataset.

Downloads the dataset from Kaggle using the official Kaggle API when credentials
are available, and falls back to a public mirror otherwise. The raw CSV is
written to ``data/raw/telco_churn.csv``.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "blastchar/telco-customer-churn"
KAGGLE_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PUBLIC_MIRROR = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_FILE = RAW_DIR / "telco_churn.csv"


def _ensure_dirs() -> None:
    """Create the ``data/raw`` directory if it does not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _has_kaggle_credentials() -> bool:
    """Return ``True`` if Kaggle API credentials are configured.

    Credentials are considered configured if either the ``KAGGLE_USERNAME`` and
    ``KAGGLE_KEY`` environment variables are set, or a ``kaggle.json`` file is
    present in the user's Kaggle config directory.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_path = Path.home() / ".kaggle" / "kaggle.json"
    return config_path.exists()


def _download_via_kaggle(target_dir: Path) -> Optional[Path]:
    """Download the dataset using the Kaggle API.

    Args:
        target_dir: Directory the dataset will be extracted into.

    Returns:
        Path to the extracted CSV file on success, ``None`` otherwise.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.warning("kaggle package not installed, skipping Kaggle download")
        return None

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            KAGGLE_DATASET, path=str(target_dir), unzip=False, quiet=False
        )
    except Exception as exc:  # noqa: BLE001 - surface any Kaggle error
        logger.warning("Kaggle download failed: %s", exc)
        return None

    archives = list(target_dir.glob("*.zip"))
    if not archives:
        return None

    with zipfile.ZipFile(archives[0]) as zf:
        zf.extractall(target_dir)
    archives[0].unlink(missing_ok=True)

    extracted = target_dir / KAGGLE_FILE
    return extracted if extracted.exists() else None


def _download_via_mirror(target_path: Path) -> Path:
    """Download the dataset from the public mirror.

    Args:
        target_path: Destination CSV path.

    Returns:
        The path the file was saved to.

    Raises:
        RuntimeError: If the HTTP request fails or returns a non-200 status.
    """
    logger.info("Downloading dataset from public mirror: %s", PUBLIC_MIRROR)
    response = requests.get(PUBLIC_MIRROR, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Mirror download failed with status {response.status_code}"
        )
    target_path.write_bytes(response.content)
    return target_path


def fetch(force: bool = False) -> Path:
    """Fetch the Telco Customer Churn dataset.

    Args:
        force: If ``True``, re-download even when a cached copy is present.

    Returns:
        Path to the local CSV file containing the raw dataset.

    Raises:
        RuntimeError: If no source is reachable.
    """
    _ensure_dirs()

    if RAW_FILE.exists() and not force:
        logger.info("Using cached dataset at %s", RAW_FILE)
        return RAW_FILE

    if _has_kaggle_credentials():
        logger.info("Attempting Kaggle download for %s", KAGGLE_DATASET)
        extracted = _download_via_kaggle(RAW_DIR)
        if extracted is not None:
            shutil.move(str(extracted), RAW_FILE)
            logger.info("Saved dataset to %s", RAW_FILE)
            return RAW_FILE
        logger.warning("Kaggle path failed, falling back to public mirror")

    _download_via_mirror(RAW_FILE)
    logger.info("Saved dataset to %s", RAW_FILE)
    return RAW_FILE


def main() -> int:
    """CLI entrypoint for ``python -m data.fetch``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    force = "--force" in sys.argv
    path = fetch(force=force)
    print(f"Dataset available at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
