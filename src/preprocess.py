"""Feature engineering and preprocessing for the Telco Churn dataset.

The :class:`Preprocessor` is fit on training data only and persisted with
``joblib``. The same artifact is reloaded at serving time so the features seen
by the model in production exactly match those seen during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CONFIG, DataConfig

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


@dataclass(frozen=True)
class SplitData:
    """Container for processed train/val/test splits.

    Attributes:
        X_train: Training feature matrix.
        y_train: Training labels (0/1).
        X_val: Validation feature matrix.
        y_val: Validation labels.
        X_test: Test feature matrix.
        y_test: Test labels.
        feature_names: Names of the columns in the processed feature matrices.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]

    @property
    def input_dim(self) -> int:
        """Return the number of input features expected by the model."""
        return int(self.X_train.shape[1])


class Preprocessor:
    """Fit/transform pipeline for the Telco Customer Churn dataset.

    Numeric columns are imputed with the median and standard-scaled; categorical
    columns are imputed with the most-frequent value and one-hot encoded. The
    binary ``Churn`` target is mapped from ``Yes/No`` to ``1/0``.
    """

    def __init__(self, data_cfg: DataConfig = CONFIG.data) -> None:
        """Initialize with the dataset's column configuration.

        Args:
            data_cfg: Data configuration providing target/id column names.
        """
        self._cfg = data_cfg
        self._pipeline: ColumnTransformer = self._build_pipeline()
        self._feature_names: List[str] = []
        self._fitted: bool = False

    @staticmethod
    def _build_pipeline() -> ColumnTransformer:
        """Construct the unfitted ``ColumnTransformer`` used internally."""
        numeric = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric, NUMERIC_COLUMNS),
                ("cat", categorical, CATEGORICAL_COLUMNS),
            ],
            remainder="drop",
        )

    @staticmethod
    def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw values and coerce dtypes.

        ``TotalCharges`` is shipped as a string with blanks for new customers;
        we coerce it to numeric. ``SeniorCitizen`` arrives as an integer flag
        but is semantically categorical.

        Args:
            df: Raw dataframe loaded from the CSV.

        Returns:
            A cleaned copy of the dataframe.
        """
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
        return df

    def _encode_target(self, series: pd.Series) -> np.ndarray:
        """Map ``Yes``/``No`` to ``1``/``0`` (passes through numeric labels)."""
        if series.dtype == object:
            return series.map({"Yes": 1, "No": 0}).astype("int64").to_numpy()
        return series.astype("int64").to_numpy()

    def fit(self, df_train: pd.DataFrame) -> "Preprocessor":
        """Fit the column transformer on the training dataframe.

        Args:
            df_train: Cleaned training dataframe including the target column.

        Returns:
            Self, to allow fluent ``fit(...).transform(...)`` style usage.
        """
        df_train = self._clean_raw(df_train)
        X = df_train[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        self._pipeline.fit(X)
        self._feature_names = list(self._pipeline.get_feature_names_out())
        self._fitted = True
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a raw feature dataframe into a model-ready matrix.

        Args:
            df: Dataframe containing at least the columns expected by the fitted
                pipeline. Extra columns are ignored.

        Returns:
            A dense ``float32`` ``ndarray`` of shape ``(n_samples, n_features)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform")
        df = self._clean_raw(df)
        X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        return self._pipeline.transform(X).astype(np.float32)

    def transform_record(self, record: Mapping[str, object]) -> np.ndarray:
        """Transform a single inference payload into a model-ready row.

        Args:
            record: Mapping from raw feature name to value (e.g. a Pydantic
                model's ``model_dump()``).

        Returns:
            A ``float32`` ``ndarray`` of shape ``(1, n_features)``.
        """
        df = pd.DataFrame([dict(record)])
        return self.transform_features(df)

    @property
    def feature_names(self) -> List[str]:
        """Names of the processed output features (post one-hot encoding)."""
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fit")
        return list(self._feature_names)

    @property
    def output_dim(self) -> int:
        """Number of columns produced by ``transform_features``."""
        return len(self.feature_names)

    def save(self, path: Path) -> Path:
        """Persist the fitted preprocessor with ``joblib``.

        Args:
            path: Destination file path. Parent directories are created.

        Returns:
            The path the artifact was written to.
        """
        if not self._fitted:
            raise RuntimeError("Refusing to save an unfitted Preprocessor")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved preprocessor to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        """Load a previously-saved :class:`Preprocessor` from disk.

        Args:
            path: Path returned by a prior call to :meth:`save`.

        Returns:
            The deserialized, fitted preprocessor.
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a Preprocessor: {type(obj)}")
        return obj


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Read the raw Telco CSV from disk.

    Args:
        csv_path: Path to the raw CSV produced by ``data/fetch.py``.

    Returns:
        Dataframe with the ``customerID`` column dropped.
    """
    df = pd.read_csv(csv_path)
    if CONFIG.data.id_column in df.columns:
        df = df.drop(columns=[CONFIG.data.id_column])
    return df


def stratified_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/val/test with target stratification.

    Args:
        df: Full dataframe including the target column.
        target_column: Column to stratify on.
        test_size: Fraction of the full dataset reserved for the test split.
        val_size: Fraction of the full dataset reserved for the val split.
        random_state: Seed for reproducibility.

    Returns:
        A ``(train, val, test)`` tuple of dataframes.
    """
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )
    relative_val = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=relative_val,
        stratify=df_trainval[target_column],
        random_state=random_state,
    )
    return df_train, df_val, df_test


def prepare_splits(
    csv_path: Path | None = None,
    save_preprocessor_to: Path | None = None,
) -> Tuple[SplitData, Preprocessor]:
    """End-to-end preprocessing: load CSV, split, fit, transform.

    Args:
        csv_path: Optional override for the raw CSV path; defaults to
            ``CONFIG.data.raw_path``.
        save_preprocessor_to: If provided, the fitted preprocessor is saved to
            this path.

    Returns:
        A tuple of ``(SplitData, Preprocessor)``. The preprocessor is fit on
        training data only.
    """
    cfg = CONFIG.data
    csv_path = csv_path or cfg.raw_path
    df = load_raw(csv_path)

    df_train, df_val, df_test = stratified_split(
        df,
        target_column=cfg.target_column,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    preprocessor = Preprocessor(cfg).fit(df_train)

    X_train = preprocessor.transform_features(df_train)
    X_val = preprocessor.transform_features(df_val)
    X_test = preprocessor.transform_features(df_test)
    y_train = preprocessor._encode_target(df_train[cfg.target_column])
    y_val = preprocessor._encode_target(df_val[cfg.target_column])
    y_test = preprocessor._encode_target(df_test[cfg.target_column])

    splits = SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=preprocessor.feature_names,
    )

    if save_preprocessor_to is not None:
        preprocessor.save(save_preprocessor_to)

    return splits, preprocessor


def class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute inverse-frequency class weights for an imbalanced binary target.

    Args:
        y: 1-D array of integer labels.

    Returns:
        Mapping from class label to its weight.
    """
    counts = np.bincount(y)
    total = counts.sum()
    return {int(c): float(total / (len(counts) * counts[c])) for c in range(len(counts))}
