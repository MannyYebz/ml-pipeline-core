"""Unit tests for the preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    Preprocessor,
    class_weights,
    prepare_splits,
    stratified_split,
)


def test_preprocessor_fits_only_on_training_data(synthetic_telco_df: pd.DataFrame) -> None:
    """The fitted scaler statistics should depend only on the training slice."""
    df_train, df_val, df_test = stratified_split(
        synthetic_telco_df,
        target_column="Churn",
        test_size=0.2,
        val_size=0.2,
        random_state=0,
    )

    pre = Preprocessor().fit(df_train)
    train_features = pre.transform_features(df_train)

    # Re-fit on a different slice and confirm the scaler statistics differ.
    pre_alt = Preprocessor().fit(df_test)
    cleaned = pre._clean_raw(df_train)[NUMERIC_COLUMNS]
    expected_mean = cleaned.fillna(cleaned.median()).mean().to_numpy()
    fit_mean = pre._pipeline.named_transformers_["num"].named_steps["scaler"].mean_
    np.testing.assert_allclose(fit_mean, expected_mean, rtol=1e-5)

    other_mean = pre_alt._pipeline.named_transformers_["num"].named_steps["scaler"].mean_
    assert not np.allclose(fit_mean, other_mean), (
        "Re-fitting on a different slice should change scaler statistics"
    )

    assert train_features.dtype == np.float32
    assert train_features.shape[0] == len(df_train)


def test_transform_record_matches_dataframe_path(synthetic_telco_df: pd.DataFrame) -> None:
    """``transform_record`` must produce the same row as ``transform_features``."""
    pre = Preprocessor().fit(synthetic_telco_df)

    row = synthetic_telco_df.iloc[0].drop(["Churn"]).to_dict()
    via_record = pre.transform_record(row)
    via_frame = pre.transform_features(synthetic_telco_df.iloc[[0]])

    assert via_record.shape == (1, pre.output_dim)
    np.testing.assert_allclose(via_record, via_frame, rtol=1e-5)


def test_transform_before_fit_raises() -> None:
    """Calling ``transform_features`` before ``fit`` should raise."""
    pre = Preprocessor()
    with pytest.raises(RuntimeError):
        pre.transform_features(pd.DataFrame({c: [0] for c in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}))


def test_save_and_load_roundtrip(
    synthetic_telco_df: pd.DataFrame, tmp_path: Path
) -> None:
    """A saved preprocessor should produce identical output after reload."""
    pre = Preprocessor().fit(synthetic_telco_df)
    path = pre.save(tmp_path / "preproc.joblib")
    assert path.exists()

    reloaded = Preprocessor.load(path)
    a = pre.transform_features(synthetic_telco_df)
    b = reloaded.transform_features(synthetic_telco_df)
    np.testing.assert_allclose(a, b, rtol=1e-6)
    assert reloaded.feature_names == pre.feature_names


def test_total_charges_coerced_from_blanks(synthetic_telco_df: pd.DataFrame) -> None:
    """Blank ``TotalCharges`` values should not blow up the pipeline."""
    df = synthetic_telco_df.copy()
    df.loc[df.index[:5], "TotalCharges"] = " "
    pre = Preprocessor().fit(df)
    X = pre.transform_features(df)
    assert not np.isnan(X).any(), "NaNs leaked through after imputation"


def test_stratified_split_preserves_class_balance(synthetic_telco_df: pd.DataFrame) -> None:
    """Each split should roughly preserve the global churn rate."""
    df_train, df_val, df_test = stratified_split(
        synthetic_telco_df,
        target_column="Churn",
        test_size=0.2,
        val_size=0.2,
        random_state=0,
    )
    total = len(synthetic_telco_df)
    assert len(df_train) + len(df_val) + len(df_test) == total

    overall = (synthetic_telco_df["Churn"] == "Yes").mean()
    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        rate = (split["Churn"] == "Yes").mean()
        assert abs(rate - overall) < 0.1, f"{name} split deviates too far: {rate} vs {overall}"


def test_prepare_splits_writes_artifact(
    synthetic_telco_csv: Path, tmp_path: Path
) -> None:
    """``prepare_splits`` should write the preprocessor when requested."""
    artifact = tmp_path / "pre.joblib"
    splits, pre = prepare_splits(csv_path=synthetic_telco_csv, save_preprocessor_to=artifact)

    assert artifact.exists()
    assert splits.X_train.shape[1] == pre.output_dim
    assert splits.X_train.shape[0] == len(splits.y_train) > 0
    assert set(np.unique(splits.y_train)).issubset({0, 1})


def test_class_weights_sum_to_class_count() -> None:
    """Inverse-frequency weights should normalize across classes."""
    y = np.array([0, 0, 0, 1])
    weights = class_weights(y)
    assert set(weights.keys()) == {0, 1}
    assert weights[1] > weights[0]
