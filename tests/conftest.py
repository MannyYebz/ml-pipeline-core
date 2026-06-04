"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _synthetic_telco_frame(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic dataframe with the Telco Churn column layout."""
    rng = np.random.default_rng(seed)

    def pick(values: list[str]) -> np.ndarray:
        return rng.choice(values, size=n_rows)

    tenure = rng.integers(0, 72, size=n_rows)
    monthly = rng.uniform(18.0, 120.0, size=n_rows).round(2)
    total = (monthly * np.maximum(tenure, 1) + rng.normal(0, 10, size=n_rows)).round(2)

    churn_logit = (
        -2.0
        + 0.04 * (monthly - 60)
        - 0.05 * tenure
        + rng.normal(0, 0.5, size=n_rows)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logit))
    churn = np.where(rng.uniform(size=n_rows) < churn_prob, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"CUST{i:05d}" for i in range(n_rows)],
            "gender": pick(["Male", "Female"]),
            "SeniorCitizen": rng.integers(0, 2, size=n_rows),
            "Partner": pick(["Yes", "No"]),
            "Dependents": pick(["Yes", "No"]),
            "tenure": tenure,
            "PhoneService": pick(["Yes", "No"]),
            "MultipleLines": pick(["Yes", "No", "No phone service"]),
            "InternetService": pick(["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": pick(["Yes", "No", "No internet service"]),
            "OnlineBackup": pick(["Yes", "No", "No internet service"]),
            "DeviceProtection": pick(["Yes", "No", "No internet service"]),
            "TechSupport": pick(["Yes", "No", "No internet service"]),
            "StreamingTV": pick(["Yes", "No", "No internet service"]),
            "StreamingMovies": pick(["Yes", "No", "No internet service"]),
            "Contract": pick(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": pick(["Yes", "No"]),
            "PaymentMethod": pick(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total.astype(str),
            "Churn": churn,
        }
    )


@pytest.fixture()
def synthetic_telco_df() -> pd.DataFrame:
    """Return a synthetic Telco-shaped dataframe with 200 rows."""
    return _synthetic_telco_frame()


@pytest.fixture()
def synthetic_telco_csv(tmp_path: Path) -> Path:
    """Write the synthetic dataframe to a temp CSV and return its path."""
    path = tmp_path / "telco.csv"
    _synthetic_telco_frame().to_csv(path, index=False)
    return path
