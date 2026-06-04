"""Pydantic schemas for the inference API.

The :class:`ChurnRequest` mirrors the raw column layout of the Telco Customer
Churn dataset so callers can post records as they appear in the source CSV.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GenderLiteral = Literal["Male", "Female"]
YesNoLiteral = Literal["Yes", "No"]
YesNoNoServiceLiteral = Literal["Yes", "No", "No internet service"]
MultipleLinesLiteral = Literal["Yes", "No", "No phone service"]
InternetServiceLiteral = Literal["DSL", "Fiber optic", "No"]
ContractLiteral = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodLiteral = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class ChurnRequest(BaseModel):
    """Single-customer prediction request payload.

    The field names and value literals match the Telco Customer Churn dataset's
    raw CSV columns verbatim, so the request can be passed straight into the
    fitted preprocessor.
    """

    model_config = ConfigDict(extra="forbid")

    gender: GenderLiteral = Field(..., description="Customer gender.")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, else 0.")
    Partner: YesNoLiteral = Field(..., description="Whether the customer has a partner.")
    Dependents: YesNoLiteral = Field(..., description="Whether the customer has dependents.")
    tenure: int = Field(..., ge=0, description="Months the customer has stayed with the company.")
    PhoneService: YesNoLiteral
    MultipleLines: MultipleLinesLiteral
    InternetService: InternetServiceLiteral
    OnlineSecurity: YesNoNoServiceLiteral
    OnlineBackup: YesNoNoServiceLiteral
    DeviceProtection: YesNoNoServiceLiteral
    TechSupport: YesNoNoServiceLiteral
    StreamingTV: YesNoNoServiceLiteral
    StreamingMovies: YesNoNoServiceLiteral
    Contract: ContractLiteral
    PaperlessBilling: YesNoLiteral
    PaymentMethod: PaymentMethodLiteral
    MonthlyCharges: float = Field(..., ge=0.0)
    TotalCharges: float = Field(..., ge=0.0)


class ChurnResponse(BaseModel):
    """Single-customer prediction response payload."""

    model_config = ConfigDict(extra="forbid")

    churn: bool = Field(..., description="Predicted churn label.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted churn probability.")


class HealthResponse(BaseModel):
    """Response payload for the ``/health`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
