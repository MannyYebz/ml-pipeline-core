"""FastAPI application exposing the churn classifier."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from api.load_model import ChurnPredictor, build_predictor
from api.schema import ChurnRequest, ChurnResponse, HealthResponse
from src.config import CONFIG

logger = logging.getLogger(__name__)


class _AppState:
    """Mutable container for the singletons attached to ``app.state``."""

    predictor: Optional[ChurnPredictor] = None
    load_error: Optional[str] = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan hook that loads the model on startup.

    Failures are captured rather than raised so the ``/health`` endpoint can
    surface a degraded state instead of the process refusing to start.
    """
    state: _AppState = app.state.app_state
    try:
        state.predictor = build_predictor()
        logger.info("Model loaded; API ready")
    except Exception as exc:  # noqa: BLE001 - surface via /health
        state.load_error = str(exc)
        logger.exception("Failed to load model on startup: %s", exc)
    try:
        yield
    finally:
        state.predictor = None


def create_app() -> FastAPI:
    """Application factory used by both uvicorn and the test client.

    Returns:
        A configured ``FastAPI`` application.
    """
    app = FastAPI(
        title="Telco Customer Churn Predictor",
        version="1.0.0",
        description="Predict customer churn probability from raw account features.",
        lifespan=_lifespan,
    )
    app.state.app_state = _AppState()

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness/readiness probe.

        Returns:
            ``status="ok"`` when the model loaded, else ``"degraded"``.
        """
        state: _AppState = app.state.app_state
        loaded = state.predictor is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_version=state.predictor.version if loaded else None,
        )

    @app.post("/predict", response_model=ChurnResponse, tags=["inference"])
    def predict(payload: ChurnRequest) -> ChurnResponse:
        """Predict churn for a single customer.

        Args:
            payload: Raw account features for one customer.

        Returns:
            A :class:`ChurnResponse` with the boolean prediction and probability.

        Raises:
            HTTPException: 503 if the model failed to load on startup.
        """
        state: _AppState = app.state.app_state
        if state.predictor is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded: {state.load_error or 'unknown error'}",
            )
        churn, probability = state.predictor.predict(payload.model_dump())
        return ChurnResponse(churn=churn, probability=probability)

    return app


app = create_app()


def main() -> int:
    """Run the API with uvicorn using the configured host/port."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "api.main:app",
        host=CONFIG.api.host,
        port=CONFIG.api.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
