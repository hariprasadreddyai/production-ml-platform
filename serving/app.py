"""Serving API: async FastAPI inference with A/B testing and metrics.

Endpoints:
    GET  /health          liveness and loaded model version
    GET  /model/info      stable/canary versions and A/B weights
    POST /predict         single prediction, A/B routed
    POST /predict/batch   batch predictions
    POST /model/reload    hot-reload a model slot
    GET  /metrics         Prometheus metrics

Run with: uvicorn serving.app:app --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from monitoring.metrics_collector import (
    PREDICTION_LATENCY,
    REQUEST_COUNT,
    get_metrics,
)
from serving.inference import ABRouter
from serving.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from serving.model_manager import ModelManager
from serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReloadRequest,
)

logger = logging.getLogger(__name__)

LABELS = ["negative", "neutral", "positive"]


def load_serving_config(path: str = "configs/serving_config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Serving config %s not found, using defaults", path)
        return {}


def build_manager(config: dict | None = None) -> ModelManager:
    config = config or {}
    ab = config.get("ab_testing", {})
    router = ABRouter(
        stable_weight=ab.get("stable_weight", 0.9),
        canary_weight=ab.get("canary_weight", 0.1),
        enabled=ab.get("enabled", True),
    )
    manager = ModelManager(router=router)
    logger.info("Model manager ready (stub predictor, no model downloaded)")
    return manager


def create_app(config: dict | None = None,
               manager: ModelManager | None = None) -> FastAPI:
    config = config or load_serving_config()
    manager = manager or build_manager(config)
    rate = config.get("rate_limiting", {})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = manager
        app.state.config = config
        yield

    app = FastAPI(
        title="MLServe",
        description="Production ML serving API with A/B testing",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=rate.get("requests_per_minute", 1000),
        burst=rate.get("burst", 50),
    )

    def _predict_one(text: str, request_id: str | None) -> PredictResponse:
        start = time.perf_counter()
        result = manager.predict(text, request_id=request_id)
        latency_ms = (time.perf_counter() - start) * 1000
        REQUEST_COUNT.labels(
            endpoint="/predict", model_version=result["model_version"]
        ).inc()
        PREDICTION_LATENCY.observe(latency_ms / 1000)
        return PredictResponse(
            label=result["label"],
            confidence=result["confidence"],
            model_version=result["model_version"],
            latency_ms=round(latency_ms, 2),
        )

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="healthy",
            model_version=manager.versions["stable"],
            canary_enabled=manager.canary_enabled,
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info():
        return ModelInfoResponse(
            stable_version=manager.versions["stable"],
            canary_version=manager.versions["canary"],
            labels=LABELS,
            ab_testing_enabled=manager.router.enabled,
            stable_weight=manager.router.stable_weight,
            canary_weight=manager.router.canary_weight,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        return _predict_one(request.text, request.request_id)

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(request: BatchPredictRequest):
        predictions = [
            _predict_one(text, None) for text in request.texts
        ]
        return BatchPredictResponse(predictions=predictions)

    @app.post("/model/reload")
    def reload_model(request: ReloadRequest):
        try:
            manager.reload(request.slot, request.model_path, request.version)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "reloaded",
            "slot": request.slot,
            "version": request.version,
        }

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(
            generate_latest(get_metrics()), media_type=CONTENT_TYPE_LATEST
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    config = load_serving_config()
    serving = config.get("serving", {})
    uvicorn.run(
        "serving.app:app",
        host=serving.get("host", "0.0.0.0"),
        port=serving.get("port", 8000),
        workers=serving.get("workers", 1),
    )
