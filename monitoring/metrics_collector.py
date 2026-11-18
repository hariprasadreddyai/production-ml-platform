"""Prometheus metrics collector for the serving layer.

Defines the standard counters, histograms, and gauges the API and training
jobs update. get_metrics() returns the registry the /metrics endpoint
exposes.
"""

from __future__ import annotations

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REQUEST_COUNT = Counter(
    "mlserve_requests_total",
    "Total prediction requests",
    ["endpoint", "model_version"],
)

REQUEST_ERRORS = Counter(
    "mlserve_request_errors_total",
    "Total failed prediction requests",
    ["endpoint"],
)

PREDICTION_LATENCY = Histogram(
    "mlserve_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

MODEL_VERSION_INFO = Gauge(
    "mlserve_model_info",
    "Currently loaded model version (1 = active)",
    ["version", "slot"],
)

GPU_UTILIZATION = Gauge(
    "mlserve_gpu_utilization_percent",
    "GPU utilization percent (0 when no GPU is present)",
)

DRIFT_SCORE = Gauge(
    "mlserve_drift_score",
    "Latest drift score per check",
    ["check", "feature"],
)


def get_metrics() -> object:
    return REGISTRY


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)
