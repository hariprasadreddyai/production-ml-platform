"""Tests for the serving API, using the stub predictor (no model download)."""

import pytest
from fastapi.testclient import TestClient

from serving.app import create_app
from serving.inference import ABRouter
from serving.model_manager import ModelManager

TEST_CONFIG = {
    "ab_testing": {"enabled": True, "stable_weight": 0.9, "canary_weight": 0.1},
    "rate_limiting": {"requests_per_minute": 100000, "burst": 1000},
}


@pytest.fixture
def client():
    manager = ModelManager(router=ABRouter(seed=42))
    return TestClient(create_app(config=TEST_CONFIG, manager=manager))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_version"] == "v1.0.0"


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["stable_version"] == "v1.0.0"
    assert body["labels"] == ["negative", "neutral", "positive"]
    assert body["ab_testing_enabled"] is True
    assert body["stable_weight"] == 0.9
    assert body["canary_weight"] == 0.1


def test_predict_positive(client):
    response = client.post("/predict", json={"text": "I love this product"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0.0 < body["confidence"] <= 1.0
    assert body["model_version"]
    assert body["latency_ms"] >= 0


def test_predict_negative(client):
    response = client.post("/predict", json={"text": "This is terrible and broken"})
    assert response.status_code == 200
    assert response.json()["label"] == "negative"


def test_predict_neutral(client):
    response = client.post("/predict", json={"text": "The box arrived on Tuesday"})
    assert response.status_code == 200
    assert response.json()["label"] == "neutral"


def test_predict_empty_text_rejected(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_batch(client):
    response = client.post(
        "/predict/batch",
        json={"texts": ["I love it", "This is awful", "The box is brown"]},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert [p["label"] for p in predictions] == ["positive", "negative", "neutral"]


def test_predict_batch_empty_rejected(client):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json={"text": "I love it"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mlserve_requests_total" in response.text
    assert "mlserve_prediction_latency_seconds" in response.text


def test_ab_routing_is_deterministic(client):
    first = client.post(
        "/predict", json={"text": "I love it", "request_id": "req-123"}
    ).json()["model_version"]
    second = client.post(
        "/predict", json={"text": "Totally different words here", "request_id": "req-123"}
    ).json()["model_version"]
    assert first == second


def test_reload_stub(client):
    response = client.post(
        "/model/reload",
        json={"slot": "canary", "model_path": "stub", "version": "stub-2.0.0"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "reloaded"
    info = client.get("/model/info").json()
    assert info["canary_version"] == "stub-2.0.0"


def test_reload_invalid_slot_rejected(client):
    response = client.post(
        "/model/reload",
        json={"slot": "prod", "model_path": "stub", "version": "x"},
    )
    assert response.status_code == 422


def test_rate_limit_blocks_when_bucket_empty():
    from serving.app import create_app as _create_app

    tight_config = {
        "ab_testing": {"enabled": False},
        "rate_limiting": {"requests_per_minute": 60, "burst": 2},
    }
    app = _create_app(config=tight_config, manager=ModelManager())
    with TestClient(app) as limited:
        assert limited.get("/health").status_code == 200
        assert limited.get("/health").status_code == 200
        assert limited.get("/health").status_code == 429
