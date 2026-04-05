from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.rate_limit import rate_limiter


def test_predict_single_returns_probability_and_risk(client, sample_transactions):
    response = client.post("/predict", json=sample_transactions[0])

    assert response.status_code == 200
    payload = response.json()
    assert payload["transaction_id"] == sample_transactions[0]["transaction_id"]
    assert 0.0 <= payload["fraud_probability"] <= 1.0
    assert payload["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert payload["prediction"] in {"LEGITIMATE", "FRAUD"}
    assert len(payload["top_risk_factors"]) == 3


def test_predict_batch_returns_aggregate_fields(client, sample_transactions):
    response = client.post("/predict/batch", json={"transactions": sample_transactions})

    assert response.status_code == 200
    payload = response.json()
    assert payload["batch_size"] == 2
    assert len(payload["predictions"]) == 2
    assert payload["fraud_count"] >= 0
    assert 0.0 <= payload["fraud_rate"] <= 1.0


def test_predict_batch_rejects_oversized_request(client, sample_transactions):
    oversized = [sample_transactions[0] for _ in range(101)]
    response = client.post("/predict/batch", json={"transactions": oversized})

    assert response.status_code == 422


def test_predict_requires_api_key_when_configured(monkeypatch, sample_transactions):
    monkeypatch.setenv("API_KEY", "test-api-key")
    get_settings.cache_clear()
    rate_limiter.clear()

    try:
        with TestClient(app) as client:
            unauthorized = client.post("/predict", json=sample_transactions[0])
            authorized = client.post(
                "/predict",
                json=sample_transactions[0],
                headers={"X-API-Key": "test-api-key"},
            )
    finally:
        rate_limiter.clear()
        get_settings.cache_clear()

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_predict_rate_limit_returns_429(monkeypatch, sample_transactions):
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("RATE_LIMIT_PREDICT_PER_MINUTE", "1")
    get_settings.cache_clear()
    rate_limiter.clear()

    try:
        with TestClient(app) as client:
            headers = {
                "X-API-Key": "test-api-key",
                "X-Forwarded-For": "198.51.100.7",
            }
            first = client.post("/predict", json=sample_transactions[0], headers=headers)
            second = client.post("/predict", json=sample_transactions[0], headers=headers)
    finally:
        rate_limiter.clear()
        get_settings.cache_clear()

    assert first.status_code == 200
    assert second.status_code == 429
