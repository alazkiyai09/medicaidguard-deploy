def test_health_endpoint_reports_model_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_loaded"] is True
    assert payload["status"] == "healthy"
    assert payload["model_version"] == "1.0.0"
    assert payload["feature_count"] >= 10


def test_metrics_endpoint_updates_after_prediction(client, sample_transactions):
    before = client.get("/metrics")
    assert before.status_code == 200
    before_payload = before.json()

    client.post("/predict", json=sample_transactions[0])

    after = client.get("/metrics")
    assert after.status_code == 200
    after_payload = after.json()

    assert after_payload["total_predictions"] >= before_payload["total_predictions"] + 1
    assert after_payload["avg_inference_time_ms"] >= 0


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "MedicaidGuard Cloud Deploy API"
