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
