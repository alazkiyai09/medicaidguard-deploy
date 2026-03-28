import sys
import types

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.dependencies import get_predictor
from app.main import app
from app.services.explainer import ExplainerService
from app.services.model_loader import ModelLoaderService


SAMPLE_TXN = {
    "transaction_id": "TXN-EDGE-001",
    "provider_id": "PRV-EDGE",
    "claim_amount": 1200.0,
    "procedure_code": "99213",
    "diagnosis_code": "J06.9",
    "provider_type": "individual",
    "patient_age": 40,
    "claim_frequency_30d": 3,
    "avg_claim_amount_90d": 900.0,
    "unique_patients_30d": 20,
    "billing_pattern_score": 0.3,
}


def test_explainer_fallback_when_shap_disabled():
    service = ExplainerService(Settings(shap_enabled=False), model=object(), feature_names=["a", "b", "c"])
    factors = service.explain({"a": 1.0, "b": 5.0, "c": 3.0})

    assert len(factors) == 3
    assert factors[0].feature == "b"


def test_explainer_with_mocked_shap(monkeypatch):
    class FakeTreeExplainer:
        def __init__(self, model) -> None:
            self.model = model

        def shap_values(self, rows):
            return [[0.1, -0.6, 0.2]]

    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(TreeExplainer=FakeTreeExplainer))

    service = ExplainerService(Settings(shap_enabled=True), model=object(), feature_names=["x", "y", "z"])
    factors = service.explain({"x": 1.0, "y": 2.0, "z": 3.0})

    assert len(factors) == 3
    assert factors[0].feature == "y"
    assert factors[0].direction == "low"


def test_model_loader_missing_local_model_raises():
    settings = Settings(model_source="local", model_path="model/does-not-exist.pkl")

    with pytest.raises(FileNotFoundError):
        ModelLoaderService(settings).load()


def test_model_loader_fallback_feature_and_metadata_paths():
    settings = Settings(
        model_source="local",
        model_path="model/model.pkl",
        feature_names_path="model/not-found-features.json",
        model_metadata_path="model/not-found-meta.json",
        model_version="9.9.9",
    )

    artifacts = ModelLoaderService(settings).load()

    assert len(artifacts.feature_names) >= 10
    assert artifacts.metadata["model_version"] == "9.9.9"


def test_predict_router_returns_500_on_predictor_exception(monkeypatch):
    class BrokenPredictor:
        def predict_one(self, payload):
            raise RuntimeError("boom")

    app.dependency_overrides[get_predictor] = lambda: BrokenPredictor()
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=SAMPLE_TXN)
            assert response.status_code == 500
            assert "boom" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_predict_batch_returns_400_when_service_batch_limit_hit(client, sample_transactions):
    predictor = client.app.state.predictor
    original_limit = predictor.settings.batch_max_size
    predictor.settings.batch_max_size = 1

    try:
        response = client.post("/predict/batch", json={"transactions": sample_transactions})
        assert response.status_code == 400
        assert "Batch size exceeds" in response.json()["detail"]
    finally:
        predictor.settings.batch_max_size = original_limit


def test_predict_returns_503_when_model_not_ready(client):
    original_predictor = client.app.state.predictor
    client.app.state.predictor = None

    try:
        response = client.post("/predict", json=SAMPLE_TXN)
        assert response.status_code == 503
    finally:
        client.app.state.predictor = original_predictor


def test_metrics_router_handles_runtime_state_error(client):
    class BrokenMetricsStore:
        def snapshot(self):
            raise RuntimeError("not this one")

    class RuntimeErrorMetricsStore:
        def snapshot(self):
            from app.dependencies import RuntimeStateError

            raise RuntimeStateError("metrics unavailable")

    original_metrics = client.app.state.metrics_store
    client.app.state.metrics_store = RuntimeErrorMetricsStore()

    try:
        response = client.get("/metrics")
        assert response.status_code == 503
        assert "metrics unavailable" in response.json()["detail"]
    finally:
        client.app.state.metrics_store = original_metrics

    # confirm non-RuntimeStateError still bubbles as 500
    with TestClient(app, raise_server_exceptions=False) as no_raise_client:
        no_raise_client.app.state.metrics_store = BrokenMetricsStore()
        response = no_raise_client.get("/metrics")
        assert response.status_code == 500
