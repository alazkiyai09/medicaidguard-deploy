import pytest

from app.config import Settings
from app.models.request import TransactionInput
from app.services.explainer import ExplainerService
from app.services.metrics_store import MetricsStore
from app.services.model_loader import ModelLoaderService
from app.services.predictor import PredictorService
from app.services.preprocessor import FeaturePreprocessor


def test_model_loader_loads_local_artifacts():
    settings = Settings(
        model_source="local",
        model_path="model/model.pkl",
        feature_names_path="model/feature_names.json",
        model_metadata_path="model/model_metadata.json",
    )

    artifacts = ModelLoaderService(settings).load()

    assert artifacts.metadata["model_version"] == "1.0.0"
    assert len(artifacts.feature_names) >= 10
    assert hasattr(artifacts.model, "predict_proba")


def test_model_loader_rejects_unknown_model_source():
    settings = Settings(model_source="invalid")
    with pytest.raises(ValueError):
        ModelLoaderService(settings).load()


def test_predictor_risk_level_boundaries():
    assert PredictorService._risk_level(0.1) == "LOW"
    assert PredictorService._risk_level(0.3) == "MEDIUM"
    assert PredictorService._risk_level(0.6) == "HIGH"
    assert PredictorService._risk_level(0.9) == "CRITICAL"


def test_predictor_batch_rejects_empty():
    settings = Settings()
    artifacts = ModelLoaderService(settings).load()
    preprocessor = FeaturePreprocessor()
    predictor = PredictorService(
        settings=settings,
        model=artifacts.model,
        model_version="1.0.0",
        feature_names=artifacts.feature_names,
        preprocessor=preprocessor,
        explainer=ExplainerService(settings, artifacts.model, artifacts.feature_names),
        metrics_store=MetricsStore(startup_ts=0.0, fraud_threshold=0.5, model_version="1.0.0"),
    )

    with pytest.raises(ValueError):
        predictor.predict_batch([])

    txn = TransactionInput(
        transaction_id="TXN-UNIT",
        provider_id="PRV-U",
        claim_amount=1000,
        procedure_code="99213",
        diagnosis_code="J06.9",
        provider_type="individual",
        patient_age=35,
        claim_frequency_30d=2,
        avg_claim_amount_90d=900,
        unique_patients_30d=15,
        billing_pattern_score=0.2,
    )
    assert predictor.predict_one(txn).transaction_id == "TXN-UNIT"
