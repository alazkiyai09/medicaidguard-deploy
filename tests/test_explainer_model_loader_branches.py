import hashlib
import pickle
import sys
import types
from pathlib import Path

import pytest

from app.config import Settings
from app.services.explainer import ExplainerService
from app.services.model_loader import ModelLoaderService
from app.services.preprocessor import FEATURE_NAMES
from app.services.simple_model import SimpleFraudModel


class UnsafeModel:
    pass


def test_explainer_shap_exception_falls_back_to_default(monkeypatch):
    class RaisingTreeExplainer:
        def __init__(self, model) -> None:
            self.model = model

        def shap_values(self, rows):
            raise RuntimeError("shap fail")

    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(TreeExplainer=RaisingTreeExplainer))

    service = ExplainerService(Settings(shap_enabled=True), model=object(), feature_names=["a", "b", "c"])
    factors = service.explain({"a": 2.0, "b": 1.0, "c": 0.5})

    assert len(factors) == 3
    assert factors[0].feature == "a"


def test_explainer_get_shap_explainer_named_steps_and_cache(monkeypatch):
    captured = {}

    class FakeTreeExplainer:
        def __init__(self, model) -> None:
            captured["model"] = model

        def shap_values(self, rows):
            return [[0.2, 0.1]]

    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(TreeExplainer=FakeTreeExplainer))

    class FakePipeline:
        named_steps = {"scaler": object(), "model": "LAST_STEP"}

    service = ExplainerService(Settings(shap_enabled=True), model=FakePipeline(), feature_names=["x", "y"])
    first = service._get_shap_explainer()
    second = service._get_shap_explainer()

    assert first is second
    assert captured["model"] == "LAST_STEP"


def test_explainer_extract_shap_vector_variants():
    service = ExplainerService(Settings(shap_enabled=False), model=object(), feature_names=[])

    class WithToList:
        def tolist(self):
            return [[0.1, 0.2, 0.3]]

    assert service._extract_shap_vector([[0.5, 0.2]]) == [0.5, 0.2]
    assert service._extract_shap_vector([[[0.7, 0.4]]]) == [0.7, 0.4]
    assert service._extract_shap_vector(WithToList()) == [0.1, 0.2, 0.3]


def test_model_loader_gcs_download_path(monkeypatch, tmp_path):
    feature_names = FEATURE_NAMES.copy()
    model = SimpleFraudModel(feature_order=feature_names, weights={}, intercept=0.0)
    payload = pickle.dumps(model)
    payload_sha256 = hashlib.sha256(payload).hexdigest()

    class FakeBlob:
        def exists(self):
            return True

        def download_to_filename(self, filename: str):
            Path(filename).write_bytes(payload)

    class FakeBucket:
        def blob(self, key: str):
            return FakeBlob()

    class FakeClient:
        def bucket(self, bucket: str):
            return FakeBucket()

    fake_storage = types.SimpleNamespace(Client=lambda: FakeClient())
    google_cloud = types.ModuleType("google.cloud")
    google_cloud.storage = fake_storage

    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud)

    settings = Settings(
        model_source="gcs",
        gcs_bucket="demo-bucket",
        gcs_model_key="models/model.pkl",
        model_sha256=payload_sha256,
        feature_names_path="model/feature_names.json",
        model_metadata_path="model/model_metadata.json",
    )

    artifacts = ModelLoaderService(settings).load()

    assert hasattr(artifacts.model, "predict_proba")
    assert len(artifacts.feature_names) >= 10


def test_model_loader_non_list_feature_names_and_non_dict_metadata(tmp_path):
    weird_features = tmp_path / "features.json"
    weird_metadata = tmp_path / "metadata.json"

    weird_features.write_text('{"unexpected": true}', encoding="utf-8")
    weird_metadata.write_text('[1, 2, 3]', encoding="utf-8")

    settings = Settings(
        model_source="local",
        model_path="model/model.pkl",
        feature_names_path=str(weird_features),
        model_metadata_path=str(weird_metadata),
        model_version="2.0.0",
    )

    artifacts = ModelLoaderService(settings).load()

    assert artifacts.feature_names == FEATURE_NAMES
    assert artifacts.metadata["model_version"] == "2.0.0"


def test_model_loader_rejects_untrusted_pickle_globals(tmp_path):
    model_path = tmp_path / "unsafe.pkl"
    metadata_path = tmp_path / "metadata.json"
    model_path.write_bytes(pickle.dumps(UnsafeModel()))
    metadata_path.write_text('{"model_version":"1.0.0"}', encoding="utf-8")

    settings = Settings(
        model_source="local",
        model_path=str(model_path),
        feature_names_path="model/feature_names.json",
        model_metadata_path=str(metadata_path),
    )

    with pytest.raises(pickle.UnpicklingError):
        ModelLoaderService(settings).load()


def test_model_loader_requires_checksum_for_gcs(monkeypatch, tmp_path):
    class FakeBlob:
        def exists(self):
            return True

        def download_to_filename(self, filename: str):
            Path(filename).write_bytes(b"payload")

    class FakeBucket:
        def blob(self, key: str):
            return FakeBlob()

    class FakeClient:
        def bucket(self, bucket: str):
            return FakeBucket()

    fake_storage = types.SimpleNamespace(Client=lambda: FakeClient())
    google_cloud = types.ModuleType("google.cloud")
    google_cloud.storage = fake_storage
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud)

    metadata_path = tmp_path / "missing_checksum_metadata.json"
    metadata_path.write_text('{"model_version":"1.0.0"}', encoding="utf-8")

    settings = Settings(
        model_source="gcs",
        gcs_bucket="demo-bucket",
        gcs_model_key="models/model.pkl",
        feature_names_path="model/feature_names.json",
        model_metadata_path=str(metadata_path),
        model_sha256="",
    )

    with pytest.raises(ValueError, match="MODEL_SHA256"):
        ModelLoaderService(settings).load()
