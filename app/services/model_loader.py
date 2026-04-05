import builtins
import hashlib
import hmac
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.config import Settings
from app.services.preprocessor import FEATURE_NAMES
from app.services.simple_model import SimpleFraudModel

logger = logging.getLogger(__name__)

_ALLOWED_PICKLE_GLOBALS = {
    ("app.services.simple_model", "SimpleFraudModel"): SimpleFraudModel,
}


@dataclass(slots=True)
class ModelArtifacts:
    model: object
    feature_names: list[str]
    metadata: dict


class ModelLoaderService:
    """Loads model artifacts from local disk or GCS."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> ModelArtifacts:
        metadata = self._load_metadata()
        model_path = self._resolve_model_path()
        self._verify_model_checksum(model_path, metadata)
        model = self._load_pickle(model_path)
        feature_names = self._load_feature_names()

        if not feature_names:
            feature_names = list(getattr(model, "feature_order", FEATURE_NAMES))

        metadata.setdefault("model_version", self.settings.model_version)
        return ModelArtifacts(model=model, feature_names=feature_names, metadata=metadata)

    def _resolve_model_path(self) -> Path:
        source = self.settings.model_source.strip().lower()
        if source == "local":
            model_path = Path(self.settings.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            return model_path

        if source == "gcs":
            if not self.settings.gcs_bucket or not self.settings.gcs_model_key:
                raise ValueError("GCS model source selected but GCS_BUCKET/GCS_MODEL_KEY not configured.")
            return self._download_from_gcs(self.settings.gcs_bucket, self.settings.gcs_model_key)

        raise ValueError(f"Unsupported MODEL_SOURCE '{self.settings.model_source}'.")

    def _download_from_gcs(self, bucket: str, key: str) -> Path:
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-cloud-storage is required for MODEL_SOURCE=gcs.") from exc

        client = storage.Client()
        blob = client.bucket(bucket).blob(key)

        if not blob.exists():
            raise FileNotFoundError(f"GCS model object not found: gs://{bucket}/{key}")

        with NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            return Path(temp_file.name)

    @staticmethod
    def _load_pickle(path: Path):
        with path.open("rb") as f:
            return _RestrictedModelUnpickler(f).load()

    def _load_feature_names(self) -> list[str]:
        path = Path(self.settings.feature_names_path)
        if not path.exists():
            return FEATURE_NAMES.copy()

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            return [str(item) for item in payload]
        return FEATURE_NAMES.copy()

    def _load_metadata(self) -> dict:
        path = Path(self.settings.model_metadata_path)
        if not path.exists():
            return {"model_version": self.settings.model_version}

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            return payload
        return {"model_version": self.settings.model_version}

    def _verify_model_checksum(self, path: Path, metadata: dict) -> None:
        expected_checksum = self._expected_model_sha256(metadata)
        model_source = self.settings.model_source.strip().lower()

        if model_source == "gcs" and not expected_checksum:
            raise ValueError("MODEL_SHA256 or metadata sha256 is required for MODEL_SOURCE=gcs.")

        if not expected_checksum:
            return

        actual_checksum = self._calculate_sha256(path)
        if not hmac.compare_digest(actual_checksum.lower(), expected_checksum.lower()):
            raise ValueError("Model SHA-256 checksum mismatch.")

    def _expected_model_sha256(self, metadata: dict) -> str:
        if self.settings.model_sha256:
            return self.settings.model_sha256.strip()

        for key in ("sha256", "model_sha256"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    @staticmethod
    def _calculate_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()


class _RestrictedModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return _ALLOWED_PICKLE_GLOBALS[(module, name)]
        if module == "builtins" and name in {"set", "frozenset"}:
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")
