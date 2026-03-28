import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.config import Settings
from app.services.preprocessor import FEATURE_NAMES


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
        model_path = self._resolve_model_path()
        model = self._load_pickle(model_path)
        feature_names = self._load_feature_names()
        metadata = self._load_metadata()

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
            return pickle.load(f)

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
