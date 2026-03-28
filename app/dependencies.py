from fastapi import HTTPException, Request, status

from app.services.metrics_store import MetricsStore
from app.services.predictor import PredictorService


class RuntimeStateError(RuntimeError):
    pass


def get_predictor(request: Request) -> PredictorService:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready yet.",
        )
    return predictor


def get_metrics_store(request: Request) -> MetricsStore:
    metrics_store = getattr(request.app.state, "metrics_store", None)
    if metrics_store is None:
        raise RuntimeStateError("Metrics store missing from app state.")
    return metrics_store
