from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import RuntimeStateError, get_metrics_store
from app.models.response import MetricsResponse
from app.services.metrics_store import MetricsStore

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def metrics(metrics_store: MetricsStore = Depends(get_metrics_store)) -> MetricsResponse:
    try:
        snapshot = metrics_store.snapshot()
    except RuntimeStateError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return MetricsResponse(**snapshot)
