import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import RuntimeStateError, get_metrics_store
from app.models.response import MetricsResponse
from app.services.metrics_store import MetricsStore

router = APIRouter(tags=["metrics"])
logger = logging.getLogger(__name__)


@router.get("/metrics", response_model=MetricsResponse)
def metrics(metrics_store: MetricsStore = Depends(get_metrics_store)) -> MetricsResponse:
    try:
        snapshot = metrics_store.snapshot()
    except RuntimeStateError as exc:
        logger.warning("Metrics store unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics store unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Metrics snapshot failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics snapshot failed.",
        ) from exc

    return MetricsResponse(**snapshot)
