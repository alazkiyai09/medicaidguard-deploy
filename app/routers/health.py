from time import time

from fastapi import APIRouter, Request

from app.models.response import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    startup_ts = float(getattr(request.app.state, "startup_ts", time()))
    feature_names = list(getattr(request.app.state, "feature_names", []))
    model_version = str(getattr(request.app.state, "model_version", "unknown"))
    model_source = str(getattr(request.app.state, "model_source", "unknown"))

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        model_source=model_source,
        feature_count=len(feature_names),
        uptime_seconds=round(time() - startup_ts, 2),
    )
