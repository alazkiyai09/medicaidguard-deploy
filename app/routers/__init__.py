from app.routers.health import router as health_router
from app.routers.metrics import router as metrics_router
from app.routers.predict import router as predict_router

__all__ = ["health_router", "metrics_router", "predict_router"]
