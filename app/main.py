from contextlib import asynccontextmanager
import logging
from time import time

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.security import require_api_key
from app.routers import health_router, metrics_router, predict_router
from app.services.explainer import ExplainerService
from app.services.metrics_store import MetricsStore
from app.services.model_loader import ModelLoaderService
from app.services.predictor import PredictorService
from app.services.preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    startup_ts = time()

    app.state.startup_ts = startup_ts
    app.state.model_source = settings.model_source
    app.state.model_loaded = False
    app.state.model_version = settings.model_version
    app.state.feature_names = []
    app.state.predictor = None

    preprocessor = FeaturePreprocessor()
    metrics_store = MetricsStore(
        startup_ts=startup_ts,
        fraud_threshold=settings.fraud_threshold,
        model_version=settings.model_version,
    )

    app.state.metrics_store = metrics_store

    try:
        artifacts = ModelLoaderService(settings).load()

        feature_names = artifacts.feature_names or preprocessor.feature_names
        model_version = str(artifacts.metadata.get("model_version", settings.model_version))

        explainer = ExplainerService(settings=settings, model=artifacts.model, feature_names=feature_names)
        predictor = PredictorService(
            settings=settings,
            model=artifacts.model,
            model_version=model_version,
            feature_names=feature_names,
            preprocessor=preprocessor,
            explainer=explainer,
            metrics_store=metrics_store,
        )

        app.state.predictor = predictor
        app.state.model_loaded = True
        app.state.model_version = model_version
        app.state.feature_names = feature_names
    except Exception as exc:
        logger.exception("Model bootstrap failed")
        app.state.model_error = str(exc)

    yield


app = FastAPI(
    title="MedicaidGuard Cloud Deploy API",
    version="0.1.0",
    description="Production inference API for Medicaid fraud detection.",
    lifespan=lifespan,
)

cors_allow_origins = get_settings().parsed_cors_allow_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(metrics_router, dependencies=[Depends(require_api_key)])
app.include_router(predict_router, dependencies=[Depends(require_api_key)])


@app.get("/")
def root() -> dict:
    return {
        "name": "MedicaidGuard Cloud Deploy API",
        "status": "running",
        "docs": "/docs",
    }
