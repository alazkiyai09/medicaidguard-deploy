from contextlib import asynccontextmanager
from time import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import health_router, metrics_router, predict_router
from app.services.explainer import ExplainerService
from app.services.metrics_store import MetricsStore
from app.services.model_loader import ModelLoaderService
from app.services.predictor import PredictorService
from app.services.preprocessor import FeaturePreprocessor


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
        app.state.model_error = str(exc)

    yield


app = FastAPI(
    title="MedicaidGuard Cloud Deploy API",
    version="0.1.0",
    description="Production inference API for Medicaid fraud detection.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(predict_router)


@app.get("/")
def root() -> dict:
    return {
        "name": "MedicaidGuard Cloud Deploy API",
        "status": "running",
        "docs": "/docs",
    }
