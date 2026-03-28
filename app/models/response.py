from pydantic import BaseModel


class RiskFactor(BaseModel):
    feature: str
    importance: float
    direction: str  # high | low


class PredictionResult(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str  # LOW | MEDIUM | HIGH | CRITICAL
    prediction: str  # LEGITIMATE | FRAUD
    confidence: float
    top_risk_factors: list[RiskFactor]
    model_version: str
    inference_time_ms: float


class BatchPredictionResult(BaseModel):
    predictions: list[PredictionResult]
    batch_size: int
    total_inference_time_ms: float
    fraud_count: int
    fraud_rate: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_source: str
    feature_count: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_predictions: int
    total_fraud_detected: int
    avg_inference_time_ms: float
    p99_inference_time_ms: float
    uptime_seconds: float
