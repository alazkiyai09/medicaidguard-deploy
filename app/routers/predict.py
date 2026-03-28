from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_predictor
from app.models.request import BatchPredictRequest, TransactionInput
from app.models.response import BatchPredictionResult, PredictionResult
from app.services.predictor import PredictorService

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResult)
def predict_single(
    payload: TransactionInput,
    predictor: PredictorService = Depends(get_predictor),
) -> PredictionResult:
    try:
        return predictor.predict_one(payload)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/batch", response_model=BatchPredictionResult)
def predict_batch(
    payload: BatchPredictRequest,
    predictor: PredictorService = Depends(get_predictor),
) -> BatchPredictionResult:
    try:
        return predictor.predict_batch(payload.transactions)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
