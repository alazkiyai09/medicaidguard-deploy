from time import perf_counter

from app.config import Settings
from app.models.request import TransactionInput
from app.models.response import BatchPredictionResult, PredictionResult
from app.services.explainer import ExplainerService
from app.services.metrics_store import MetricsStore
from app.services.preprocessor import FeaturePreprocessor


class PredictorService:
    """Prediction orchestration: preprocess -> model -> explain -> format."""

    def __init__(
        self,
        settings: Settings,
        model,
        model_version: str,
        feature_names: list[str],
        preprocessor: FeaturePreprocessor,
        explainer: ExplainerService,
        metrics_store: MetricsStore,
    ) -> None:
        self.settings = settings
        self.model = model
        self.model_version = model_version
        self.feature_names = feature_names
        self.preprocessor = preprocessor
        self.explainer = explainer
        self.metrics_store = metrics_store

    def predict_one(self, transaction: TransactionInput) -> PredictionResult:
        started = perf_counter()

        feature_row = self.preprocessor.transform(transaction)
        matrix = self.preprocessor.to_matrix([feature_row], self.feature_names)
        probability = self._extract_positive_probability(self.model.predict_proba(matrix)[0])

        risk_level = self._risk_level(probability)
        prediction = "FRAUD" if probability >= self.settings.fraud_threshold else "LEGITIMATE"
        confidence = max(probability, 1.0 - probability)

        risk_factors = self.explainer.explain(feature_row)

        elapsed_ms = (perf_counter() - started) * 1000
        self.metrics_store.record_prediction(probability=probability, inference_ms=elapsed_ms, batch_size=1)

        return PredictionResult(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(probability, 6),
            risk_level=risk_level,
            prediction=prediction,
            confidence=round(confidence, 6),
            top_risk_factors=risk_factors,
            model_version=self.model_version,
            inference_time_ms=round(elapsed_ms, 3),
        )

    def predict_batch(self, transactions: list[TransactionInput]) -> BatchPredictionResult:
        if not transactions:
            raise ValueError("Batch cannot be empty.")
        if len(transactions) > self.settings.batch_max_size:
            raise ValueError(f"Batch size exceeds configured maximum ({self.settings.batch_max_size}).")

        started = perf_counter()

        feature_rows = self.preprocessor.transform_batch(transactions)
        matrix = self.preprocessor.to_matrix(feature_rows, self.feature_names)
        raw_probabilities = self.model.predict_proba(matrix)

        results: list[PredictionResult] = []
        fraud_count = 0
        for transaction, feature_row, probability_vector in zip(
            transactions, feature_rows, raw_probabilities, strict=True
        ):
            probability = self._extract_positive_probability(probability_vector)
            risk_level = self._risk_level(probability)
            prediction = "FRAUD" if probability >= self.settings.fraud_threshold else "LEGITIMATE"
            confidence = max(probability, 1.0 - probability)

            if prediction == "FRAUD":
                fraud_count += 1

            self.metrics_store.record_prediction(probability=probability, inference_ms=0.0, batch_size=len(transactions))

            results.append(
                PredictionResult(
                    transaction_id=transaction.transaction_id,
                    fraud_probability=round(probability, 6),
                    risk_level=risk_level,
                    prediction=prediction,
                    confidence=round(confidence, 6),
                    top_risk_factors=self.explainer.explain(feature_row),
                    model_version=self.model_version,
                    inference_time_ms=0.0,
                )
            )

        total_elapsed_ms = (perf_counter() - started) * 1000
        per_item_ms = total_elapsed_ms / len(results)
        for row in results:
            row.inference_time_ms = round(per_item_ms, 3)

        fraud_rate = fraud_count / len(results)

        return BatchPredictionResult(
            predictions=results,
            batch_size=len(results),
            total_inference_time_ms=round(total_elapsed_ms, 3),
            fraud_count=fraud_count,
            fraud_rate=round(fraud_rate, 6),
        )

    @staticmethod
    def _extract_positive_probability(probability_vector) -> float:
        if isinstance(probability_vector, (list, tuple)) and len(probability_vector) >= 2:
            value = float(probability_vector[1])
            return max(0.0, min(1.0, value))

        value = float(probability_vector)
        return max(0.0, min(1.0, value))

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability < 0.25:
            return "LOW"
        if probability < 0.50:
            return "MEDIUM"
        if probability < 0.80:
            return "HIGH"
        return "CRITICAL"
