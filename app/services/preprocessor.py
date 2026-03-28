import math

from app.models.request import TransactionInput

FEATURE_NAMES = [
    "claim_amount",
    "claim_amount_log",
    "procedure_code_freq",
    "diagnosis_code_freq",
    "provider_type_bin",
    "patient_age",
    "age_bucket",
    "claim_frequency_30d",
    "avg_claim_amount_90d",
    "claim_amount_to_avg_ratio",
    "unique_patients_30d",
    "billing_pattern_score",
]


class FeaturePreprocessor:
    """Feature engineering pipeline for MedicaidGuard inference."""

    def __init__(self) -> None:
        self.procedure_frequency = {
            "99213": 0.82,
            "99214": 0.74,
            "93000": 0.34,
            "87086": 0.41,
            "A0429": 0.11,
        }
        self.diagnosis_frequency = {
            "J06.9": 0.63,
            "E11.9": 0.59,
            "I10": 0.55,
            "M54.5": 0.32,
            "R69": 0.19,
        }

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()

    def transform(self, transaction: TransactionInput) -> dict[str, float]:
        claim_amount = float(transaction.claim_amount)
        avg_claim_amount = float(transaction.avg_claim_amount_90d)

        claim_amount_log = math.log1p(claim_amount)
        procedure_code_freq = self._frequency_encode(transaction.procedure_code, self.procedure_frequency)
        diagnosis_code_freq = self._frequency_encode(transaction.diagnosis_code, self.diagnosis_frequency)

        provider_type_bin = 1.0 if transaction.provider_type.lower().strip() == "organization" else 0.0
        age_bucket = float(self._age_bucket(transaction.patient_age))

        avg_safe = max(avg_claim_amount, 1.0)
        claim_amount_to_avg_ratio = claim_amount / avg_safe

        return {
            "claim_amount": claim_amount,
            "claim_amount_log": claim_amount_log,
            "procedure_code_freq": procedure_code_freq,
            "diagnosis_code_freq": diagnosis_code_freq,
            "provider_type_bin": provider_type_bin,
            "patient_age": float(transaction.patient_age),
            "age_bucket": age_bucket,
            "claim_frequency_30d": float(transaction.claim_frequency_30d),
            "avg_claim_amount_90d": avg_claim_amount,
            "claim_amount_to_avg_ratio": claim_amount_to_avg_ratio,
            "unique_patients_30d": float(transaction.unique_patients_30d),
            "billing_pattern_score": float(transaction.billing_pattern_score),
        }

    def transform_batch(self, transactions: list[TransactionInput]) -> list[dict[str, float]]:
        return [self.transform(transaction) for transaction in transactions]

    def to_matrix(self, feature_rows: list[dict[str, float]], feature_names: list[str]) -> list[list[float]]:
        matrix: list[list[float]] = []
        for row in feature_rows:
            matrix.append([float(row.get(feature_name, 0.0)) for feature_name in feature_names])
        return matrix

    @staticmethod
    def _age_bucket(age: int) -> int:
        if age < 18:
            return 0
        if age < 40:
            return 1
        if age < 65:
            return 2
        return 3

    @staticmethod
    def _frequency_encode(value: str, table: dict[str, float]) -> float:
        return float(table.get(value, 0.05))
