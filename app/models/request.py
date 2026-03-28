from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Single transaction for fraud prediction."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    provider_id: str = Field(..., description="Healthcare provider ID")
    claim_amount: float = Field(..., ge=0, description="Claim dollar amount")
    procedure_code: str = Field(..., description="Medical procedure code (CPT)")
    diagnosis_code: str = Field(..., description="ICD-10 diagnosis code")
    provider_type: str = Field(..., description="'individual' or 'organization'")
    patient_age: int = Field(..., ge=0, le=120)
    claim_frequency_30d: int = Field(..., ge=0, description="Claims in last 30 days")
    avg_claim_amount_90d: float = Field(..., ge=0, description="Avg claim amount 90d")
    unique_patients_30d: int = Field(..., ge=0, description="Unique patients 30d")
    billing_pattern_score: float = Field(..., ge=0, le=1, description="Billing pattern anomaly score")


class BatchPredictRequest(BaseModel):
    """Batch of transactions for fraud prediction."""

    transactions: list[TransactionInput] = Field(..., min_length=1, max_length=100)
