from app.models.request import TransactionInput
from app.services.preprocessor import FEATURE_NAMES, FeaturePreprocessor


def test_preprocessor_transform_outputs_expected_features():
    preprocessor = FeaturePreprocessor()
    txn = TransactionInput(
        transaction_id="TXN-1",
        provider_id="PRV-1",
        claim_amount=12000,
        procedure_code="99213",
        diagnosis_code="J06.9",
        provider_type="individual",
        patient_age=50,
        claim_frequency_30d=10,
        avg_claim_amount_90d=6000,
        unique_patients_30d=40,
        billing_pattern_score=0.8,
    )

    row = preprocessor.transform(txn)

    assert set(row.keys()) == set(FEATURE_NAMES)
    assert row["provider_type_bin"] == 0.0
    assert row["claim_amount_to_avg_ratio"] == 2.0


def test_preprocessor_unknown_codes_fallback_and_matrix_ordering():
    preprocessor = FeaturePreprocessor()
    txn = TransactionInput(
        transaction_id="TXN-2",
        provider_id="PRV-2",
        claim_amount=500,
        procedure_code="UNKNOWN",
        diagnosis_code="UNKNOWN",
        provider_type="organization",
        patient_age=30,
        claim_frequency_30d=2,
        avg_claim_amount_90d=1000,
        unique_patients_30d=20,
        billing_pattern_score=0.2,
    )

    row = preprocessor.transform(txn)
    matrix = preprocessor.to_matrix([row], FEATURE_NAMES)

    assert row["procedure_code_freq"] == 0.05
    assert row["diagnosis_code_freq"] == 0.05
    assert row["provider_type_bin"] == 1.0
    assert len(matrix[0]) == len(FEATURE_NAMES)
