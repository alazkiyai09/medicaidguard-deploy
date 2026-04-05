import numpy as np
import pandas as pd

np.random.seed(42)
n = 100

data = pd.DataFrame(
    {
        "transaction_id": [f"TXN-{i:04d}" for i in range(n)],
        "provider_id": [f"PRV-{np.random.randint(1000, 9999)}" for _ in range(n)],
        "claim_amount": np.random.exponential(3000, n).clip(50, 50000).round(2),
        "procedure_code": np.random.choice(["99213", "99214", "99215", "99211", "99212"], n),
        "diagnosis_code": np.random.choice(["J06.9", "M54.5", "Z00.00", "I10", "E11.9"], n),
        "provider_type": np.random.choice(["individual", "organization"], n, p=[0.6, 0.4]),
        "patient_age": np.random.randint(18, 90, n),
        "claim_frequency_30d": np.random.poisson(8, n).clip(0, 50),
        "avg_claim_amount_90d": np.random.exponential(2500, n).clip(100, 30000).round(2),
        "unique_patients_30d": np.random.poisson(40, n).clip(1, 200),
        "billing_pattern_score": np.random.beta(2, 5, n).round(3),
    }
)

data.to_csv("sample_batch_100.csv", index=False)
