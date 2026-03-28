from locust import HttpUser, between, task


class MedicaidGuardLoadUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        self.sample_payload = {
            "transaction_id": "TXN-LOAD-001",
            "provider_id": "PRV-LOAD",
            "claim_amount": 15000.0,
            "procedure_code": "99213",
            "diagnosis_code": "J06.9",
            "provider_type": "individual",
            "patient_age": 45,
            "claim_frequency_30d": 12,
            "avg_claim_amount_90d": 8500.0,
            "unique_patients_30d": 45,
            "billing_pattern_score": 0.73,
        }

    @task(80)
    def predict_single(self):
        self.client.post("/predict", json=self.sample_payload, name="predict_single")

    @task(15)
    def predict_batch(self):
        payload = {"transactions": [self.sample_payload for _ in range(10)]}
        self.client.post("/predict/batch", json=payload, name="predict_batch_10")

    @task(5)
    def health_check(self):
        self.client.get("/health", name="health")
