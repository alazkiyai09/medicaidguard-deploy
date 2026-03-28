import math


class SimpleFraudModel:
    """Portable deterministic model used as bundled artifact for deployment demos/tests."""

    def __init__(self, feature_order: list[str], weights: dict[str, float], intercept: float = -2.2) -> None:
        self.feature_order = feature_order
        self.weights = weights
        self.intercept = intercept

    def predict_proba(self, rows) -> list[list[float]]:
        probabilities: list[list[float]] = []
        for row in rows:
            score = float(self.intercept)
            for feature_name, value in zip(self.feature_order, row, strict=True):
                score += self.weights.get(feature_name, 0.0) * float(value)

            fraud_prob = 1.0 / (1.0 + math.exp(-score))
            legit_prob = 1.0 - fraud_prob
            probabilities.append([legit_prob, fraud_prob])

        return probabilities
