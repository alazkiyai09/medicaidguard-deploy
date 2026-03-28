from app.config import Settings
from app.models.response import RiskFactor


class ExplainerService:
    """SHAP-based explainer with deterministic fallback when SHAP/model support is unavailable."""

    def __init__(self, settings: Settings, model, feature_names: list[str]) -> None:
        self.settings = settings
        self.model = model
        self.feature_names = feature_names
        self._explainer = None

    def explain(self, feature_row: dict[str, float]) -> list[RiskFactor]:
        if self.settings.shap_enabled:
            shap_factors = self._explain_with_shap(feature_row)
            if shap_factors:
                return shap_factors

        return self._explain_fallback(feature_row)

    def _explain_with_shap(self, feature_row: dict[str, float]) -> list[RiskFactor]:
        explainer = self._get_shap_explainer()
        if explainer is None:
            return []

        ordered_values = [float(feature_row.get(feature_name, 0.0)) for feature_name in self.feature_names]

        try:
            shap_values = explainer.shap_values([ordered_values])
            vector = self._extract_shap_vector(shap_values)
        except Exception:
            return []

        factors: list[RiskFactor] = []
        for feature_name, shap_value in zip(self.feature_names, vector, strict=False):
            score = float(shap_value)
            factors.append(
                RiskFactor(
                    feature=feature_name,
                    importance=round(abs(score), 6),
                    direction="high" if score >= 0 else "low",
                )
            )

        factors.sort(key=lambda item: item.importance, reverse=True)
        return factors[:3]

    def _get_shap_explainer(self):
        if self._explainer is not None:
            return self._explainer

        try:
            import shap
        except ImportError:
            return None

        model_for_explainer = self.model
        if hasattr(model_for_explainer, "named_steps"):
            named_steps = getattr(model_for_explainer, "named_steps")
            if named_steps:
                model_for_explainer = list(named_steps.values())[-1]

        try:
            self._explainer = shap.TreeExplainer(model_for_explainer)
            return self._explainer
        except Exception:
            return None

    @staticmethod
    def _extract_shap_vector(shap_values):
        if isinstance(shap_values, list):
            if not shap_values:
                return []
            first = shap_values[0]
            if isinstance(first, list):
                if first and isinstance(first[0], list):
                    return first[0]
                return first
            if hasattr(first, "tolist"):
                first = first.tolist()
            if first and isinstance(first[0], list):
                return first[0]
            return first

        if hasattr(shap_values, "tolist"):
            values = shap_values.tolist()
        else:
            values = shap_values

        if values and isinstance(values[0], list):
            return values[0]
        return values

    def _explain_fallback(self, feature_row: dict[str, float]) -> list[RiskFactor]:
        scored = []
        for feature_name, raw_value in feature_row.items():
            value = float(raw_value)
            scored.append((feature_name, abs(value), "high" if value >= 0 else "low"))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RiskFactor(feature=feature, importance=round(importance, 6), direction=direction)
            for feature, importance, direction in scored[:3]
        ]
