from threading import Lock
from time import time


class MetricsStore:
    """Thread-safe in-memory metrics collector for inference traffic."""

    def __init__(self, startup_ts: float, fraud_threshold: float, model_version: str) -> None:
        self.startup_ts = startup_ts
        self.fraud_threshold = fraud_threshold
        self.model_version = model_version

        self.total_predictions = 0
        self.total_fraud_detected = 0
        self.inference_latencies_ms: list[float] = []
        self.batch_sizes: list[int] = []
        self._lock = Lock()

    def record_prediction(self, probability: float, inference_ms: float, batch_size: int = 1) -> None:
        with self._lock:
            self.total_predictions += 1
            if probability >= self.fraud_threshold:
                self.total_fraud_detected += 1

            self.inference_latencies_ms.append(float(inference_ms))
            self.batch_sizes.append(int(batch_size))

            if len(self.inference_latencies_ms) > 10000:
                self.inference_latencies_ms = self.inference_latencies_ms[-5000:]
            if len(self.batch_sizes) > 5000:
                self.batch_sizes = self.batch_sizes[-2500:]

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self.inference_latencies_ms)
            total_predictions = int(self.total_predictions)
            total_fraud_detected = int(self.total_fraud_detected)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p99_latency = self._percentile(latencies, 0.99)

        return {
            "total_predictions": total_predictions,
            "total_fraud_detected": total_fraud_detected,
            "avg_inference_time_ms": round(avg_latency, 4),
            "p99_inference_time_ms": round(p99_latency, 4),
            "uptime_seconds": round(time() - self.startup_ts, 2),
        }

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile))))
        return float(ordered[index])
