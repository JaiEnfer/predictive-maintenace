from dataclasses import dataclass

@dataclass
class Metrics:
    total_requests: int = 0
    total_predictions: int = 0
    fail_soon_predictions: int = 0

metrics = Metrics()
