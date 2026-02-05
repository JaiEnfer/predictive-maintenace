from pydantic import BaseModel
from typing import List, Dict


class SensorData(BaseModel):
    sensor_values: List[float]  # must match number of sensor features (21)


class PredictionResponse(BaseModel):
    will_fail_soon: int
    failure_probability: float

class TurbofanRow(BaseModel):
    # a dict
    values: Dict[str, float]