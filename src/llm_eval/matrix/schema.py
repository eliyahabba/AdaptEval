

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ObservationRow(BaseModel):
    dataset: str
    split: Optional[str] = None
    task_type: str
    question_id: str
    model_name: str
    model_family: Optional[str] = None
    model_size_params: Optional[str] = None
    prompt_variant: Optional[str] = None
    metric_name: str
    raw_score: float
    normalized_score: float
    is_higher_better: bool = True
    timestamp: str
    run_id: Optional[str] = None
    snapshot_id: Optional[str] = None

    @field_validator("normalized_score")
    @classmethod
    def _normalized_in_range(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError("normalized_score must be in [0, 100]")
        return v


