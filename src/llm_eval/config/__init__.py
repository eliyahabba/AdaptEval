

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional


class MetricConfig(BaseModel):
    higher_is_better: bool
    min: Optional[float] = None
    max: Optional[float] = None
    method: str = "minmax"  # minmax|zscore|quantile


class AppConfig(BaseSettings):
    storage_dir: str = "data/processed"
    decay: float = 0.9
    metrics: Dict[str, MetricConfig] = {}

    model_config = {
        "validate_assignment": True,
        "extra": "ignore",
    }


def load_yaml_config(defaults_path: str, metrics_path: str) -> AppConfig:
    import yaml

    with open(defaults_path, "r") as f:
        defaults = yaml.safe_load(f) or {}
    with open(metrics_path, "r") as f:
        metrics_data = yaml.safe_load(f) or {}

    metrics = metrics_data
    if "metrics" in defaults:
        # allow override via defaults
        metrics = defaults.get("metrics", metrics)

    cfg = AppConfig(
        storage_dir=defaults.get("defaults", {}).get("storage_dir", "data/processed"),
        decay=defaults.get("defaults", {}).get("decay", 0.9),
        metrics={k: MetricConfig(**v) for k, v in metrics.items()},
    )
    return cfg


