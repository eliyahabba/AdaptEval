### Normalization

Metric registry maps `metric_name` to normalization strategy, producing `normalized_score ∈ [0,100]`.

Rules:
- Min-max with configured `min/max` and `higher_is_better`.
- Z-score→CDF requires `mean/std` via metadata. No generic fallback: metrics must be explicitly configured.

Configure per-metric in `src/llm_eval/config/metrics.yaml`.

Run examples
------------

Python:
```python
from src.llm_eval.config import load_yaml_config
from src.llm_eval.normalization import MetricRegistry

cfg = load_yaml_config("src/llm_eval/config/defaults.yaml", "src/llm_eval/config/metrics.yaml")
reg = MetricRegistry(cfg)
res = reg.normalize("binary_acc", raw_score=1.0, metadata={})
print(res.normalized)
```



