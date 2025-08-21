

from dataclasses import dataclass
from typing import Mapping, Any, Callable

from src.llm_eval.config import AppConfig
from src.llm_eval.normalization.base import AbstractNormalizer, NormalizationResult
from src.llm_eval.normalization.rules import minmax_normalize, zscore_cdf_normalize


@dataclass
class MetricRegistry:
    config: AppConfig

    def normalize(self, metric_name: str, raw_score: float, metadata: Mapping[str, Any]) -> NormalizationResult:  # noqa: E501
        metric_cfg = self.config.metrics.get(metric_name)
        if metric_cfg is None:
            raise KeyError(f"Unknown metric '{metric_name}'. Configure it in metrics.yaml")

        method = metric_cfg.method

        if method == "minmax":
            if metric_cfg.min is None or metric_cfg.max is None:
                raise ValueError(f"Metric '{metric_name}' requires min and max for minmax normalization")
            value = minmax_normalize(raw_score, float(metric_cfg.min), float(metric_cfg.max), metric_cfg.higher_is_better)  # noqa: E501
            return NormalizationResult(
                normalized=value,
                method="minmax",
                params={"min": metric_cfg.min, "max": metric_cfg.max, "higher_is_better": metric_cfg.higher_is_better},  # noqa: E501
            )

        if method == "zscore":
            mean = float(metadata.get("mean"))
            std = float(metadata.get("std"))
            higher = bool(metric_cfg.higher_is_better)
            value = zscore_cdf_normalize(raw_score, mean, std, higher)
            return NormalizationResult(
                normalized=value,
                method="zscore_cdf",
                params={"mean": mean, "std": std, "higher_is_better": higher},
            )

        raise ValueError(f"Unsupported normalization method '{method}' for metric '{metric_name}'")


