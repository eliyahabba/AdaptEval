

from dataclasses import dataclass
from typing import Mapping, Any
import pandas as pd

from src.llm_eval.matrix.schema import ObservationRow
from src.llm_eval.normalization import MetricRegistry
from src.llm_eval.utils import get_logger


logger = get_logger(__name__)


@dataclass
class MatrixBuilder:
    registry: MetricRegistry

    def build(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize, returning standard long-format DataFrame."""
        rows = []
        for _, r in raw_df.iterrows():
            metric = str(r["metric_name"])  # required
            raw_score = float(r["raw_score"])  # required
            metadata: Mapping[str, Any] = {}
            norm = self.registry.normalize(metric, raw_score, metadata)
            normalized = norm.normalized
            is_higher = self.registry.config.metrics.get(metric, None)
            higher = is_higher.higher_is_better if is_higher else bool(metadata.get("higher_is_better", True))  # noqa: E501
            row = ObservationRow(
                dataset=str(r.get("dataset", "unknown")),
                split=r.get("split"),
                task_type=str(r.get("task_type", "unknown")),
                question_id=str(r.get("question_id", "")),
                model_name=str(r.get("model_name", "")),
                model_family=r.get("model_family"),
                model_size_params=str(r.get("model_size_params")) if r.get("model_size_params") is not None else None,  # noqa: E501
                prompt_variant=r.get("prompt_variant"),
                metric_name=metric,
                raw_score=raw_score,
                normalized_score=float(normalized),
                is_higher_better=higher,
                timestamp=str(r.get("timestamp", "")),
                run_id=r.get("run_id"),
                snapshot_id=r.get("snapshot_id"),
            )
            rows.append(row.model_dump())
        df = pd.DataFrame(rows)
        logger.info("Built matrix with %d rows", len(df))
        return df


