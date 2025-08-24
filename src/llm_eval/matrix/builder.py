

from dataclasses import dataclass
from typing import Mapping, Any
import pandas as pd

from llm_eval.matrix.schema import ObservationRow
from llm_eval.normalization import MetricRegistry
from llm_eval.utils import get_logger


logger = get_logger(__name__)


@dataclass
class MatrixBuilder:
    registry: MetricRegistry
    use_irt_normalization: bool = True  # Use IRT-style normalization if True, else standard normalization
    irt_method: str = 'direct'  # 'direct' or 'normalized'

    def build(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize, returning standard long-format DataFrame."""
        
        if self.use_irt_normalization:
            return self._build_with_irt_normalization(raw_df)
        else:
            return self._build_with_standard_normalization(raw_df)
    
    def _build_with_standard_normalization(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Original normalization method - per-score normalization."""
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
                metric_name=metric,
                raw_score=raw_score,
                normalized_score=float(normalized),
                is_higher_better=higher,
                run_id=r.get("run_id"),
            )
            rows.append(row.model_dump())
        df = pd.DataFrame(rows)
        logger.info("Built matrix with %d rows using standard normalization", len(df))
        return df
    
    def _build_with_irt_normalization(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """IRT-style normalization - per-scenario optimal thresholds."""
        # First create basic structure without normalization
        rows = []
        for _, r in raw_df.iterrows():
            metric = str(r["metric_name"])
            raw_score = float(r["raw_score"])
            is_higher = self.registry.config.metrics.get(metric, None)
            higher = is_higher.higher_is_better if is_higher else True
            
            row = ObservationRow(
                dataset=str(r.get("dataset", "unknown")),
                split=r.get("split"),
                task_type=str(r.get("task_type", "unknown")),
                question_id=str(r.get("question_id", "")),
                model_name=str(r.get("model_name", "")),
                model_family=r.get("model_family"),
                model_size_params=str(r.get("model_size_params")) if r.get("model_size_params") is not None else None,
                metric_name=metric,
                raw_score=raw_score,
                normalized_score=0.0,  # Will be computed below
                is_higher_better=higher,
                run_id=r.get("run_id"),
            )
            rows.append(row.model_dump())
        
        df = pd.DataFrame(rows)
        
        # Apply IRT normalization to the entire dataset
        df = self.registry.normalize_with_irt_thresholds(df, method=self.irt_method)
        
        logger.info("Built matrix with %d rows using IRT normalization (%s method)", len(df), self.irt_method)
        if hasattr(df, 'attrs') and 'irt_thresholds' in df.attrs:
            logger.info("IRT thresholds computed: %s", df.attrs['irt_thresholds'])
        
        return df


