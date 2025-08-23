

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from llm_eval.selection.interfaces import QuestionSelector, ModelProfile
from llm_eval.selection.cold_start import simple_cold_start_theta
from .training import fit_2pl_parameters, TrainingConfig
from .anchors import find_anchor_items, AnchorConfig
from .estimation import estimate_theta_from_anchors, expected_correctness, EstimationConfig


@dataclass
class TinyBenchmarksSelector(QuestionSelector):
    """Selector implementing the TinyBenchmarks tutorial pipeline.

    Steps:
      1) Fit/estimate 2PL item parameters from existing matrix.
      2) Choose anchor items distributed across difficulty.
      3) Estimate target model ability from (optional) anchor responses; if not provided,
         fall back to cold-start prior.
      4) Rank remaining questions by Fisher information at estimated theta.
    """

    anchors_per_level: int = 3
    levels: int = 10
    use_pyirt_if_available: bool = True
    num_epochs: int = 300
    threshold: float = 50.0

    def _fisher_information(self, theta: float, a: float, b: float) -> float:
        p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        return float((a ** 2) * p * (1 - p))

    def _fit_items(self, matrix_df: pd.DataFrame) -> pd.DataFrame:
        tcfg = TrainingConfig(
            model_type="2pl",
            threshold=self.threshold,
            num_epochs=self.num_epochs,
            seed=0,
        )
        params = fit_2pl_parameters(matrix_df, tcfg)
        if not {"a", "b"}.issubset(params.columns):
            raise ValueError("Fitted item parameters missing required columns a,b")
        return params

    def _choose_anchors(self, item_params: pd.DataFrame) -> list[str]:
        acfg = AnchorConfig(per_level=self.anchors_per_level, levels=self.levels)
        return find_anchor_items(item_params, acfg)

    def select(self, model: ModelProfile, k: int, matrix_df: pd.DataFrame) -> list[str]:
        # 1) Fit 2PL item params
        params = self._fit_items(matrix_df)

        # 2) Choose anchor items
        anchor_ids = self._choose_anchors(params)

        # 3) Estimate theta. In this MVP we do not have the new model's anchor responses yet.
        #    So we fall back to cold start. In a future flow, callers can supply a Series of
        #    observed anchor responses and we would plug them into estimate_theta_from_anchors.
        theta = simple_cold_start_theta(model)

        # 4) Score items by Fisher information at theta; exclude anchors from selection set
        infos: list[tuple[str, float]] = []
        for qid, row in params.iterrows():
            if str(qid) in set(anchor_ids):
                continue
            a = float(row["a"])
            b = float(row["b"])
            info = self._fisher_information(theta, a, b)
            infos.append((str(qid), info))
        infos.sort(key=lambda x: x[1], reverse=True)
        return [qid for qid, _ in infos[:k]]


