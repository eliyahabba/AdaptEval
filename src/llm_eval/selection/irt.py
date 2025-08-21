

import numpy as np
import pandas as pd

from src.llm_eval.selection.interfaces import QuestionSelector, ModelProfile
from src.llm_eval.selection.cold_start import simple_cold_start_theta


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class IRT2PLSelector(QuestionSelector):
    """Lightweight 2PL-style selector.

    Heuristically estimates item discrimination (a) and difficulty (b) from existing normalized
    probabilities p = normalized_score/100 across models. Then for a target model ability theta,
    selects items maximizing Fisher information: I(θ) = a^2 * p(1-p).
    """

    def __init__(self, default_theta: float | None = None) -> None:
        self.default_theta = default_theta

    def _estimate_item_parameters(self, matrix_df: pd.DataFrame) -> pd.DataFrame:
        df = matrix_df.copy()
        df["p"] = (df["normalized_score"].astype(float) / 100.0).clip(1e-3, 1 - 1e-3)
        grouped = df.groupby("question_id")["p"]
        p_mean = grouped.mean()
        p_std = grouped.std().fillna(0.1)
        # heuristic: discrimination proportional to spread; difficulty from logit of mean
        a = (p_std / (p_mean * (1 - p_mean))).clip(0.1, 3.0)
        logit = np.log(p_mean / (1 - p_mean))
        b = -logit  # higher mean => easier => lower difficulty
        params = pd.DataFrame({"a": a, "b": b})
        return params

    def _fisher_information(self, theta: float, a: float, b: float) -> float:
        p = float(sigmoid(a * (theta - b)))
        return float((a ** 2) * p * (1 - p))

    def select(self, model: ModelProfile, k: int, matrix_df: pd.DataFrame) -> list[str]:
        theta = self.default_theta if self.default_theta is not None else simple_cold_start_theta(model)
        params = self._estimate_item_parameters(matrix_df)
        params["info"] = [self._fisher_information(theta, float(a), float(b)) for a, b in zip(params["a"], params["b"])]
        top = params.sort_values("info", ascending=False).head(k)
        return list(top.index)


