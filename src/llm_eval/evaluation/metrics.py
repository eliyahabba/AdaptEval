

import pandas as pd


def estimate_error_to_full_eval(partial_scores: pd.Series, full_scores: pd.Series) -> float:
    """Return RMSE between partial and full normalized scores for overlap of indices."""
    idx = partial_scores.index.intersection(full_scores.index)
    if len(idx) == 0:
        return float("nan")
    diff = partial_scores.loc[idx] - full_scores.loc[idx]
    return float((diff.pow(2).mean()) ** 0.5)


