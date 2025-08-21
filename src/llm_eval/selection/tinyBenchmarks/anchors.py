

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class AnchorConfig:
    per_level: int = 5  # number of anchors to keep per difficulty bin
    levels: int = 10     # number of difficulty bins


def compute_anchor_scores(item_params: pd.DataFrame) -> pd.Series:
    """Compute an anchor-worthiness score for each item.

    Uses Fisher information at a small grid of abilities and averages it,
    preferring items that are informative across abilities.
    """
    if not {"a", "b"}.issubset(item_params.columns):
        raise ValueError("item_params must have columns 'a' and 'b'")
    thetas = np.linspace(-2.0, 2.0, 9)
    infos = []
    for theta in thetas:
        p = 1.0 / (1.0 + np.exp(-item_params["a"] * (theta - item_params["b"])) )
        info = (item_params["a"] ** 2) * p * (1 - p)
        infos.append(info)
    mean_info = pd.concat(infos, axis=1).mean(axis=1)
    return mean_info


def find_anchor_items(item_params: pd.DataFrame, config: AnchorConfig | None = None) -> list[str]:
    """Select anchor items distributed across difficulty.

    - Bin by difficulty (b)
    - Within each bin, rank by anchor score and take top-k
    """
    cfg = config or AnchorConfig()
    if item_params.empty:
        return []
    anchorscore = compute_anchor_scores(item_params)
    df = item_params.copy()
    df["anchor_score"] = anchorscore
    # Bin by difficulty b into cfg.levels quantiles
    # Fallback: if not enough distinct values, use cut on range
    try:
        df["b_bin"] = pd.qcut(df["b"], q=cfg.levels, duplicates="drop")
    except Exception:
        df["b_bin"] = pd.cut(df["b"], bins=cfg.levels)
    picked: list[str] = []
    for _, group in df.groupby("b_bin"):
        if len(group) == 0:
            continue
        top = group.sort_values("anchor_score", ascending=False).head(cfg.per_level)
        picked.extend([str(i) for i in top.index.tolist()])
    return picked


