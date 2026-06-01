"""Ranking-stability metrics computed from saved per-model predictions.

These metrics are reviewer-requested additions that sit *alongside* the existing
MAE and Spearman rho. They are computed purely from prediction outputs already
written by the chain-linking pipeline (the per-model ``validation_*`` files), so
they trigger **no** model re-evaluation and can be applied to existing paper runs
as well as new ones.

Metric definitions (all per scenario x method x eval-split), given a model's true
full-evaluation accuracy ``acc`` and its estimated accuracy ``hat``:

- ``mae``                  : mean_m |hat_m - acc_m|  (matches Eq. 3, recomputed here).
- ``spearman_rho``         : Spearman correlation of (acc, hat) rankings.
- ``top{k}_overlap``       : |topk(acc) ∩ topk(hat)| / k  (top-k rank stability).
- ``pairwise_flip_rate``   : fraction of model pairs whose relative order disagrees
                             between true and predicted rankings (discordant / total).
- ``adjacent_flip_rate``   : fraction of consecutive pairs in the TRUE ranking that
                             are inverted under the predicted ranking (hardest, nearby
                             discriminations).
- ``adjacent_gap_mae``     : mean |Δtrue - Δhat| over consecutive true-ranked pairs
                             (how well score spacing between neighbours is preserved).
- ``top_model_abs_error``  : |hat - acc| for the true best model.
- ``top1_identified``      : 1.0 if argmax(hat) == argmax(acc) else 0.0.
"""
from __future__ import annotations

import numpy as np

try:  # scipy ships with scikit-learn; fall back to a numpy implementation if absent.
    from scipy.stats import spearmanr as _spearmanr
except Exception:  # pragma: no cover
    _spearmanr = None


def _spearman(true: np.ndarray, pred: np.ndarray) -> float:
    if len(true) < 2:
        return float("nan")
    if _spearmanr is not None:
        rho = _spearmanr(true, pred).correlation
        return float(rho) if rho is not None else float("nan")
    tr = np.argsort(np.argsort(true)).astype(float)
    pr = np.argsort(np.argsort(pred)).astype(float)
    if tr.std() == 0 or pr.std() == 0:
        return float("nan")
    return float(np.corrcoef(tr, pr)[0, 1])


def _pairwise_flip_rate(true: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of unordered pairs whose order disagrees between true and pred."""
    n = len(true)
    if n < 2:
        return float("nan")
    dt = true[:, None] - true[None, :]
    dp = pred[:, None] - pred[None, :]
    iu = np.triu_indices(n, k=1)
    st, sp = np.sign(dt[iu]), np.sign(dp[iu])
    nonzero = st != 0
    if not nonzero.any():
        return float("nan")
    discordant = (st[nonzero] != sp[nonzero]).sum()
    return float(discordant) / float(nonzero.sum())


def compute_rank_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute all ranking-stability metrics for one (true, pred) vector pair."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(true) | np.isnan(pred))
    true, pred = true[mask], pred[mask]
    n = len(true)
    out: dict[str, float] = {"n_models": float(n)}
    if n == 0:
        return out

    out["mae"] = float(np.mean(np.abs(pred - true)))
    out["spearman_rho"] = _spearman(true, pred)

    # Higher accuracy = better rank. Order indices best-first.
    true_order = np.argsort(-true, kind="mergesort")
    pred_order = np.argsort(-pred, kind="mergesort")
    for k in k_values:
        kk = min(k, n)
        if kk < 1:
            continue
        top_true = set(true_order[:kk].tolist())
        top_pred = set(pred_order[:kk].tolist())
        out[f"top{k}_overlap"] = len(top_true & top_pred) / float(kk)

    out["pairwise_flip_rate"] = _pairwise_flip_rate(true, pred)

    if n >= 2:
        # Consecutive neighbours in the true ranking.
        t_sorted = true[true_order]
        p_sorted = pred[true_order]
        adj_true_gap = t_sorted[:-1] - t_sorted[1:]
        adj_pred_gap = p_sorted[:-1] - p_sorted[1:]
        out["adjacent_flip_rate"] = float(np.mean(adj_pred_gap < 0))
        out["adjacent_gap_mae"] = float(np.mean(np.abs(adj_true_gap - adj_pred_gap)))

    top_idx = int(true_order[0])
    out["top_model_abs_error"] = float(abs(pred[top_idx] - true[top_idx]))
    out["top1_identified"] = float(int(pred_order[0]) == top_idx)
    return out
