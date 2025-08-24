

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class EstimationConfig:
    # Ability estimation from anchor responses
    max_iter: int = 50
    tol: float = 1e-4
    # lambda blending (gp-IRT) per dataset
    lambdas_by_dataset: dict[str, float] | None = None


def estimate_theta_from_anchors(
    item_params: pd.DataFrame,
    anchor_responses: pd.Series,
    init_theta: float = 0.0,
    config: EstimationConfig | None = None,
) -> float:
    """Estimate ability theta using MLE on 2PL with anchor responses.

    anchor_responses: Series indexed by question_id with binary {0,1} correctness.
    """
    cfg = config or EstimationConfig()
    # Filter to anchors present in params
    common = item_params.index.intersection(anchor_responses.index)
    if len(common) == 0:
        return init_theta
    a = item_params.loc[common, "a"].astype(float).values
    b = item_params.loc[common, "b"].astype(float).values
    y = anchor_responses.loc[common].astype(float).values

    theta = float(init_theta)
    for _ in range(cfg.max_iter):
        z = a * (theta - b)
        p = 1.0 / (1.0 + np.exp(-z))
        # Gradient and Hessian for 2PL log-likelihood
        grad = np.sum(a * (y - p))
        hess = -np.sum((a ** 2) * p * (1 - p)) - 1e-6
        step = grad / hess
        theta_new = theta - step
        if abs(theta_new - theta) < cfg.tol:
            theta = theta_new
            break
        theta = theta_new
    return float(theta)


def expected_correctness(item_params: pd.DataFrame, theta: float) -> pd.Series:
    """Return expected correctness for each item at ability theta under 2PL."""
    z = item_params["a"].astype(float) * (theta - item_params["b"].astype(float))
    p = 1.0 / (1.0 + np.exp(-z))
    return p.astype(float)


def blend_anchor_and_irt(
    preds_anchor: pd.Series,
    preds_irt: pd.Series,
    lambdas_by_dataset: dict[str, float] | None,
    item_to_dataset: pd.Series | None = None,
) -> pd.Series:
    """Blend predictions as in gp-IRT: lambda*data + (1-lambda)*irt.

    If no per-dataset lambdas are provided, defaults to simple average.
    """
    if lambdas_by_dataset is None or item_to_dataset is None:
        return 0.5 * preds_anchor + 0.5 * preds_irt
    lam = item_to_dataset.map(lambda d: float(lambdas_by_dataset.get(str(d), 0.5))).astype(float)
    return lam * preds_anchor + (1.0 - lam) * preds_irt


