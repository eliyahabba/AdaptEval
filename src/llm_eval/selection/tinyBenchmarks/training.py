

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:  # Optional heavy deps
    import py_irt  # type: ignore  # noqa: F401
    import pyro  # type: ignore  # noqa: F401
    import torch  # type: ignore  # noqa: F401
    _PYIRT_AVAILABLE = True
except Exception:
    _PYIRT_AVAILABLE = False


@dataclass
class TrainingConfig:
    model_type: str = "2pl"
    threshold: float = 50.0  # binarization of normalized_score for py-irt path
    num_epochs: int = 500
    seed: int | None = 0


def _heuristic_estimate_item_parameters(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic estimation of IRT 2PL item parameters from cross-model scores.

    Returns a DataFrame indexed by question_id with columns ["a", "b"].
    This mirrors the lightweight approach used by the built-in IRT selector.
    """
    if not {"question_id", "normalized_score"}.issubset(matrix_df.columns):
        raise ValueError("matrix_df must include question_id and normalized_score")
    df = matrix_df.copy()
    df["p"] = (df["normalized_score"].astype(float) / 100.0).clip(1e-3, 1 - 1e-3)
    grouped = df.groupby("question_id")["p"]
    p_mean = grouped.mean()
    p_std = grouped.std().fillna(0.1)
    # discrimination ~ spread; difficulty ~ -logit(mean)
    a = (p_std / (p_mean * (1 - p_mean))).clip(0.1, 3.0)
    logit = np.log(p_mean / (1 - p_mean))
    b = -logit
    params = pd.DataFrame({"a": a, "b": b})
    return params


def _to_pyirt_jsonl_rows(df: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    if not {"model_name", "question_id", "normalized_score"}.issubset(df.columns):
        raise ValueError("matrix_df missing required columns for py-irt training")
    correct = (df["normalized_score"].astype(float) >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for subject_id, item_id, y in zip(df["model_name"], df["question_id"], correct):
        rows.append({
            "subject_id": str(subject_id),
            "item_id": str(item_id),
            "response": int(y),
        })
    return rows


def _train_with_pyirt(rows: list[dict[str, Any]], cfg: TrainingConfig) -> pd.DataFrame:  # pragma: no cover - stochastic/slow
    import json
    import tempfile
    from pathlib import Path
    from py_irt.training import IrtModelTrainer
    from py_irt.config import IrtConfig

    with tempfile.TemporaryDirectory() as td:
        data_path = Path(td) / "data.jsonl"
        with open(data_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        irt_cfg = IrtConfig(
            model_type=cfg.model_type,
            num_epochs=cfg.num_epochs,
            seed=cfg.seed,
            dataset_path=str(data_path),
            validate_every=0,
        )
        trainer = IrtModelTrainer(irt_cfg)
        trainer.train()
        item_params = trainer.model.item_param_store  # type: ignore[attr-defined]
        df = pd.DataFrame({
            str(k): {"a": float(v.get("a", 1.0)), "b": float(v.get("b", 0.0))}
            for k, v in item_params.items()
        }).T
        df.index.name = "question_id"
        return df


def fit_2pl_parameters(matrix_df: pd.DataFrame, config: TrainingConfig | None = None) -> pd.DataFrame:
    """Fit or estimate 2PL parameters per item.

    If py-irt is available, trains a 2PL model; otherwise, uses a heuristic estimation.

    Returns a DataFrame indexed by question_id with columns ["a", "b"].
    """
    cfg = config or TrainingConfig()
    if _PYIRT_AVAILABLE:
        rows = _to_pyirt_jsonl_rows(matrix_df, threshold=cfg.threshold)
        return _train_with_pyirt(rows, cfg)
    return _heuristic_estimate_item_parameters(matrix_df)


