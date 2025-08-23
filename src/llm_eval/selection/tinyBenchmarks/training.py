

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import py_irt
import pyro
import torch
import json
import tempfile
from pathlib import Path
from py_irt.training import IrtModelTrainer
from py_irt.config import IrtConfig


@dataclass
class TrainingConfig:
    model_type: str = "multidim_2pl"  # align with notebook/irt.py usage
    threshold: float = 50.0  # binarization of normalized_score for py-irt path
    num_epochs: int = 2000   # notebook default
    seed: int | None = 42
    # Additional hyperparameters often used by py-irt CLI/API
    dims: int | None = 10
    lr: float | None = 0.1
    lr_decay: float | None = 0.9999
    dropout: float | None = 0.5
    hidden: int | None = 100
    priors: str | None = "hierarchical"
    deterministic: bool | None = True
    log_every: int | None = 200
    device: str | None = None  # 'cuda' or 'cpu'
    validate_every: int | None = 0


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
    # Convert to expected py-irt JSONL format:
    # {"subject_id": "...", "responses": {"<item_id>": <0|1>, ...}}
    correct = (df["normalized_score"].astype(float) >= threshold).astype(int)
    df_tmp = df.copy()
    df_tmp["subject_id"] = df_tmp["model_name"].astype(str)
    df_tmp["item_id"] = df_tmp["question_id"].astype(str)
    df_tmp["y"] = correct.astype(int)

    grouped = df_tmp.groupby("subject_id")
    rows: list[dict[str, Any]] = []
    for subject_id, grp in grouped:
        responses: dict[str, int] = {}
        for item_id, y in zip(grp["item_id"], grp["y"]):
            responses[str(item_id)] = int(y)
        rows.append({
            "subject_id": str(subject_id),
            "responses": responses,
        })
    return rows


def _train_with_pyirt(rows: list[dict[str, Any]], cfg: TrainingConfig) -> pd.DataFrame:  # pragma: no cover - stochastic/slow

    with tempfile.TemporaryDirectory() as td:
        data_path = Path(td) / "data.jsonl"
        with open(data_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Build minimal config then enrich attributes if available in this py-irt version
        irt_cfg = IrtConfig(
            model_type=cfg.model_type,
            num_epochs=cfg.num_epochs,
            seed=cfg.seed,
            dataset_path=str(data_path),
            validate_every=cfg.validate_every if cfg.validate_every is not None else 0,
        )
        for name in [
            "dims", "lr", "lr_decay", "dropout", "hidden", "priors",
            "deterministic", "log_every", "device",
        ]:
            if hasattr(irt_cfg, name):
                val = getattr(cfg, name)
                if val is not None:
                    setattr(irt_cfg, name, val)

        # Instantiate trainer robustly across API variants
        trainer = IrtModelTrainer(data_path=data_path, config=irt_cfg)  # type: ignore[arg-type]
        trainer.train()
        item_params = trainer.model.item_param_store  # type: ignore[attr-defined]

        # Normalize parameters into scalar a,b per item (handle multidim keys)
        records: dict[str, dict[str, float]] = {}
        for k, v in item_params.items():
            a_val: float
            b_val: float
            if isinstance(v, dict):
                if "a" in v or "b" in v:
                    a_val = float(v.get("a", 1.0))
                    b_val = float(v.get("b", 0.0))
                else:
                    # py-irt multidim often uses keys like 'disc' and 'diff'
                    disc = v.get("disc", v.get("a", 1.0))
                    diff = v.get("diff", v.get("b", 0.0))
                    try:
                        a_arr = np.asarray(disc, dtype=float)
                        a_val = float(np.linalg.norm(a_arr))  # collapse to scalar
                    except Exception:
                        a_val = float(disc)  # type: ignore[arg-type]
                    try:
                        b_arr = np.asarray(diff, dtype=float)
                        b_val = float(np.mean(b_arr))  # collapse if vector
                    except Exception:
                        b_val = float(diff)  # type: ignore[arg-type]
            else:
                # Unknown structure; default conservative values
                a_val, b_val = 1.0, 0.0
            records[str(k)] = {"a": a_val, "b": b_val}

        df = pd.DataFrame(records).T
        df.index.name = "question_id"
        return df


def fit_2pl_parameters(matrix_df: pd.DataFrame, config: TrainingConfig | None = None) -> pd.DataFrame:
    """Fit or estimate 2PL parameters per item.

    If py-irt is available, trains a 2PL model; otherwise, uses a heuristic estimation.

    Returns a DataFrame indexed by question_id with columns ["a", "b"].
    """
    cfg = config or TrainingConfig()
    rows = _to_pyirt_jsonl_rows(matrix_df, threshold=cfg.threshold)
    try:
        return _train_with_pyirt(rows, cfg)
    except Exception:
        # Fallback to heuristic estimation for robustness
        return _heuristic_estimate_item_parameters(matrix_df)


