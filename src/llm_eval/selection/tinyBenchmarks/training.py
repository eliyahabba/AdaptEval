

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
    # Validation/workflow helpers
    dims_search: list[int] | None = None  # e.g., [5, 10]
    val_stride: int = 5  # take every Nth model as validation when searching dims
    number_item_per_scenario: int = 100  # for lambda heuristic like in notebook


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
def _compute_thresholds_per_dataset(matrix_df: pd.DataFrame, candidates: np.ndarray | None = None) -> dict[str, float]:
    """Find per-dataset binarization thresholds that best match average probabilities.

    Mirrors notebook logic where thresholds per scenario are chosen to minimize the
    difference between binarized mean and raw mean correctness.
    """
    if candidates is None:
        candidates = np.linspace(0.01, 0.99, 100)
    if "dataset" not in matrix_df.columns:
        # Single global threshold on normalized_score/100
        p = (matrix_df["normalized_score"].astype(float) / 100.0)
        diffs = [float(np.mean(np.abs((p >= c).astype(int).groupby(matrix_df["model_name"]).mean() - p.groupby(matrix_df["model_name"]).mean()))) for c in candidates]
        return {"__global__": float(candidates[int(np.argmin(diffs))])}
    thresholds: dict[str, float] = {}
    for ds, grp in matrix_df.groupby("dataset"):
        p = (grp["normalized_score"].astype(float) / 100.0)
        # group by model to average like notebook (per subject)
        diffs = [float(np.mean(np.abs((p >= c).astype(int).groupby(grp["model_name"]).mean() - p.groupby(grp["model_name"]).mean()))) for c in candidates]
        thresholds[str(ds)] = float(candidates[int(np.argmin(diffs))])
    return thresholds


def _binarize_with_thresholds(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    p = (df["normalized_score"].astype(float) / 100.0)
    if "__global__" in thresholds or "dataset" not in df.columns:
        c = thresholds.get("__global__", 0.5)
        return (p >= c).astype(int)
    # per-dataset
    cs = df["dataset"].map(lambda d: thresholds.get(str(d), 0.5)).astype(float)
    return (p >= cs.values).astype(int)


def _build_rows_from_matrix(matrix_df: pd.DataFrame, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    df_tmp = matrix_df.copy()
    df_tmp["y"] = _binarize_with_thresholds(df_tmp, thresholds)
    df_tmp["subject_id"] = df_tmp["model_name"].astype(str)
    df_tmp["item_id"] = df_tmp["question_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for subject_id, grp in df_tmp.groupby("subject_id"):
        responses: dict[str, int] = {str(i): int(y) for i, y in zip(grp["item_id"], grp["y"])}
        rows.append({"subject_id": str(subject_id), "responses": responses})
    return rows


def _estimate_theta_from_seen(item_params: pd.DataFrame, seen_responses: pd.Series, init_theta: float = 0.0) -> float:
    # Reuse the same MLE as in estimation, generalized for any seen subset
    common = item_params.index.intersection(seen_responses.index)
    if len(common) == 0:
        return init_theta
    a = item_params.loc[common, "a"].astype(float).values
    b = item_params.loc[common, "b"].astype(float).values
    y = seen_responses.loc[common].astype(float).values
    theta = float(init_theta)
    for _ in range(50):
        z = a * (theta - b)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = np.sum(a * (y - p))
        hess = -np.sum((a ** 2) * p * (1 - p)) - 1e-6
        step = grad / hess
        theta_new = theta - step
        if abs(theta_new - theta) < 1e-4:
            theta = theta_new
            break
        theta = theta_new
    return float(theta)


def _evaluate_dims_on_validation(matrix_df: pd.DataFrame, params_by_d: dict[int, pd.DataFrame], thresholds: dict[str, float], val_models: list[str]) -> tuple[int, dict[str, float]]:
    """Return best D and per-dataset validation errors like the notebook's errors2."""
    # Prepare binarized responses for val models
    df = matrix_df[matrix_df["model_name"].astype(str).isin(val_models)].copy()
    df["y"] = _binarize_with_thresholds(df, thresholds)
    # Define seen/unseen split by alternating items globally
    all_items = sorted(df["question_id"].astype(str).unique())
    seen_items = set(all_items[::2])
    unseen_items = set(all_items[1::2])

    avg_errors_per_d: dict[int, dict[str, float]] = {}
    for D, params in params_by_d.items():
        ds_errors: dict[str, list[float]] = {}
        for model_name, grp in df.groupby("model_name"):
            # Build series of seen responses
            seen = grp[grp["question_id"].astype(str).isin(seen_items)]
            seen_series = pd.Series(seen["y"].values, index=seen["question_id"].astype(str).values)
            theta = _estimate_theta_from_seen(params, seen_series)
            # Predict on unseen per dataset and compute abs error vs true
            for ds, gds in grp.groupby("dataset") if "dataset" in grp.columns else [("__all__", grp)]:
                g_unseen = gds[gds["question_id"].astype(str).isin(unseen_items)]
                if g_unseen.empty:
                    continue
                items_idx = g_unseen["question_id"].astype(str).values
                sub_params = params.loc[params.index.intersection(items_idx)]
                z = sub_params["a"].astype(float) * (theta - sub_params["b"].astype(float))
                p_hat = 1.0 / (1.0 + np.exp(-z))
                true = g_unseen["y"].astype(float).values
                # Aggregate error per model per dataset
                err = float(np.abs(p_hat.mean() - true.mean()))
                ds_key = str(ds)
                ds_errors.setdefault(ds_key, []).append(err)
        # Average per dataset
        avg_errors_per_d[D] = {ds: float(np.mean(es)) for ds, es in ds_errors.items() if es}

    # Choose best D by macro-average across datasets
    best_D = min(avg_errors_per_d.keys(), key=lambda d: np.mean(list(avg_errors_per_d[d].values())) if avg_errors_per_d[d] else float("inf"))
    return best_D, avg_errors_per_d.get(best_D, {})


def fit_2pl_parameters(matrix_df: pd.DataFrame, config: TrainingConfig | None = None) -> pd.DataFrame:
    """Fit or estimate 2PL parameters per item.

    If py-irt is available, trains a 2PL model; otherwise, uses a heuristic estimation.

    Returns a DataFrame indexed by question_id with columns ["a", "b"].
    """
    cfg = config or TrainingConfig()
    # Compute per-dataset thresholds
    thresholds = _compute_thresholds_per_dataset(matrix_df)
    # If searching over dims, build train/val split by models
    if cfg.dims_search:
        models = sorted(matrix_df["model_name"].astype(str).unique())
        val_models = models[::max(1, int(cfg.val_stride))]
        train_models = [m for m in models if m not in set(val_models)]
        train_df = matrix_df[matrix_df["model_name"].astype(str).isin(train_models)]
        params_by_d: dict[int, pd.DataFrame] = {}
        for D in cfg.dims_search:
            cfg_d = TrainingConfig(**{**cfg.__dict__, "dims": D})
            rows = _build_rows_from_matrix(train_df, thresholds)
            try:
                params_by_d[D] = _train_with_pyirt(rows, cfg_d)
            except Exception:
                params_by_d[D] = _heuristic_estimate_item_parameters(train_df)
        best_D, ds_val_errors = _evaluate_dims_on_validation(matrix_df, params_by_d, thresholds, val_models)
        # Final train on all models using best_D
        cfg_final = TrainingConfig(**{**cfg.__dict__, "dims": best_D})
    else:
        cfg_final = cfg
        ds_val_errors = {}
    # Train final
    rows_all = _build_rows_from_matrix(matrix_df, thresholds)
    try:
        params = _train_with_pyirt(rows_all, cfg_final)
    except Exception:
        params = _heuristic_estimate_item_parameters(matrix_df)
    # Attach thresholds to params as attributes for downstream (not persisted here)
    params.attrs["thresholds"] = thresholds
    params.attrs["val_errors_by_dataset"] = ds_val_errors
    # Compute lambdas per dataset using notebook-like heuristic when possible
    lambdas: dict[str, float] = {}
    if "dataset" in matrix_df.columns:
        # variance of raw probabilities per scenario across models
        p_all = (matrix_df["normalized_score"].astype(float) / 100.0)
        for ds, grp in matrix_df.groupby("dataset"):
            v = float(np.var((grp["normalized_score"].astype(float) / 100.0).values, ddof=0))
            b = float(ds_val_errors.get(str(ds), 0.05))  # fallback small error
            denom = v / max(1, cfg.number_item_per_scenario) / 4.0  # approximate scaling
            lamb = (b * b) / (denom + (b * b))
            lambdas[str(ds)] = float(lamb)
    params.attrs["lambdas_by_dataset"] = lambdas
    return params


