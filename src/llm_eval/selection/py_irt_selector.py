

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.llm_eval.selection.interfaces import QuestionSelector, ModelProfile
from src.llm_eval.selection.cold_start import simple_cold_start_theta


@dataclass
class _PyIrtAvailable:
    ok: bool
    reason: str | None = None


def _check_py_irt_available() -> _PyIrtAvailable:
    try:
        import py_irt  # noqa: F401
        import pyro  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on optional deps
        return _PyIrtAvailable(False, str(e))
    return _PyIrtAvailable(True)


class PyIRTSelector(QuestionSelector):
    """Selector backed by py-irt 2PL model.

    Notes
    -----
    - Requires optional dependencies: torch, pyro-ppl, py-irt.
    - Binarizes `normalized_score` into correctness with threshold (default 50).
    - Fits a 2PL model and scores items by Fisher information at target theta.
    """

    def __init__(self, threshold: float = 50.0, num_epochs: int = 500, seed: int | None = 0) -> None:
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.seed = seed

    def _to_pyirt_jsonl_rows(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        # Expect columns: model_name, question_id, normalized_score
        if not {"model_name", "question_id", "normalized_score"}.issubset(df.columns):
            raise ValueError("matrix_df missing required columns for py-irt")
        # binarize
        correct = (df["normalized_score"].astype(float) >= self.threshold).astype(int)
        rows: list[dict[str, Any]] = []
        for subject_id, item_id, y in zip(df["model_name"], df["question_id"], correct):
            rows.append({
                "subject_id": str(subject_id),
                "item_id": str(item_id),
                "response": int(y),
            })
        return rows

    def _fit_pyirt(self, rows: list[dict[str, Any]]):  # pragma: no cover - slow/stochastic
        import json
        import tempfile
        from pathlib import Path

        from py_irt.training import IrtModelTrainer
        from py_irt.config import IrtConfig

        # Write jsonl to a temp file because trainer expects a file path
        with tempfile.TemporaryDirectory() as td:
            data_path = Path(td) / "data.jsonl"
            with open(data_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            cfg = IrtConfig(
                model_type="2pl",
                num_epochs=self.num_epochs,
                seed=self.seed,
                dataset_path=str(data_path),
                validate_every=0,
            )
            trainer = IrtModelTrainer(cfg)
            trainer.train()
            # Extract learned parameters
            item_params = trainer.model.item_param_store  # type: ignore[attr-defined]
            # item_params is mapping item_id -> {"a": float, "b": float} for 2PL
            return {str(k): {"a": float(v["a"]), "b": float(v["b"]) } for k, v in item_params.items()}

    @staticmethod
    def _fisher_information(theta: float, a: float, b: float) -> float:
        p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        return float((a ** 2) * p * (1 - p))

    def select(self, model: ModelProfile, k: int, matrix_df: pd.DataFrame) -> list[str]:
        avail = _check_py_irt_available()
        if not avail.ok:
            raise RuntimeError(f"py-irt not available: {avail.reason}")

        theta = simple_cold_start_theta(model)
        rows = self._to_pyirt_jsonl_rows(matrix_df)
        params = self._fit_pyirt(rows)

        infos: list[tuple[str, float]] = []
        for item_id, ab in params.items():
            a = float(ab.get("a", 1.0))
            b = float(ab.get("b", 0.0))
            info = self._fisher_information(theta, a, b)
            infos.append((item_id, info))

        infos.sort(key=lambda x: x[1], reverse=True)
        return [qid for qid, _ in infos[:k]]


