import pandas as pd
from src.llm_eval.selection import NaiveVarianceSelector, MITVSelector, ModelProfile, TinyBenchmarksSelector


def _load_matrix():
    return pd.read_csv("examples/tiny_dataset.csv")


def test_naive_selector_on_demo():
    df = _load_matrix()
    # fabricate normalized_score for test if missing
    if "normalized_score" not in df.columns:
        df = df.assign(normalized_score=(df["raw_score"].astype(float) * (100.0 if df["metric_name"].iloc[0] == "binary_acc" else 1.0)))  # noqa: E501
    sel = NaiveVarianceSelector()
    q = sel.select(ModelProfile(model_name="X"), 2, df)
    assert len(q) == 2


def test_tiny_benchmarks_selector_on_demo():
    df = _load_matrix()
    if "normalized_score" not in df.columns:
        df = df.assign(normalized_score=(df["raw_score"].astype(float) * 100.0))
    sel = TinyBenchmarksSelector()
    q = sel.select(ModelProfile(model_name="X"), 2, df)
    assert len(q) == 2


def test_mitv_selector_on_demo():
    df = _load_matrix()
    if "normalized_score" not in df.columns:
        df = df.assign(normalized_score=(df["raw_score"].astype(float) * 100.0))
    sel = MITVSelector()
    q = sel.select(ModelProfile(model_name="X"), 3, df)
    assert len(q) == 3

