import pandas as pd
from src.llm_eval.config import load_yaml_config
from src.llm_eval.normalization import MetricRegistry
from src.llm_eval.matrix import MatrixBuilder


def test_build_matrix():
    cfg = load_yaml_config("src/llm_eval/config/defaults.yaml", "src/llm_eval/config/metrics.yaml")
    reg = MetricRegistry(cfg)
    builder = MatrixBuilder(reg)
    df = pd.read_csv("examples/tiny_dataset.csv")
    matrix = builder.build(df)
    assert {"normalized_score", "metric_name", "model_name", "question_id"}.issubset(matrix.columns)
    assert matrix["normalized_score"].between(0, 100).all()


