import pandas as pd
from src.llm_eval.ingestion import LocalCSVSource


def test_local_csv_ingestion(tmp_path):
    src = LocalCSVSource("examples/tiny_dataset.csv")
    df = src.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


