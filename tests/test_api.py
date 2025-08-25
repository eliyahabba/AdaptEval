from fastapi.testclient import TestClient
from src.llm_eval.serving.api import app


def test_select_questions_endpoint(monkeypatch, tmp_path):
    # Create a tiny matrix parquet
    import pandas as pd
    from src.llm_eval.matrix import MatrixStorage
    df = pd.DataFrame({
        "dataset": ["d"],
        "task_type": ["binary_qa"],
        "question_id": ["q1"],
        "model_name": ["A"],
        "metric_name": ["binary_acc"],
        "raw_score": [1.0],
        "normalized_score": [100.0],
        "timestamp": ["2024-01-01T00:00:00Z"],
    })
    path = tmp_path / "m.parquet"
    MatrixStorage(str(path)).save(df)

    client = TestClient(app)
    resp = client.post("/select_questions", json={
        "method": "irt",
        "k": 1,
        "model_name": "X",
        "matrix_path": str(path),
    })
    assert resp.status_code == 200
    assert "questions" in resp.json()


