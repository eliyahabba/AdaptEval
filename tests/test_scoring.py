import pandas as pd
from src.llm_eval.scoring import SnapshotManager, Leaderboard


def test_scoring_snapshot(tmp_path):
    # minimal matrix
    df = pd.DataFrame({
        "model_name": ["A", "A", "B"],
        "normalized_score": [80.0, 90.0, 70.0],
        "question_id": ["q1", "q2", "q1"],
    })
    out = tmp_path / "snap.parquet"
    mgr = SnapshotManager(str(out))
    sid = mgr.create_snapshot(df)
    assert isinstance(sid, str)
    scores = mgr.aggregate_scores(df)
    lb = Leaderboard().from_scores(scores)
    assert set(lb.columns) == {"model_name", "cumulative_score"}


