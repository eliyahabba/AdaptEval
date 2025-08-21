## AdaptEval (llm-eval): Modular LLM Evaluation MVP

AdaptEval is a modular, extensible system for evaluating LLMs across datasets, tasks, and metrics, with adaptive question selection (IRT and naive baselines), normalization to a common 0–100 scale, time-aware scoring with snapshots, and both CLI and API.

### Quickstart
1) Install (Python 3.11+):
```bash
pip install -e .
```
or with Poetry:
```bash
poetry install
poetry run llm-eval --help
```

2) End-to-end demo using `examples/tiny_dataset.csv`:
```bash
llm-eval ingest --source local_files --path examples/tiny_dataset.csv --out data/processed/matrix.parquet
llm-eval select --method irt --k 3 --model-name new_model --matrix data/processed/matrix.parquet --out data/processed/selection.json
llm-eval score --matrix data/processed/matrix.parquet --out data/processed/leaderboard.parquet
llm-eval simulate --matrix data/processed/matrix.parquet --method irt --k 8 --model-name new_model
```

3) Serve API:
```bash
uvicorn llm_eval.serving.api:app --host 0.0.0.0 --port 8000
```
Endpoints: POST `/select_questions`, `/score_model`, `/snapshot`.

### Architecture
- Ingestion → Normalization → Matrix → Selection → Scoring → Serving
- Pluggable ingestion sources (CSV now; HF adapters stubbed)
- Metric registry and monotone normalization (min-max, z-score→CDF)
- Parquet storage; easy to switch to DuckDB/Arrow IPC
- IRT 2PL-lite selector and naive baselines (variance/entropy)
- Time-aware scoring with snapshots and leaderboard

### Assumptions
- IRT uses a lightweight heuristic on normalized scores; replaceable with full estimation later.
- For non-binary metrics, normalized score is a proxy probability for IRT.
- Metric ranges configurable in `src/llm_eval/config/metrics.yaml`; fallback to z-score→CDF.

See module READMEs and docstrings for details. Future work: Bayesian IRT (3PL), contextual priors, improved estimators, streaming ingestion.


