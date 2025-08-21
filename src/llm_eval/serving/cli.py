

from pathlib import Path
import json
import typer
import pandas as pd

from src.llm_eval.config import load_yaml_config
from src.llm_eval.utils import get_logger
from src.llm_eval.ingestion import LocalCSVSource
try:
    from src.llm_eval.ingestion.hf_datasets import HuggingFaceDatasetSource  # type: ignore
except Exception:  # pragma: no cover - optional dep
    HuggingFaceDatasetSource = None  # type: ignore
from src.llm_eval.normalization import MetricRegistry
from src.llm_eval.matrix import MatrixBuilder, MatrixStorage
from src.llm_eval.selection import NaiveVarianceSelector, IRT2PLSelector, MITVSelector, ModelProfile
try:
    from src.llm_eval.selection import PyIRTSelector  # type: ignore
except Exception:  # pragma: no cover - optional dep
    PyIRTSelector = None  # type: ignore
try:
    from src.llm_eval.selection import TinyBenchmarksSelector  # type: ignore
except Exception:  # pragma: no cover - optional dep
    TinyBenchmarksSelector = None  # type: ignore
from src.llm_eval.scoring import SnapshotManager, Leaderboard
from src.llm_eval.evaluation import simulate_selection_impact


app = typer.Typer(add_completion=False)
logger = get_logger(__name__)


@app.command()
def ingest(source: str = typer.Option("local_files"), path: str = typer.Option(...), out: str = typer.Option(...)):  # noqa: E501
    cfg = load_yaml_config(
        str(Path(__file__).parents[2] / "config" / "defaults.yaml"),
        str(Path(__file__).parents[2] / "config" / "metrics.yaml"),
    )
    if source != "local_files":
        raise typer.BadParameter("Only local_files supported in MVP")
    src = LocalCSVSource(path)
    raw = src.load()
    registry = MetricRegistry(cfg)
    builder = MatrixBuilder(registry)
    matrix = builder.build(raw)
    MatrixStorage(out).save(matrix)
    logger.info("Ingested and saved to %s", out)


@app.command()
def select(method: str = typer.Option("naive"), k: int = typer.Option(10), model_name: str = typer.Option(...), matrix: str = typer.Option(...), out: str = typer.Option(...)):  # noqa: E501
    df = MatrixStorage(matrix).load()
    profile = ModelProfile(model_name=model_name)
    if method == "naive":
        selector = NaiveVarianceSelector()
    elif method == "irt":
        selector = IRT2PLSelector()
    elif method in {"mitv", "interview"}:
        selector = MITVSelector()
    elif method in {"py-irt", "py_irt"}:
        if PyIRTSelector is None:
            raise typer.BadParameter("py-irt method requested but optional dependencies not installed. Install with extras: pip install .[irt]")
        selector = PyIRTSelector()
    elif method in {"tinyb", "tinybench", "tiny-benchmarks", "tiny_benchmarks"}:
        if TinyBenchmarksSelector is None:
            raise typer.BadParameter("TinyBenchmarks selector not available")
        selector = TinyBenchmarksSelector()
    else:
        raise typer.BadParameter("Unknown method")
    questions = selector.select(profile, k, df)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"questions": questions}, f)
    logger.info("Selection written to %s", out)


@app.command()
def ingest_multi(
    inputs: list[str] = typer.Argument(..., help="List of sources: csv:PATH or hf:DATASET[:SPLIT]"),
    out: str = typer.Option(..., help="Output Parquet path for unified matrix"),
    dump_raw_dir: str | None = typer.Option(None, help="Optional directory to dump per-source raw CSVs"),
):
    """Ingest multiple sources (CSV/HF), merge, normalize and write a unified matrix.

    Examples:
      llm-eval ingest-multi csv:examples/tiny_dataset.csv hf:boolq:validation --out data/processed/matrix.parquet
    """
    cfg = load_yaml_config(
        str(Path(__file__).parents[2] / "config" / "defaults.yaml"),
        str(Path(__file__).parents[2] / "config" / "metrics.yaml"),
    )
    raw_frames: list[pd.DataFrame] = []
    for item in inputs:
        if item.startswith("csv:"):
            path = item.split(":", 1)[1]
            df = LocalCSVSource(path).load()
            if dump_raw_dir:
                Path(dump_raw_dir).mkdir(parents=True, exist_ok=True)
                pd.DataFrame(df).to_csv(Path(dump_raw_dir) / (Path(path).stem + "_raw.csv"), index=False)
            raw_frames.append(df)
        elif item.startswith("hf:"):
            if HuggingFaceDatasetSource is None:
                raise typer.BadParameter("HF source requested but optional dependencies not installed. Install with extras: pip install .[hf]")
            rest = item.split(":", 1)[1]
            if ":" in rest:
                ds_name, split = rest.split(":", 1)
            else:
                ds_name, split = rest, "test"
            df = HuggingFaceDatasetSource(ds_name, split=split).load()
            if dump_raw_dir:
                Path(dump_raw_dir).mkdir(parents=True, exist_ok=True)
                pd.DataFrame(df).to_csv(Path(dump_raw_dir) / (f"{ds_name}_{split}_raw.csv"), index=False)
            raw_frames.append(df)
        else:
            raise typer.BadParameter(f"Unrecognized input format: {item}")

    if not raw_frames:
        raise typer.BadParameter("No valid inputs provided")

    merged = pd.concat(raw_frames, ignore_index=True, sort=False)
    registry = MetricRegistry(cfg)
    builder = MatrixBuilder(registry)
    matrix = builder.build(merged)
    MatrixStorage(out).save(matrix)
    logger.info("Ingested %d sources and saved unified matrix to %s", len(raw_frames), out)


@app.command()
def score(matrix: str = typer.Option(...), out: str = typer.Option(...)):
    df = MatrixStorage(matrix).load()
    snap = SnapshotManager(out)
    snapshot_id = snap.create_snapshot(df)
    scores = snap.aggregate_scores(df)
    lb = Leaderboard().from_scores(scores)
    MatrixStorage(out).save(df.assign(snapshot_id=snapshot_id))
    logger.info("Leaderboard top:\n%s", lb.head().to_string(index=False))


@app.command()
def simulate(matrix: str = typer.Option(...), method: str = typer.Option("irt"), k: int = typer.Option(8), model_name: str = typer.Option("candidate")):
    df = MatrixStorage(matrix).load()
    profile = ModelProfile(model_name=model_name)
    if method == "irt":
        selector = IRT2PLSelector()
    elif method in {"py-irt", "py_irt"}:
        if PyIRTSelector is None:
            raise typer.BadParameter("py-irt method requested but optional dependencies not installed. Install with extras: pip install .[irt]")
        selector = PyIRTSelector()
    elif method in {"mitv", "interview"}:
        selector = MITVSelector()
    else:
        selector = NaiveVarianceSelector()
    res = simulate_selection_impact(selector, profile, k, df)
    logger.info("Selected %d questions (coverage=%.2f)", len(res.selected_questions), res.coverage)


if __name__ == "__main__":
    app()


