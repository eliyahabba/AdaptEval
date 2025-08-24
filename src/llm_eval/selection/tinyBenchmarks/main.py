from __future__ import annotations

from pathlib import Path
import json
import typer
import pandas as pd

from llm_eval.matrix import MatrixStorage
from .training import fit_2pl_parameters, TrainingConfig
from .anchors import find_anchor_items, AnchorConfig
from .estimation import estimate_theta_from_anchors, expected_correctness, blend_anchor_and_irt, EstimationConfig


app = typer.Typer(add_completion=False, help="TinyBenchmarks workflow: train IRT, find anchors, estimate.")


@app.command()
def train_irt(
    matrix: str = typer.Option(..., help="Path to matrix Parquet produced by MatrixBuilder"),
    out: str = typer.Option(..., help="Output path for learned item params Parquet (a,b per question_id)"),
    model_type: str = typer.Option("multidim_2pl"),
    threshold: float = typer.Option(50.0, help="Binarization threshold for normalized_score"),
    num_epochs: int = typer.Option(2000),
    seed: int = typer.Option(42),
    dims: int = typer.Option(10),
    lr: float = typer.Option(0.1),
    lr_decay: float = typer.Option(0.9999),
    dropout: float = typer.Option(0.5),
    hidden: int = typer.Option(100),
    priors: str = typer.Option("hierarchical"),
    deterministic: bool = typer.Option(True),
    log_every: int = typer.Option(200),
    device: str | None = typer.Option(None, help="'cuda' or 'cpu' (None lets py-irt decide)"),
    dims_search: str | None = typer.Option(None, help="Optional comma-separated list for D search, e.g. '5,10'"),
    val_stride: int = typer.Option(5, help="Validation stride over models for D search"),
    number_item_per_scenario: int = typer.Option(100, help="For lambda heuristic like notebook"),
):
    df = MatrixStorage(matrix).load()
    cfg = TrainingConfig(
        model_type=model_type,
        threshold=threshold,
        num_epochs=num_epochs,
        seed=seed,
        dims=dims,
        lr=lr,
        lr_decay=lr_decay,
        dropout=dropout,
        hidden=hidden,
        priors=priors,
        deterministic=deterministic,
        log_every=log_every,
        device=device,
        dims_search=[int(x) for x in dims_search.split(",")] if dims_search else None,
        val_stride=val_stride,
        number_item_per_scenario=number_item_per_scenario,
    )
    params = fit_2pl_parameters(df, cfg)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params.to_parquet(out_path)
    typer.echo(f"Saved item parameters to {out}")


@app.command()
def anchors(
    item_params: str = typer.Option(..., help="Path to item params Parquet (from train-irt)"),
    out: str = typer.Option(..., help="Output JSON path with anchors list"),
    per_level: int = typer.Option(5),
    levels: int = typer.Option(10),
):
    params = pd.read_parquet(item_params)
    acfg = AnchorConfig(per_level=per_level, levels=levels)
    anchor_ids = find_anchor_items(params, acfg)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"anchors": anchor_ids}, f)
    typer.echo(f"Saved {len(anchor_ids)} anchors to {out}")


@app.command()
def estimate(
    item_params: str = typer.Option(..., help="Path to item params Parquet (from train-irt)"),
    anchors_path: str = typer.Option(..., help="Path to anchors JSON (from anchors)"),
    out: str = typer.Option(..., help="Output Parquet with predictions per question_id and theta"),
    # Option A: Provide anchor responses directly
    anchor_responses_csv: str | None = typer.Option(None, help="CSV with columns: question_id,response(0/1)"),
    # Option B: Derive anchor responses from matrix for a specific model
    matrix: str | None = typer.Option(None, help="Path to matrix Parquet containing the candidate model"),
    model_name: str | None = typer.Option(None, help="Candidate model name in the matrix"),
    threshold: float = typer.Option(50.0, help="Binarization threshold when deriving responses from matrix"),
    blend_lambda: bool = typer.Option(False, help="Blend anchor and IRT predictions using gp-IRT style"),
):
    params = pd.read_parquet(item_params)
    lambdas_by_dataset = params.attrs.get("lambdas_by_dataset", {}) if hasattr(params, "attrs") else {}
    with open(anchors_path, "r") as f:
        data = json.load(f)
    anchor_ids = [str(x) for x in data.get("anchors", [])]
    if not anchor_ids:
        raise typer.BadParameter("No anchors found in anchors JSON")

    # Prepare anchor responses series
    if anchor_responses_csv:
        ar = pd.read_csv(anchor_responses_csv)
        if not {"question_id", "response"}.issubset(ar.columns):
            raise typer.BadParameter("anchor_responses_csv must have columns question_id,response")
        responses = pd.Series(ar["response"].astype(int).values, index=ar["question_id"].astype(str).values)
    else:
        if matrix is None or model_name is None:
            raise typer.BadParameter("Provide either anchor_responses_csv or both matrix and model_name")
        df = MatrixStorage(matrix).load()
        sub = df[(df["model_name"].astype(str) == str(model_name)) & (df["question_id"].astype(str).isin(anchor_ids))]
        if sub.empty:
            raise typer.BadParameter("No matching anchor responses found for the given model_name in matrix")
        correct = (sub["normalized_score"].astype(float) >= float(threshold)).astype(int)
        responses = pd.Series(correct.values, index=sub["question_id"].astype(str).values)

    theta = estimate_theta_from_anchors(params, responses)
    preds_irt = expected_correctness(params, theta)
    # Simple anchor-only estimate per item: project anchor mean to all items (baseline)
    anchor_mean = float(responses.astype(float).mean()) if len(responses) else 0.5
    preds_anchor = pd.Series(anchor_mean, index=preds_irt.index)
    if blend_lambda and lambdas_by_dataset:
        item_to_dataset = None
        # If item_params preserved dataset column, use it; otherwise skip
        if "dataset" in params.columns:
            item_to_dataset = params["dataset"]
        preds = blend_anchor_and_irt(preds_anchor, preds_irt, lambdas_by_dataset, item_to_dataset)
    else:
        preds = preds_irt
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame({"question_id": preds.index.astype(str), "predicted_p": preds.values})
    res.to_parquet(out_path)
    typer.echo(f"Estimated theta={theta:.4f}; saved predictions to {out}")


if __name__ == "__main__":
    app()














