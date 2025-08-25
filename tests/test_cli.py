from typer.testing import CliRunner
from src.llm_eval.serving.cli import app


def test_cli_ingest_and_select(tmp_path):
    runner = CliRunner()
    out_ingest = tmp_path / "matrix.parquet"
    result_ingest = runner.invoke(app, [
        "ingest",
        "--source", "local_files",
        "--path", "examples/tiny_dataset.csv",
        "--out", str(out_ingest),
    ])
    assert result_ingest.exit_code == 0, result_ingest.output

    out_matrix = out_ingest

    out_select = tmp_path / "selection.json"
    result_select = runner.invoke(app, [
        "select",
        "--method", "irt",
        "--k", "2",
        "--model-name", "new_model",
        "--matrix", str(out_matrix),
        "--out", str(out_select),
    ])
    assert result_select.exit_code == 0, result_select.output


