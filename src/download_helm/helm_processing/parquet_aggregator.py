from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLUMNS: List[str] = [
    "dataset_name",
    "hf_split",
    "hf_index",
    "model_name",
    "model_family",
    "evaluation_method_name",
    "evaluation_score",
]


def _batch_iter(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def aggregate_to_parquet(
    input_dir: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 10,
) -> Path:
    """Aggregate many HELM CSVs into a single Parquet file.

    - Reads files in batches of `batch_size` to keep memory bounded
    - Keeps only REQUIRED_COLUMNS
    - Normalizes hf_split: 'valid' -> 'validation'
    """

    base_dir = Path(__file__).parent
    in_dir = input_dir or (base_dir / "converted_data")
    out_path = output_path or (base_dir / "extracted" / "helm_aggregated.parquet")

    in_dir = in_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    csv_files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {in_dir}")

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for batch in _batch_iter(csv_files, batch_size):
        frames: List[pd.DataFrame] = []
        for csv_path in batch:
            try:
                df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS)
            except Exception as e:
                print(f"[skip] Failed reading {csv_path.name}: {e}")
                continue

            # Normalize split naming
            if "hf_split" in df.columns:
                df["hf_split"] = df["hf_split"].replace({"valid": "validation"})

            frames.append(df)

        if not frames:
            continue

        batch_df = pd.concat(frames, ignore_index=True)

        # Ensure stable column order and types where possible
        batch_df = batch_df.reindex(columns=REQUIRED_COLUMNS)

        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total_rows += len(batch_df)
        print(f"Wrote batch of {len(batch_df):,} rows (total {total_rows:,})")

    if writer is not None:
        writer.close()
    else:
        raise RuntimeError("No data written. Check input CSV files.")

    print(f"Done. Aggregated {total_rows:,} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    aggregate_to_parquet()


