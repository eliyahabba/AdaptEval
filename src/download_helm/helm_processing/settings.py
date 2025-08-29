"""
Centralized project configuration and registries.

Edit this file to update environment-specific paths, dataset/model registries,
and frequently changed constants. This module is intentionally minimal and free
of side-effects so it can be safely imported anywhere.

Guidelines:
- Keep values deterministic and static; do not compute values from runtime state.
- If you change paths, ensure they exist or are created by call sites (not here).
- If you add dataset/model entries, follow existing key naming conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------------------
# Paths (environment-specific)
# --------------------------------------------------------------------------------------

# Path to the HF mapping data directory used by `convert_cluade.py`.
# NOTE: This value mirrors the previously hardcoded path to preserve behavior.
HF_MAP_DATA_DIR: Path = Path(__file__).parent.parent.parent.parent / "data" / "hf_map_data"

# Path to model metadata CSV file used by model utilities
MODEL_METADATA_CSV: Path = Path(__file__).parent / "model_metadata.csv"

# Default CSVs used by CLI defaults in download/processor scripts. Kept identical
# to existing per-script defaults to avoid behavior changes.
DEFAULT_CSV_FILE_DOWNLOAD: Path = Path(__file__).parent.parent.parent / "download_helm" / "download" / "helm_lite_v1.13.0.csv"
DEFAULT_CSV_FILE_PROCESSOR: Path = Path(__file__).parent.parent.parent / "download_helm" / "download" / "helm_lite_v1.13.0.csv"

# Subdirectory names (relative to the module locations that use them)
DOWNLOADS_SUBDIR: str = "downloads"
OUTPUT_SUBDIR: str = "converted_data"


# --------------------------------------------------------------------------------------
# HELM download & processing settings
# --------------------------------------------------------------------------------------

# Versions to search for when downloading HELM files (kept identical ordering)
HELM_VERSIONS: List[str] = [f"v1.{i}.0" for i in range(14)]  # v1.0.0 to v1.13.0

# Default starting version
DEFAULT_START_VERSION: str = "v1.0.0"

# Base URL template for HELM lite assets (unchanged)
HELM_LITE_BASE_URL_TEMPLATE: str = (
    "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/{version}"
)

# File types fetched per task (unchanged)
HELM_FILE_TYPES: List[str] = [
    "run_spec",
    "stats",
    "per_instance_stats",
    "instances",
    "scenario_state",
    "display_predictions",
    "display_requests",
    "scenario",
]

# Concurrency for `ProcessPoolExecutor` in `helm_data_processor.py`
PROCESS_POOL_MAX_WORKERS: int = 16


# --------------------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------------------

# Map dataset base names to HF repos. Mirrors previous in-file mapping.
DATASET_REGISTRY: Dict[str, str] = {
    "med_qa": "bigbio/med_qa",
    "openbook_qa": "allenai/openbookqa",
    "mmlu": "cais/mmlu",
    "gsm": "openai/gsm8k",
    "gsm8k": "openai/gsm8k",
    "legalbench": "nguha/legalbench",
    "narrativeqa": "deepmind/narrativeqa",
    "wmt14": "wmt/wmt14",
}

# Optional: central place for known model metadata (names -> info). Currently
# the project derives model info from `model_metadata.csv`. Keep this mapping
# empty to preserve behavior, but leave as an extension point for future use.
MODEL_REGISTRY: Dict[str, dict] = {}


# --------------------------------------------------------------------------------------
# Small utility constants
# --------------------------------------------------------------------------------------

# Progress bar format used in `helm_data_processor.py` (kept identical)
TQDM_BAR_FORMAT: str = (
    "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


