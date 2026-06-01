"""Reference vs test model splits (random / temporal / family-held-out).

This module is the single place that decides which models are *reference* (used to
calibrate IRT item parameters) and which are *held-out test* models. It is used by
``chain_linking_parallel.py`` to support the reviewer-requested split protocols
while keeping the original random 75/25 split (the published paper result)
bit-for-bit identical.

Split modes
-----------
random:
    The paper default — legacy global-RNG ``np.random.seed`` + ``np.random.choice``
    over the sorted model list. Reproduces the published numbers exactly.

time_ordered / noniid:
    efficbench-style temporal split (``run_experiment.py`` + ``process_lb_data``):
      1. Resolve a date per model from Open LLM Leaderboard evaluation revisions
         (preferred), falling back to Hugging Face model-repo metadata.
      2. Sort models newest-first.
      3. Test = first ``test_ratio`` fraction (newest); reference = older remainder.

family_holdout:
    Hold out entire model families as the test set (e.g. all llama-2 / qwen models).
    Uses the same deterministic partition mechanism; the held-out model ids come
    from ``model_families.load_holdout_models`` (an approved, reviewed list).

Dates are cached on disk (``data/input/model_release_dates.json`` for LB,
``model_dates_mmlu.json`` for MMLU) so cluster runs need no network access.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

# Project root: .../AdaptEval/src/experiments/chain_linking/model_splits.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

EFFICBENCH_DATE_SCENARIO = "harness_gsm8k_5"
# Default per-suite date caches (ISO strings or {date, source} objects).
DATES_CACHE_LB = PROJECT_ROOT / "data" / "input" / "model_release_dates.json"
DATES_CACHE_MMLU = PROJECT_ROOT / "data" / "input" / "model_dates_mmlu.json"
DEFAULT_DATES_CACHE = DATES_CACHE_LB
FALLBACK_DATE_ISO = "1970-01-01T00:00:00+00:00"

HF_DATASET_TREE_API = "https://huggingface.co/api/datasets/{id}/tree/main"
HF_DATASET_API = "https://huggingface.co/api/datasets/{id}"
HF_MODEL_API = "https://huggingface.co/api/models/{id}"

DateSource = Literal[
    "leaderboard_results",
    "leaderboard_dataset",
    "hf_model_created",
    "hf_model_modified",
    "unknown",
]

_RESULTS_DATE_RE = re.compile(
    r"results[_-](\d{4})[_-](\d{2})[_-](\d{2})",
    re.IGNORECASE,
)
_REVISION_DIR_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T",
)


def dates_cache_for_suite(data_source_mode: str) -> Path:
    """Return the on-disk date cache for a given data source mode."""
    if data_source_mode in ("mmlu_split", "mmlu_fields"):
        return DATES_CACHE_MMLU
    return DATES_CACHE_LB


@dataclass(frozen=True)
class ModelDateRecord:
    """Resolved temporal metadata for one LB pickle model id."""

    date: datetime
    source: DateSource

    def to_cache_entry(self) -> dict[str, str]:
        dt = self.date
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return {"date": dt.isoformat(), "source": self.source}


def lb_model_id_to_hf_repo(model_id: str) -> str:
    """Convert LB pickle model id to Hugging Face repo id (org/model)."""
    name = str(model_id)
    for prefix in ("open-llm-leaderboard/details_", "open-llm-leaderboard/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("__", "/")


def parse_efficbench_date_token(token: str) -> datetime | None:
    """Parse efficbench date strings (``%Y_%m_%d`` or ``%Y-%m-%d`` from first 10 chars)."""
    if not token:
        return None
    head = token[:10].replace("_", "-")
    try:
        return datetime.strptime(head, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _fetch_hf_json(url: str, retries: int = 4, pause_s: float = 1.0) -> object | None:
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as response:
                return json.load(response)
        except HTTPError as err:
            if err.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(pause_s * (attempt + 1))
                continue
            return None
        except URLError:
            if attempt + 1 < retries:
                time.sleep(pause_s * (attempt + 1))
                continue
            return None
    return None


def _dates_from_hf_tree(entries: list) -> list[datetime]:
    """Extract evaluation timestamps from an open-llm-leaderboard/details_* tree listing."""
    found: list[datetime] = []
    for entry in entries:
        path = entry.get("path", "") if isinstance(entry, dict) else str(entry)
        match = _RESULTS_DATE_RE.search(path)
        if match:
            y, mo, d = match.groups()
            found.append(datetime(int(y), int(mo), int(d), tzinfo=timezone.utc))
            continue
        rev = _REVISION_DIR_RE.match(path)
        if rev:
            y, mo, d = rev.groups()
            found.append(datetime(int(y), int(mo), int(d), tzinfo=timezone.utc))
    return found


def fetch_leaderboard_evaluation_date(model_id: str, pause_s: float = 0.0) -> ModelDateRecord | None:
    """Latest Open LLM Leaderboard evaluation date (efficbench GSM8K proxy).

    Reads ``open-llm-leaderboard/details_<org>__<model>`` on Hugging Face — the same
    repository efficbench fills via ``load_dataset(model_id, 'harness_gsm8k_5')``.
    Works for models not in efficbench's pickle if they were evaluated on the LB.
    """
    dataset_id = quote(model_id, safe="/")
    tree = _fetch_hf_json(HF_DATASET_TREE_API.format(id=dataset_id))
    if pause_s:
        time.sleep(pause_s)
    if isinstance(tree, list):
        candidates = _dates_from_hf_tree(tree)
        if candidates:
            return ModelDateRecord(max(candidates), "leaderboard_results")

    payload = _fetch_hf_json(HF_DATASET_API.format(id=dataset_id))
    if isinstance(payload, dict):
        dt = _parse_iso_timestamp(payload.get("lastModified") or payload.get("createdAt"))
        if dt is not None:
            return ModelDateRecord(dt, "leaderboard_dataset")
    return None


def fetch_hf_model_repo_date(
    model_id: str,
    pause_s: float = 0.0,
) -> ModelDateRecord | None:
    """Hugging Face model-repo timestamps (https://huggingface.co/<org>/<model>).

    ``createdAt`` is when the weights repo was first published; ``lastModified`` is the
    latest commit. Prefer ``createdAt`` for "model release" ordering when leaderboard
    details are missing. Returns None for deleted/private repos (HTTP 401/404).
    """
    repo = quote(lb_model_id_to_hf_repo(model_id), safe="/")
    payload = _fetch_hf_json(HF_MODEL_API.format(id=repo))
    if pause_s:
        time.sleep(pause_s)
    if not isinstance(payload, dict):
        return None

    # HF uses 2022-03-02 for repos created before tracking began — treat as unknown.
    created = _parse_iso_timestamp(payload.get("createdAt"))
    if created is not None and created.year > 2022:
        return ModelDateRecord(created, "hf_model_created")

    modified = _parse_iso_timestamp(payload.get("lastModified"))
    if modified is not None:
        return ModelDateRecord(modified, "hf_model_modified")
    return None


def fetch_model_temporal_date(
    model_id: str,
    pause_s: float = 0.12,
) -> ModelDateRecord:
    """Resolve a sortable date for any LB-style model id.

    Priority:
      1. Leaderboard ``details_*`` result revisions (best match to efficbench).
      2. Leaderboard ``details_*`` dataset created/updated metadata.
      3. Underlying HF model repo ``createdAt`` / ``lastModified``.
    """
    record = fetch_leaderboard_evaluation_date(model_id, pause_s=pause_s)
    if record is not None:
        return record

    record = fetch_hf_model_repo_date(model_id, pause_s=pause_s * 0.5)
    if record is not None:
        return record

    return ModelDateRecord(
        datetime.fromisoformat(FALLBACK_DATE_ISO.replace("Z", "+00:00")),
        "unknown",
    )


def _parse_cache_entry(raw: str | dict) -> tuple[str, DateSource]:
    """Support legacy cache values (plain ISO string) and enriched {date, source} objects."""
    if isinstance(raw, str):
        if raw.startswith("9999-") or raw.startswith("1970-01-01"):
            return raw, "unknown"
        return raw, "leaderboard_results"
    if isinstance(raw, dict):
        date = raw.get("date", FALLBACK_DATE_ISO)
        source = raw.get("source", "unknown")
        return date, source  # type: ignore[return-value]
    return FALLBACK_DATE_ISO, "unknown"


def _cache_date_iso(raw: str | dict) -> str:
    return _parse_cache_entry(raw)[0]


def build_model_gsm8k_dates(
    model_ids: Iterable[str],
    cache_path: Path | None = None,
    refresh: bool = False,
) -> dict[str, str]:
    """Build or load per-model dates (ISO strings) for sorting.

    Enriched entries in the JSON cache also store ``source``; this function returns
    only the date string for backward compatibility. Use ``build_model_date_records``
    for full metadata.
    """
    records = build_model_date_records(model_ids, cache_path=cache_path, refresh=refresh)
    return {m: rec.to_cache_entry()["date"] for m, rec in records.items()}


def build_model_date_records(
    model_ids: Iterable[str],
    cache_path: Path | None = None,
    refresh: bool = False,
) -> dict[str, ModelDateRecord]:
    """Build or load dated records with provenance (``source`` field in cache)."""
    cache_path = cache_path or DEFAULT_DATES_CACHE
    models = sorted(set(model_ids))
    cached_raw: dict[str, str | dict] = {}
    if cache_path.exists() and not refresh:
        cached_raw = json.loads(cache_path.read_text())
        if all(m in cached_raw for m in models):
            return {
                m: ModelDateRecord(
                    datetime.fromisoformat(_cache_date_iso(cached_raw[m]).replace("Z", "+00:00")),
                    _parse_cache_entry(cached_raw[m])[1],
                )
                for m in models
            }

    cache: dict[str, dict[str, str]] = {}
    for m, raw in cached_raw.items():
        iso, source = _parse_cache_entry(raw)
        cache[m] = {"date": iso, "source": source}

    to_fetch = list(models) if refresh else [m for m in models if m not in cache]
    if not refresh:
        to_fetch = sorted(
            set(to_fetch) | {m for m in models if cache.get(m, {}).get("source") == "unknown"}
        )

    for i, model_id in enumerate(to_fetch):
        record = fetch_model_temporal_date(model_id)
        entry = record.to_cache_entry()
        cache[model_id] = entry
        if record.source == "unknown":
            print(f"   Warning: no online date for {model_id} ({lb_model_id_to_hf_repo(model_id)})")
        if (i + 1) % 50 == 0:
            print(f"   Resolved dates for {i + 1}/{len(to_fetch)} models...")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    return {
        m: ModelDateRecord(
            datetime.fromisoformat(cache[m]["date"].replace("Z", "+00:00")),
            cache[m]["source"],  # type: ignore[arg-type]
        )
        for m in models
    }


def order_models_newest_first(
    models: Iterable[str],
    model_dates: dict[str, str] | None = None,
    dates_cache_path: Path | None = None,
) -> list[str]:
    """Return model ids sorted newest-first (matches efficbench ``process_lb_data``)."""
    model_list = sorted(set(models))
    if model_dates is None:
        model_dates = build_model_gsm8k_dates(model_list, cache_path=dates_cache_path)
    return sorted(
        model_list,
        key=lambda m: (model_dates.get(m, FALLBACK_DATE_ISO), m),
        reverse=True,
    )


def efficbench_noniid_test_indices(n_models: int, test_ratio: float = 0.25) -> list[int]:
    """Row indices used as test set in efficbench ``split='noniid'`` for lb/mmlu."""
    n_test = max(1, int(n_models * test_ratio))
    return list(range(n_test))


def split_reference_test_models(
    models: Iterable[str],
    test_ratio: float = 0.25,
    split_mode: str = "random",
    seed: int = 42,
    model_dates: dict[str, str] | None = None,
    dates_cache_path: Path | None = None,
    assume_pickle_ordered: bool = False,
    holdout_models: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Split models into reference (train) and held-out test sets.

    random:
        Uniform random 75/25 split (paper Section 4.1 default).

    time_ordered, noniid:
        efficbench temporal split — sort by resolved date (newest first), test =
        first ``test_ratio`` of models.

    family_holdout:
        Hold out entire model families: ``test`` = models in ``holdout_models``
        (the approved family set, see ``model_families``), ``reference`` = the rest.
        ``test_ratio`` is ignored.

    assume_pickle_ordered:
        If True with time_ordered/noniid, skip re-sorting (efficbench-sorted pickle).
    """
    model_list = list(dict.fromkeys(models))
    if len(model_list) < 2:
        raise ValueError(f"Need at least 2 models to split, got {len(model_list)}")

    if split_mode == "random":
        # IMPORTANT: replicate the paper (HEAD) random split BIT-FOR-BIT — legacy
        # global RNG (np.random.seed + np.random.choice) over the sorted model list.
        # Do NOT switch to np.random.default_rng: it yields a different partition for
        # the same seed and would break reproducibility of the published numbers.
        import numpy as np

        base_models = sorted(set(model_list))
        np.random.seed(seed)
        n_test = max(1, int(len(base_models) * test_ratio))
        test_models = set(np.random.choice(base_models, size=n_test, replace=False))
        train_models = set(base_models) - test_models
        return train_models, test_models

    n_test = max(1, int(len(model_list) * test_ratio))
    n_test = min(n_test, len(model_list) - 1)

    if split_mode in ("time_ordered", "noniid"):
        if assume_pickle_ordered:
            ordered = model_list
        else:
            ordered = order_models_newest_first(
                model_list, model_dates=model_dates, dates_cache_path=dates_cache_path,
            )
        test_models = set(ordered[:n_test])
        train_models = set(ordered[n_test:])
        return train_models, test_models

    if split_mode == "family_holdout":
        if not holdout_models:
            raise ValueError(
                "family_holdout requires holdout_models (the approved family set). "
                "Generate/approve it via `python -m src.experiments.chain_linking.model_families`."
            )
        present = set(model_list)
        test_models = present & set(holdout_models)
        if not test_models:
            raise ValueError("None of the held-out family models are present in this suite.")
        train_models = present - test_models
        if not train_models:
            raise ValueError("Family holdout would leave no reference models.")
        return train_models, test_models

    raise ValueError(
        f"Unknown split_mode {split_mode!r}. Expected 'random', 'time_ordered', "
        f"'noniid', or 'family_holdout'."
    )


# Backwards-compatible aliases
build_model_release_dates = build_model_gsm8k_dates


def fetch_model_release_date(model_id: str, pause_s: float = 0.12) -> datetime | None:
    record = fetch_model_temporal_date(model_id, pause_s=pause_s)
    return None if record.source == "unknown" else record.date


def fetch_gsm8k_submission_date(model_id: str, pause_s: float = 0.12) -> datetime | None:
    return fetch_model_release_date(model_id, pause_s=pause_s)
