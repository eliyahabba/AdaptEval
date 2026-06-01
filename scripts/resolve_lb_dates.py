#!/usr/bin/env python3
"""Re-resolve fallback/unknown model dates in a date cache and merge them back.

The shared ``build_model_date_records`` short-circuits when every model is already
present in the cache (even if some entries are fallbacks), so it will not re-fetch
prior rate-limited failures. This tool targets exactly those fallback entries:

  1. Load the cache (LB by default).
  2. Find every entry whose resolved source is ``unknown`` (stored fallback dates
     ``9999-12-31`` or ``1970-01-01``, or ``{"source": "unknown"}``).
  3. Re-fetch each from the public Hugging Face API (leaderboard ``details_*``
     result revisions, then HF model-repo metadata).
  4. Write the cache back in the enriched ``{"date", "source"}`` format.

Run OFF-CLUSTER (needs internet). Idempotent: only fallback entries are re-fetched,
so re-running after a rate-limit pause keeps resolving the remainder.

Usage:
    python scripts/resolve_lb_dates.py                       # LB cache
    python scripts/resolve_lb_dates.py --suite mmlu          # MMLU cache
    python scripts/resolve_lb_dates.py --cache path/to.json  # explicit cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.experiments.chain_linking.model_splits import (  # noqa: E402
    DATES_CACHE_LB,
    DATES_CACHE_MMLU,
    FALLBACK_DATE_ISO,
    fetch_model_temporal_date,
    lb_model_id_to_hf_repo,
    _parse_cache_entry,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["lb", "mmlu"], default="lb")
    ap.add_argument("--cache", type=Path, default=None,
                    help="Explicit cache path (overrides --suite).")
    ap.add_argument("--pause", type=float, default=0.2,
                    help="Seconds to pause between model fetches (politeness).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only attempt the first N unknowns (for a quick partial pass).")
    args = ap.parse_args()

    cache_path = args.cache or (DATES_CACHE_LB if args.suite == "lb" else DATES_CACHE_MMLU)
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        return 1

    raw = json.loads(cache_path.read_text())
    # Normalise to enriched {date, source} entries.
    cache: dict[str, dict[str, str]] = {}
    for m, v in raw.items():
        iso, source = _parse_cache_entry(v)
        cache[m] = {"date": iso, "source": source}

    unknown = sorted(m for m, e in cache.items() if e["source"] == "unknown")
    if args.limit:
        unknown = unknown[: args.limit]
    print(f"Cache: {cache_path}")
    print(f"Total models: {len(cache)} | fallback/unknown to resolve: {len(unknown)}")
    if not unknown:
        print("Nothing to resolve.")
        return 0

    t0 = time.time()
    resolved = 0
    still: list[str] = []
    src_counter: Counter[str] = Counter()
    for i, model_id in enumerate(unknown):
        rec = fetch_model_temporal_date(model_id, pause_s=args.pause)
        cache[model_id] = rec.to_cache_entry()
        src_counter[rec.source] += 1
        if rec.source == "unknown":
            still.append(model_id)
        else:
            resolved += 1
        if (i + 1) % 25 == 0:
            print(f"   {i + 1}/{len(unknown)} attempted "
                  f"({resolved} resolved, {len(still)} still unknown)...", flush=True)

    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    final_unknown = sum(1 for e in cache.values() if e["source"] == "unknown")
    print(f"\nDone in {time.time() - t0:.0f}s. Re-fetched {len(unknown)} fallbacks.")
    print(f"Resolved this run: {resolved} | still unknown this run: {len(still)}")
    print(f"Source breakdown (this run): {dict(src_counter)}")
    print(f"\nCache now: {len(cache)} models, {len(cache) - final_unknown} dated, "
          f"{final_unknown} genuinely unknown.")
    if still:
        print(f"\nGenuinely unknown after research ({len(still)}) — "
              f"these get fallback date {FALLBACK_DATE_ISO[:10]} (oldest -> reference set):")
        for m in still:
            print(f"  {lb_model_id_to_hf_repo(m)}   ({m})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
