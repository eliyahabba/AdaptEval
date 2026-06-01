"""Rule-based model-family classification for the family-held-out split.

The family-held-out experiment holds entire model families aside as the test set,
to probe how well calibration generalises to a lineage absent from the reference
pool. Because the Open LLM Leaderboard contains hundreds of community fine-tunes
with messy names, we do NOT try to classify every model. Instead we:

  1. Match models against a small set of *conservative* per-family rules
     (``include`` patterns minus ``exclude`` patterns) on the Hugging Face repo id.
  2. Assign a model to a family only when it matches exactly one family.
  3. Report every model that matches more than one family (or only a weak hint) as
     ``ambiguous`` so a human can decide — we never force these in or out.

The proposed held-out set is written to ``config/holdout_families_<suite>.json`` for
review before any experiment runs. Matching is purely string-based (no network).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.experiments.chain_linking.model_splits import PROJECT_ROOT, lb_model_id_to_hf_repo

HOLDOUT_FAMILIES_DIR = PROJECT_ROOT / "config"


def holdout_config_for_suite(data_source_mode: str) -> Path:
    """Per-suite approved family-holdout config path."""
    suite = "mmlu" if data_source_mode in ("mmlu_split", "mmlu_fields") else "lb"
    return HOLDOUT_FAMILIES_DIR / f"holdout_families_{suite}.json"


@dataclass(frozen=True)
class FamilyRule:
    """Conservative include/exclude regexes for one model family."""
    name: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = field(default_factory=tuple)

    def matches(self, repo: str) -> bool:
        if any(re.search(p, repo) for p in self.exclude):
            return False
        return any(re.search(p, repo) for p in self.include)


# Distinctive lineage tokens. Patterns run on the lowercased "org/model" repo id.
# Kept deliberately tight to avoid sweeping in unrelated fine-tunes/merges.
FAMILY_RULES: tuple[FamilyRule, ...] = (
    FamilyRule(
        name="llama2",
        # Require a token boundary after the version digit so model SIZE strings like
        # "llama-34b" are NOT misread as a version ("llama-2"/"llama-3").
        include=(r"llama[-_ ]?2\b",),
        exclude=(r"llama[-_ ]?3\b", r"tinyllama", r"codellama"),
    ),
    FamilyRule(
        name="llama3",
        include=(r"llama[-_ ]?3\b",),
        exclude=(r"llama[-_ ]?2\b", r"tinyllama", r"codellama"),
    ),
    FamilyRule(
        name="qwen",
        include=(r"\bqwen", r"/qwen"),
        exclude=(),
    ),
    FamilyRule(
        name="mistral",
        include=(r"\bmistral", r"/mistral"),
        exclude=(r"mixtral",),  # Mixtral (MoE) is a distinct lineage; leave for review.
    ),
)


@dataclass
class FamilyAssignment:
    """Result of classifying a set of model ids into families."""
    by_family: dict[str, list[str]]
    ambiguous: dict[str, list[str]]   # model_id -> families it matched (>1)
    unmatched: list[str]

    def counts(self) -> dict[str, int]:
        return {fam: len(ids) for fam, ids in self.by_family.items()}


def classify_model(model_id: str) -> list[str]:
    """Return the list of families whose rules match ``model_id`` (0, 1, or many)."""
    repo = lb_model_id_to_hf_repo(model_id).lower()
    return [rule.name for rule in FAMILY_RULES if rule.matches(repo)]


def build_family_assignment(model_ids: Iterable[str]) -> FamilyAssignment:
    """Classify ``model_ids`` into families, isolating ambiguous/unmatched models."""
    by_family: dict[str, list[str]] = {rule.name: [] for rule in FAMILY_RULES}
    ambiguous: dict[str, list[str]] = {}
    unmatched: list[str] = []

    for mid in sorted(set(model_ids)):
        fams = classify_model(mid)
        if len(fams) == 1:
            by_family[fams[0]].append(mid)
        elif len(fams) > 1:
            ambiguous[mid] = fams
        else:
            unmatched.append(mid)

    by_family = {fam: ids for fam, ids in by_family.items() if ids}
    return FamilyAssignment(by_family=by_family, ambiguous=ambiguous, unmatched=unmatched)


def propose_holdout_families(
    model_ids: Iterable[str],
    families: list[str] | None = None,
    top_k: int = 3,
    min_models: int = 5,
) -> tuple[dict[str, list[str]], FamilyAssignment]:
    """Pick the held-out families.

    If ``families`` is given, hold out exactly those. Otherwise choose the ``top_k``
    families with the most members (and at least ``min_models``). Returns
    (holdout {family: [ids]}, full assignment) — the full assignment exposes the
    ambiguous/unmatched lists so they can be surfaced for human review.
    """
    assignment = build_family_assignment(model_ids)
    if families is None:
        ranked = sorted(assignment.by_family.items(), key=lambda kv: len(kv[1]), reverse=True)
        families = [fam for fam, ids in ranked if len(ids) >= min_models][:top_k]
    holdout = {fam: assignment.by_family.get(fam, []) for fam in families}
    return holdout, assignment


def write_holdout_config(
    holdout: dict[str, list[str]],
    assignment: FamilyAssignment,
    path: Path,
) -> Path:
    """Persist the proposed held-out families + review metadata to JSON."""
    payload = {
        "families": sorted(holdout),
        "holdout_models": {fam: sorted(ids) for fam, ids in holdout.items()},
        "n_holdout": sum(len(v) for v in holdout.values()),
        "review": {
            "all_family_counts": assignment.counts(),
            "ambiguous": assignment.ambiguous,
            "n_unmatched": len(assignment.unmatched),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_holdout_models(path: Path) -> set[str]:
    """Load the approved held-out model ids from the family config."""
    if not path.exists():
        raise FileNotFoundError(
            f"Family-holdout config not found at {path}. Generate it first with "
            f"`python -m src.experiments.chain_linking.model_families --data-source-mode lb` "
            f"and approve the list."
        )
    payload = json.loads(path.read_text())
    models: set[str] = set()
    for ids in payload.get("holdout_models", {}).values():
        models.update(ids)
    return models


def _load_model_ids(data_source_mode: str) -> list[str]:
    """Load all model ids for a suite without building the full response frames."""
    from src.experiments.equating.cross_dataset_equating import (
        ExperimentConfig,
        load_all_datasets,
    )

    datasets = load_all_datasets(ExperimentConfig(data_source_mode=data_source_mode))
    models: set[str] = set()
    for df in datasets.values():
        models.update(df["model_name"].astype(str).unique())
    return sorted(models)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Propose held-out model families for review.")
    ap.add_argument("--data-source-mode", default="lb",
                    choices=["lb", "lb_only", "tinybenchmarks", "mmlu_split", "mmlu_fields"])
    ap.add_argument("--families", nargs="*", default=None,
                    help="Explicit families to hold out (default: auto-pick top-k by count).")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output config path (default: config/holdout_families_<suite>.json).")
    args = ap.parse_args()

    out_path = args.out or holdout_config_for_suite(args.data_source_mode)
    model_ids = _load_model_ids(args.data_source_mode)
    holdout, assignment = propose_holdout_families(model_ids, families=args.families, top_k=args.top_k)
    path = write_holdout_config(holdout, assignment, out_path)

    print(f"\nScanned {len(model_ids)} models ({args.data_source_mode}).")
    print(f"Family counts (all matched): {assignment.counts()}")
    print(f"\nProposed held-out families -> {args.families or f'top-{args.top_k}'}:")
    for fam, ids in holdout.items():
        print(f"  {fam}: {len(ids)} models")
    print(f"Total held out: {sum(len(v) for v in holdout.values())} / {len(model_ids)}")
    if assignment.ambiguous:
        print(f"\nAMBIGUOUS (matched >1 family, NOT held out — review): {len(assignment.ambiguous)}")
        for mid, fams in list(assignment.ambiguous.items())[:20]:
            print(f"  {mid} -> {fams}")
    print(f"\nWrote proposal to {path}. Review/approve before running the family-holdout experiment.")


if __name__ == "__main__":
    main()
