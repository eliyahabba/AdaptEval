"""MMLU subject groupings for the out-of-domain stress test.

The stress test calibrates on a *narrow* slice of MMLU (a skill group such as STEM,
or a tighter math cluster) and adds only that slice's subjects sequentially along
the chain. We implement it by filtering the loaded dataset dict down to the subjects
in the requested group before base/chain/target assignment, so the existing chain
machinery is reused unchanged.

Dataset names look like ``harness_hendrycksTest_<subject>_5``; matching is on the
``<subject>`` token so it is robust to prefixes/suffixes.
"""
from __future__ import annotations

import re

# MMLU STEM category (standard 4-way MMLU grouping: STEM / humanities / social / other).
MMLU_STEM_SUBJECTS: frozenset[str] = frozenset({
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "machine_learning",
})

# Tighter math-only cluster.
MMLU_MATH_SUBJECTS: frozenset[str] = frozenset({
    "abstract_algebra", "college_mathematics", "elementary_mathematics",
    "high_school_mathematics", "high_school_statistics",
})

SUBJECT_GROUPS: dict[str, frozenset[str]] = {
    "stem": MMLU_STEM_SUBJECTS,
    "math": MMLU_MATH_SUBJECTS,
}

_SUBJECT_RE = re.compile(r"hendrycks[tT]est[_-](?P<subject>.+?)(?:[_-]\d+)?$")


def dataset_subject(dataset_name: str) -> str:
    """Extract the bare MMLU subject token from a dataset name."""
    m = _SUBJECT_RE.search(dataset_name)
    if m:
        return m.group("subject")
    return dataset_name


def datasets_in_group(dataset_names, group: str) -> list[str]:
    """Return the subset of ``dataset_names`` belonging to ``group`` (stem|math)."""
    if group not in SUBJECT_GROUPS:
        raise ValueError(f"Unknown subject group {group!r}. Expected one of {list(SUBJECT_GROUPS)}.")
    wanted = SUBJECT_GROUPS[group]
    return [d for d in dataset_names if dataset_subject(d) in wanted]
