"""Integrity of the committed trial logs: one record per cell, and the trial counts the paper reports."""
import json
from collections import Counter
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/v2"
PRE_REGISTERED = sorted(V2.glob("trackA_gpt-4.1*_eval.jsonl"))
POST_HOC_DIRS = ("ablation", "repeats", "full_spec", "haiku", "family", "addendum7", "addendum8")
POST_HOC = sorted(f for d in POST_HOC_DIRS for f in (V2 / d).glob("*.jsonl")
                  if not f.name.endswith("_transcripts.jsonl") and not f.name.startswith("pilot"))


def records(path):
    return [json.loads(line) for line in path.open()]


@pytest.mark.parametrize("path", PRE_REGISTERED + POST_HOC + sorted(ROOT.glob("results/trials_*_eval.jsonl")), ids=lambda p: str(p.relative_to(ROOT)))
def test_one_record_per_model_task_condition_trial(path):
    cells = Counter((r["model"], r["task_id"], r["condition"], r.get("trial")) for r in records(path))
    assert [c for c, n in cells.items() if n > 1] == []


def test_pre_registered_trial_count():
    assert sum(len(records(f)) for f in PRE_REGISTERED) == 5005


def test_post_hoc_trial_count():
    if not all((V2 / d).is_dir() for d in POST_HOC_DIRS):
        pytest.skip("post-hoc logs not present")
    assert sum(len(records(f)) for f in POST_HOC) == 14002
