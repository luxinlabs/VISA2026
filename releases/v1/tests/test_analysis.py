import pytest
from agentisa.analysis import mcnemar_exact, holm, paired_effect, wilson


def rows(pairs):
    out = []
    for i, (a, b) in enumerate(pairs):
        for c, v in (("prose", a), ("structured", b)):
            out.append({"model": "m", "task_id": f"t{i}", "template": f"T{i // 2}", "condition": c,
                        "first_attempt_semantic_correct": v})
    return out


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 6) - 2 / 64) < 1e-12
    assert abs(mcnemar_exact(3, 3) - 1.0) < 1e-12


def test_holm_monotone():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_paired_effect_label_and_missing():
    r = rows([(0, 1), (0, 1), (1, 1), (1, 0)])
    r.append({"model": "m", "task_id": "orphan", "template": "X", "condition": "prose", "first_attempt_semantic_correct": 1})
    r.append({"model": "other", "task_id": "t0", "template": "T0", "condition": "prose", "first_attempt_semantic_correct": 1})
    e = paired_effect(r, "m", "prose", "structured", n_boot=200)
    assert e["contrast"] == "structured - prose" and e["paired_tasks"] == 4
    assert e["missing_unpaired_tasks"] == ["orphan"]
    assert abs(e["effect"] - 0.25) < 1e-12
    assert e["discordant"] == {"structured_only": 2, "prose_only": 1}


def test_string_booleans_rejected_or_parsed():
    r = rows([(0, 1)]); r[0]["first_attempt_semantic_correct"] = "false"
    assert paired_effect(r, "m", "prose", "structured", n_boot=10)["effect"] == 1
    r[0]["first_attempt_semantic_correct"] = "maybe"
    with pytest.raises(ValueError):
        paired_effect(r, "m", "prose", "structured", n_boot=10)


def test_wilson_bounds():
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0 < hi < 0.35
