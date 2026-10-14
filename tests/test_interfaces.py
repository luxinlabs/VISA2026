import copy, json
import pytest
from agentisa import interfaces, fact_inventory
from agentisa.interfaces import render, audit_equivalence, tool_get_facts, CONDITIONS, task_ops
from agentisa.tasks import all_tasks


def test_audit_passes_for_every_task():
    for t in all_tasks():
        assert audit_equivalence(t)["matching_fact_ids"]


def test_audit_detects_tampered_prose(monkeypatch):
    t = all_tasks("dev")[0]
    real = interfaces.render_prose
    monkeypatch.setattr(interfaces, "render_prose", lambda fs: real(fs[:-1]))
    with pytest.raises(ValueError):
        audit_equivalence(t)


def test_no_fact_ids_in_prose_and_ids_in_structured():
    t = all_tasks("dev")[0]
    p, s = render(t, "prose"), render(t, "structured")
    assert not any(i in p["content"] for i in p["fact_ids"])
    assert all(i in s["content"] for i in s["fact_ids"])


def test_condition_fact_sets():
    t = all_tasks("dev")[0]
    ids = {c: set(render(t, c)["fact_ids"]) for c in CONDITIONS}
    assert ids["none"] == set()
    assert ids["minimal"] == ids["syntax_only_structured"] < ids["prose"] == ids["structured"] == ids["interactive"]
    assert render(t, "interactive")["content"] == ""


def test_tool_is_restricted_to_task_ops():
    t = [t for t in all_tasks("dev") if t["template"] == "matmul"][0]
    ops = task_ops(t)
    out, ids = tool_get_facts(ops, "reduce")
    assert "error" in out and ids == []
    out, ids = tool_get_facts(ops, "stablehlo.dot_general", "shape")
    assert ids and all(i.startswith("DOT.SHP") for i in ids)


def test_fact_ids_unique_and_anchored():
    ids = [f["id"] for f in fact_inventory.FACTS]
    assert len(ids) == len(set(ids))
    assert all(f["source_anchor"].startswith(fact_inventory.SPEC) for f in fact_inventory.FACTS)
