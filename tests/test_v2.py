import pytest
from agentisa.tasks import all_tasks
from agentisa.interfaces import render, select_facts, audit_equivalence
from agentisa.fact_inventory import FACTS_BY_ID

jax = pytest.importorskip("jax")


def test_syntax_facts_carry_same_code_in_both_renderings():
    f = FACTS_BY_ID["BID.SYN.1"]
    for code in ("array<i64: 1>", "array<i64: ...>"):
        assert code in f["nl"] and code in f["formal"]


def test_kernelbench_tasks_load_and_reference_passes():
    from agentisa.harness import grade
    kb = all_tasks("dev", suite="kernelbench")
    assert kb and all(t["source"] == "kernelbench" for t in kb)
    t = kb[0]
    assert audit_equivalence(t)["matching_fact_ids"]
    assert grade(t["sol"], t).semantic_correct
    assert not grade(t["sol"].replace("stablehlo.add", "stablehlo.subtract", 1), t).semantic_correct or "stablehlo.add" not in t["sol"]


def test_interactive_core_delivers_only_module_facts_upfront():
    t = all_tasks("dev")[0]
    r = render(t, "interactive_core")
    assert set(r["upfront_fact_ids"]) == {f["id"] for f in select_facts(["module"])}
    assert "module" not in r["available_ops"] and r["content"]


def test_agent_tools_static_check_and_example():
    from agentisa.agent import _static_check, _run_example
    t = [t for t in all_tasks("dev") if t["template"] == "relu"][0]
    assert _static_check(t["sol"], t)[0]
    assert _run_example(t["sol"], t)[0]
    ok, msg = _run_example(t["sol"].replace("stablehlo.maximum", "stablehlo.minimum"), t)
    assert not ok and "Mismatch" in msg
    ok, msg = _static_check("func.func @main() {}", t)
    assert not ok
