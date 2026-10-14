"""Agent-facing interface treatments rendered from one canonical fact inventory.

Conditions (all receive the identical task statement, signature and list of relevant operations):
  none                    no documentation (model prior only)
  minimal                 syntax facts only, natural-language rendering
  syntax_only_structured  syntax facts only, structured rendering (negative control: format without semantics)
  prose                   all facts, natural-language paragraphs (fact.nl), no IDs
  structured              all facts, JSON contract (fact.formal) keyed by category, with IDs
  interactive             no facts upfront; the same structured facts are retrievable through a bounded tool
  interactive_forced      POST-HOC exploratory (added 2026-10-13 after observing zero tool use): as interactive,
                          but the first model turn is required to call the tool (tool_choice="required")
The fact set for a task is the 'module' facts plus the facts of the task's relevant operations.
"""
import json
from .fact_inventory import FACTS, OP_SUMMARY

CONDITIONS = ["none", "minimal", "syntax_only_structured", "prose", "structured", "interactive"]
CATEGORIES = ["syntax", "type", "shape", "semantics"]


def task_ops(task):
    return ["module"] + [o for o in task["ops"] if o != "module"]


def select_facts(ops, categories=None):
    cats = set(categories or CATEGORIES)
    order = {o: i for i, o in enumerate(ops)}
    fs = [f for f in FACTS if f["op"] in order and f["category"] in cats]
    return sorted(fs, key=lambda f: (order[f["op"]], CATEGORIES.index(f["category"]), FACTS.index(f)))


def _title(op):
    return "Program structure and types" if op == "module" else (
        "Element-wise operations" if op == "elementwise" else f"stablehlo.{op}")


def render_prose(facts):
    parts, cur = [], None
    for f in facts:
        if f["op"] != cur:
            cur = f["op"]; parts.append(f"\n### {_title(cur)}\n")
        parts.append(f["nl"] + " ")
    return "".join(parts).strip()


def render_structured(facts):
    ops = []
    for f in facts:
        if not ops or ops[-1]["operation"] != _title(f["op"]):
            ops.append({"operation": _title(f["op"])})
        ops[-1].setdefault(f["category"], []).append({"id": f["id"], "rule": f["formal"]})
    return json.dumps({"stablehlo_contract": ops}, indent=1, ensure_ascii=False)


def render(task, condition):
    """Returns dict(condition, fact_ids, content). content is placed before the task statement."""
    ops = task_ops(task)
    if condition == "none":
        return {"condition": condition, "fact_ids": [], "content": ""}
    if condition in ("minimal", "syntax_only_structured"):
        facts = select_facts(ops, ["syntax"])
        content = render_prose(facts) if condition == "minimal" else render_structured(facts)
        return {"condition": condition, "fact_ids": [f["id"] for f in facts], "content": content}
    if condition == "prose":
        facts = select_facts(ops)
        return {"condition": condition, "fact_ids": [f["id"] for f in facts], "content": render_prose(facts)}
    if condition == "structured":
        facts = select_facts(ops)
        return {"condition": condition, "fact_ids": [f["id"] for f in facts], "content": render_structured(facts)}
    if condition in ("interactive", "interactive_forced"):
        # available (not delivered) facts; delivered facts are logged per trial from tool calls
        return {"condition": condition, "fact_ids": [f["id"] for f in select_facts(ops)], "content": "",
                "available_ops": ops}
    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------- interactive tool
TOOL_SPEC = [{
    "type": "function",
    "function": {
        "name": "get_facts",
        "description": "Return the StableHLO documentation facts for one operation, optionally restricted to one category.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "Operation group name, e.g. 'broadcast_in_dim', 'reduce', 'module'."},
                "category": {"type": "string", "enum": CATEGORIES + ["all"],
                             "description": "syntax, type, shape, semantics, or all."},
            },
            "required": ["operation"],
        },
    },
}]


def tool_get_facts(available_ops, operation, category="all"):
    op = (operation or "").strip().replace("stablehlo.", "")
    if op not in available_ops:
        return {"error": f"Unknown or unavailable operation '{operation}'. Available: {available_ops}"}, []
    cats = CATEGORIES if category in (None, "", "all") else [category]
    if any(c not in CATEGORIES for c in cats):
        return {"error": f"Unknown category '{category}'. Use one of {CATEGORIES + ['all']}"}, []
    facts = select_facts([op], cats)
    return json.loads(render_structured(facts)), [f["id"] for f in facts]


def interactive_preamble(available_ops):
    lines = "\n".join(f"- {o}: {OP_SUMMARY[o]}" for o in available_ops)
    return ("Documentation is not included in this message. You can look it up with the get_facts tool "
            "(arguments: operation, and optionally category = syntax | type | shape | semantics | all). "
            f"Documentation is available for these operation groups:\n{lines}\n"
            "You may make at most {max_calls} tool calls in total for this task.")


# ---------------------------------------------------------------- equivalence audit
def audit_equivalence(task, tokenizer=None):
    """Automated audit of the prose/structured pair for one task (necessary, not sufficient):
    identical fact-ID sets; every fact's nl sentence occurs verbatim in prose and its formal rule in
    structured; no fact ID leaks into prose; and token lengths are recorded."""
    p, s = render(task, "prose"), render(task, "structured")
    if p["fact_ids"] != s["fact_ids"]:
        raise ValueError("Fact ID sequences differ between prose and structured")
    by_id = {f["id"]: f for f in FACTS}
    for fid in p["fact_ids"]:
        f = by_id[fid]
        if f["nl"] not in p["content"]:
            raise ValueError(f"{fid}: nl rendering missing from prose")
        if json.dumps(f["formal"], ensure_ascii=False)[1:-1] not in s["content"]:
            raise ValueError(f"{fid}: formal rendering missing from structured")
        if fid in p["content"]:
            raise ValueError(f"{fid}: fact ID leaked into prose")
    out = {"n_facts": len(p["fact_ids"]), "matching_fact_ids": True, "human_semantic_audit_required": True}
    if tokenizer is not None:
        out.update({c: len(tokenizer.encode(render(task, c)["content"])) for c in CONDITIONS})
    return out
