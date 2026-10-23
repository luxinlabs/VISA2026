"""Agent-facing interface treatments rendered from one canonical fact inventory.

Conditions (all receive the identical task statement, signature and list of relevant operations):
  none                    no documentation (model prior only)
  minimal                 syntax facts only, natural-language rendering
  syntax_only_structured  syntax facts only, structured rendering (negative control: format without semantics)
  prose                   all facts, natural-language paragraphs (fact.nl), no IDs
  structured              all facts, JSON contract (fact.formal) keyed by category, with IDs
  interactive             no facts upfront; the same structured facts are retrievable through a bounded tool
  interactive_core        v2: cross-cutting 'module' facts delivered upfront (structured), operation facts on demand
  interactive_forced      (v1: POST-HOC exploratory; v2: pre-registered) (added 2026-10-13 after observing zero tool use): as interactive,
                          but the first model turn is required to call the tool (tool_choice="required")
  full_spec               (camera-ready, Addendum 5) the whole pinned StableHLO specification instead of curated facts
The fact set for a task is the 'module' facts plus the facts of the task's relevant operations.
"""
import functools, hashlib, json
from pathlib import Path
from .fact_inventory import FACTS, OP_SUMMARY

CONDITIONS = ["none", "minimal", "syntax_only_structured", "prose", "structured", "interactive"]
# v2 (pre-registered 2026-10-14): forced first retrieval becomes a planned condition, plus core-upfront hybrid
CONDITIONS_V2 = CONDITIONS + ["interactive_forced", "interactive_core"]
# Reviewer-requested ablation (2026-10-15): the structured contract's JSON wrapper, IDs and category grouping,
# but each rule carries the prose sentence (fact.nl) instead of the formal predicate.
ABLATION = ["structured_nl"]
# Reviewer-requested (camera-ready, 2026-10-19): the raw pinned spec (91k o200k tokens) as the documentation.
ABLATION += ["full_spec"]
# Mechanism test (camera-ready, Addendum 7): contract C with every quoted code span moved out of the JSON strings
# into a plain code block keyed by fact ID, so no example needs JSON escaping; content otherwise identical to C.
ABLATION += ["structured_rawcode"]
_ROOT = Path(__file__).resolve().parents[2]
# The pinned spec comes from the StableHLO clone described in data/DATA_SOURCES.md (commit in data/versions.lock.json);
# a standalone copy at data/vendor/stablehlo_spec.md is also accepted. Either must match the pinned SHA-256.
SPEC_CANDIDATES = (_ROOT / "data/vendor/stablehlo/docs/spec.md", _ROOT / "data/vendor/stablehlo_spec.md")


@functools.lru_cache(maxsize=1)
def pinned_spec_text():
    lock = json.loads((_ROOT / "data/versions.lock.json").read_text())
    for path in SPEC_CANDIDATES:
        if path.exists():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == lock["stablehlo_spec_sha256"]:
                return data.decode("utf-8")
    raise FileNotFoundError(
        "full_spec needs the pinned StableHLO spec: clone https://github.com/openxla/stablehlo into data/vendor/stablehlo "
        f"and check out commit {lock['stablehlo_spec_commit']} (see data/DATA_SOURCES.md); "
        f"spec.md must match sha256 {lock['stablehlo_spec_sha256']}")
INTERACTIVE = ("interactive", "interactive_forced", "interactive_core")
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


def render_structured(facts, field="formal"):
    ops = []
    for f in facts:
        if not ops or ops[-1]["operation"] != _title(f["op"]):
            ops.append({"operation": _title(f["op"])})
        ops[-1].setdefault(f["category"], []).append({"id": f["id"], "rule": f[field]})
    # v2: compact JSON so the structured rendering is length-matched to prose (v1 used indent=1, +31% tokens)
    return json.dumps({"stablehlo_contract": ops}, ensure_ascii=False, separators=(",", ":"))


def render_structured_rawcode(facts):
    import re
    blocks, out = [], []
    for f in facts:
        k = 0
        def lift(m):
            nonlocal k
            k += 1; ref = f"{f['id']}#{k}"; blocks.append(f"{ref}: {m.group(1)}"); return f"[code {ref}]"
        out.append(dict(f, formal=re.sub(r"`([^`]*\"[^`]*)`", lift, f["formal"])))
    body = render_structured(out)
    return body + ("\n\nCode referenced above:\n```\n" + "\n".join(blocks) + "\n```" if blocks else "")


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
    if condition == "structured_nl":
        facts = select_facts(ops)
        return {"condition": condition, "fact_ids": [f["id"] for f in facts], "content": render_structured(facts, field="nl")}
    if condition == "structured_rawcode":
        facts = select_facts(ops)
        return {"condition": condition, "fact_ids": [f["id"] for f in facts], "content": render_structured_rawcode(facts)}
    if condition == "full_spec":
        return {"condition": condition, "fact_ids": [], "content": pinned_spec_text()}
    if condition == "structured":
        facts = select_facts(ops)
        return {"condition": condition, "fact_ids": [f["id"] for f in facts], "content": render_structured(facts)}
    if condition == "interactive_core":
        core = select_facts(["module"])
        return {"condition": condition, "fact_ids": [f["id"] for f in select_facts(ops)],
                "content": render_structured(core), "upfront_fact_ids": [f["id"] for f in core],
                "available_ops": [o for o in ops if o != "module"]}
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


def interactive_preamble(available_ops, core_upfront=False):
    lines = "\n".join(f"- {o}: {OP_SUMMARY[o]}" for o in available_ops)
    head = ("The program-structure and syntax rules that apply to every operation are included below; documentation "
            "for the individual operations is not. " if core_upfront else "Documentation is not included in this message. ")
    return (head + "You can look it up with the get_facts tool "
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
        out.update({c: len(tokenizer.encode(render(task, c)["content"])) for c in CONDITIONS_V2})
    return out
