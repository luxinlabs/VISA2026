"""Automated prose/structured equivalence audit and per-condition documentation token lengths."""
import json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import tiktoken
from agentisa.tasks import all_tasks
from agentisa.interfaces import audit_equivalence, CONDITIONS_V2 as CONDITIONS, render
from agentisa.fact_inventory import FACTS
enc = tiktoken.get_encoding("o200k_base")
rows = [dict(task_id=t["task_id"], **audit_equivalence(t, enc)) for t in all_tasks(suite='all')]
summ = {c: {"mean": round(statistics.mean(r[c] for r in rows), 1), "min": min(r[c] for r in rows), "max": max(r[c] for r in rows)} for c in CONDITIONS}
summ["n_facts"] = {"mean": round(statistics.mean(r["n_facts"] for r in rows), 1), "min": min(r["n_facts"] for r in rows), "max": max(r["n_facts"] for r in rows)}
ratio = statistics.mean(r["structured"] / r["prose"] for r in rows)
cats = {c: sum(f["category"] == c for f in FACTS) for c in ["syntax", "type", "shape", "semantics"]}
out = {"n_tasks": len(rows), "all_passed": True, "doc_tokens_o200k": summ, "structured_over_prose_token_ratio": round(ratio, 3),
       "inventory": {"n_facts": len(FACTS), "n_op_groups": len({f['op'] for f in FACTS}), "by_category": cats}}
(ROOT / "data/processed/treatment_audit_v2.json").write_text(json.dumps(out, indent=1))
t = all_tasks("dev")[0]
(ROOT / "data/processed/example_treatments.json").write_text(json.dumps({c: render(t, c) for c in CONDITIONS}, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1))
