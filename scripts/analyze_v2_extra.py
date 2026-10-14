"""Additional analyses of existing Study 1/2 logs (no new trials):
(a) confound check: structured-prose on the identical 114 synthetic tasks, Study 1 vs Study 2;
(b) retrieval mechanism: correctness by whether the agent retrieved the cross-cutting module facts (observational);
(c) per-suite share of failed first attempts that passed the verifier (does semantics ever decide outcomes?);
(d) mean input tokens next to accuracy for upfront vs on-demand conditions."""
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.analysis import paired_effect, complete_tasks, diagnostic_class, wilson
from agentisa.interfaces import CONDITIONS_V2
S = {"gpt-4.1-nano-2025-04-14": "nano", "gpt-4.1-mini-2025-04-14": "mini", "gpt-4.1-2025-04-14": "4.1"}
v1 = [json.loads(l) for f in glob.glob(str(ROOT / "results/trials_gpt-4.1*_eval.jsonl")) for l in open(f)]
raw = [json.loads(l) for f in glob.glob(str(ROOT / "results/v2/trackA_gpt-4.1*_eval.jsonl")) for l in open(f)]
keep = {m: complete_tasks(raw, m, CONDITIONS_V2) for m in S}
v2 = [r for r in raw if r["task_id"] in keep[r["model"]]]
out = {}
# (a)
out["confound_check_synthetic"] = {}
for m in S:
    s2 = [r for r in v2 if r["model"] == m and r.get("source") == "synthetic"]
    e1 = paired_effect(v1, m, "prose", "structured"); e2 = paired_effect(s2, m, "prose", "structured")
    out["confound_check_synthetic"][S[m]] = {
        "study1": {"effect": e1["effect"], "ci": e1["ci95_cluster_bootstrap"], "n": e1["paired_tasks"], "p": e1["mcnemar_exact_p"],
                   "prose": e1["rate_prose"], "structured": e1["rate_structured"]},
        "study2": {"effect": e2["effect"], "ci": e2["ci95_cluster_bootstrap"], "n": e2["paired_tasks"], "p": e2["mcnemar_exact_p"],
                   "prose": e2["rate_prose"], "structured": e2["rate_structured"]}}
# (b)
out["retrieval_mechanism"] = {}
for m in S:
    for c in ("interactive", "interactive_forced"):
        rs = [r for r in v2 if r["model"] == m and r["condition"] == c and r["tool_calls"] > 0]
        if not rs: continue
        w = [r for r in rs if "MOD.SYN.3" in r["retrieved_fact_ids"]]; wo = [r for r in rs if "MOD.SYN.3" not in r["retrieved_fact_ids"]]
        f = lambda xs: (round(sum(r["first_attempt_semantic_correct"] for r in xs) / len(xs), 3), len(xs)) if xs else (None, 0)
        ae = lambda xs: round(sum(diagnostic_class(r["attempts"][0]) == "attr_encoding" for r in xs) / len(xs), 3) if xs else None
        out["retrieval_mechanism"][f"{S[m]}/{c}"] = {"with_module_rule": f(w), "without": f(wo), "attr_err_with": ae(w), "attr_err_without": ae(wo),
                                                    "share_trials_retrieving_module": round(len(w) / len(rs), 3)}
# (c)
out["semantic_share"] = {}
for m in S:
    for src in ("synthetic", "kernelbench"):
        rs = [r for r in v2 if r["model"] == m and r.get("source") == src]
        fails = [r for r in rs if not r["first_attempt_semantic_correct"]]
        ex = [r for r in fails if r["attempts"] and r["attempts"][0]["verified"]]
        allatt = [a for r in rs for a in r["attempts"]]
        out["semantic_share"][f"{S[m]}/{src}"] = {"failed_first": len(fails), "failed_but_verified": len(ex),
            "share": round(len(ex) / max(1, len(fails)), 4), "wrong_values_any_attempt": sum(a["error_type"] == "semantic_mismatch" for a in allatt),
            "executed_any_attempt": sum(a["verified"] for a in allatt)}
# (d)
out["tokens_vs_accuracy"] = {}
for m in S:
    for c in ("prose", "structured", "interactive", "interactive_forced", "interactive_core"):
        rs = [r for r in v2 if r["model"] == m and r["condition"] == c]
        out["tokens_vs_accuracy"][f"{S[m]}/{c}"] = {"acc": round(sum(r["first_attempt_semantic_correct"] for r in rs) / len(rs), 3),
            "first_call_input_tokens_mean": None, "input_tokens_mean": round(sum(r["input_tokens"] for r in rs) / len(rs)),
            "tool_calls_mean": round(sum(r["tool_calls"] for r in rs) / len(rs), 2)}
(ROOT / "results/v2/analysis_v2_extra.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
