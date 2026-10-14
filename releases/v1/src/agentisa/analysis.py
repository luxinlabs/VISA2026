"""Paired analysis of collected trial records (no model execution).

Fixes over the original scaffold: explicit missing/exclusion accounting, per-model filtering,
correct labels for arbitrary contrasts, template-cluster bootstrap (instances of one template are
not independent), exact McNemar on task-level discordant pairs, Holm correction, secondary endpoints.
"""
import math, random, statistics
from collections import defaultdict, Counter

CONTRASTS = [  # (b, a): effect = rate(b) - rate(a); pre-specified, Holm-adjusted as one family
    ("structured", "prose"),               # primary (RQ1)
    ("interactive", "structured"),         # RQ2
    ("prose", "minimal"),                  # value of semantic facts
    ("minimal", "none"),                   # value of syntax documentation
    ("syntax_only_structured", "minimal"), # negative control: format without semantics
]


def _bool(v):
    if isinstance(v, str):
        if v.lower() in ("true", "1"): return True
        if v.lower() in ("false", "0"): return False
        raise ValueError(f"not a boolean: {v!r}")
    return bool(v)


def wilson(k, n, z=1.959964):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b (a wins) and c (b wins)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i]); adj = [0.0] * len(pvals); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i])); adj[i] = run
    return adj


def _quantile(sorted_vals, q):
    pos = q * (len(sorted_vals) - 1); lo = math.floor(pos); hi = math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def index(rows, model, outcome):
    """task_id -> condition -> mean outcome over trials; also task_id -> template."""
    by = defaultdict(lambda: defaultdict(list)); tmpl = {}
    for r in rows:
        if r["model"] != model: continue
        by[r["task_id"]][r["condition"]].append(int(_bool(r[outcome])))
        tmpl[r["task_id"]] = r["template"]
    return {t: {c: statistics.mean(v) for c, v in d.items()} for t, d in by.items()}, tmpl


def paired_effect(rows, model, a, b, outcome="first_attempt_semantic_correct", seed=20261013, n_boot=10000):
    idx, tmpl = index(rows, model, outcome)
    tasks = sorted(t for t, d in idx.items() if a in d and b in d)
    missing = sorted(t for t, d in idx.items() if (a in d) != (b in d))
    if not tasks: raise ValueError(f"No paired tasks for {b} vs {a} ({model})")
    eff = {t: idx[t][b] - idx[t][a] for t in tasks}
    clusters = defaultdict(list)
    for t in tasks: clusters[tmpl[t]].append(eff[t])
    keys = sorted(clusters); rng = random.Random(seed)
    boot_c, boot_t = [], []
    vals = [eff[t] for t in tasks]
    for _ in range(n_boot):
        pick = [clusters[k] for k in rng.choices(keys, k=len(keys))]
        boot_c.append(sum(sum(p) for p in pick) / sum(len(p) for p in pick))
        boot_t.append(statistics.mean(rng.choices(vals, k=len(vals))))
    boot_c.sort(); boot_t.sort()
    bwin = sum(1 for t in tasks if idx[t][b] > idx[t][a]); awin = sum(1 for t in tasks if idx[t][a] > idx[t][b])
    return {"model": model, "contrast": f"{b} - {a}", "outcome": outcome, "paired_tasks": len(tasks),
            "clusters": len(keys), "missing_unpaired_tasks": missing,
            f"rate_{a}": statistics.mean(idx[t][a] for t in tasks), f"rate_{b}": statistics.mean(idx[t][b] for t in tasks),
            "effect": statistics.mean(vals),
            "ci95_cluster_bootstrap": [_quantile(boot_c, .025), _quantile(boot_c, .975)],
            "ci95_task_bootstrap": [_quantile(boot_t, .025), _quantile(boot_t, .975)],
            "discordant": {f"{b}_only": bwin, f"{a}_only": awin}, "mcnemar_exact_p": mcnemar_exact(awin, bwin)}


def condition_table(rows, model):
    out = {}
    by = defaultdict(list)
    for r in rows:
        if r["model"] == model: by[r["condition"]].append(r)
    for c, rs in by.items():
        n = len(rs); k1 = sum(_bool(r["first_attempt_semantic_correct"]) for r in rs)
        kr = sum(_bool(r["correct_after_repair"]) for r in rs); kv = sum(_bool(r["first_attempt_verified"]) for r in rs)
        ok = [r for r in rs if _bool(r["correct_after_repair"])]
        out[c] = {"n": n, "first_attempt": k1 / n, "first_attempt_ci95": wilson(k1, n),
                  "after_repair": kr / n, "after_repair_ci95": wilson(kr, n), "first_attempt_verified": kv / n,
                  "mean_input_tokens": statistics.mean(r["input_tokens"] for r in rs),
                  "mean_output_tokens": statistics.mean(r["output_tokens"] for r in rs),
                  "mean_input_tokens_if_solved": statistics.mean(r["input_tokens"] for r in ok) if ok else None,
                  "mean_tool_calls": statistics.mean(r["tool_calls"] for r in rs),
                  "mean_attempts": statistics.mean(len(r["attempts"]) for r in rs),
                  "first_attempt_error_types": dict(Counter(r["first_attempt_error_type"] or "none" for r in rs))}
    return out


def family_table(rows, model, outcome="first_attempt_semantic_correct"):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["model"] == model: agg[r["family"]][r["condition"]].append(int(_bool(r[outcome])))
    return {f: {c: (statistics.mean(v), len(v)) for c, v in d.items()} for f, d in agg.items()}


def retrieval_stats(rows, model):
    rs = [r for r in rows if r["model"] == model and r["condition"] == "interactive"]
    if not rs: return {}
    cats = Counter(q[1] or "all" for r in rs for q in r["tool_queries"])
    return {"n": len(rs), "mean_tool_calls": statistics.mean(r["tool_calls"] for r in rs),
            "share_no_tool_call": statistics.mean(r["tool_calls"] == 0 for r in rs),
            "mean_fraction_available_facts_retrieved": statistics.mean(
                r["n_retrieved_facts"] / max(1, len(r["available_fact_ids"])) for r in rs),
            "query_categories": dict(cats)}


def full_report(rows, models, outcomes=("first_attempt_semantic_correct", "correct_after_repair")):
    rep = {"models": {}}
    for m in models:
        mr = {"conditions": condition_table(rows, m), "families": family_table(rows, m),
              "retrieval": retrieval_stats(rows, m), "contrasts": {}}
        for oc in outcomes:
            res = [paired_effect(rows, m, a, b, oc) for b, a in CONTRASTS]
            for r, p in zip(res, holm([r["mcnemar_exact_p"] for r in res])):
                r["holm_p"] = p
            mr["contrasts"][oc] = res
        rep["models"][m] = mr
    return rep


# ---------------------------------------------------------------- post-hoc diagnostic taxonomy
# Defined after inspecting first-attempt diagnostics (post-hoc, descriptive). Replaces the heuristic
# subclass recorded at run time, which mislabels e.g. attribute-encoding errors that mention "dimensions".
import re as _re
_TAX = [
    ("unknown_op", r"is unknown|unregistered dialect|unregistered operation|does not exist|forbidden|Operations not allowed"),
    ("attr_encoding", r"dense array attribute|Attribute that models the dimension|requires attribute|unknown StableHLO attribute|"
                      r"invalid kind of attribute|Which comparison operation|to be one of|expected valid keyword"),
    ("literal", r"literal|floating point value not valid"),
    ("shape_type", r"expects different type|broadcast_dimensions|compatible types|inferred type|inferred shape|"
                   r"reduction-region|Precision Config|same shape|cannot be concatenated|size of operand|op requires|failed to verify|must be|"
                   r"does not match|mismatch|incompatible|rank|permutation|dimension"),
    ("assembly_syntax", r"expected|custom op|redefinition|undeclared|use of undefined"),
]


def diagnostic_class(attempt):
    if attempt["semantic_correct"]: return "correct"
    if attempt["error_type"] == "semantic_mismatch": return "wrong_values"
    if attempt["error_type"] in ("signature_mismatch", "no_main", "no_program"): return "signature_format"
    d = attempt.get("diagnostic") or ""
    for name, pat in _TAX:
        if _re.search(pat, d): return name
    return "other"
