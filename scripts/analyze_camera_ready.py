"""Camera-ready analyses of existing Study 1/2 and ablation logs (no new trials), answering reviewer requests:
(a) cost: per condition, mean input tokens, model calls, tool calls, median trial wall-clock and list-price USD per task
    solved after <=2 repairs (all repairs and tool calls included);
(b) test-retest: the None condition's prompt is byte-identical in Study 1 and Study 2 on the shared synthetic tasks,
    so the pairs measure run-to-run variation at temperature 0 (different days, same snapshots);
(c) JSON penalty: first-attempt failure classes for B, C' and C on paired tasks, plus checks for copied JSON artifacts
    (escaped quotes, fact IDs) in the submitted programs;
(d)-(h) the post-hoc runs of Addenda 4-8. Their logs are not part of the public artifact, so each of those sections is
skipped when its logs are absent. Writes results/v2/analysis_camera_ready.json and LaTeX tables to --fig-dir."""
import argparse, glob, json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.analysis import complete_tasks, diagnostic_class, paired_effect
from agentisa.interfaces import CONDITIONS_V2
from agentisa.pricing import trial_cost_usd
ap = argparse.ArgumentParser(); ap.add_argument("--fig-dir", default=str(ROOT / "results/v2/tables"), help="where the LaTeX tables go")
FIG = Path(ap.parse_args().fig_dir); FIG.mkdir(parents=True, exist_ok=True)
S = {"gpt-4.1-nano-2025-04-14": "nano", "gpt-4.1-mini-2025-04-14": "mini", "gpt-4.1-2025-04-14": "4.1"}
NAME = {"none": "None", "minimal": "Minimal (A)", "syntax_only_structured": "Syntax-only str. (N)", "prose": "Prose (B)",
        "structured": "Structured (C)", "interactive": "Interactive (D)", "interactive_forced": "Int. forced (D$^\\ast$)",
        "interactive_core": "Int. core (D$^+$)"}
def jl(rel):
    """Rows of one JSONL log, or None when the (post-hoc, unpublished) log is absent."""
    path = ROOT / rel
    return [json.loads(l) for l in open(path)] if path.exists() else None
load = lambda pat: [json.loads(l) for f in sorted(glob.glob(str(ROOT / pat))) if "transcripts" not in f for l in open(f)]
v1, raw, ab = load("results/trials_gpt-4.1*_eval.jsonl"), load("results/v2/trackA_gpt-4.1*_eval.jsonl"), load("results/v2/ablation/abl_*.jsonl")
keep = {m: complete_tasks(raw, m, CONDITIONS_V2) for m in S}
v2 = [r for r in raw if r["task_id"] in keep[r["model"]]]
hk = jl("results/v2/haiku/trackA_claude-haiku-5-5_eval.jsonl")  # Haiku 5.5 (Addendum 5), complete tasks only
if hk is not None:
    hk_keep = complete_tasks(hk, "claude-haiku-5-5", CONDITIONS_V2); hk = [r for r in hk if r["task_id"] in hk_keep]
out = {}
# (a)
out["cost"] = {}
for m in S:
    for c in CONDITIONS_V2:
        rs = [r for r in v2 if r["model"] == m and r["condition"] == c]; n = len(rs)
        usd = sum(trial_cost_usd(r) for r in rs)
        solved = sum(r["correct_after_repair"] for r in rs); el = sorted(r["elapsed_seconds"] for r in rs)
        out["cost"][f"{S[m]}/{c}"] = {"n": n, "first": round(sum(r["first_attempt_semantic_correct"] for r in rs) / n, 3),
            "final": round(solved / n, 3), "input_tokens_mean": round(sum(r["input_tokens"] for r in rs) / n),
            "output_tokens_mean": round(sum(r["output_tokens"] for r in rs) / n), "model_calls_mean": round(sum(r["model_calls"] for r in rs) / n, 2),
            "tool_calls_mean": round(sum(r["tool_calls"] for r in rs) / n, 2), "median_seconds": el[n // 2],
            "usd_total": round(usd, 4), "usd_per_solved": round(usd / solved, 4) if solved else None}
# (b)
out["test_retest_none"] = {}
for m in S:
    a = {r["task_id"]: r for r in v1 if r["model"] == m and r["condition"] == "none"}
    b = {r["task_id"]: r for r in raw if r["model"] == m and r["condition"] == "none"}
    ks = sorted(set(a) & set(b))
    out["test_retest_none"][S[m]] = {"pairs": len(ks),
        "identical_first_program": sum(a[k]["attempts"][0]["program"] == b[k]["attempts"][0]["program"] for k in ks),
        "same_first_label": sum(a[k]["first_attempt_semantic_correct"] == b[k]["first_attempt_semantic_correct"] for k in ks),
        "same_first_error_class": sum(diagnostic_class(a[k]["attempts"][0]) == diagnostic_class(b[k]["attempts"][0]) for k in ks),
        "correct_study1": sum(a[k]["first_attempt_semantic_correct"] for k in ks), "correct_study2": sum(b[k]["first_attempt_semantic_correct"] for k in ks)}
# (c)
out["json_penalty_failures"] = {}
for m in S:
    ids = keep[m] & {r["task_id"] for r in ab if r["model"] == m}
    for c in ("prose", "structured_nl", "structured"):
        rs = [r for r in v2 + ab if r["model"] == m and r["condition"] == c and r["task_id"] in ids]
        progs = [r["attempts"][0].get("program") or "" for r in rs]
        out["json_penalty_failures"][f"{S[m]}/{c}"] = {"n": len(rs), "classes": dict(Counter(diagnostic_class(r["attempts"][0]) for r in rs).most_common()),
            "programs_with_escaped_quotes": sum('\\"' in p for p in progs),
            "programs_with_fact_ids": sum(bool(re.search(r"[A-Z]{2,}\.(SYN|TYP|SHP|SEM)\.\d", p)) for p in progs),
            "output_tokens_mean": round(sum(r["output_tokens"] for r in rs) / len(rs))}
# (d) Addendum 4 repeats: runs 1-2 (first attempt only) next to the pre-registered run 0, nano and mini, B and C
out["repeats"] = {}
for m in list(S)[:2]:
    f = ROOT / f"results/v2/repeats/rep_{S[m]}.jsonl"
    if not f.exists(): continue
    reps = [r for r in map(json.loads, open(f)) if r["model"] == m]
    runs = [r for r in v2 if r["model"] == m and r["condition"] in ("prose", "structured")] + reps
    cell = {}
    for r in runs: cell.setdefault((r["task_id"], r["condition"]), {})[r["trial"]] = r["first_attempt_semantic_correct"]
    full = {k: v for k, v in cell.items() if len(v) == 3}
    eff = lambda rows: (lambda e: {"effect": round(e["effect"], 4), "ci": [round(x, 4) for x in e["ci95_cluster_bootstrap"]],
                                   "discordant": e["discordant"], "p": e["mcnemar_exact_p"], "n": e["paired_tasks"]})(paired_effect(rows, m, "prose", "structured"))
    out["repeats"][S[m]] = {"cells_with_3_runs": len(full), "trials_new": len(reps),
        "unanimous": {c: [sum(len(set(v.values())) == 1 for k, v in full.items() if k[1] == c), sum(1 for k in full if k[1] == c)] for c in ("prose", "structured")},
        "rate_by_run": {c: [round(sum(r["first_attempt_semantic_correct"] for r in runs if r["condition"] == c and r["trial"] == k) /
                              max(1, sum(1 for r in runs if r["condition"] == c and r["trial"] == k)), 4) for k in range(3)] for c in ("prose", "structured")},
        "c_minus_b_by_run": [eff([r for r in runs if r["trial"] == k]) for k in range(3)],
        "c_minus_b_mean_of_3": eff(runs),
        "usd": round(sum(trial_cost_usd(r) for r in reps), 4)}
# (e) Addendum 5b: full pinned spec vs prose, first attempt; comparator = pre-registered prose run, robustness = mean of 3 prose runs
out["full_spec"] = {}
for m in list(S)[:2]:
    f = ROOT / f"results/v2/full_spec/fs_{S[m]}.jsonl"
    if not f.exists(): continue
    fs = [r for r in map(json.loads, open(f)) if r["model"] == m]
    ids = {r["task_id"] for r in fs}
    prose = [r for r in v2 if r["model"] == m and r["condition"] == "prose" and r["task_id"] in ids]
    rep_rows = jl(f"results/v2/repeats/rep_{S[m]}.jsonl")  # Addendum 4; only for the mean-of-three robustness check
    reps = [r for r in rep_rows or [] if r["condition"] == "prose" and r["task_id"] in ids]
    summ = lambda e: {"effect": round(e["effect"], 4), "ci": [round(x, 4) for x in e["ci95_cluster_bootstrap"]], "discordant": e["discordant"],
                      "p": e["mcnemar_exact_p"], "n": e["paired_tasks"], "rate_prose": round(e["rate_prose"], 4), "rate_full_spec": round(e["rate_full_spec"], 4)}
    out["full_spec"][S[m]] = {"tasks": len(ids), "vs_prose_run0": summ(paired_effect(prose + fs, m, "prose", "full_spec")),
        "vs_prose_mean3": summ(paired_effect(prose + reps + fs, m, "prose", "full_spec")) if rep_rows is not None else None,
        "input_tokens_mean": round(sum(r["input_tokens"] for r in fs) / len(fs)),
        "cached_share": round(sum(r.get("cached_input_tokens", 0) for r in fs) / sum(r["input_tokens"] for r in fs), 3),
        "usd_per_trial": round(sum(map(trial_cost_usd, fs)) / len(fs), 5), "usd_total": round(sum(map(trial_cost_usd, fs)), 3),
        "classes": dict(Counter(diagnostic_class(r["attempts"][0]) for r in fs).most_common()),
        "classes_prose": dict(Counter(diagnostic_class(r["attempts"][0]) for r in prose).most_common())}
# (f) Addendum 6: GPT-5.4-mini, Haiku 4.5, Qwen3-Coder; five conditions, first attempt; Holm over four contrasts
from agentisa.analysis import holm
FAM = {"gpt-5.4-mini-2026-03-17": ("GPT-5.4-mini", "gpt54mini"), "claude-haiku-4-5-20251001": ("Haiku 4.5", "haiku45"),
       "qwen/qwen3-coder": ("Qwen3-Coder", "qwen3coder")}
C6 = ["none", "minimal", "prose", "structured", "interactive"]
K6 = [("none", "minimal"), ("minimal", "prose"), ("prose", "structured"), ("structured", "interactive")]
out["family"] = {}
for m, (name, f) in FAM.items():
    path = ROOT / f"results/v2/family/{f}.jsonl"
    if not path.exists(): continue
    raw6 = [r for r in map(json.loads, open(path)) if r["model"] == m]
    ids = complete_tasks(raw6, m, C6); rs = [r for r in raw6 if r["task_id"] in ids]
    res = [paired_effect(rs, m, a, b) for a, b in K6]
    for e, p in zip(res, holm([e["mcnemar_exact_p"] for e in res])): e["holm_p"] = p
    fails = [r for r in rs if not r["first_attempt_semantic_correct"]]
    rate = lambda c, src=None: round(sum(r["first_attempt_semantic_correct"] for r in rs if r["condition"] == c and src in (None, r.get("source"))) /
                                     max(1, sum(1 for r in rs if r["condition"] == c and src in (None, r.get("source")))), 4)
    out["family"][name] = {"model": m, "complete_tasks": len(ids), "trials": len(raw6),
        "rates": {c: rate(c) for c in C6}, "rates_kernelbench": {c: rate(c, "kernelbench") for c in C6}, "rates_synthetic": {c: rate(c, "synthetic") for c in C6},
        "contrasts": [{"contrast": e["contrast"], "effect": round(e["effect"], 4), "ci": [round(x, 4) for x in e["ci95_cluster_bootstrap"]],
                       "discordant": e["discordant"], "holm_p": e["holm_p"]} for e in res],
        "static_share_of_failures": round(sum(1 for r in fails if not r["attempts"][0]["verified"]) / max(1, len(fails)), 4),
        "tool_calls_interactive": round(sum(r["tool_calls"] for r in rs if r["condition"] == "interactive") / max(1, sum(1 for r in rs if r["condition"] == "interactive")), 2),
        "usd": round(sum(trial_cost_usd(r) for r in raw6), 3),
        "served_by": dict(Counter(x for r in raw6 for x in r.get("served_by", [])).most_common(5))}
# descriptive: documentation gain (prose - none) against prior strength (none), first attempt, every model
prior = {}
for m, s in S.items():
    rs = [r for r in v2 if r["model"] == m]
    pr_ = lambda c: sum(r["first_attempt_semantic_correct"] for r in rs if r["condition"] == c) / sum(1 for r in rs if r["condition"] == c)
    prior[s] = {"none": round(pr_("none"), 4), "prose": round(pr_("prose"), 4), "released": "2025-04"}
if hk is not None:
    hr = lambda c: sum(r["first_attempt_semantic_correct"] for r in hk if r["condition"] == c) / sum(1 for r in hk if r["condition"] == c)
    prior["Haiku 5.5"] = {"none": round(hr("none"), 4), "prose": round(hr("prose"), 4), "released": "2026-10"}
for name, v in out["family"].items():
    prior[name] = {"none": v["rates"]["none"], "prose": v["rates"]["prose"],
                   "released": {"GPT-5.4-mini": "2026-03", "Haiku 4.5": "2025-10", "Qwen3-Coder": "2025-07"}[name]}
for v in prior.values(): v["gain"] = round(v["prose"] - v["none"], 4)
out["prior_vs_gain"] = prior
# (g) Addendum 7: C'' (JSON without escaped code) vs C and B for GPT-4.1; Qwen full protocol; equivalence (TOST, +-5 pp)
from agentisa.analysis import index as _index, _quantile
import random
def ci90(rows, m, a, b, outcome="first_attempt_semantic_correct"):
    idx, tmpl = _index(rows, m, outcome)
    tasks = sorted(t for t, d in idx.items() if a in d and b in d)
    cl = {}
    for t in tasks: cl.setdefault(tmpl[t], []).append(idx[t][b] - idx[t][a])
    keys = sorted(cl); rng = random.Random(20261013); bs = []
    for _ in range(10000):
        pick = [cl[k] for k in rng.choices(keys, k=len(keys))]; bs.append(sum(map(sum, pick)) / sum(map(len, pick)))
    bs.sort(); return [round(_quantile(bs, .05), 4), round(_quantile(bs, .95), 4)]
out["rawcode"] = {}
for m, f in [("gpt-4.1-nano-2025-04-14", "nano"), ("gpt-4.1-mini-2025-04-14", "mini"), ("gpt-4.1-2025-04-14", "41")]:
    path = ROOT / f"results/v2/addendum7/rawcode_{f}.jsonl"
    if not path.exists(): continue
    rc = [r for r in map(json.loads, open(path)) if r["model"] == m]; ids = {r["task_id"] for r in rc}
    base = [r for r in v2 if r["model"] == m and r["condition"] in ("prose", "structured") and r["task_id"] in ids]
    res = [paired_effect(base + rc, m, "structured", "structured_rawcode"), paired_effect(base + rc, m, "structured_rawcode", "prose")]
    for e, p in zip(res, holm([e["mcnemar_exact_p"] for e in res])): e["holm_p"] = p
    rt = lambda c: round(sum(r["first_attempt_semantic_correct"] for r in base + rc if r["condition"] == c) / len(ids), 4)
    out["rawcode"][S[m]] = {"tasks": len(ids), "rates": {c: rt(c) for c in ("prose", "structured", "structured_rawcode")},
        "contrasts": [{"contrast": e["contrast"], "effect": round(e["effect"], 4), "ci": [round(x, 4) for x in e["ci95_cluster_bootstrap"]],
                       "discordant": e["discordant"], "holm_p": e["holm_p"]} for e in res],
        "classes": {c: dict(Counter(diagnostic_class(r["attempts"][0]) for r in base + rc if r["condition"] == c).most_common()) for c in ("structured", "structured_rawcode")}}
qf = ROOT / "results/v2/addendum7/qwen3coder_full.jsonl"
if qf.exists():
    q = [json.loads(l) for l in open(qf)]; m = "qwen/qwen3-coder"
    ids = complete_tasks(q, m, ["prose", "structured", "interactive"]); q = [r for r in q if r["task_id"] in ids]
    out["qwen_full"] = {"tasks": len(ids), "usd": round(sum(trial_cost_usd(r) for r in q), 3)}
    for oc in ("first_attempt_semantic_correct", "correct_after_repair"):
        res = [paired_effect(q, m, "prose", "structured", oc), paired_effect(q, m, "structured", "interactive", oc)]
        for e, p in zip(res, holm([e["mcnemar_exact_p"] for e in res])): e["holm_p"] = p
        out["qwen_full"][oc] = {"rates": {c: round(sum(r[oc] for r in q if r["condition"] == c) / len(ids), 4) for c in ("prose", "structured", "interactive")},
            "contrasts": [{"contrast": e["contrast"], "effect": round(e["effect"], 4), "ci": [round(x, 4) for x in e["ci95_cluster_bootstrap"]], "holm_p": e["holm_p"]} for e in res]}
out["equivalence"] = {}
eqsets = {"Haiku 5.5": ("claude-haiku-5-5", hk)} if hk is not None else {}
for m, (name, f) in FAM.items():
    if (ROOT / f"results/v2/family/{f}.jsonl").exists():
        rr = [r for r in map(json.loads, open(ROOT / f"results/v2/family/{f}.jsonl")) if r["model"] == m]
        k = complete_tasks(rr, m, C6); eqsets[name] = (m, [r for r in rr if r["task_id"] in k])
for name, (m, rr) in eqsets.items():
    for a, b in (("prose", "structured"), ("structured", "interactive")):
        lo, hi = ci90(rr, m, a, b)
        out["equivalence"][f"{name}/{b}-{a}"] = {"ci90": [lo, hi], "equivalent_5pp": lo > -0.05 and hi < 0.05}
# (h) Addendum 8: repeats of A-None and D-C (runs 1-2) next to the pre-registered run 0; nano and mini
out["repeats8"] = {}
for m in list(S)[:2]:
    f8 = ROOT / f"results/v2/addendum8/rep_{S[m]}.jsonl"
    if not f8.exists(): continue
    r8 = [r for r in map(json.loads, open(f8)) if r["model"] == m]
    r4 = jl(f"results/v2/repeats/rep_{S[m]}.jsonl")  # Addendum 4 structured runs 1-2 are needed for D-C over three runs
    if r4 is None: continue
    r4 = [r for r in r4 if r["condition"] == "structured"]
    runs = [r for r in v2 if r["model"] == m and r["condition"] in ("none", "minimal", "structured", "interactive")] + r8 + r4
    summ = lambda e: {"effect": round(e["effect"], 4), "ci": [round(x, 4) for x in e["ci95_cluster_bootstrap"]], "discordant": e["discordant"], "p": e["mcnemar_exact_p"]}
    res = {}
    for a_, b_ in (("none", "minimal"), ("structured", "interactive")):
        res[f"{b_}-{a_}"] = {"by_run": [summ(paired_effect([r for r in runs if r["trial"] == k], m, a_, b_)) for k in range(3)],
                             "mean_of_3": summ(paired_effect(runs, m, a_, b_))}
    cell = {}
    for r in runs: cell.setdefault((r["task_id"], r["condition"]), {})[r["trial"]] = r["first_attempt_semantic_correct"]
    res["unanimous"] = {c: [sum(len(set(v.values())) == 1 for k, v in cell.items() if k[1] == c and len(v) == 3), sum(1 for k, v in cell.items() if k[1] == c and len(v) == 3)]
                        for c in ("none", "minimal", "interactive")}
    res["usd"] = round(sum(trial_cost_usd(r) for r in r8), 3)
    out["repeats8"][S[m]] = res
(ROOT / "results/v2/analysis_camera_ready.json").write_text(json.dumps(out, indent=1))
# LaTeX (minor camera-ready): Table II with list-price cents per solved task next to the correctness columns
L2 = ["\\begin{tabular}{@{}l" + "rrr" * 3 + "@{}}", "\\toprule",
      "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{x}}}" for x in S.values()) + " \\\\",
      "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3)),
      "Condition & " + " & ".join(["1st", "rep.", "\\textcent"] * 3) + " \\\\", "\\midrule"]
for c in CONDITIONS_V2:
    cells = []
    for s_ in S.values():
        x = out["cost"][f"{s_}/{c}"]; u = x["usd_per_solved"]
        cells.append(f"{100 * x['first']:.1f} & {100 * x['final']:.1f} & " + ("--" if u is None else (f"{100 * u:.0f}" if u >= 0.1 else f"{100 * u:.1f}")))
    L2.append(NAME[c].replace("$^\\ast$", "*").replace("$^+$", "+") + " & " + " & ".join(cells) + " \\\\")
L2 += ["\\bottomrule", "\\end{tabular}"]
(FIG / "tab_v2_conditions_cost.tex").write_text("\n".join(L2))
# LaTeX: one row per model; first attempt; contrasts with Holm stars (GPT-4.1 and Haiku 5.5: the eight-contrast Holm of
# Table III; Addendum 6 models: Holm over these four); D tool calls per trial
hv = ROOT / "results/v2/analysis_v2_haiku.json"
if hv.exists() and hk is not None and out["family"]:
    rep2 = json.loads(hv.read_text())["overall"]["models"]
    allrows = raw + hk
    star = lambda e: f"${100 * e['effect']:+.1f}" + ("^\\ast" if e["holm_p"] < 0.05 else "") + "$"  # math mode: true minus sign
    want = {"minimal - none", "prose - minimal", "structured - prose", "interactive - structured"}
    rowsT = []
    for m, lab in [("gpt-4.1-nano-2025-04-14", "GPT-4.1 nano"), ("gpt-4.1-mini-2025-04-14", "GPT-4.1 mini"), ("gpt-4.1-2025-04-14", "GPT-4.1"),
                   ("qwen/qwen3-coder", "Qwen3-Coder"), ("claude-haiku-4-5-20251001", "Haiku 4.5"), ("gpt-5.4-mini-2026-03-17", "GPT-5.4-mini"),
                   ("claude-haiku-5-5", "Haiku 5.5")]:
        rel = {"gpt-4.1-nano-2025-04-14": "25-04", "gpt-4.1-mini-2025-04-14": "25-04", "gpt-4.1-2025-04-14": "25-04", "qwen/qwen3-coder": "25-07",
               "claude-haiku-4-5-20251001": "25-10", "gpt-5.4-mini-2026-03-17": "26-03", "claude-haiku-5-5": "26-10"}[m]
        if m in rep2:
            cs = {e["contrast"]: e for e in rep2[m]["contrasts"]["first_attempt_semantic_correct"] if e["contrast"] in want}
            n = rep2[m]["conditions"]["none"]["n"]; none = rep2[m]["conditions"]["none"]["first_attempt"]
            keepm = complete_tasks([r for r in allrows if r["model"] == m], m, CONDITIONS_V2)
            dr = [r for r in allrows if r["model"] == m and r["condition"] == "interactive" and r["task_id"] in keepm]
        else:
            name = {"qwen/qwen3-coder": "Qwen3-Coder", "claude-haiku-4-5-20251001": "Haiku 4.5", "gpt-5.4-mini-2026-03-17": "GPT-5.4-mini"}[m]
            if name not in out["family"]: continue
            v = out["family"][name]; cs = {e["contrast"]: e for e in v["contrasts"]}; n = v["complete_tasks"]; none = v["rates"]["none"]
            dr = None; calls = v["tool_calls_interactive"]
        if dr is not None: calls = round(sum(r["tool_calls"] for r in dr) / len(dr), 2)
        rowsT.append(f"{lab} & {rel} & {n} & {100 * none:.1f} & " + " & ".join(star(cs[k]) for k in ["minimal - none", "prose - minimal", "structured - prose", "interactive - structured"]) + f" & {calls:.1f} \\\\")
    tab = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
           "Model & Rel. & $n$ & None & A$-$None & B$-$A & C$-$B & D$-$C & Calls \\\\", "\\midrule"] + rowsT + ["\\bottomrule", "\\end{tabular}"]
    (FIG / "tab_families.tex").write_text("\n".join(tab))
# LaTeX: input tokens, cents per solved task, median seconds, per model
C = out["cost"]; L = []
for c in CONDITIONS_V2:
    g = [C[f"{s}/{c}"] for s in S.values()]
    cents = lambda x: "--" if x["usd_per_solved"] is None else (f"{100*x['usd_per_solved']:.0f}" if x["usd_per_solved"] >= 0.1 else f"{100*x['usd_per_solved']:.1f}")
    L.append(f"{NAME[c]} & " + " & ".join(f"{x['input_tokens_mean']/1000:.1f}" for x in g) + " & " + " & ".join(cents(x) for x in g)
             + " & " + " & ".join(f"{x['median_seconds']:.0f}" for x in g) + " \\\\")
tab = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
       "& \\multicolumn{3}{c}{Input ktok/trial} & \\multicolumn{3}{c}{\\textcent{} per solved} & \\multicolumn{3}{c}{Median s} \\\\",
       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
       "Condition & nano & mini & 4.1 & nano & mini & 4.1 & nano & mini & 4.1 \\\\", "\\midrule"] + L + ["\\bottomrule", "\\end{tabular}"]
(FIG / "tab_cost.tex").write_text("\n".join(tab))
print(json.dumps({k: out[k] for k in ("repeats", "full_spec")}, indent=1))
for k, v in C.items(): print(k, v)
