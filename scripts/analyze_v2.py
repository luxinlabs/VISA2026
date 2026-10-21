"""Study 2 (v2) Track A analysis: pre-registered contrasts per model, overall and per suite; complete-task rule;
pre-registered diagnostic taxonomy; cost. Writes results/v2/analysis_v2.json, paper tables and figures."""
import argparse, glob, json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.analysis import full_report, complete_tasks, CONTRASTS_V2, diagnostic_class, wilson
from agentisa.interfaces import CONDITIONS_V2 as CONDS

ap = argparse.ArgumentParser(); ap.add_argument("--fig-dir", default=None); ap.add_argument("--fig-height", type=float, default=2.6)
ap.add_argument("--crop", action="store_true", help="camera-ready: crop the figure margin"); ap.add_argument("--with-haiku", action="store_true", help="camera-ready: add the Addendum 5 Claude Haiku 5.5 run (JSON to analysis_v2_haiku.json)")
A = ap.parse_args()
MODELS = ["gpt-4.1-nano-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]
SHORT = {"gpt-4.1-nano-2025-04-14": "nano", "gpt-4.1-mini-2025-04-14": "mini", "gpt-4.1-2025-04-14": "4.1"}
PRICE = {"gpt-4.1-nano-2025-04-14": (0.10, 0.40), "gpt-4.1-mini-2025-04-14": (0.40, 1.60), "gpt-4.1-2025-04-14": (2.00, 8.00)}
if A.with_haiku:
    MODELS.append("claude-haiku-5-5"); SHORT["claude-haiku-5-5"] = "Haiku 5.5"; PRICE["claude-haiku-5-5"] = (0.10, 0.50)
CLAB = {"none": "None", "minimal": "Minimal (A)", "syntax_only_structured": "Syntax-only struct. (N)", "prose": "Prose (B)",
        "structured": "Structured (C)", "interactive": "Interactive (D)", "interactive_forced": "Int. forced (D*)",
        "interactive_core": "Int. core (D+)"}
raw = [json.loads(l) for f in sorted(glob.glob(str(ROOT / "results/v2/trackA_gpt-4.1*_eval.jsonl"))) for l in open(f) if l.strip()]
if A.with_haiku:
    raw += [json.loads(l) for l in open(ROOT / "results/v2/haiku/trackA_claude-haiku-5-5_eval.jsonl") if l.strip()]
models = [m for m in MODELS if any(r["model"] == m for r in raw)]
keep = {m: complete_tasks(raw, m, CONDS) for m in models}
rows = [r for r in raw if r["task_id"] in keep[r["model"]]]
rep = {"n_raw_trials": len(raw), "complete_tasks": {m: len(keep[m]) for m in models},
       "dropped_incomplete_trials": len(raw) - len(rows)}
rep["overall"] = full_report(rows, models, contrasts=CONTRASTS_V2)
rep["by_suite"] = {}
for suite in ("synthetic", "kernelbench"):
    sr = [r for r in rows if r.get("source") == suite]
    if sr:
        rep["by_suite"][suite] = full_report(sr, [m for m in models if any(r["model"] == m for r in sr)],
                                             outcomes=("first_attempt_semantic_correct",), contrasts=CONTRASTS_V2)
rep["taxonomy"] = {m: {c: dict(Counter(diagnostic_class(r["attempts"][0]) for r in rows if r["model"] == m and r["condition"] == c))
                       for c in CONDS} for m in models}
rep["cost_usd"] = {m: round(sum(r["input_tokens"] / 1e6 * PRICE[m][0] + r["output_tokens"] / 1e6 * PRICE[m][1]
                                for r in raw if r["model"] == m), 2) for m in models}
out = ROOT / "results/v2"; (out / ("analysis_v2_haiku.json" if A.with_haiku else "analysis_v2.json")).write_text(json.dumps(rep, indent=1, default=str))

# ---- tables
fig = Path(A.fig_dir) if A.fig_dir else ROOT / "paper/figures"; fig.mkdir(parents=True, exist_ok=True)
pct = lambda x: f"{100 * x:.1f}"
def cond_table(report, key):
    L = ["\\begin{tabular}{l" + "rr" * len(models) + "}", "\\toprule",
         "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[m]}}}" for m in models) + " \\\\",
         "Condition & " + " & ".join(["1st", "rep."] * len(models)) + " \\\\", "\\midrule"]
    for c in CONDS:
        L.append(CLAB[c] + " & " + " & ".join(f"{pct(report['models'][m]['conditions'][c]['first_attempt'])} & {pct(report['models'][m]['conditions'][c]['after_repair'])}" for m in models) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (fig / key).write_text("\n".join(L))
cond_table(rep["overall"], "tab_v2_conditions.tex")
L = ["\\begin{tabular}{llrrrr}", "\\toprule", "Model & Contrast & $\\Delta$ (pp) & 95\\% CI & disc. & $p_\\text{Holm}$ \\\\", "\\midrule"]
for m in models:
    for i, e in enumerate(rep["overall"]["models"][m]["contrasts"]["first_attempt_semantic_correct"]):
        b, a = e["contrast"].split(" - "); d = e["discordant"]
        p = "$<$0.001" if e["holm_p"] < 0.001 else f"{e['holm_p']:.3f}"
        L.append((SHORT[m] if i == 0 else "") + f" & {CLAB[b].split(' (')[0]} -- {CLAB[a].split(' (')[0]} & {100 * e['effect']:+.1f} & "
                 f"[{100 * e['ci95_cluster_bootstrap'][0]:+.1f}, {100 * e['ci95_cluster_bootstrap'][1]:+.1f}] & {d[b + '_only']}/{d[a + '_only']} & {p} \\\\")
    L.append("\\midrule" if m != models[-1] else "\\bottomrule")
L.append("\\end{tabular}")
(fig / "tab_v2_contrasts.tex").write_text("\n".join(L))
if A.with_haiku:  # camera-ready: one column block per model (full-width table)
    res = {m: rep["overall"]["models"][m]["contrasts"]["first_attempt_semantic_correct"] for m in models}
    W = ["\\begin{tabular}{l" + "rrr" * len(models) + "}", "\\toprule",
         "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{SHORT[m]} ($n$={len(keep[m])})}}" for m in models) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(models))),
         "Contrast & " + " & ".join(["$\\Delta$", "95\\% CI", "$p_\\text{Holm}$"] * len(models)) + " \\\\", "\\midrule"]
    for i, e0 in enumerate(res[models[0]]):
        b_, a_ = e0["contrast"].split(" - ")
        cells = []
        for m in models:
            e = res[m][i]; pv = "$<$0.001" if e["holm_p"] < 0.001 else f"{e['holm_p']:.3f}"
            cells.append(f"{100 * e['effect']:+.1f} & [{100 * e['ci95_cluster_bootstrap'][0]:+.1f}, {100 * e['ci95_cluster_bootstrap'][1]:+.1f}] & {pv}")
        W.append(f"{CLAB[b_].split(' (')[0]} -- {CLAB[a_].split(' (')[0]} & " + " & ".join(cells) + " \\\\")
    W += ["\\bottomrule", "\\end{tabular}"]
    (fig / "tab_v2_contrasts_wide.tex").write_text("\n".join(W))

# ---- figure: per-condition first-attempt correctness, by suite
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, never Type 3 (IEEE PDF eXpress)
                     "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"]; MK = ["o", "s", "D", "^"]
suites = [("synthetic", "Synthetic (114 tasks)"), ("kernelbench", "KernelBench-derived")]
f1, axes = plt.subplots(1, 2, figsize=(7.0, A.fig_height), sharey=True)
for ax, (s, title) in zip(axes, suites):
    if s not in rep["by_suite"]: continue
    R = rep["by_suite"][s]
    for j, m in enumerate(models):
        if m not in R["models"]: continue
        for i, c in enumerate(CONDS):
            d = R["models"][m]["conditions"][c]; y = len(CONDS) - 1 - i + (j - (len(models) - 1) / 2) * (0.22 if len(models) <= 3 else 0.17)
            lo, hi = d["first_attempt_ci95"]
            ax.plot([100 * lo, 100 * hi], [y, y], color=COL[j], lw=1.2)
            ax.plot(100 * d["first_attempt"], y, MK[j], color=COL[j], ms=4.5, mec="#fcfcfb", mew=0.8, label=SHORT[m] if i == 0 else None)
    n = {SHORT[m]: R["models"][m]["conditions"]["none"]["n"] for m in R["models"]}
    full = "/".join(k for k in n if k != "4.1")
    ax.set_title(f"{title.split(' (')[0]}\n({full} n={n['nano']}, 4.1 n={n['4.1']} tasks)", fontsize=8); ax.set_xlim(0, 100)
    ax.set_xlabel("First-attempt semantic correctness (%)"); ax.grid(axis="x", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
axes[0].set_yticks(range(len(CONDS))); axes[0].set_yticklabels([CLAB[c] for c in CONDS][::-1])
h, l = axes[0].get_legend_handles_labels()
f1.legend(h, l, loc="upper center", ncol=len(models), frameon=False, fontsize=7.5, bbox_to_anchor=(0.6, 1.0))
f1.tight_layout(rect=(0, 0, 1, 0.90))
crop = dict(bbox_inches="tight", pad_inches=0.02) if (A.with_haiku or A.crop) else {}  # camera-ready: no empty margin above the legend
if A.with_haiku:
    for ax in axes: ax.set_xlim(-1, 101)  # Haiku's 100% markers are not clipped
f1.savefig(fig / "fig_v2_conditions.pdf", **crop); f1.savefig(fig / "fig_v2_conditions.png", dpi=200, **crop)
print(json.dumps({"complete_tasks": rep["complete_tasks"], "cost": rep["cost_usd"]}))
for m in models:
    c = rep["overall"]["models"][m]["conditions"]
    print(SHORT[m], {k: (round(v["first_attempt"], 3), round(v["after_repair"], 3)) for k, v in c.items()})
    for e in rep["overall"]["models"][m]["contrasts"]["first_attempt_semantic_correct"]:
        print("   ", e["contrast"], round(100 * e["effect"], 1), [round(100 * x, 1) for x in e["ci95_cluster_bootstrap"]], e["discordant"], round(e["holm_p"], 4))
