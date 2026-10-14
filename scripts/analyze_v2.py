"""Study 2 (v2) Track A analysis: pre-registered contrasts per model, overall and per suite; complete-task rule;
pre-registered diagnostic taxonomy; cost. Writes results/v2/analysis_v2.json, paper tables and figures."""
import glob, json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.analysis import full_report, complete_tasks, CONTRASTS_V2, diagnostic_class, wilson
from agentisa.interfaces import CONDITIONS_V2 as CONDS

MODELS = ["gpt-4.1-nano-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]
SHORT = {"gpt-4.1-nano-2025-04-14": "4.1-nano", "gpt-4.1-mini-2025-04-14": "4.1-mini", "gpt-4.1-2025-04-14": "4.1"}
PRICE = {"gpt-4.1-nano-2025-04-14": (0.10, 0.40), "gpt-4.1-mini-2025-04-14": (0.40, 1.60), "gpt-4.1-2025-04-14": (2.00, 8.00)}
CLAB = {"none": "None", "minimal": "Minimal (A)", "syntax_only_structured": "Syntax-only struct. (N)", "prose": "Prose (B)",
        "structured": "Structured (C)", "interactive": "Interactive (D)", "interactive_forced": "Int. forced (D*)",
        "interactive_core": "Int. core (D+)"}
raw = [json.loads(l) for f in sorted(glob.glob(str(ROOT / "results/v2/trackA_gpt-4.1*_eval.jsonl"))) for l in open(f) if l.strip()]
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
out = ROOT / "results/v2"; (out / "analysis_v2.json").write_text(json.dumps(rep, indent=1, default=str))

# ---- tables
fig = ROOT / "paper/figures"; fig.mkdir(parents=True, exist_ok=True)
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

# ---- figure: per-condition first-attempt correctness, by suite
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
COL = ["#2a78d6", "#eb6834", "#1baf7a"]; MK = ["o", "s", "D"]
suites = [("synthetic", "Synthetic (114 tasks)"), ("kernelbench", "KernelBench-derived")]
f1, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, (s, title) in zip(axes, suites):
    if s not in rep["by_suite"]: continue
    R = rep["by_suite"][s]
    for j, m in enumerate(models):
        if m not in R["models"]: continue
        for i, c in enumerate(CONDS):
            d = R["models"][m]["conditions"][c]; y = len(CONDS) - 1 - i + (j - 1) * 0.22
            lo, hi = d["first_attempt_ci95"]
            ax.plot([100 * lo, 100 * hi], [y, y], color=COL[j], lw=1.2)
            ax.plot(100 * d["first_attempt"], y, MK[j], color=COL[j], ms=4.5, mec="#fcfcfb", mew=0.8, label=SHORT[m] if i == 0 else None)
    n = {m: R["models"][m]["conditions"]["none"]["n"] for m in R["models"]}
    ax.set_title(f"{title.split(' (')[0]} (n={max(n.values())} tasks)", fontsize=8.5); ax.set_xlim(0, 100)
    ax.set_xlabel("First-attempt semantic correctness (%)"); ax.grid(axis="x", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
axes[0].set_yticks(range(len(CONDS))); axes[0].set_yticklabels([CLAB[c] for c in CONDS][::-1])
h, l = axes[0].get_legend_handles_labels()
f1.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.6, 1.0))
f1.tight_layout(rect=(0, 0, 1, 0.92)); f1.savefig(fig / "fig_v2_conditions.pdf"); f1.savefig(fig / "fig_v2_conditions.png", dpi=200)
print(json.dumps({"complete_tasks": rep["complete_tasks"], "cost": rep["cost_usd"]}))
for m in models:
    c = rep["overall"]["models"][m]["conditions"]
    print(SHORT[m], {k: (round(v["first_attempt"], 3), round(v["after_repair"], 3)) for k, v in c.items()})
    for e in rep["overall"]["models"][m]["contrasts"]["first_attempt_semantic_correct"]:
        print("   ", e["contrast"], round(100 * e["effect"], 1), [round(100 * x, 1) for x in e["ci95_cluster_bootstrap"]], e["discordant"], round(e["holm_p"], 4))
