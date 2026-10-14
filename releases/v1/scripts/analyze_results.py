"""Analyze trial JSONL files: per-condition rates, pre-specified paired contrasts, failure classes,
retrieval stats; writes results/analysis.json, LaTeX tables and PDF figures for the paper."""
import argparse, glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.analysis import full_report, CONTRASTS

p = argparse.ArgumentParser()
p.add_argument("--inputs", default=str(ROOT / "results/trials_gpt-4.1*_eval.jsonl"))
p.add_argument("--outdir", default=str(ROOT / "results"))
p.add_argument("--figdir", default=str(ROOT / "paper/figures"))
a = p.parse_args()
rows = [json.loads(l) for f in sorted(glob.glob(a.inputs)) for l in open(f) if l.strip()]
MODELS = [m for m in ["gpt-4.1-nano-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"] if any(r["model"] == m for r in rows)]
SHORT = {"gpt-4.1-nano-2025-04-14": "4.1-nano", "gpt-4.1-mini-2025-04-14": "4.1-mini", "gpt-4.1-2025-04-14": "4.1"}
CONDS = ["none", "minimal", "syntax_only_structured", "prose", "structured", "interactive"]
CLAB = {"none": "None", "minimal": "Minimal (A)", "syntax_only_structured": "Syntax-only struct. (N)",
        "prose": "Prose (B)", "structured": "Structured (C)", "interactive": "Interactive (D)"}
rep = full_report(rows, MODELS)
# completeness
exp = {(r["task_id"]) for r in rows}
rep["completeness"] = {m: {c: sum(1 for r in rows if r["model"] == m and r["condition"] == c) for c in CONDS} for m in MODELS}
# failure classes (first attempt), pooled categories
def fclass(r):
    e = r["first_attempt_error_type"]
    if r["first_attempt_semantic_correct"]: return "correct"
    if e in ("static_syntax",): return "syntax"
    if e in ("static_shape",): return "shape"
    if e in ("static_type", "static_other"): return "type/attr"
    if e in ("semantic_mismatch",): return "wrong values"
    if e in ("signature_mismatch", "no_main", "no_program"): return "signature/format"
    return "runtime/other"
rep["failure_classes"] = {m: {c: dict(Counter(fclass(r) for r in rows if r["model"] == m and r["condition"] == c)) for c in CONDS} for m in MODELS}
cost = {"gpt-4.1-nano-2025-04-14": (0.10, 0.40), "gpt-4.1-mini-2025-04-14": (0.40, 1.60), "gpt-4.1-2025-04-14": (2.00, 8.00)}
rep["api_cost_usd_estimate_list_price"] = {m: round(sum(r["input_tokens"] for r in rows if r["model"] == m) / 1e6 * cost[m][0] +
                                              sum(r["output_tokens"] for r in rows if r["model"] == m) / 1e6 * cost[m][1], 2) for m in MODELS}
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
(out / "analysis.json").write_text(json.dumps(rep, indent=1, default=str))

# ---------------- LaTeX tables
fig = Path(a.figdir); fig.mkdir(parents=True, exist_ok=True)
pct = lambda x: f"{100 * x:.1f}"
L = ["\\begin{tabular}{l" + "rr" * len(MODELS) + "}", "\\toprule",
     "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[m]}}}" for m in MODELS) + " \\\\",
     "Condition & " + " & ".join(["1st", "rep."] * len(MODELS)) + " \\\\", "\\midrule"]
for c in CONDS:
    L.append(CLAB[c] + " & " + " & ".join(f"{pct(rep['models'][m]['conditions'][c]['first_attempt'])} & {pct(rep['models'][m]['conditions'][c]['after_repair'])}" for m in MODELS) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(fig / "tab_conditions.tex").write_text("\n".join(L))
L = ["\\begin{tabular}{llrrrr}", "\\toprule", "Model & Contrast & $\\Delta$ (pp) & 95\\% CI (cluster) & disc. & $p_\\text{Holm}$ \\\\", "\\midrule"]
for m in MODELS:
    for i, e in enumerate(rep["models"][m]["contrasts"]["first_attempt_semantic_correct"]):
        b, aa = e["contrast"].split(" - ")
        d = e["discordant"]
        L.append((SHORT[m] if i == 0 else "") + f" & {CLAB[b].split(' (')[0]} -- {CLAB[aa].split(' (')[0]} & {100 * e['effect']:+.1f} & "
                 f"[{100 * e['ci95_cluster_bootstrap'][0]:+.1f}, {100 * e['ci95_cluster_bootstrap'][1]:+.1f}] & "
                 f"{d[b + '_only']}/{d[aa + '_only']} & {'$<$0.001' if e['holm_p'] < 0.001 else format(e['holm_p'], '.3f')} \\\\")
    L.append("\\midrule" if m != MODELS[-1] else "\\bottomrule")
L.append("\\end{tabular}")
(fig / "tab_contrasts.tex").write_text("\n".join(L))

# ---------------- figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
COL = ["#2a78d6", "#eb6834", "#1baf7a"]; MK = ["o", "s", "D"]
fig1, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for ax, key, title in [(axes[0], "first_attempt", "First attempt"), (axes[1], "after_repair", "After $\\leq$2 repairs")]:
    for j, m in enumerate(MODELS):
        for i, c in enumerate(CONDS):
            d = rep["models"][m]["conditions"][c]; y = len(CONDS) - 1 - i + (j - 1) * 0.22
            lo, hi = d[key + "_ci95"]
            ax.plot([100 * lo, 100 * hi], [y, y], color=COL[j], lw=1.2, solid_capstyle="round")
            ax.plot(100 * d[key], y, MK[j], color=COL[j], ms=4.5, mec="#fcfcfb", mew=0.8, label=SHORT[m] if i == 0 else None)
    ax.set_xlim(0, 100); ax.set_title(title, fontsize=8.5, color="#0b0b0b"); ax.set_xlabel("Semantic correctness (%)")
    ax.grid(axis="x", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
axes[0].set_yticks(range(len(CONDS))); axes[0].set_yticklabels([CLAB[c] for c in CONDS][::-1])
h, l = axes[0].get_legend_handles_labels()
fig1.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.5, handletextpad=0.3, bbox_to_anchor=(0.6, 1.0))
fig1.tight_layout(rect=(0, 0, 1, 0.92)); fig1.savefig(fig / "fig_conditions.pdf"); fig1.savefig(fig / "fig_conditions.png", dpi=200)

fig2, ax = plt.subplots(figsize=(3.4, 2.4))
for j, m in enumerate(MODELS):
    for i, e in enumerate(rep["models"][m]["contrasts"]["first_attempt_semantic_correct"]):
        y = len(CONTRASTS) - 1 - i + (j - 1) * 0.22; lo, hi = e["ci95_cluster_bootstrap"]
        ax.plot([100 * lo, 100 * hi], [y, y], color=COL[j], lw=1.2)
        ax.plot(100 * e["effect"], y, MK[j], color=COL[j], ms=4.5, mec="#fcfcfb", mew=0.8, label=SHORT[m] if i == 0 else None)
ax.axvline(0, color="#52514e", lw=0.7)
ax.set_yticks(range(len(CONTRASTS)))
ax.set_yticklabels([f"{CLAB[b].split(' (')[0]} − {CLAB[aa].split(' (')[0]}" for b, aa in CONTRASTS][::-1], fontsize=7)
ax.set_xlabel("Paired $\\Delta$, first attempt (pp)"); ax.grid(axis="x", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=6.5, handletextpad=0.3, bbox_to_anchor=(0.35, 1.13))
fig2.tight_layout(); fig2.savefig(fig / "fig_contrasts.pdf"); fig2.savefig(fig / "fig_contrasts.png", dpi=200)
print(json.dumps({m: {c: [round(rep['models'][m]['conditions'][c]['first_attempt'], 3), round(rep['models'][m]['conditions'][c]['after_repair'], 3)] for c in CONDS} for m in MODELS}, indent=1))
for m in MODELS:
    for e in rep["models"][m]["contrasts"]["first_attempt_semantic_correct"]:
        print(SHORT[m], e["contrast"], round(100 * e["effect"], 1), [round(100 * v, 1) for v in e["ci95_cluster_bootstrap"]], e["discordant"], round(e["holm_p"], 4))
print("completeness", rep["completeness"]); print("cost", rep["api_cost_usd_estimate_list_price"])
