"""Reviewer-requested ablation (Addendum 3): C' = JSON wrapper + IDs + grouping with prose sentences.
Contrasts per model on paired tasks: C' - B (wrapper/IDs/grouping) and C - C' (formal predicates vs sentences),
Holm over the two within model; also strong-oracle labels for the same contrasts. Writes JSON + LaTeX table."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.analysis import paired_effect, holm

M = {"gpt-4.1-nano-2025-04-14": ("nano", "abl_nano"), "gpt-4.1-mini-2025-04-14": ("mini", "abl_mini"),
     "gpt-4.1-2025-04-14": ("4.1", "abl_full")}
out, L = {}, []
for m, (short, f) in M.items():
    ab = [json.loads(l) for l in open(ROOT / f"results/v2/ablation/{f}.jsonl")]
    ids = {r["task_id"] for r in ab}
    main = [json.loads(l) for l in open(ROOT / f"results/v2/trackA_{m}_eval.jsonl")]
    rows = [r for r in main if r["task_id"] in ids and r["condition"] in ("prose", "structured")] + ab
    res = [paired_effect(rows, m, "prose", "structured_nl"), paired_effect(rows, m, "structured_nl", "structured")]
    for r, p in zip(res, holm([r["mcnemar_exact_p"] for r in res])): r["holm_p"] = p
    rate = lambda c: sum(r["first_attempt_semantic_correct"] for r in rows if r["condition"] == c) / sum(1 for r in rows if r["condition"] == c)
    out[short] = {"n_tasks": len(ids), "rates": {c: rate(c) for c in ("prose", "structured_nl", "structured")},
                  "contrasts": res, "cost_usd": None}
    pfmt = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    L.append(f"{short} & {len(ids)} & {100*rate('prose'):.1f} & {100*rate('structured_nl'):.1f} & {100*rate('structured'):.1f} & "
             f"{100*res[0]['effect']:+.1f} [{100*res[0]['ci95_cluster_bootstrap'][0]:+.1f}, {100*res[0]['ci95_cluster_bootstrap'][1]:+.1f}] ({pfmt(res[0]['holm_p'])}) & "
             f"{100*res[1]['effect']:+.1f} [{100*res[1]['ci95_cluster_bootstrap'][0]:+.1f}, {100*res[1]['ci95_cluster_bootstrap'][1]:+.1f}] ({pfmt(res[1]['holm_p'])}) \\\\")
(ROOT / "results/v2/ablation/analysis_ablation.json").write_text(json.dumps(out, indent=1, default=str))
tab = ["\\begin{tabular}{lrrrrll}", "\\toprule",
       "Model & $n$ & B & C$'$ & C & C$'-$B (wrapper) & C$-$C$'$ (predicates) \\\\", "\\midrule"] + L + ["\\bottomrule", "\\end{tabular}"]
(ROOT / "paper/figures/tab_ablation.tex").write_text("\n".join(tab))
for k, v in out.items():
    print(k, v["n_tasks"], {c: round(x, 3) for c, x in v["rates"].items()},
          [(e["contrast"], round(100 * e["effect"], 1), [round(100 * x, 1) for x in e["ci95_cluster_bootstrap"]], e["discordant"], round(e["holm_p"], 4)) for e in v["contrasts"]])
