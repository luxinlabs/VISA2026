"""Re-grade accepted first-attempt programs on the 9 KernelBench tasks whose oracle missed a mutant, using 5
fresh signed inputs and the validated reference program as comparator. No model calls."""
import glob, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.tasks import all_tasks
from agentisa.evaluator import run_program
weak = set(json.loads((ROOT / "data/processed/mutation_survivor_probe_kernelbench.json").read_text())["tasks_with_detectable_survivor"])
tasks = {t["task_id"]: t for t in all_tasks("eval", suite="kernelbench") if t["task_id"] in weak}
rows = [json.loads(l) for f in glob.glob(str(ROOT / "results/v2/trackA_gpt-4.1*_eval.jsonl")) for l in open(f)]
acc = [r for r in rows if r["task_id"] in weak and r["first_attempt_semantic_correct"]]

def check(r):
    t = tasks[r["task_id"]]; rng = np.random.default_rng(11)
    cases = [[(rng.standard_normal(a.shape).astype(a.dtype) if a.dtype.kind == "f" else a) for a in t["cases"][0]] for _ in range(5)]
    ref = run_program(t["sol"], t["signature"], cases); got = run_program(r["attempts"][0]["program"], t["signature"], cases)
    if got["stage"] != "executed": return r, "fails_on_signed"
    ok = all(np.allclose(a, b, rtol=1e-3, atol=1e-4, equal_nan=True) for a, b in zip(ref["outputs"], got["outputs"]))
    return r, "still_correct" if ok else "wrong_on_signed"

with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(check, acc))
from collections import Counter
print("weak tasks", len(weak), "accepted first attempts on them", len(acc), Counter(s for _, s in res))
flip = [(r["model"], r["task_id"], r["condition"]) for r, s in res if s != "still_correct"]
print(flip)
(ROOT / "results/v2/regrade_signed.json").write_text(json.dumps({"weak_tasks": sorted(weak), "accepted": len(acc),
    "result": dict(Counter(s for _, s in res)), "flipped": flip}, indent=1))
