"""Classify KernelBench mutation survivors: run mutant and reference on 5 fresh SIGNED random inputs
(standard normal; integer args kept). Differences => the held-out oracle cases (KernelBench's torch.rand,
non-negative) were too weak; identical => likely semantically equivalent (e.g. dead running-stat updates)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from agentisa.tasks import all_tasks
from agentisa.evaluator import run_program
from mutation_test import mutants

tasks = {t["task_id"]: t for t in all_tasks("eval", suite="kernelbench")}
surv = json.loads((ROOT / "data/processed/mutation_test_kernelbench.json").read_text())["survivors"]

def probe(item):
    tid, name = item; t = tasks[tid]
    prog = dict(mutants(t["sol"]))[name]
    rng = np.random.default_rng(7)
    cases = [[(rng.standard_normal(a.shape).astype(a.dtype) if a.dtype.kind == "f" else a) for a in t["cases"][0]] for _ in range(5)]
    r = run_program(t["sol"], t["signature"], cases); m = run_program(prog, t["signature"], cases)
    if r["stage"] != "executed" or m["stage"] != "executed":
        return tid, name, "probe_error"
    diff = any(not np.allclose(a, b, rtol=1e-3, atol=1e-4, equal_nan=True) for a, b in zip(r["outputs"], m["outputs"]))
    return tid, name, "detectable_with_signed_inputs" if diff else "likely_equivalent"

with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(probe, [tuple(s) for s in surv]))
from collections import Counter
c = Counter(r[2] for r in res)
out = {"survivors": len(res), "classification": dict(c), "detail": res,
       "tasks_with_detectable_survivor": sorted({r[0] for r in res if r[2] == "detectable_with_signed_inputs"})}
(ROOT / "data/processed/mutation_survivor_probe_kernelbench.json").write_text(json.dumps(out, indent=1))
print(dict(c), "tasks affected:", len(out["tasks_with_detectable_survivor"]))
