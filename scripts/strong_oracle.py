"""Strengthened-oracle sensitivity analysis (reviewer W1). No model calls.

1. For every KernelBench-derived eval task, build 4 new held-out cases and compute expected outputs with PyTorch
   eager (the independent oracle), using the task's own parameters from case 0 and new inputs:
     signed N(0,1); signed N(0,3); signed with 20% exact zeros; signed with sparse +-8 outliers.
   Task-appropriate fallback: if PyTorch yields NaN/Inf on signed inputs, use |x| (non-negative, same cases).
   Integer inputs (e.g. class targets) are kept from the original cases.
   Synthetic tasks get 4 new cases from their own trap-aware generators and numpy references.
2. Gate: each task's reference StableHLO program must pass its strong cases (else the task is reported, not used).
3. Re-grade every accepted Study 2 program (first attempt, and final attempt after repair) on the strong cases.
4. Re-probe all KernelBench mutation survivors on the strong cases.
Outputs results/v2/strong_oracle.json (+ per-trial labels in results/v2/strong_labels.jsonl).
"""
import glob, json, sys, types
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from agentisa.tasks import all_tasks, compare, N_CASES, _default_gen
from agentisa.evaluator import run_program

KB = ROOT / "data/processed/kernelbench/results"
OUT = ROOT / "results/v2"


def kb_strong_cases(task):
    import torch
    r = json.loads((KB / f"{task['task_id']}.json").read_text())
    mod = types.ModuleType("kb"); exec(compile(r["pytorch_source"], task["task_id"], "exec"), mod.__dict__)
    torch.manual_seed(0)
    model = mod.Model(*mod.get_init_inputs())
    names = [n for n, _ in model.named_parameters()]
    base = task["cases"][0]; npar = r["n_params"]
    rng = np.random.default_rng(int(task["task_id"].split("_")[-1]) + 77)

    def make(kind, signed=True):
        out = list(base[:npar])
        for a in base[npar:]:
            if a.dtype.kind != "f":
                out.append(a.copy()); continue
            if kind == 0: x = rng.standard_normal(a.shape)
            elif kind == 1: x = 3 * rng.standard_normal(a.shape)
            elif kind == 2: x = rng.standard_normal(a.shape); x[rng.random(a.shape) < 0.2] = 0.0
            elif kind == 3: x = rng.standard_normal(a.shape); m = rng.random(a.shape) < 0.05; x[m] = 8 * np.sign(x[m] + 1e-9)
            else: x = -np.abs(rng.standard_normal(a.shape)) - 0.1  # all-negative boundary (max-init / ReLU traps)
            out.append((x if signed else np.abs(x)).astype(a.dtype))
        return out

    def expect(args):
        with torch.no_grad():
            y = torch.func.functional_call(model, dict(zip(names, [torch.from_numpy(a.copy()) for a in args[:npar]])),
                                           tuple(torch.from_numpy(a.copy()) for a in args[npar:]))
        return y.detach().numpy()
    cases, dist = [], "signed"
    for k in range(5):
        c = make(k, True); y = expect(c)
        if not np.all(np.isfinite(y)):
            dist = "nonnegative_fallback"; break
        cases.append((c, y))
    if dist != "signed":
        rng = np.random.default_rng(int(task["task_id"].split("_")[-1]) + 77)
        cases = [(c, expect(c)) for c in (make(k, False) for k in range(4))]  # all-negative case dropped for non-negative-domain tasks
    return [c for c, _ in cases], [y.astype(np.float32) if y.dtype == np.float64 else y for _, y in cases], dist


def syn_strong_cases(task):
    gen = task.get("gen") or _default_gen(task["args"])
    rng = np.random.default_rng(task["seed"] + 991)
    cases = [gen(rng, c % N_CASES) for c in range(5)]
    from agentisa.tasks import NP
    return cases, [np.asarray(task["ref"](*c)).astype(NP[task["out"][1]]) for c in cases], "generator"


def passes(prog, task, cases, exp):
    if not prog: return False, "no_program"
    res = run_program(prog, task["signature"], cases)
    if res["stage"] != "executed": return False, res["stage"]
    for a, e in zip(res["outputs"], exp):
        ok, msg = compare(a, e, task)
        if not ok: return False, "wrong_values"
    return True, ""


if __name__ == "__main__":
    tasks = {t["task_id"]: t for t in all_tasks("eval", suite="all")}
    strong = {}
    for tid, t in tasks.items():
        c, e, d = (kb_strong_cases(t) if t.get("source") == "kernelbench" else syn_strong_cases(t))
        strong[tid] = (c, e, d)
    # gate: reference must pass strong cases
    with ThreadPoolExecutor(4) as ex:
        gate = dict(zip(strong, ex.map(lambda tid: passes(tasks[tid]["sol"], tasks[tid], *strong[tid][:2]), strong)))
    bad_gate = sorted(t for t, (ok, _) in gate.items() if not ok)
    print("reference fails strong cases:", bad_gate, flush=True)
    rows = [json.loads(l) for f in sorted(glob.glob(str(OUT / "trackA_gpt-4.1*_eval.jsonl"))) for l in open(f) if l.strip()]
    jobs = []
    for i, r in enumerate(rows):
        if r["task_id"] in bad_gate: continue
        if r["first_attempt_semantic_correct"]: jobs.append((i, "first", r["attempts"][0]["program"]))
        if r["correct_after_repair"]: jobs.append((i, "final", r["attempts"][-1]["program"]))

    def regrade(job):
        i, which, prog = job; r = rows[i]; t = tasks[r["task_id"]]
        return i, which, passes(prog, t, *strong[r["task_id"]][:2])
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(regrade, jobs))
    flips = [(rows[i]["model"], rows[i]["task_id"], rows[i]["condition"], which, why) for i, which, (ok, why) in res if not ok]
    labels = {(i, w): ok for i, w, (ok, _) in res}
    with open(OUT / "strong_labels.jsonl", "w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"model": r["model"], "task_id": r["task_id"], "condition": r["condition"],
                                 "first_strong": bool(r["first_attempt_semantic_correct"] and labels.get((i, "first"), False) and r["task_id"] not in bad_gate),
                                 "final_strong": bool(r["correct_after_repair"] and labels.get((i, "final"), False) and r["task_id"] not in bad_gate)}) + "\n")
    # mutation survivors re-probe
    from mutation_test import mutants
    surv = json.loads((ROOT / "data/processed/mutation_test_kernelbench.json").read_text())["survivors"]

    def probe(s):
        # Several mutants can share a name (identical attribute lists), so find the ones that actually survived the
        # original oracle cases, then test each of those on the strong cases.
        tid, name = s; t = tasks[tid]
        if tid in bad_gate: return tid, name, "task_excluded"
        same = [p for n, p in mutants(t["sol"]) if n == name]
        survivors = [p for p in same if passes(p, t, t["cases"], [np.asarray(e) for e in t["expected"]])[0]]
        if not survivors: return tid, name, "no_survivor_reproduced"
        res = [passes(p, t, *strong[tid][:2]) for p in survivors]
        return tid, name, "survives_strong_oracle" if any(ok for ok, _ in res) else "killed_by_strong_oracle"
    with ThreadPoolExecutor(4) as ex:
        sp = list(ex.map(probe, [tuple(s) for s in surv]))
    prev = {(d[0], d[1]): d[2] for d in json.loads((ROOT / "data/processed/mutation_survivor_probe_kernelbench.json").read_text())["detail"]}
    out = {"strong_case_distribution": dict(Counter(v[2] for v in strong.values())),
           "reference_fails_strong": bad_gate,
           "regraded": {"first_attempt_accepted": sum(1 for j in jobs if j[1] == "first"), "final_accepted": sum(1 for j in jobs if j[1] == "final")},
           "flips": flips, "n_flips": dict(Counter(f[3] for f in flips)),
           "flips_by_model_condition": dict(Counter(f"{f[0]}|{f[2]}|{f[3]}" for f in flips)),
           "mutation_resurvey": dict(Counter(x[2].split("(")[0] for x in sp)),
           "previously_unprobed": [(t, n, s) for t, n, s in sp if prev.get((t, n)) == "probe_error"],
           "still_surviving": [(t, n) for t, n, s in sp if s == "survives_strong_oracle"]}
    (OUT / "strong_oracle.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("still_surviving",)}, indent=1, default=str)[:4000])
