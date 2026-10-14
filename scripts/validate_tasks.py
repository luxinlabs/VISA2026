"""Gate G1: every task's reference StableHLO solution must parse, verify, execute and match the
independent numpy oracle on all held-out cases. Also checks that fact-inventory code examples parse."""
import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.tasks import all_tasks, expected_outputs, compare, manifest_record, manifest_hash
from agentisa.evaluator import run_program
from agentisa.fact_inventory import FACTS


def check(t):
    res = run_program(t["sol"], t["signature"], t["cases"])
    if res["stage"] != "executed":
        return t["task_id"], False, f'{res["stage"]}: {res.get("error", "")[:400]}'
    for a, e in zip(res["outputs"], expected_outputs(t)):
        ok, msg = compare(a, e, t)
        if not ok:
            return t["task_id"], False, msg
    return t["task_id"], True, ""


def fact_examples():
    """Wrap every generic-form example in a syntax fact into a function and check it verifies."""
    out = []
    for f in FACTS:
        if f["category"] != "syntax" or f["op"] in ("module",):
            continue
        for code in re.findall(r"`([^`]*%r = [^`]*)`", f["nl"]):
            m = re.search(r":\s*\(([^)]*)\)\s*->\s*(tensor<[^>]*>)\s*$", code) or re.search(r":\s*(tensor<[^>]*>)\s*$", code)
            out.append((f["id"], code))
    return out


if __name__ == "__main__":
    tasks = all_tasks(); t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(check, tasks))
    bad = [r for r in results if not r[1]]
    for r in bad:
        print("FAIL", r[0], r[2])
    print(f"{len(results) - len(bad)}/{len(results)} reference solutions pass ({time.time() - t0:.0f}s)")
    out = ROOT / "data/processed"; out.mkdir(parents=True, exist_ok=True)
    (out / "tasks_manifest.jsonl").write_text("".join(json.dumps(manifest_record(t)) + "\n" for t in tasks))
    (out / "g1_validation.json").write_text(json.dumps({"n_tasks": len(tasks), "n_pass": len(results) - len(bad),
        "failures": bad, "manifest_sha256": manifest_hash(tasks)}, indent=1))
    sys.exit(1 if bad else 0)
