"""Run trials: every (task, condition) pair in a seeded random order, fresh conversation per trial.

Resumable: trials already present in --out are skipped. Writes trial records (JSONL) and full transcripts.
Example:
  python scripts/run_experiment.py --provider openai --model gpt-4.1-mini-2025-04-14 --split eval --workers 8
"""
import argparse, hashlib, json, os, random, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import yaml
from agentisa.tasks import all_tasks, manifest_hash
from agentisa.interfaces import CONDITIONS, CONDITIONS_V2
from agentisa.agent import run_agent_trial
from agentisa.harness import run_trial
from agentisa.models import ChatClient
from agentisa.fact_inventory import FACTS

p = argparse.ArgumentParser()
p.add_argument("--provider", required=True, choices=["openai", "ollama"])
p.add_argument("--model", required=True)
p.add_argument("--split", default="eval", choices=["dev", "eval", "all"])
p.add_argument("--conditions", default=None)
p.add_argument("--track", default="A", choices=["A", "B"], help="A = interface isolation (harness repairs); B = closed-loop agent")
p.add_argument("--suite", default="all", choices=["synthetic", "kernelbench", "all"])
p.add_argument("--trials", type=int, default=1)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--limit", type=int, default=0, help="first N tasks only (pilot)")
p.add_argument("--templates", default="", help="comma-separated template names (pilot)")
p.add_argument("--out", default=None)
p.add_argument("--max-usd", type=float, default=0.0, help="hard stop once list-price spend of THIS run reaches this (0 = no paid calls allowed for openai)")
a = p.parse_args()

# API key: read from the project's .env (git-ignored); never from shell rc files
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip().removeprefix("export ").strip(), _v.strip().strip("\"'"))
cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
lock = json.loads((ROOT / "data/versions.lock.json").read_text())
budget = cfg["budget"]
tasks = all_tasks(None if a.split == "all" else a.split, suite=a.suite)
if a.templates:
    keep = set(a.templates.split(",")); tasks = [t for t in tasks if t["template"] in keep]
if a.limit:
    tasks = tasks[:a.limit]
conds = (a.conditions or ",".join(cfg["tracks"][a.track]["conditions"])).split(",")
assert all(c in CONDITIONS_V2 for c in conds), conds
safe = a.model.replace(":", "_").replace("/", "_")
out = Path(a.out or ROOT / f"results/v2/track{a.track}_{safe}_{a.split}.jsonl")
tr_out = out.with_name(out.stem + "_transcripts.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for line in out.read_text().splitlines():
        if line.strip():
            r = json.loads(line); done.add((r["task_id"], r["condition"], r["trial"]))
# v2: task-blocked order (tasks shuffled; conditions shuffled within each task) so that a budget stop leaves
# complete paired tasks rather than scattered missing cells
rng = random.Random(cfg["seed"])
order = list(tasks); rng.shuffle(order)
jobs = []
for t in order:
    cs = list(conds); rng.shuffle(cs)
    jobs += [(t, c, k) for c in cs for k in range(a.trials) if (t["task_id"], c, k) not in done]
meta = dict(toolchain=lock, study="v2", track=a.track, manifest_sha256=manifest_hash(all_tasks(suite="all")),
            fact_inventory_sha256=hashlib.sha256(json.dumps(FACTS, sort_keys=True).encode()).hexdigest(),
            temperature=cfg["model"]["temperature"])
client = ChatClient(a.provider, a.model, temperature=cfg["model"]["temperature"])
PRICE = {"gpt-4.1-nano-2025-04-14": (0.10, 0.40), "gpt-4.1-mini-2025-04-14": (0.40, 1.60), "gpt-4.1-2025-04-14": (2.00, 8.00)}
if a.provider == "openai":
    if a.model not in PRICE: sys.exit(f"no price for {a.model}; add it before running")
    if a.max_usd <= 0: sys.exit("refusing to make paid API calls without an explicit --max-usd budget")
spent = 0.0; stop = threading.Event()
wlock = threading.Lock(); n_ok = 0; t0 = time.time()
print(f"{len(jobs)} trials to run ({len(done)} already done) -> {out}", flush=True)


def job(t, c, k):
    if stop.is_set():
        return None, None
    fn = run_trial if a.track == "A" else run_agent_trial
    log, msgs = fn(client, t, c, dict(budget, max_output_tokens=cfg["model"]["max_output_tokens"]), trial=k)
    log.setdefault("source", t.get("source"))
    log.update(meta)
    return log, msgs


with ThreadPoolExecutor(a.workers, thread_name_prefix="w") as ex:
    futs = {ex.submit(job, *j): j for j in jobs}
    for i, f in enumerate(as_completed(futs), 1):
        t, c, k = futs[f]
        try:
            log, msgs = f.result()
        except Exception:
            print(f"ERROR {t['task_id']} {c}: {traceback.format_exc()[-800:]}", flush=True); continue
        if log is None:
            continue
        if a.provider == "openai":
            pin, pout = PRICE[a.model]
            spent += log["input_tokens"] / 1e6 * pin + log["output_tokens"] / 1e6 * pout
            if spent >= a.max_usd and not stop.is_set():
                stop.set(); print(f"BUDGET STOP: spent ${spent:.2f} >= ${a.max_usd:.2f}; remaining trials skipped", flush=True)
        with wlock:
            with out.open("a") as fh:
                fh.write(json.dumps(log) + "\n")
            with tr_out.open("a") as fh:
                fh.write(json.dumps({"task_id": t["task_id"], "condition": c, "trial": k, "model": a.model,
                                     "messages": msgs}) + "\n")
        n_ok += log.get("first_attempt_semantic_correct", log.get("semantic_correct", False))
        if i % 20 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] correct so far {n_ok}/{i}  spent ${spent:.2f}  {time.time() - t0:.0f}s", flush=True)
