"""Convert KernelBench level-1/2 PyTorch tasks into StableHLO tasks (docs/REAL_WORLD_CONVERSION.md).

Per task (run in a subprocess for isolation):
  1. Shrink top-level integer size literals (>16) in the source so programs stay small; derived globals recompute.
  2. Instantiate the Model (training mode, as KernelBench evaluates it); parameters become explicit @main args.
  3. Oracle = PyTorch eager output on 3 held-out cases (fresh parameter init + inputs per case seed).
  4. Reference solution = torch.export -> torchax -> JAX lowering to StableHLO text (never shown to agents).
  5. Validation (separately, scripts/validate_tasks.py): the reference must verify, run on XLA:CPU and match the
     PyTorch oracle. Excluded tasks are logged with reasons (dropout/randomness, multiple outputs, export failure,
     unsupported ops, mismatch).
"""
import json, re, subprocess, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data/vendor/kernelbench/KernelBench"
OUT = ROOT / "data/processed/kernelbench"
SMALL = [6, 7, 8, 9, 10, 5, 11, 12, 13]


def _distinct_small(v, vals):
    if v not in vals:
        vals[v] = SMALL[len(vals) % len(SMALL)]
    return vals[v]


def _pow2_div(limit):
    def f(v, vals):  # divide by the smallest power of two that brings v to <= limit (keeps divisibility)
        w = v
        while w > limit and w % 2 == 0:
            w //= 2
        vals[v] = w if w <= limit else limit
        return vals[v]
    return f


STRATEGIES = {"distinct_small": _distinct_small, "pow2_le16": _pow2_div(16), "pow2_le8": _pow2_div(8),
              "pow2_le32": _pow2_div(32)}


def shrink_source(src, strategy="distinct_small"):
    """Replace every integer literal > 16 on top-level (unindented) statement lines, including inside
    expressions and tuples (e.g. `N = 2048 * 2`, `shape = (4096,)`). Equal originals map to equal small values,
    so equalities such as height == width are preserved; derived globals recompute at exec time."""
    vals = {}

    def sv(m):
        v = int(m.group(1))
        if v <= 16:
            return m.group(1)
        return str(STRATEGIES[strategy](v, vals))
    lines = []
    for line in src.splitlines():
        top = line and not line[0].isspace() and not re.match(r"(def|class|import|from|@|#|return)\b", line)
        lines.append(re.sub(r"(?<![\w.])(\d+)(?![\w.])", sv, line) if top else line)
    return "\n".join(lines) + "\n", vals


MAX_ELEMS, MAX_TOTAL = 65536, 262144


def convert_one(path, task_id, outdir):
    reasons = []
    for strat in STRATEGIES:
        res = _convert_one(path, task_id, outdir, strat)
        if res["status"] == "converted" or "stochastic" in res["reason"] or "non-tensor" in res["reason"]:
            res["shrink_strategy"] = strat
            if reasons: res["earlier_attempts"] = reasons
            return res
        reasons.append(f"{strat}: {res['reason'][:160]}")
    return {"status": "excluded", "reason": " || ".join(reasons)}


def _convert_one(path, task_id, outdir, strategy):
    import numpy as np, torch, jax
    jax.config.update("jax_enable_x64", True)  # keep int64 (argmax/argmin/targets) as i64 in the lowering
    from torchax.export import exported_program_to_jax
    src = Path(path).read_text()
    if re.search(r"dropout|Dropout|torch\.rand\w*\(.*\)\s*\*|bernoulli|multinomial", src.split("def get_inputs")[0]):
        return {"status": "excluded", "reason": "stochastic op in forward (dropout/random)"}
    small, mapping = shrink_source(src, strategy)
    mod = types.ModuleType("kb")
    try:
        exec(compile(small, str(path), "exec"), mod.__dict__)
    except Exception as e:
        return {"status": "excluded", "reason": f"source exec failed: {type(e).__name__}: {str(e)[:160]}"}
    cases, outs = [], []
    for k in range(3):
        torch.manual_seed(1000 + k)
        try:
            model = mod.Model(*mod.get_init_inputs())
            _ = mod.get_inputs()
        except Exception as e:
            return {"status": "excluded", "reason": f"model build failed: {type(e).__name__}: {str(e)[:160]}"}
        torch.manual_seed(1000 + k)
        model = mod.Model(*mod.get_init_inputs())
        params = dict(model.named_parameters())
        inputs = mod.get_inputs()
        if not all(isinstance(x, torch.Tensor) for x in inputs):
            return {"status": "excluded", "reason": "non-tensor forward inputs"}
        names = list(params)
        args = [p.detach().clone() for p in params.values()] + [x.clone() for x in inputs]
        try:
            with torch.no_grad():
                y = torch.func.functional_call(model, dict(zip(names, args[:len(names)])), tuple(args[len(names):]))
        except Exception as e:
            return {"status": "excluded", "reason": f"forward failed: {type(e).__name__}: {str(e)[:160]}"}
        if not isinstance(y, torch.Tensor):
            return {"status": "excluded", "reason": "multiple or non-tensor outputs"}
        cases.append([a.numpy().copy() for a in args]); outs.append(y.detach().numpy().copy())
    sizes = [a.size for a in cases[0]] + [outs[0].size]
    if max(sizes) > MAX_ELEMS or sum(sizes) > MAX_TOTAL:
        return {"status": "excluded", "reason": f"too large after shrinking: max {max(sizes)}, total {sum(sizes)} elements"}
    if any(a.dtype not in (np.float32, np.int64, np.int32, np.bool_) for a in cases[0]) or outs[0].dtype not in (np.float32, np.int64, np.int32, np.bool_):
        return {"status": "excluded", "reason": f"unsupported dtype {[str(a.dtype) for a in cases[0]]} -> {outs[0].dtype}"}

    class Fn(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.m = model

        def forward(self, *flat):
            return torch.func.functional_call(self.m, dict(zip(names, flat[:len(names)])), tuple(flat[len(names):]), strict=False)
    targs = tuple(torch.from_numpy(a) for a in cases[0])
    try:
        ep = torch.export.export(Fn(), targs)
        w, jfn = exported_program_to_jax(ep)
        f = lambda *a: jfn(w, a)
        sol = jax.jit(f).lower(*cases[0]).as_text()
    except Exception as e:
        return {"status": "excluded", "reason": f"export/lowering failed: {type(e).__name__}: {str(e)[:200]}"}
    sol = re.sub(r"\s*\{jax\.result_info = \"[^\"]*\"\}", "", sol)
    if sol.count("-> (tensor") and not re.search(r"-> \(tensor<[^)]*>\) \{", sol):
        pass
    ops = sorted(set(re.findall(r"\b((?:stablehlo|chlo)\.\w+)\b", sol)))
    np.savez(outdir / f"{task_id}.npz", **{f"c{c}_a{i}": a for c, args in enumerate(cases) for i, a in enumerate(args)},
             **{f"y{c}": y for c, y in enumerate(outs)})
    pnames = names + [f"input{i}" for i in range(len(cases[0]) - len(names))]
    return {"status": "converted", "source_file": str(Path(path).relative_to(ROOT / "data/vendor/kernelbench")),
            "size_map": {str(k): v for k, v in mapping.items()}, "pytorch_source": small,
            "arg_names": pnames, "arg_shapes": [list(a.shape) for a in cases[0]], "arg_dtypes": [str(a.dtype) for a in cases[0]],
            "out_shape": list(outs[0].shape), "out_dtype": str(outs[0].dtype), "n_params": len(names),
            "reference_ops": ops, "sol": sol}


if __name__ == "__main__":
    if len(sys.argv) == 4:  # worker mode
        path, tid, outdir = sys.argv[1:]
        try:
            res = convert_one(path, tid, Path(outdir))
        except Exception as e:
            res = {"status": "excluded", "reason": f"model build/run failed: {type(e).__name__}: {str(e)[:200]}"}
        print("@@RESULT@@" + json.dumps(res)); sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    files = [(lvl, p) for lvl in ("level1", "level2") for p in sorted((KB / lvl).glob("*.py"))]

    (OUT / "results").mkdir(exist_ok=True)

    def run(item):
        lvl, p = item
        tid = f"kb_{lvl[-1]}_{p.stem.split('_')[0]}"
        done = OUT / "results" / f"{tid}.json"
        if done.exists():
            return json.loads(done.read_text())
        try:
            r = subprocess.run([sys.executable, __file__, str(p), tid, str(OUT)], capture_output=True, text=True, timeout=300)
            line = [l for l in r.stdout.splitlines() if l.startswith("@@RESULT@@")]
            res = json.loads(line[0][10:]) if line else {"status": "excluded", "reason": "worker crash: " + r.stderr[-300:]}
        except subprocess.TimeoutExpired:
            res = {"status": "excluded", "reason": "conversion timeout"}
        res.update(task_id=tid, level=int(lvl[-1]), upstream_file=p.name)
        done.write_text(json.dumps(res))
        return res
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(run, files))
    (OUT / "conversion.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    from collections import Counter
    print(Counter(r["status"] for r in results))
    for r in results:
        if r["status"] != "converted":
            print(r["task_id"], r["upstream_file"], "|", r["reason"][:150])
