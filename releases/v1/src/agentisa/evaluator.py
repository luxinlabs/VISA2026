"""Pinned StableHLO evaluator: parse+verify (MLIR/StableHLO dialect in jaxlib) and execute (XLA CPU).

Generated programs are evaluated in a separate subprocess with a wall-clock timeout and an
operation allow-list (no custom_call, no foreign dialects). The parent process never imports JAX.
Correctness is decided by the independent numpy oracle in tasks.py, not here.
"""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ALLOWED_NON_STABLEHLO = {"func.func", "func.return", "func.call", "builtin.module"}
FORBIDDEN = {"stablehlo.custom_call", "stablehlo.infeed", "stablehlo.outfeed", "stablehlo.send",
             "stablehlo.recv", "stablehlo.rng", "stablehlo.rng_bit_generator"}


def classify_static_error(msg: str) -> str:
    """Heuristic sub-class of a parse/verify diagnostic (MLIR verifies during parsing)."""
    m = msg.lower()
    if re.search(r"expected|custom op|undeclared|unknown|redefinition|invalid|unexpected|does not exist|unbalanced|use of value", m) \
            and not re.search(r"dimension|shape|rank|size", m):
        return "static_syntax"
    if re.search(r"dimension|shape|rank|size|permutation|broadcast|index|indices|padding|out of bounds", m):
        return "static_shape"
    if re.search(r"type|i1|f32|i32|i64|element", m):
        return "static_type"
    return "static_other"


def _worker(req_path: str, out_path: str):
    import jax, jax.extend  # noqa: F401
    from jax._src.interpreters import mlir as jmlir
    from jax._src.lib import xla_client as xc
    from jaxlib.mlir import ir
    req = json.loads(Path(req_path).read_text())
    res = {"stage": None}
    try:
        with jmlir.make_ir_context():
            module = ir.Module.parse(req["program"])
            names, mains = [], []

            def walk(op):
                names.append(op.name)
                if op.name == "func.func" and op.attributes["sym_name"].value == "main":
                    mains.append(op)
                for region in op.regions:
                    for block in region.blocks:
                        for inner in block.operations:
                            walk(inner.operation)
            walk(module.operation)
            bad = sorted({n for n in names if n in FORBIDDEN or
                          (not n.startswith("stablehlo.") and n not in ALLOWED_NON_STABLEHLO)})
            if bad:
                res.update(stage="forbidden_op", error=f"Operations not allowed: {bad}")
                raise StopIteration
            if len(mains) != 1:
                res.update(stage="no_main", error="Program must define exactly one func.func @main")
                raise StopIteration
            sig = str(mains[0].attributes["function_type"].value)
            res["signature"] = sig
            if " ".join(sig.split()) != " ".join(req["expected_signature"].split()):
                res.update(stage="signature_mismatch",
                           error=f"@main has type {sig} but the task requires {req['expected_signature']}")
                raise StopIteration
            bytecode = jmlir.module_to_bytecode(module)
    except StopIteration:
        Path(out_path).write_text(json.dumps(res)); return
    except Exception as e:  # MLIR parse/verify diagnostic
        msg = str(e).replace("Unable to parse module assembly:\n", "")
        res.update(stage="static_error", error=msg[:2000], subclass=classify_static_error(msg))
        Path(out_path).write_text(json.dumps(res)); return
    try:
        backend = jax.extend.backend.get_backend("cpu")
        exe = backend.compile_and_load(bytecode, xc.DeviceList(tuple(backend.local_devices()[:1])),
                                       xc.CompileOptions())
    except Exception as e:
        res.update(stage="compile_error", error=str(e)[:2000])
        Path(out_path).write_text(json.dumps(res)); return
    data = np.load(req["inputs_npz"])
    outs = {}
    try:
        for c in range(req["n_cases"]):
            args = [jax.device_put(data[f"c{c}_a{i}"]) for i in range(req["n_args"])]
            out = exe.execute(args)
            if len(out) != 1:
                raise RuntimeError(f"expected 1 result, got {len(out)}")
            outs[f"c{c}"] = np.asarray(out[0])
    except Exception as e:
        res.update(stage="runtime_error", error=str(e)[:2000])
        Path(out_path).write_text(json.dumps(res)); return
    np.savez(req["outputs_npz"], **outs)
    res["stage"] = "executed"
    Path(out_path).write_text(json.dumps(res))


def run_program(program: str, expected_signature: str, cases: list, timeout: float = 60.0) -> dict:
    """cases: list of argument lists (numpy arrays). Returns dict with stage, error, outputs."""
    with tempfile.TemporaryDirectory(prefix="agentisa_") as td:
        td = Path(td)
        arrays = {f"c{c}_a{i}": a for c, args in enumerate(cases) for i, a in enumerate(args)}
        np.savez(td / "in.npz", **arrays)
        req = {"program": program, "expected_signature": expected_signature, "n_cases": len(cases),
               "n_args": len(cases[0]) if cases else 0, "inputs_npz": str(td / "in.npz"),
               "outputs_npz": str(td / "out.npz")}
        (td / "req.json").write_text(json.dumps(req))
        env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false",
                   TF_CPP_MIN_LOG_LEVEL="3", OMP_NUM_THREADS="1")
        try:
            p = subprocess.run([sys.executable, "-I", "-c",
                                "import sys; sys.path.insert(0, sys.argv[3]); "
                                "from agentisa.evaluator import _worker; _worker(sys.argv[1], sys.argv[2])",
                                str(td / "req.json"), str(td / "res.json"), str(Path(__file__).resolve().parents[1])],
                               capture_output=True, text=True, timeout=timeout, env=env, cwd=td)
        except subprocess.TimeoutExpired:
            return {"stage": "timeout", "error": f"evaluation exceeded {timeout}s"}
        if not (td / "res.json").exists():
            return {"stage": "crash", "error": (p.stderr or "")[-1500:]}
        res = json.loads((td / "res.json").read_text())
        if res["stage"] == "executed":
            z = np.load(td / "out.npz")
            res["outputs"] = [z[f"c{c}"] for c in range(len(cases))]
        return res
