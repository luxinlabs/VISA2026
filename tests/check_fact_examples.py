"""Every code example in a syntax fact must verify under the pinned toolchain (run with the .venv python)."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentisa.fact_inventory import FACTS
from jax._src.interpreters import mlir as jmlir
from jaxlib.mlir import ir


def wrap(code):
    code = code.strip()
    if '"stablehlo.convolution"' in code:
        return f"func.func @main(%x: tensor<2x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<2x4x8x8xf32> {{\n  {code}\n  return %r : tensor<2x4x8x8xf32>\n}}"
    if '"stablehlo.reduce_window"' in code:
        return f"func.func @main(%x: tensor<2x3x8x8xf32>, %init: tensor<f32>) -> tensor<2x3x4x4xf32> {{\n  {code}\n  return %r : tensor<2x3x4x4xf32>\n}}"
    if code.startswith("%r = chlo."):
        return f"func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {{\n  {code}\n  return %r : tensor<4xf32>\n}}"
    if '"stablehlo.reduce"' in code:
        return f"func.func @main(%x: tensor<4x3xf32>, %init: tensor<f32>) -> tensor<4xf32> {{\n  {code}\n  return %r : tensor<4xf32>\n}}"
    if code.startswith("func.func"):
        body = code.replace("{ ... return %r", "{ %r = stablehlo.add %x, %x : tensor<4x3xf32> return %r")
        return body
    ins, out = None, None
    m = re.search(r":\s*\((.*)\)\s*->\s*(tensor<[^>]*>)\s*$", code)
    if m:
        ins, out = [s.strip() for s in re.split(r",\s*(?=tensor)", m.group(1)) if s.strip()], m.group(2)
    else:  # pretty unary/binary form `... : T`
        t = re.search(r":\s*(tensor<[^>]*>)\s*$", code).group(1); out = t
    names = re.findall(r"%[a-z_]+\b", code.split("=", 1)[1].split(":")[0])
    names = [n for n in dict.fromkeys(names) if n not in ("%r",)]
    if ins is None:
        ins = [out] * len(names)
    if code.startswith("%c ="):
        names, ins = [], []
        out = re.search(r"(tensor<[^>]*>)\s*$", code).group(1)
    code = code.replace("%c =", "%r =")
    args = ", ".join(f"{n}: {t}" for n, t in zip(names, ins))
    return f"func.func @main({args}) -> {out} {{\n  {code}\n  return %r : {out}\n}}"


ok = bad = 0
with jmlir.make_ir_context():
    for f in FACTS:
        if f["category"] not in ("syntax",) and f["id"] != "CONST.SEM.1":
            continue
        for code in re.findall(r"`([^`]+)`", f.get("nl_v1", f["nl"])):
            if "%" not in code and "dense" not in code or "stablehlo.NAME" in code or "lo0" in code or "dense<[0, 1]>" in code:
                continue
            if code.startswith("dense<"):
                code = f"%c = stablehlo.constant {code} : tensor<f32>"
            try:
                ir.Module.parse(wrap(code)); ok += 1
            except Exception as e:
                bad += 1; print("FAIL", f["id"], code[:120], "\n   ", str(e)[:300])
print(f"{ok} examples verify, {bad} fail")
sys.exit(1 if bad else 0)
