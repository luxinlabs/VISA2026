"""Oracle sensitivity: apply single textual mutations to each eval reference solution; a mutant that
still verifies and runs must be rejected by the oracle (unless it is semantically equivalent)."""
import json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from agentisa.tasks import all_tasks
from agentisa.harness import grade

SWAPS = [("stablehlo.add", "stablehlo.subtract"), ("stablehlo.subtract", "stablehlo.add"),
         ("stablehlo.multiply", "stablehlo.add"), ("stablehlo.divide", "stablehlo.multiply"),
         ('"stablehlo.maximum"', '"stablehlo.minimum"'), ("stablehlo.maximum", "stablehlo.minimum"),
         ("direction LE", "direction LT"), ("direction GT", "direction GE"), ("direction EQ", "direction NE"),
         ("dense<0xFF800000>", "dense<0.0>"), ("dense<0.0>", "dense<1.0>"), ("stablehlo.sqrt", "stablehlo.rsqrt"),
         ("stablehlo.exponential", "stablehlo.negate")]


def mutants(sol):
    out = []
    for a, b in SWAPS:
        if a in sol: out.append((f"{a}->{b}", sol.replace(a, b, 1)))
    for m in re.finditer(r"array<i64: ([0-9, ]+)>", sol):  # perturb integer-list attributes
        vals = [int(v) for v in m.group(1).split(",")]
        for alt in ([vals[::-1]] if len(vals) > 1 and vals != vals[::-1] else []) + [[v + 1 for v in vals[:1]] + vals[1:]]:
            out.append((f"attr {vals}->{alt}", sol[:m.start(1)] + ", ".join(map(str, alt)) + sol[m.end(1):]))
    for m in re.finditer(r"contracting_dimensions = \[(\d)\]", sol):
        out.append(("contracting+1", sol[:m.start(1)] + str(int(m.group(1)) ^ 1) + sol[m.end(1):]))
    return out


def run(item):
    t, name, prog = item
    ev = grade(prog, t)
    return t["template"], name, ev.verified, ev.semantic_correct


SUITE = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
tasks = all_tasks("eval", suite=SUITE)
items = [(t, n, p) for t in tasks for n, p in mutants(t["sol"])]
with ThreadPoolExecutor(3) as ex:
    res = list(ex.map(run, items))
execd = [r for r in res if r[2]]
survivors = [r for r in execd if r[3]]
out = {"mutants": len(res), "statically_rejected": len(res) - len(execd), "executed": len(execd),
       "killed_by_oracle": len(execd) - len(survivors), "survivors": sorted({(r[0], r[1]) for r in survivors})}
(ROOT / f"data/processed/mutation_test_{SUITE}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
