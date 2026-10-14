"""Validate externally converted task candidates (KernelBench/TritonBench/StableHLO tests) against
data/task_schema.json and emit an evaluation manifest containing ONLY oracle_validated tasks.
Excluded and unvalidated records are written to a separate exclusion log with reasons."""
import argparse, json
from pathlib import Path
import jsonschema
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument('--source', choices=['stablehlo', 'kernelbench', 'tritonbench'], required=True)
p.add_argument('--input', required=True); p.add_argument('--output', required=True); a = p.parse_args()
schema = json.loads((ROOT / 'data/task_schema.json').read_text())
records = [json.loads(s) for s in Path(a.input).read_text().splitlines() if s.strip()]
seen, keep, excl = set(), [], []
for i, r in enumerate(records):
    jsonschema.validate(r, schema)
    if r['source'] != a.source: raise ValueError(f'record {i}: source {r["source"]} != {a.source}')
    if r['task_id'] in seen: raise ValueError(f'record {i}: duplicate task_id {r["task_id"]}')
    seen.add(r['task_id'])
    if r['validation_status'] == 'oracle_validated': keep.append(r)
    else: excl.append({**r, 'exclusion_reason': r.get('exclusion_reason') or f'status={r["validation_status"]}'})
out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(''.join(json.dumps(r) + '\n' for r in keep))
out.with_name(out.stem + '_excluded.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in excl))
print(f'{len(keep)} oracle-validated tasks kept, {len(excl)} excluded/unvalidated logged')
