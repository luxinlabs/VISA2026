"""Render all interface treatments for one task (for inspection / the paper's example figure)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentisa.tasks import all_tasks
from agentisa.interfaces import render, audit_equivalence, CONDITIONS
p = argparse.ArgumentParser(); p.add_argument('--task', default='bias_add_row__0'); p.add_argument('--out', required=True); a = p.parse_args()
t = {t["task_id"]: t for t in all_tasks()}[a.task]
audit_equivalence(t)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({c: render(t, c) for c in CONDITIONS}, indent=2, ensure_ascii=False))
print('Rendered; automated audit passed (human semantic audit still required).')
