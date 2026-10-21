# AgentISA-Bench

Artifact for **"What Makes a Virtual ISA Agent-Programmable? A Controlled Study of Agent-Facing Documentation for StableHLO"**, by Luxin Zhang. VISA 2026 Workshop, co-located with MICRO 2026; non-archival. The paper is posted on the workshop website. `paper/` holds the submitted version.

The study fixes one IR (StableHLO), a pinned verifier and executor, the tasks and a harness-driven repair loop. It varies only how documentation reaches an LLM agent:

- no documentation;
- syntax facts only;
- prose;
- a length-matched JSON contract with the same facts and examples;
- retrieval through a tool (free, forced, or with the cross-cutting rules given upfront).

Correctness is judged on 267 held-out tasks (114 synthetic, 153 lowered from KernelBench) by independent numpy and PyTorch oracles.

## Main results
- **Pre-registered study (5,005 trials, GPT-4.1 nano / mini / 4.1).**
  - Syntax-only documentation added 16–33 pp of first-attempt correctness.
  - With matched content, the JSON contract scored below prose (−6.0, −7.9, −12.1 pp). This reverses our earlier, unmatched comparison.
  - On-demand access trailed upfront delivery, because the agents did not call the tool.
  - Over 99% of failures were rejected statically.
- **Post-hoc checks after review (10,862 trials; each under a protocol addendum frozen before its first trial).**
  - The presence effect held for six of seven models (GPT-4.1 ×3, Qwen3-Coder, Claude Haiku 4.5, GPT-5.4-mini; the exception is Claude Haiku 5.5, which already writes StableHLO).
  - Curated facts beat the raw 91k-token specification by 12–26 pp.
  - The JSON penalty and the uptake failure were large only for GPT-4.1.

## Repository layout
| Path | Content |
|---|---|
| `src/agentisa/` | Fact inventory (96 facts anchored to the pinned spec), interface treatments and retrieval tool, task generator and oracles, sandboxed evaluator, trial harness, model clients (OpenAI, Anthropic, OpenRouter, Ollama), statistics |
| `scripts/` | Task validation, KernelBench conversion, treatment audit, mutation tests, stronger-oracle re-grade, experiment runner, analyses |
| `configs/experiment.yaml` | Frozen Study 2 configuration |
| `data/processed/` | Task manifests (synthetic and KernelBench-derived), validation (gate G1), treatment audits, mutation-test results |
| `docs/` | Pre-registration (`PREREGISTRATION_V2.md`), protocol, reproducibility notes |
| `results/` | Study 1 logs (`trials_*`), Study 2 pre-registered logs and transcripts (`results/v2/trackA_*`), ablation, file hashes (`FREEZE.sha256`) |
| `releases/v1/` | Frozen snapshot of Study 1 |
| `paper/` | LaTeX source and PDF of the submitted paper |
| `tests/` | Unit tests |

The post-hoc runs reported in the camera-ready paper (repeats, raw specification, further models, mechanism test) are not included: their logs and protocol addenda 4–8 are available from the author on request. The code that produced them is included.

## Setup
Python 3.12; exact dependencies are in `requirements.lock.txt`.
```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -m pytest -q tests
```
API keys go in a git-ignored `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`). The runner refuses paid calls without an explicit `--max-usd` cap.

## Reproduce
These steps need no API calls:
```bash
.venv/bin/python scripts/validate_tasks.py        # gate G1: every reference program passes its oracle
.venv/bin/python tests/check_fact_examples.py     # every code example in the fact inventory verifies
.venv/bin/python scripts/audit_treatments.py      # fact equivalence and token lengths of the treatments
.venv/bin/python scripts/mutation_test.py         # oracle sensitivity
.venv/bin/python scripts/analyze_v2.py            # Study 2 tables and contrasts from the committed logs
.venv/bin/python scripts/analyze_ablation.py      # JSON-with-prose ablation
```
KernelBench conversion needs a local KernelBench clone in `data/vendor/kernelbench` (see `data/DATA_SOURCES.md`):
```bash
.venv/bin/python scripts/convert_kernelbench.py
```
Running trials costs money. Each run is resumable, ordered task by task, and stops at its `--max-usd` cap. Example:
```bash
.venv/bin/python scripts/run_experiment.py --provider openai --model gpt-4.1-mini-2025-04-14 \
    --split eval --suite all --track A --workers 6 --max-usd 12
```
Other providers: `--provider anthropic` (e.g. `claude-haiku-4-5-20251001`), `--provider openrouter` (e.g. `qwen/qwen3-coder`). Useful flags: `--conditions`, `--repairs 0` (first attempt only), `--trials N --trial-start K` (repeats).

## License
- Code: MIT (`LICENSE`).
- Data (fact inventory, synthetic tasks, protocols, logs): CC BY 4.0 (`LICENSE-DATA.md`).
- KernelBench-derived content remains under KernelBench's MIT License (`THIRD_PARTY_NOTICES.md`).

## Citation
See `CITATION.cff`.

## AI use
An AI coding assistant (Anthropic Claude, via Claude Code) implemented much of the code and drafted parts of the paper under the author's direction; see the AI-use disclosure in the paper. The evaluated models are objects of study.
