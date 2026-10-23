# AgentISA-Bench

Artifact for **"What Makes a Virtual ISA Agent-Programmable? A Controlled Study of Agent-Facing Documentation for StableHLO"**, by Luxin Zhang. VISA 2026 Workshop, co-located with MICRO 2026; non-archival. The paper is posted on the workshop website.

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
  - On-demand access trailed upfront delivery (−4 to −32 pp): GPT-4.1 never called the tool unless forced, and nano stopped retrieving when part of the documentation was given upfront.
  - Over 99% of failures were rejected statically.
- **Post-hoc checks after review (14,002 trials; each under a protocol addendum frozen before its first trial).**
  - The presence effect held for six of seven models (GPT-4.1 ×3, Qwen3-Coder, Claude Haiku 4.5, GPT-5.4-mini; the exception is Claude Haiku 5.5, which already writes StableHLO).
  - Curated facts beat the raw 91k-token specification by 12–26 pp.
  - The JSON penalty and the uptake failure were large only for GPT-4.1 mini and 4.1.

## Repository layout
| Path | Content |
|---|---|
| `src/agentisa/` | Fact inventory (96 facts anchored to the pinned spec), interface treatments and retrieval tool, task generator and oracles, sandboxed evaluator, trial harness, model clients (OpenAI, Anthropic, OpenRouter, Ollama), statistics |
| `scripts/` | Task validation, KernelBench conversion, treatment audit, mutation tests, stronger-oracle re-grade, experiment runner, analyses |
| `configs/experiment.yaml` | Frozen Study 2 configuration |
| `data/processed/` | Task manifests (synthetic and KernelBench-derived), converted KernelBench tasks with oracle inputs (`kernelbench/`), validation (gate G1), treatment audits, mutation-test results |
| `docs/` | Pre-registration (`PREREGISTRATION_V2.md`), protocol, reproducibility notes |
| `results/` | Study 1 logs (`trials_*`), Study 2 pre-registered logs and transcripts (`results/v2/trackA_*`), post-hoc trial records (`results/v2/<addendum>/`), analyses and camera-ready tables (`results/v2/tables/`), file hashes (`FREEZE.sha256`) |
| `releases/v1/` | Frozen snapshot of Study 1 |
| `tests/` | Unit tests |
| `REVIEW.md`, `legacy/` | Review of the original scaffold and its illustrative contract; historical only (every issue it marks [B] or [M] is fixed in the current code) |

The post-hoc runs reported in the camera-ready paper (14,002 trials under Addenda 3–8: ablations, repeats, raw specification, four further models) are included. Their trial records are in `results/v2/{ablation,repeats,full_spec,haiku,family,addendum7,addendum8}/`, each directory with the hashes frozen before its first trial (`FREEZE.sha256`); the addenda are in `docs/PREREGISTRATION_V2.md`. The ablation's transcripts are in the repository; the others (274 MB) are attached to the [camera-ready release](https://github.com/luxinlabs/VISA2026/releases/tag/camera-ready) as `visa2026-posthoc-transcripts.tar.xz` (unpack it in the repository root). Development pilots are not included.

`results/v2/FREEZE.sha256` lists the files hashed before the Study 2 runs. Check it against commit `af6c04c` (Addendum 1, made before the first valid trial): two entries, `src/agentisa/analysis.py` and the first `configs/experiment.yaml` line, still carry hashes from before that commit. Later commits added the post-hoc conditions and logging fields; the documentation rendered for the eight pre-registered conditions is unchanged. The code hashed before Addenda 7 and 8 is commit `4547ff7`. Addenda 4–6 hashed intermediate versions of `scripts/run_experiment.py` and `src/agentisa/{harness,interfaces,models}.py` that were not committed separately; their hashes therefore cannot be checked against the history.

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
.venv/bin/python scripts/validate_tasks.py        # gate G1: all 328 reference programs pass their oracles
.venv/bin/python tests/check_fact_examples.py     # every code example in the fact inventory verifies
.venv/bin/python scripts/audit_treatments.py      # fact equivalence and token lengths of the treatments
.venv/bin/python scripts/mutation_test.py         # oracle sensitivity
.venv/bin/python scripts/analyze_v2.py            # Study 2 tables and contrasts from the committed logs
.venv/bin/python scripts/analyze_ablation.py      # JSON-with-prose ablation
.venv/bin/python scripts/analyze_v2.py --with-haiku   # Claude Haiku 5.5 run (Addendum 5)
.venv/bin/python scripts/analyze_camera_ready.py  # post-hoc analyses and camera-ready Tables II and V (results/v2/tables/)
```
The converted KernelBench tasks are committed in `data/processed/kernelbench/`. To regenerate them from KernelBench, you need a local clone in `data/vendor/kernelbench` (see `data/DATA_SOURCES.md`):
```bash
.venv/bin/python scripts/convert_kernelbench.py
```
Running trials costs money. Each run is resumable, ordered task by task, and stops at its `--max-usd` cap. Example:
```bash
.venv/bin/python scripts/run_experiment.py --provider openai --model gpt-4.1-mini-2025-04-14 \
    --split eval --suite all --track A --workers 6 --max-usd 12
```
The raw-specification condition (`--conditions full_spec`) needs the StableHLO clone at the pinned commit (see `data/DATA_SOURCES.md`); its spec is checked against the hash in `data/versions.lock.json`. Other providers: `--provider anthropic` (e.g. `claude-haiku-4-5-20251001`), `--provider openrouter` (e.g. `qwen/qwen3-coder`). Useful flags: `--conditions`, `--repairs 0` (first attempt only), `--trials N --trial-start K` (repeats).

## License
- Code: MIT (`LICENSE`).
- Data (fact inventory, synthetic tasks, protocols, logs): CC BY 4.0 (`LICENSE-DATA.md`).
- KernelBench-derived content remains under KernelBench's MIT License (`THIRD_PARTY_NOTICES.md`).

## Citation
See `CITATION.cff`.

## AI use
An AI coding assistant (Anthropic Claude, via Claude Code) implemented much of the code and drafted parts of the paper under the author's direction; see the AI-use disclosure in the paper. The evaluated models are objects of study.
