# Study 2 (v2) protocol — FROZEN 2026-10-14 before any eval-split trial (hashes in results/v2/FREEZE.sha256)

Study 1 (v1, `releases/v1/`, results in `results/trials_*`) is complete and unchanged. Study 2 fixes the
v1 design problems found in review and extends the task population. It is a new study, not a re-analysis.

## Changes from v1 and why
| v1 problem | v2 fix |
|---|---|
| Syntax facts: prose carried a concrete instance, structured an abstract schema (confound in B-C and N-A) | Every syntax fact carries the SAME schema and the SAME concrete instance in both renderings; only the wrapper differs |
| Structured 31% longer than prose (length confound) | Compact JSON: structured/prose token ratio 0.98 over all 328 tasks |
| 114 small synthetic tasks only; oracle rarely decisive | + 176 KernelBench level-1/2 tasks lowered from PyTorch (153 eval), with PyTorch as the independent oracle |
| Interactive-forced was post-hoc | Pre-registered condition |
| Implication "cross-cutting facts must not depend on retrieval" untested | New `interactive_core`: module facts upfront, operation facts on demand |
| One-shot generation with harness-driven repair is not an agent | New Track B: closed-loop agent with verify / run_example / submit tools |
| Failure taxonomy post-hoc | `diagnostic_class` (analysis.py) is fixed in advance for v2 |
| Fact inventory lacked ops needed by real models | +19 facts: convolution, reduce_window, math, logical, CHLO erf, variadic reduce (96 facts, 23 groups; all 29 code examples verify) |

## Tasks
* Synthetic: 38 templates x 3 eval instances = 114 tasks (38 clusters). Same as v1.
* KernelBench: level 1-2 (MIT), sizes shrunk deterministically, parameters as explicit @main arguments, training-mode
  semantics as KernelBench evaluates. 200 -> 177 converted -> 176 pass G1 (reference lowering verified and matching the
  PyTorch oracle on 3 cases, rtol 1e-3 / atol 1e-4). Exclusions logged in data/processed/kernelbench/results/*.json.
  10% hash split = dev (24), eval = 152 tasks, each its own cluster.
* Eval population: 266 tasks (114 synthetic + 152 KernelBench). G1: 328/328 reference solutions pass.

## Track A — interface isolation (as v1)
Conditions: none, minimal, syntax_only_structured, prose, structured, interactive, interactive_forced, interactive_core.
Primary endpoint: first-attempt semantic correctness. Primary contrast: structured - prose.
Pre-specified contrasts (Holm within model): structured-prose, interactive-structured, interactive_forced-interactive,
interactive_core-interactive, interactive_core-structured, prose-minimal, minimal-none, syntax_only_structured-minimal.
Secondary: correct after <=2 repairs; per-suite (synthetic / KernelBench) estimates.

## Track B — closed-loop agent
Conditions: none, prose, structured, interactive, interactive_core. Tools: get_facts (interactive conditions), verify,
run_example (one visible example with expected output), submit. Budget 15 tool calls, 20 model calls. Endpoint:
submitted program semantically correct on all oracle cases (2 never shown). Contrasts: structured-prose,
interactive_core-structured, interactive-structured, prose-none (Holm within model). Secondary: tool usage, tokens.

## Scope approved by the project owner (2026-10-14)
Track A only, total API budget hard-capped at $35 (list price, enforced by the runner). Track B (closed-loop agent) is
implemented and unit-tested but NOT run in this study; it is reported as a design, with no results.
Dev-split pilot: 32 trials on gpt-4.1-nano ($0.04), plumbing only, not analysed.

## Models
gpt-4.1-nano-2025-04-14 (cap $5), gpt-4.1-mini-2025-04-14 (cap $12), then gpt-4.1-2025-04-14 with the remaining budget.
Temperature 0, one trial per cell, fresh conversation per trial. Task-blocked order: tasks shuffled (seed 20261014),
conditions shuffled within task. If a cap stops a run early, only tasks with all 8 conditions are analysed for that
model, and the number of completed tasks is reported (the stop point is outcome-blind: it depends only on spend).

## Statistics
Paired differences; 95% cluster bootstrap (template for synthetic, task for KernelBench), exact McNemar on task-level
discordant pairs, Holm correction. Suite-level and family-level analyses are secondary/exploratory.

## Addendum 1 (2026-10-14, before any valid eval trial)
The first launch of the nano/mini eval runs produced 0 trials: every call returned HTTP 401 because the runner let a
revoked key inherited from the shell environment take precedence over the project `.env` key. `run_experiment.py`
now lets `.env` override the environment (one-line change; no effect on prompts, tasks, grading or analysis). Its
hash in FREEZE.sha256 was updated accordingly; no trial data existed at the time.

## Addendum 2 (2026-10-14, during the eval runs; robustness only)
The gpt-4.1-nano run stalled twice with worker threads blocked for >1 h in SSL reads on half-dead connections
(the HTTP client timeout did not fire). The runner was restarted (resumable; completed trials kept, in-flight
trials re-run) and `models.py` gained a 150 s hard watchdog per API call. No prompt, grading or analysis change.
The gpt-4.1 run stopped at its budget cap after 733/2136 trials (task-blocked order, outcome-blind stop).

## Erratum (2026-10-14, after the runs; text only)
The "Tasks" bullets above state "dev (24), eval = 152 tasks" and "Eval population: 266 tasks (114 synthetic + 152
KernelBench)". The frozen, hashed manifest (`data/processed/tasks_manifest_all.jsonl`, hash in FREEZE.sha256) — the
authoritative artifact that the runs used — contains 23 KernelBench dev and 153 KernelBench eval tasks, i.e. 267 eval
tasks in total. The prose was written before the final G1 exclusion was applied and was not updated. No task, code or
analysis changed; the paper reports the manifest counts (153 / 267).

## Addendum 3 (2026-10-15, before any ablation trial; reviewer-requested, exploratory)
Condition `structured_nl` (C'): the structured contract's compact JSON wrapper, fact IDs and category grouping, with each
rule holding the prose sentence (fact.nl) instead of the formal predicate. Contrasts: C' - B (JSON wrapper, IDs and
grouping) and C - C' (formal predicates vs sentences), Holm over these two within model, first-attempt endpoint.
Mean documentation length: B 1466, C 1483, C' 1822 o200k tokens (JSON escaping of full sentences).
Runs: nano and mini on all 267 eval tasks; gpt-4.1 on its 91 complete Study 2 tasks only, so all contrasts are paired.
Budget approved by the owner: $8 total (caps nano $1, mini $2.5, gpt-4.1 $4.5). Same harness, prompts and grading.

## Addendum 4 (2026-10-19, camera-ready, before any repeat trial; reviewer-requested, exploratory)
Repetition study for within-cell noise (reviewer 85B). Conditions prose (B) and structured (C); models nano and mini
on all 267 eval tasks; two new trials per cell (trial indices 1 and 2, which are also the API seeds; Study 2 used 0).
First attempt only (`--repairs 0`): the first attempt is generated by the identical prompt with or without later
repairs, so these trials are comparable with the Study 2 first attempts on the primary endpoint. Same harness, prompts,
grading and snapshots; `run_experiment.py` gained only the `--trial-start` and `--repairs` flags.
Analysis, fixed now: (i) per cell, agreement of the first-attempt label across the three runs (share of unanimous cells,
per condition and model); (ii) C - B separately in each of runs 1 and 2 (cluster-bootstrap CI, exact McNemar), next to
the pre-registered run 0; (iii) C - B on per-task mean correctness over the three runs (cluster-bootstrap CI). The
pre-registered Study 2 results are not replaced. Budget approved by the owner: about $3 total (caps nano $0.70,
mini $2.30, list price); estimated $2.44.

## Addendum 5 (2026-10-19, camera-ready, before any eval trial of these runs; reviewer-requested, exploratory)
Two additions requested by reviewers (85A, 85B, 85C). Budget approved by the owner: $10 total (list price).
(a) Second model family (85B, 85C). `claude-haiku-5-5` (Claude Haiku 5.5, Anthropic API; the served model ID is logged per
call) on all 267 eval tasks, all eight Track A conditions, full protocol (<=2 repairs, 6 tool calls), same prompts, tool,
grading and analysis. Differences forced by the API, fixed now: thinking disabled (GPT-4.1 has no reasoning phase);
sampling parameters at their defaults, because this model rejects temperature 0; no seed parameter. Contrasts: the eight
pre-specified Study 2 contrasts, Holm within model; cap $4.00.
(b) Raw specification (85A). Condition `full_spec`: the documentation is the whole pinned spec (commit 23d2f926, sha256 in
data/versions.lock.json; 91k o200k tokens) instead of curated facts; same task text. nano and mini, all 267 eval tasks,
trial index 0, first attempt only (`--repairs 0`). Contrast: full_spec - prose on the first attempt, with the
pre-registered Study 2 prose run as comparator (cluster bootstrap, exact McNemar); robustness: against the mean of the
three prose first attempts (Addendum 4). Also reported: input tokens and list-price cost per trial (OpenAI cached-input
price applied to cached tokens). Caps nano $1.50, mini $3.50.
Budget stops follow the Study 2 rule (task-blocked order; only complete tasks analysed; number reported).
Dev-split pilot (plumbing only, not analysed): 24 Haiku trials and 4 full_spec nano trials, $0.03.

## Addendum 6 (2026-10-20, camera-ready, before any eval trial of these runs; exploratory)
Purpose: Addendum 5 found that Claude Haiku 5.5 did not reproduce the GPT-4.1 effects, but it differs from GPT-4.1 in
both vendor and release date (18 months). Three models separate the two factors:
(a) `gpt-5.4-mini-2026-03-17` (same vendor as GPT-4.1, newer): no sampling parameters (reasoning model), lowest
    reasoning effort ("none"), the 2048-token cap covers reasoning and visible output.
(b) `claude-haiku-4-5-20251001` (other vendor, six months after GPT-4.1): temperature 0, no thinking.
(c) `qwen/qwen3-coder` via OpenRouter (open weights, July 2025): temperature 0; host pinned to DeepInfra without
    fallback; the serving host is logged per call.
Each on all 267 eval tasks, conditions none, minimal, prose, structured, interactive; trial index 0; first attempt only
(`--repairs 0`, tool budget unchanged). Same prompts, tool, grading. Contrasts per model (Holm over the four):
minimal-none, prose-minimal, structured-prose, interactive-structured. Descriptive, across all six models: documentation
gain (prose - none) against prior strength (none). Caps (list price): (a) $5.50, (b) $6.50, (c) $2.50; Study 2 stop rule.
Dev pilot (plumbing only, not analysed): 20 trials per model, $0.09. Code changes since Addendum 5: provider settings
for these models, OpenRouter host pinning, per-call `served_by` logging, no retries on programming errors.

## Addendum 7 (2026-10-20, camera-ready, before any eval trial of these runs; exploratory)
Budget approved by the owner: $10 total (list price).
(a) Equivalence (no new trials). For C-B and D-C of every model other than GPT-4.1: two one-sided tests at alpha 0.05 with
a +-5 pp margin, i.e. the 90% cluster-bootstrap CI (10k resamples, same clusters and seed as the main analysis) lies
within [-5, +5] pp. The margin is smaller than every GPT-4.1 C-B point estimate (-6.0, -7.9, -12.1); it was chosen after
the Addendum 6 results were seen, which the paper states. Computed before (b); recomputed for Haiku 4.5 after (b).
(b) Complete Addendum 6 for claude-haiku-4-5-20251001: resume the same run (same settings) for the tasks skipped at its
budget stop; cap $2.30.
(c) Full protocol for qwen/qwen3-coder (<=2 repairs, 6 tool calls) on prose, structured, interactive, all 267 tasks,
trial 0, separate file; endpoints first attempt and after repair; contrasts C-B and D-C; cap $3.50.
(d) Mechanism test for the GPT-4.1 JSON penalty. Condition C'' = `structured_rawcode`: contract C with every quoted code
span moved out of the JSON strings into one plain code block keyed by fact ID (no JSON escaping; audited: all spans
preserved, 0 escaped quotes left; mean 1,616 vs 1,469 o200k tokens for C). nano and mini on 267 tasks, 4.1 on its 91
Study 2 tasks; trial 0, first attempt. Contrasts C''-C (escaping) and B-C'' (residual), Holm over the two within model.
Caps nano $0.30, mini $0.70, 4.1 $1.00. Dev pilot (plumbing only): 4 nano trials, $0.00.

## Addendum 8 (2026-10-21, camera-ready, before any trial of these runs; reviewer-requested, exploratory)
Completes the repetition study of Addendum 4 for two more contrasts (Reviewer B suggested 2-3 key contrasts). Conditions
none, minimal and interactive; nano and mini; all 267 eval tasks; trial indices 1 and 2 (API seeds), first attempt only
(`--repairs 0`, tool budget unchanged); structured runs 1-2 exist from Addendum 4. Same harness, prompts and grading.
Analysis, fixed now (as in Addendum 4): A-None and D-C in each of runs 1 and 2 next to the pre-registered run 0
(cluster-bootstrap CI, exact McNemar), the mean over the three runs, and the share of cells with the same label in all
three runs. Budget: the $3.02 left of Addendum 7's approved $10 (caps nano $0.80, mini $2.20).
