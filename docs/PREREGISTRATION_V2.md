# Study 2 (v2) protocol — DRAFT, to be frozen (hashes in results/v2/FREEZE.sha256) before any eval-split trial

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

## Models
Pending owner approval of API spend. Candidates: gpt-4.1-nano/mini/2025-04-14 (continuity with v1).
Temperature 0, one trial per cell, seeded random order, fresh conversation per trial.

## Statistics
Paired differences; 95% cluster bootstrap (template for synthetic, task for KernelBench), exact McNemar on task-level
discordant pairs, Holm correction. Suite-level and family-level analyses are secondary/exploratory.
