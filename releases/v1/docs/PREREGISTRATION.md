# Frozen protocol (2026-10-13, before any evaluation-split trial)

Development-split pilots (results/pilot/) were used only to debug the harness. The prompts, fact inventory,
renderers, task suite, oracle, budgets and analysis plan below were frozen before the eval split was run.
A SHA-256 of the frozen files is recorded in `results/FREEZE.sha256`.

* Tasks: 38 templates x 3 eval instances = 114 tasks (dev instance 0 never evaluated in the main study).
  All 152 reference solutions pass parse/verify/execute + oracle (gate G1, data/processed/g1_validation.json).
* Conditions: none, minimal, syntax_only_structured, prose, structured, interactive (6). One trial per
  task x condition at temperature 0 and seed 0; seeded random trial order; fresh conversation per trial.
* Models: gpt-4.1-nano-2025-04-14, gpt-4.1-mini-2025-04-14, gpt-4.1-2025-04-14 (OpenAI API, dated snapshots).
  The local qwen2.5-7b-instruct model was piloted on the dev split and dropped before the eval split
  because it hit the floor (0/12 first-attempt and after-repair successes on two of the easiest
  templates; it never invoked the tool). This exclusion is reported.
* Budgets: up to 2 repairs with verifier/oracle feedback; up to 6 tool calls (interactive); 12 model calls;
  2048 output tokens per call; 60 s evaluation timeout.
* Primary endpoint: first-attempt semantic correctness (verified AND matches the independent numpy oracle
  on all 3 held-out cases). Primary contrast: structured - prose, per model.
* Secondary: correctness after repair; verification rate; tokens; tool calls; first-attempt failure class.
* Statistics: paired differences; 95% CIs by template-cluster bootstrap (10k; primary) and task bootstrap;
  exact McNemar on task-level discordant pairs; Holm correction over the 5 pre-specified contrasts
  (structured-prose, interactive-structured, prose-minimal, minimal-none, syntax_only_structured-minimal)
  within each model and outcome. Family-level breakdowns are exploratory.
* Exclusions: none planned. API failures after retries are logged and reported as missing; trials are
  not re-run selectively based on outcome.

## Addendum A (2026-10-13, during the eval run; post-hoc, exploratory)
Partway through the eval run, gpt-4.1 had made zero tool calls in 56/56 `interactive` trials, so the frozen
interactive-vs-structured contrast measures tool *uptake*, not the access mechanism. We added one
exploratory condition, `interactive_forced`, identical to `interactive` except that the first model turn
uses `tool_choice="required"`. Frozen conditions, prompts, budgets and analyses are unchanged. Results for
`interactive_forced` are reported separately as post-hoc and are not part of the Holm family.
