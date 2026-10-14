# Pre-registered experiment protocol (DRAFT — freeze before running)

## Population and splits
Use official StableHLO tests/spec as primary source. Candidate tasks require independent oracle, supported operations, and provenance. 30 development tasks, 100 initial frozen tasks, 300–600 aspirational expanded unique validated tasks. Prevent leakage between development and evaluation using upstream IDs and semantic duplicate checks. Stratify by operation family. Exclude unsupported cases only under predeclared rules and log counts.

## Treatment construction
Every operation has a pinned canonical fact inventory with unique IDs and source anchors. Human auditor checks source fidelity and that B/C contain exactly the same facts, examples and restrictions. Automated equality of fact IDs is necessary, **not sufficient**. Count tokens and record length. Introspection uses same inventory; bound retrieval requests and total budget. Test both equal-token and equal-total-budget sensitivity analyses if feasible.

## Primary endpoint
First-attempt semantic correctness (verified AND passed held-out independent oracle). Primary contrast C–B on paired tasks. Freeze prompt templates, model version, temperature, sampling seeds when available, verifier, oracle, timeout and task selection before evaluating. Randomize condition order; fresh session per trial.

## Secondary endpoints
Correct after at most two repairs, repair attempts, input/output tokens, tool calls, latency, parse/verify success, failure class. Report costs conditional on success and unconditionally to avoid survivorship bias. Inspect errors with blinded coding where possible.

## Oracle design
Reference implementations must be independent of the generated StableHLO program. Use seeded random cases, boundary cases, nontrivial shapes, tolerances by dtype/operation and metamorphic properties. Execute generated code only in sandbox with CPU/memory/time limits. Review numerical semantics and unsupported operations before inclusion. Avoid LLM judging.

## Statistical plan
Paired task-cluster bootstrap 10k resamples for difference in correctness; report 95% CI and task count. For exactly one binary observation per task/condition, McNemar exact test optional. With multiple trials, do not use naive McNemar across dependent repeats; cluster by task or use hierarchical models. Family breakdown exploratory; avoid multiple uncorrected significance claims. Record all exclusions and missing outcomes.

## Scale and budget
At 300 validated tasks × 4 treatments × 2 repeated trials = 2,400 first-attempt generations, before repairs and ablations. Plan API budget and inference throughput before execution. At 600 tasks, 4,800 first attempts. No claim these have been executed or that all source tasks can be translated.

## Pilot gates
1 Pin commits and install compiler/interpreter. 2 Verify 10 programs and compare with independent reference. 3 Audit 10 fact inventories. 4 Complete 30-task pilot. 5 Freeze protocol and 100-task evaluation. 6 Scale to 300–600 only if valid and resources permit.
