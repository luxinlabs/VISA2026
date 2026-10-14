# AgentISA-Bench: What Makes a Virtual ISA Programmable by AI Agents?

## Abstract
AI coding agents increasingly generate and repair accelerator programs, yet benchmarks typically score model output rather than the design of the compiler interface agents consume. We propose AgentISA-Bench, a controlled experimental framework that varies semantic information availability, representation and access mechanism while holding a virtual IR, model, task and budget fixed. A StableHLO study compares minimal documentation, human-readable semantic specifications, machine-readable contracts containing identical canonical facts, and on-demand semantic introspection. We specify independent correctness oracles, paired evaluation, negative controls, confidence intervals and failure analyses. We also propose external task transfer from public KernelBench and TritonBench collections, conditional on validated lowering to StableHLO. The study seeks to identify measurable design principles for agent-facing virtual ISAs. **No empirical findings are claimed in this proposal.**

## Problem and novelty
KernelBench, TritonBench and related agentic kernel benchmarks measure program generation and performance. CIRBench and KernelCraft are important related baselines for IR manipulation and unfamiliar accelerator targets, respectively; exact bibliographic details and scope should be independently verified before submission. Our variable is the **agent-facing semantic interface to one fixed representation**, not the model or accelerator. The strongest novelty is a fact-equivalent, controlled comparison of semantic representation and on-demand access, rather than another kernel-generation leaderboard.

## Research questions
RQ1: Does structured representation change first-attempt semantic correctness relative to information-equivalent prose? RQ2: Does on-demand semantic introspection alter repair success and cost versus upfront delivery of the same facts? RQ3: Are differences concentrated in shape, type, broadcasting, reduction and indexing constraints? RQ4: Do effects persist on translated real-world operator tasks?

## Contributions (proposed, not demonstrated)
1. Operational agent-programmability metrics conditioned on model, task, interface and budget.
2. Canonical, versioned semantic fact inventory with paired prose/structured renderers and equivalence audit.
3. Tool-based semantic introspection interface, bounded by an equalized retrieval budget.
4. Paired benchmark protocol with independent execution oracles and external validity analysis.

## Experimental materials
Primary: pinned OpenXLA StableHLO specification, operation tests, interpreter and verifier. External: ScalingIntelligence/KernelBench and thunlp/TritonBench. Large-scale target: 300–600 **validated unique tasks**, contingent on conversion feasibility; target is not an assertion of available native StableHLO tasks. A 30-task development pilot and 100-task frozen initial evaluation are planned, followed by scaling. Include real-world kernels only where the reference semantics are representable and the lowering is validated.

## Design
Four treatments: A minimal syntax; B complete prose; C structured canonical facts; D queryable canonical facts. B and C must have exactly matching fact IDs and no hidden examples. D has the same total available facts; report actually retrieved facts and token/tool budget. Fix model version, system instruction, temperature, retry limits, tool access, task wording and verifier version. Randomize condition order and isolate sessions. Include structured syntax-only negative control and shape/type ablations.

## Ground truth
A task contains reference implementation, input generator, shape/type constraints, tolerance policy and held-out inputs. Generated code must parse, verify, execute within limits and match independent reference outputs. Supplement with metamorphic tests; no finite test suite proves equivalence. Exclude unsupported operations before outcome inspection. Distinguish verifier pass from semantic pass. Never use LLM grading as primary correctness.

## Metrics and analysis
Primary: paired difference in first-attempt semantic correctness, C minus B. Secondary: correctness after fixed repair budget, repair attempts, tool calls, input/output tokens, elapsed time, compilation/verification error types and semantic failure types. Bootstrap over task IDs, keeping repeated trials clustered. Report paired discordance and McNemar test for one-trial binary comparison. Stratify by operation family and source, but label exploratory comparisons. Pre-register primary comparison, stopping rule and exclusion criteria. Do not treat translated versions of the same task as independent.

## Threats
Documentation length, tokenizer artifacts, model training exposure, translation errors, reference oracle errors, insufficient power, task selection, unsupported StableHLO features, and limited generalization from a high-level IR to accelerator ISAs. Mitigate with fact audit, blinded task selection, provenance tracking, version pinning, held-out tests, sensitivity analyses, and transparent limitations.

## VISA paper outline (4–6 pages, IEEE)
1 Introduction & question (0.6p); 2 Related work and gap (0.5p); 3 interface treatments and task construction (1.1p); 4 oracle and statistical protocol (1.0p); 5 results **only after experiments** (1.2p); 6 limitations/conclusion (0.5p); references (remaining). Do not fabricate tables, graphs, results or acceptance claims.

## Decision gates
G0: pin toolchain; G1: run 10 tasks parse/verify/execute; G2: independently audit fact equivalence; G3: freeze 100-task evaluation; G4: launch larger task suite only if valid. Until these pass, paper is a methodology/position proposal, not a completed benchmark.
