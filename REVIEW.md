# Strict code & design review of the original AgentISA-Bench scaffold (2026-10-13)

> Historical record. This review covers the original scaffold (now in `legacy/`), not the current code; every issue marked [B] or [M] was fixed on 2026-10-13 (see the last section).

Scope: every file in the original archive (≈100 lines of Python, configs, docs). The two unit tests pass
(`PYTHONPATH=src pytest` → 2 passed), but they only test a tautology (see T1). Severity: **[B]** blocks a valid
experiment / invalidates the primary claim, **[M]** produces wrong or misleading numbers, **[m]** minor.

## 1. The experimental manipulation itself (interfaces.py)

- **[B] I1. "prose" is not prose.** `render(..., "prose")` (`src/agentisa/interfaces.py:19`) emits
  `fact.id: statement` lines, which is a machine-oriented key/value list. `structured` (`:21`) is the *same
  statements* wrapped in JSON. The primary contrast C−B (pre-registered RQ1) therefore measures
  "newline list vs JSON quoting", not "human prose vs structured contract". The fact IDs in the prose condition
  are also a leak of structure into the human condition.
- **[B] I2. B and C are not information-equivalent.** `structured` includes `record["operation"]`; `prose` does
  not. The category field is dropped in both, but anything structural (which fact constrains which operand,
  which attribute) exists in neither. The claimed equivalence holds only at the level of fact IDs.
- **[B] I3. The interactive condition (D) does not exist.** `render(..., "interactive")` returns a
  sentence naming a tool `get_semantic_facts(operation, ids)` plus a `tool_catalog` dict, but (a) no tool
  is implemented anywhere, (b) the agent is never told which IDs/categories exist, so it cannot form a query,
  and (c) if `tool_catalog` were shown to the model, it would leak every fact upfront, collapsing D into C.
  `fact_ids` for D reports all facts as delivered, while the protocol says to report *retrieved* facts.
- **[M] I4. "minimal" = facts with category `syntax`.** Minimal is defined by a category filter of the same
  inventory, so how "minimal" it is depends on how authors happen to label facts. There is no syntax-only
  negative control renderer even though `configs/experiment.yaml` declares `syntax_only_structured`, and the
  declared ablations (`remove_shape_facts`, `remove_type_facts`) are not implemented.
- **[M] I5. `audit_equivalence` is tautological** (`:26-29`). It only compares fact-ID sets. Both renderers
  iterate the same list, so the check can never fail; it does not check that statements are identical,
  that nothing else (operation name, ordering, examples) differs, or that token lengths are comparable.
- **[M] I6. `validate_facts` accepts placeholder provenance** (`:9`). `"REVIEW_AGAINST_PINNED_SPEC"` and
  `spec_revision: "ILLUSTRATIVE_NOT_PINNED"` pass, so the provenance requirement is not enforced. The category
  enum in `contracts/fact_schema.json` is never validated (jsonschema is a dev dependency but unused).
- **[M] I7. The only contract is semantically incomplete.** `contracts/example_broadcast.json` omits the
  constraints that actually make `broadcast_in_dim` hard: `size(broadcast_dimensions) = rank(operand)`,
  dims unique and `< rank(result)`, `dim(operand,i) ∈ {1, dim(result, broadcast_dimensions[i])}`, and
  element type equality. A treatment built from it cannot test RQ3 (shape/broadcast failures).

## 2. Harness (harness.py)

- **[B] H1. There is no trial runner.** Nothing implements generation → evaluation → repair loop, condition
  randomization, fresh sessions, budgets, or writes the JSONL fields listed in `docs/REPRODUCIBILITY.md`.
  `configs/experiment.yaml` is never read by any code.
- **[M] H2. Parse vs verify are conflated** (`:27-29`). `parse_and_verify` returns one bool and `evaluate`
  sets `parsed=False, verified=False` together, so the protocol's required distinction (parse/verify/semantic
  failure classes) cannot be measured.
- **[M] H3. Runtime failures crash the run.** `evaluator.execute` exceptions (compile errors, wrong arity,
  dtype mismatch, timeouts) propagate out of `evaluate`; one bad generation aborts the whole experiment
  instead of being logged as a failure. `execution_timeout_seconds` is declared but never enforced, and there is no sandbox.
- **[m] H4. `oracle_cases=len(cases)` is reported even when evaluation stopped at the first failing case.**
- **[M] H5. `ModelAgent.repair(program, diagnostic, budget)` drops the task and the interface.** A repair turn
  would no longer see the documentation, so "correct after repair" would not be conditioned on the
  treatment, and D can't use its tool during repair. No token or tool-call accounting is returned.
- **[m] H6. The oracle compares only output values.** It doesn't check output shape/dtype, or that the program's
  signature matches the task's declared signature.

## 3. Analysis (analysis.py, analyze_results.py)

- **[M] A1. Silent exclusion.** Tasks missing either condition are dropped from `pairs` (`analysis.py:10`)
  without being counted or reported, which violates the protocol rule "record all exclusions and missing outcomes".
- **[M] A2. Mislabelled output.** The result key is hard-coded `effect_structured_minus_prose` even when
  `a`/`b` are other conditions.
- **[M] A3. Pools across models/repeats.** Rows are never filtered by `model_version`; mixed-model logs are
  silently averaged together.
- **[M] A4. McNemar (promised in protocol) is not implemented**; neither are secondary endpoints, repair
  endpoint, stratification by family, or failure-type tables.
- **[m] A5.** `int(r["first_attempt_semantic_correct"])` crashes on `"true"`/`"false"` strings; percentile
  indices use `int(.975*n)` rather than a standard quantile definition (off-by-one, negligible at 10k).
  `seed`/`n_boot` duplicate the config instead of reading it.

## 4. Data pipeline (prepare_manifest.py, task_schema.json)

- **[M] D1. `prepare_manifest.py` doesn't validate against `data/task_schema.json`**, and it copies
  `excluded` and unvalidated `candidate` records into the evaluation manifest. Excluded tasks would be
  evaluated. Missing keys raise bare `KeyError`.
- **[B] D2. There are no tasks, references, or oracles at all**, and no extractor from StableHLO tests. Also,
  StableHLO's public `.mlir` tests are very likely in pretraining corpora, so tasks copied from them
  have a contamination problem the threats section mentions but doesn't mitigate.

## 5. Tests

- **[M] T1.** `test_fact_ids_equal` asserts `set(x) == set(x)` through two renderers of the same list; it
  can't fail. `test_minimal_subset` only checks a count. Harness and analysis are untested; tests rely on
  `PYTHONPATH=src` (no conftest or editable install).

## 6. Design and statistics

- **[M] S1. Power.** With 100 paired binary tasks and plausible discordance (~15–25%), the minimum detectable
  paired difference at 80% power is roughly 10–14 points. The proposal should say this, since a null C−B
  result at n=100 is weakly informative.
- **[m] S2.** `temperature: 0` with "2 repeated trials" yields near-duplicate repeats (deterministic local
  models). Repeats should use a fixed nonzero temperature and seeds, or be dropped.
- **[M] S3. D confounds access mode with interaction turns.** Retrieval adds turns *and* token budget; the
  protocol mentions equal-budget sensitivity but the config has a single `max_total_tokens` with no rule
  for counting retrieved text.
- **[M] S4. Leakage of the solution through operation choice.** Providing facts only for the ops the
  reference uses tells the model which ops to use. The design should either name required ops in the task
  for all conditions (making the variable "how to use an op correctly") or provide a fixed, task-independent
  mini-spec.
- **[m] S5.** Related-work URLs under PMLR v306 (CIRBench, KernelCraft, TritonGym) are flagged "verify" in
  the repo; they must not be cited until verified.

## Disposition

Everything marked [B]/[M] above is fixed in the implementation added on 2026-10-13 (see `README.md` and
`docs/IMPLEMENTATION_NOTES.md`). The original scaffold files are kept for reference.
