# Reproducibility and reporting checklist

- [ ] Record exact git SHAs, license snapshots and dependency lockfile
- [ ] Pin compiler/interpreter, model identifier and tokenizer
- [ ] Freeze all prompts, contracts and task split hashes
- [ ] Independently audit source-to-contract fact fidelity
- [ ] Independently audit oracle and numerical tolerances
- [ ] Log excluded tasks with reasons before outcomes
- [ ] Save trial-level records, seeds, timing, tool calls and tokens
- [ ] Use isolated sandbox and resource caps for generated programs
- [ ] Report paired confidence intervals and failure categories
- [ ] Report negative findings, not just improvements
- [ ] Do not claim completed experiments until logs exist

## Expected trial JSONL fields
`task_id`, `source`, `family`, `condition`, `model_version`, `trial_id`, `first_attempt_semantic_correct`, `verified`, `correct_after_repair`, `repair_attempts`, `input_tokens`, `output_tokens`, `tool_calls`, `elapsed_seconds`, `error_type`, `toolchain_commit`, `oracle_version`, `fact_inventory_hash`.
