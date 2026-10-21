# Data license

Except as noted below, the data in this repository is licensed under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/ (legal code:
https://creativecommons.org/licenses/by/4.0/legalcode).

Copyright (c) 2026 Luxin Zhang. When you reuse it, cite the paper (see `CITATION.cff`).

**Covered:** the fact inventory (`src/agentisa/fact_inventory.py` and `contracts/`), the synthetic
task suite and manifests in `data/processed/`, the treatment audits, the validation and mutation-test
outputs, the protocols in `docs/` and `releases/`, and the trial logs and transcripts in `results/`.

**Not covered (third-party terms apply):**
- KernelBench-derived content, i.e. `data/processed/kernelbench/` and the PyTorch module sources embedded in
  `data/processed/tasks_manifest_kernelbench.jsonl`, `data/processed/tasks_manifest_all.jsonl` and in
  the KernelBench trial transcripts. It remains under KernelBench's MIT License; see
  `THIRD_PARTY_NOTICES.md`.
- Model outputs in the transcripts are reproduced for research purposes. Their use may also be
  subject to the respective model provider's terms.

Source code is licensed separately under the MIT License (`LICENSE`).
