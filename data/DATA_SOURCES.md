# Public source registry and licensing due diligence

| Source | Official URL | Role | Required adaptation |
|---|---|---|---|
| OpenXLA StableHLO | https://github.com/openxla/stablehlo | Primary semantic specification, verifier/interpreter tests | Pin commit; derive task prompts, references and held-out inputs |
| KernelBench | https://github.com/ScalingIntelligence/KernelBench | Realistic PyTorch operator and model tasks | Filter representable operators; convert to StableHLO; independently validate |
| TritonBench | https://github.com/thunlp/TritonBench | Real-world Triton operator cases | Filter/translate supported semantics; validate references and shape constraints |
| LLVM MLIR | https://github.com/llvm/llvm-project/tree/main/mlir/test | Optional IR diagnostic tests | Do not conflate parser tests with executable semantic tasks |

Dataset size varies by version, subsets and definitions; compute counts from pinned manifests, not web descriptions. Do not assume every upstream task is eligible. Preserve upstream copyright, license, provenance and attribution. Verify redistribution rights for tasks, prompts and outputs before publication. Do not ingest arbitrary untrusted code without sandboxing.

## Acquisition (instructions only; NOT executed)
```bash
git clone https://github.com/openxla/stablehlo.git data/vendor/stablehlo
git clone https://github.com/ScalingIntelligence/KernelBench.git data/vendor/kernelbench
git clone https://github.com/thunlp/TritonBench.git data/vendor/tritonbench
# Record exact commit hashes and licenses in data/versions.lock.json.
```

For StableHLO task extraction, implement a human-reviewed candidate export into `data/raw/stablehlo_candidates.jsonl`; do not blindly treat every `.mlir` test as a benchmark task. KernelBench/TritonBench require explicit conversion and oracle review.
