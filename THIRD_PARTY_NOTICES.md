# Third-party notices

## KernelBench (MIT License)
The KernelBench-derived tasks (`data/processed/kernelbench/`, `data/processed/tasks_manifest_kernelbench.jsonl`,
`data/processed/tasks_manifest_all.jsonl` and the related trial transcripts) embed PyTorch module
sources from KernelBench, https://github.com/ScalingIntelligence/KernelBench. They were shrunk and
lowered to StableHLO by `scripts/convert_kernelbench.py`. The KernelBench license:

```
MIT License

Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini
Scaling Intelligence Lab, Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## StableHLO specification (Apache License 2.0)
The fact inventory is anchored to the StableHLO specification, https://github.com/openxla/stablehlo
(commit 23d2f926; see `data/versions.lock.json`). The specification is not redistributed here. The
`full_spec` condition reads it from a local clone in `data/vendor/stablehlo` (see `data/DATA_SOURCES.md`).
