# Real-world task conversion policy

KernelBench and TritonBench contain useful external programming tasks, but they are not directly StableHLO programs. Convert only tasks with expressible reference semantics. For each task:
1. Record upstream repo, commit, original path/ID, license and source task family.
2. Preserve reference mathematical behavior and shape/type preconditions; identify unsupported stateful or hardware-specific semantics.
3. Author a StableHLO task description and an independently executable reference implementation.
4. Validate source-vs-reference outputs on random and edge-case inputs, then verify interpreter support.
5. Record conversion reviewer, date, test seeds and excluded reasons.
6. Prevent duplicate or near-duplicate tasks from inflating statistical power.
7. Split by upstream task identity, not translated variant.

**Do not assume a Triton GPU performance test is a StableHLO semantic test.** No speedup claims without hardware and profiling. Source data must be downloaded and reviewed in a future execution phase.
