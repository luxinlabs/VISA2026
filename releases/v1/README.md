# v1 snapshot (study 1, run 2026-10-13)

Code as of the end of study 1. Hashes in FREEZE.sha256 were taken *before* the eval runs. Five files differ from
those hashes because of documented post-freeze, additive edits made while/after the frozen runs were in flight
(the running processes had already imported the frozen modules):
- harness.py / interfaces.py / models.py / run_experiment.py: added the post-hoc `interactive_forced` condition
  (tool_choice="required" on the first turn); existing conditions unchanged.
- analysis.py: added the post-hoc diagnostic taxonomy (`diagnostic_class`).
- PREREGISTRATION.md: Addendum A.
The byte-exact pre-edit versions of these files were not preserved (no VCS); this is reported as a limitation.
