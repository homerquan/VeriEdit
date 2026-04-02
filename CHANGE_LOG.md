# VeriEdit Change Log

This file tracks incremental architecture and implementation work, especially the gap between Photoshop-grade manual restoration and the current autonomous workflow.

## 2026-04-01

### Completed
- Added a deeper roadmap to [SPECS.md](/Users/homer/Projects/VeriEdit/SPECS.md) describing why Photoshop still outperforms the agent workflow and which features should close that gap incrementally.
- Added reproducible local restoration fixture generation in [scripts/generate_test_pairs.py](/Users/homer/Projects/VeriEdit/scripts/generate_test_pairs.py) and expanded [tests/data](/Users/homer/Projects/VeriEdit/tests/data).
- Added stronger classical restoration capacity with OpenCV, scikit-image, and SciPy-oriented tools including `shadow_highlight_balance`, `bilateral_denoise`, and `small_defect_heal`.
- Added richer diagnostics signals such as scratch/fade/edge-damage estimates.
- Improved closed-loop behavior so planner/retry avoid repeatedly using tools that only roll back.
- Added manual review pack generation in [scripts/generate_manual_review_samples.py](/Users/homer/Projects/VeriEdit/scripts/generate_manual_review_samples.py).

### Completed In This Pass
- Added region-aware diagnostics artifacts and persisted defect maps for manual inspection and future region-scoped editing.
- Added lightweight branch-and-compare execution so some tools can try safer variants before committing.
- Added localized planning hints based on detected damage regions.
- Verified the full suite after these changes: `11 passed`.
- Added mask-aware patch review metrics so the reviewer can distinguish edits inside defect regions from unintended changes outside them.
- Added LangGraph-style observability artifacts with node and tool traces for each run.
- Added a standalone per-edit summary markdown artifact for each photo editing run.
- Added `manual_eval` generation as both a reusable Python module, a standalone script, and a CLI command to create one markdown file with source/reference/result images plus tool and trace details.

### Not Completed Yet
- Layer-based execution stack.
Why not: current workflow still commits flattened image states step by step. A true layer/mask stack needs broader schema and report changes.

- Patch-level reviewer with per-region scoring.
Why not: reviewer is still mostly image-level. Region-level review needs tighter coupling to diagnostics masks and execution masks.

- Human approval agent.
Why not: useful, but outside the current autonomous CLI-focused implementation.

- Semantic preservation models.
Why not: would likely require optional model assets or external inference dependencies beyond the current classical MVP.
