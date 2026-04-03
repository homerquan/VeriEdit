# VeriEdit Change Log

This file tracks incremental architecture and implementation work, especially the gap between Photoshop-grade manual restoration and the current autonomous workflow.

## 2026-04-02

### Completed
- Replaced the previous LangGraph runtime layer with an AG2-based multi-agent runtime under [src/veriedit/runtime](/Users/homer/Projects/VeriEdit/src/veriedit/runtime).
- Added AG2 chat-history artifacts to each run for easier inspection of runtime-level agent communication.
- Introduced structured inter-agent handoffs so policy, diagnostics, planning, execution, review, human approval, and retry now pass explicit summaries and payloads instead of relying only on hidden shared-state mutation.
- Added a tool selector layer with ranked recommendations, parameter hints, and scope-aware reasoning so planning is more grounded in tool capabilities and less dependent on one large heuristic branch chain.
- Marked key tools with explicit capability tags and edit scopes to support better autonomous tool selection.
- Restricted autonomous planning away from manual-only tools such as clone/healing brush variants while keeping them available in direct paint workflows.
- Added `--allow-tool` to `veriedit edit`, allowing one or more tools to be explicitly whitelisted for debugging or tool-capacity evaluation.
- Updated reports and manual-eval output to include structured handoffs and better reflect AG2-era runtime behavior.
- Cleaned up the main user docs so runtime, CLI, and workflow examples match the current implementation.
- Reworked the planner into a diagnostics-first staged planner that records `detected_problems`, `repair_strategy`, and `feedback_applied`, so agent communication now carries explicit restoration intent and retry feedback.
- Improved retry-to-planner guidance so retries can explicitly steer the next plan toward local repair, narrower edit footprints, and less aggressive sharpen/denoise choices.
- Changed the workflow loop to re-run diagnostics on the current working image every iteration instead of planning repeatedly from the original input only.
- Added iterative planning history and review history to reports so multi-round behavior is visible and debuggable.
- Switched planning to smaller per-iteration batches, with stage selection that can move from local repair to tone/reference/detail work across rounds.
- Added a simple tool-rotation rule so the planner is less likely to repeat the same batch after weak progress.
- Added a true manual `clone_stamp` tool with aligned stroke-path cloning, separate from the older transformed patch clone tool, and exposed it through `veriedit paint --tool clone`.
- Added clone-tool UX improvements: `--spacing` for clone stamp, automatic source-point suggestion for clone tools, and preview overlay images showing the source/target layout before review.

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
- Added workflow observability artifacts with node and tool traces for each run.
- Added a standalone per-edit summary markdown artifact for each photo editing run.
- Added `manual_eval` generation as both a reusable Python module, a standalone script, and a CLI command to create one markdown file with source/reference/result images plus tool and trace details.
- Added mask-aware layered execution for local repair tools so dust, scratch, and small-heal steps are composited back only inside detected defect regions with feathered masks.
- Added executor safeguards that penalize and roll back local repair candidates when they alter preserved regions too broadly.
- Added a Human Approval Agent and runtime gate that pauses ambiguous runs with a self-contained manual-eval markdown and a `human_approval.json` decision artifact.
- Added a `manual-approve` CLI command to record human approval or rejection for a run.
- Reworked the CLI into a `veriedit` REPL shell with an ASCII banner, guided edit/paint flows, and spinner-based progress feedback while keeping subcommands scriptable.
- Added a non-generative `paint_strokes` tool plus a `veriedit paint` command for manual brush-based touch-up with soft, round, and square pens.
- Added Photoshop-inspired retouch tools: `spot_healing_brush`, `healing_brush`, `clone_source_paint`, and `masked_curves_adjustment`.
- Extended `veriedit paint` so it can drive healing and clone workflows from explicit source and target coordinates.
- Changed run storage to short alphanumeric run ids and self-contained run folders, with everything now stored under `/tmp/veriedit/<run_id>` by default.
- Changed final output handling so the result image always lives inside the run folder; `output_path` now controls the filename within that folder.

### Not Completed Yet
- Layer-based execution stack.
Why not: current workflow still commits flattened image states step by step. A true layer/mask stack needs broader schema and report changes.

- Patch-level reviewer with per-region scoring.
Why not: reviewer is still mostly image-level. Region-level review needs tighter coupling to diagnostics masks and execution masks.

- Human approval agent.
Why not: basic approval gating now exists, but resume-after-approval orchestration is not yet implemented.

- Semantic preservation models.
Why not: would likely require optional model assets or external inference dependencies beyond the current classical MVP.
