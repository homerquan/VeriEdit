# VeriEdit

`veriedit` is a non-generative image editing orchestrator that combines classical image processing, a multi-agent review loop, and auditable reporting.

By default, each run is stored in its own folder under `/tmp/veriedit/<run_id>`.

## Highlights

- Explicit image editing tools only
- LangGraph workflow with policy, diagnostics, planning, execution, review, and retry
- Optional Gemini-assisted planning and review with deterministic fallbacks
- Interactive REPL shell and scriptable CLI under one `veriedit` entrypoint
- Manual paint/brush tool for localized touch-up with soft, round, and square pens
- Photoshop-style local repair tools including spot healing, healing brush, clone-style source painting, and masked curves adjustment
- JSON and Markdown reports plus intermediate artifacts

## Quick start

```bash
pip install -e .
veriedit
```

That opens the interactive shell. For direct one-shot commands, you can still run:

```bash
veriedit edit \
  --input input.jpg \
  --prompt "Clean dust, reduce yellow cast, and lightly sharpen." \
  --output result.png
```

That creates a short run id such as `/tmp/veriedit/ab12cd34/` and stores the final image, reports, step snapshots, and logs together in that folder. If you pass `--output`, VeriEdit uses that as the result filename inside the run folder.

For manual brush-based touch-up:

```bash
veriedit paint \
  --input input.jpg \
  --output output/painted.png \
  --pen soft \
  --size 10 \
  --opacity 0.55 \
  --stroke "120,84 128,88 136,90"
```

Healing-style repair example:

```bash
veriedit paint \
  --tool heal \
  --input input.jpg \
  --output output/healed.png \
  --source-point 120,84 \
  --stroke "220,90" \
  --size 12 \
  --blend-mode replace \
  --feather 3
```

## Python API

```python
from veriedit import edit_image

result = edit_image(
    source_image="input.jpg",
    prompt="Restore this scan naturally and keep it realistic.",
    reference_image="reference.jpg",
    output_path="result.png",
)
```
