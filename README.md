# VeriEdit

`veriedit` is a non-generative image editing orchestrator that combines classical image processing, a multi-agent review loop, and auditable reporting.

## Highlights

- Explicit image editing tools only
- LangGraph workflow with policy, diagnostics, planning, execution, review, and retry
- Optional Gemini-assisted planning and review with deterministic fallbacks
- CLI and Python API
- JSON and Markdown reports plus intermediate artifacts

## Quick start

```bash
pip install -e .
veriedit edit \
  --input input.jpg \
  --prompt "Clean dust, reduce yellow cast, and lightly sharpen." \
  --output output/result.png
```

## Python API

```python
from veriedit import edit_image

result = edit_image(
    source_image="input.jpg",
    prompt="Restore this scan naturally and keep it realistic.",
    reference_image="reference.jpg",
    output_path="output/result.png",
)
```
