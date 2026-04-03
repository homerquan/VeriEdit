# VeriEdit

## 1. Project Summary

Build a Python pip package for **tool-based image editing orchestration**.

The system takes:
- **one required source image**
- **an optional reference image** describing the desired texture, feel, tone, or restoration target
- **a natural-language prompt**

The system produces:
- an **edited image**
- a **structured edit report** explaining what tools were used, why they were used, and what checks passed or failed

### Core principle
This project must **not use image generative AI** for pixel synthesis or patch generation.

Instead, it must use:
- traditional / deterministic / procedural image editing operations
- classical vision techniques
- restoration and enhancement tools
- optional vision-language review models for planning and closed-loop evaluation

The LLM is used for:
- planning
- tool selection
- critique / review
- retry strategy
- structured reasoning over results

The LLM is **not** used to generate replacement image content.

---

## 2. Goals

### Primary goals
- Build a reusable Python package installable via `pip`
- Accept **1 or 2 images + prompt**
- Use **tool-driven editing only**
- Support **closed-loop review and retry**
- Support an **AG2 multi-agent runtime workflow**
- Default to **Gemini Flash** for orchestration / review
- Make outputs auditable and reproducible

### Non-goals
- No text-to-image generation
- No inpainting / outpainting with diffusion or generative models
- No “hallucinated” patching of missing regions
- No hidden pixel synthesis marketed as restoration

---

## 3. Example Use Cases

### Supported
- Remove dust / speckles from scanned photos
- Correct white balance / exposure / contrast
- Reduce noise in old photos
- Improve sharpness conservatively
- Match the tonal feel of a reference image without copying content
- Restore faded photos using non-generative enhancement
- Straighten, crop, de-skew, de-yellow, and clean scanned images
- Improve local defects when achievable with procedural tools
- Retouch small blemishes using non-generative clone / heal style operators available in traditional SDKs

### Not supported
- Invent missing faces, hands, objects, or scenery
- Fill large missing regions using generative models
- Change identity or fabricate evidence
- “Make it look untouched” in a deceptive / forensic-evasion sense

---

## 4. High-Level Architecture

```text
CLI / Python API
      |
      v
Input Loader + Validation
      |
      v
AG2 Runtime Orchestrator
  ├─ Planner Agent
  ├─ Tool Executor Agent
  ├─ Reviewer Agent
  └─ Retry / Policy Agent
      |
      v
Image Editing Tool Layer
  ├─ OpenCV
  ├─ Pillow
  ├─ scikit-image
  └─ optional domain-specific restoration helpers
      |
      v
Artifacts
  ├─ edited image
  ├─ diff previews
  ├─ execution trace
  └─ JSON report
```

---

## 5. Recommended Tech Stack

### Package + runtime
- Python 3.11+
- `pyproject.toml`-based package
- `uv` or `pip` install support
- `pydantic` for schemas
- `typer` for CLI
- `rich` for logs / terminal UX

### Orchestration
- `ag2`
- direct Python agent/tool orchestration around a shared workflow state

### Image editing SDKs
Use the following stack:
- **OpenCV** (`opencv-python-headless`) — primary editing / filtering / transforms
- **Pillow** — file IO, metadata-safe conversions, compositing basics
- **scikit-image** — restoration, denoise, histogram, metrics
- **numpy** — array ops

### Why this stack
This is the best practical fit for a **non-generative, programmable, auditable** editing pipeline:
- mature
- fast
- scriptable
- strong classical vision coverage
- easy to package for CLI and backend use

Optional later additions:
- `rawpy` for RAW photo workflows
- `piexif` for EXIF preservation control
- `onnxruntime` for optional non-generative enhancement models

### LLM / VLM
- **Default orchestration model:** configurable Gemini Flash model
- Default config target: `gemini-3-flash`
- Production fallback: `gemini-2.5-flash` if the requested model name is unavailable in the deployed environment

The Gemini API supports multimodal input, function calling, structured outputs, and Files API workflows, which fit planning/review well. Current Google docs show `gemini-2.5-flash` as a stable Flash model and the Files API for media upload/handling. citeturn101608search2turn101608search3turn101608search0

---

## 6. Package Name Suggestion

Suggested pip package name:
- `veriedit`

Other options:
- `veriedit`
- `groundededit`
- `faithfulfix`

Repo structure:

```text
veriedit/
  pyproject.toml
  README.md
  src/veriedit/
    __init__.py
    cli.py
    config.py
    schemas.py
    workflow.py
    runtime/
      ag2_runtime.py
    policy.py
    io/
      loader.py
      writer.py
    tools/
      base.py
      exposure.py
      denoise.py
      sharpen.py
      color.py
      geometry.py
      retouch.py
      texture.py
      compare.py
    agents/
      planner.py
      executor.py
      reviewer.py
      retry.py
    metrics/
      iq_metrics.py
      similarity.py
    reports/
      report_builder.py
  tests/
  examples/
```

---

## 7. Inputs and Outputs

### Inputs
#### Python API
```python
result = edit_image(
    source_image="old_photo.jpg",
    prompt="Clean dust, reduce yellow cast, and lightly improve sharpness.",
    reference_image="desired_mood.jpg",  # optional
    allowed_tools=["stroke_paint"],      # optional debug restriction
)
```

#### CLI
```bash
veriedit edit \
  --input old_photo.jpg \
  --prompt "Clean dust, reduce yellow cast, and lightly improve sharpness." \
  --reference desired_mood.jpg \
  --output-folder /tmp/veriedit
```

### Output artifacts
- edited image file
- optional side-by-side preview
- optional diff / mask previews
- JSON execution report
- Markdown summary report
- AG2 chat history and runtime trace artifacts

### Debugging control
The main `veriedit edit` entrypoint should allow an optional tool whitelist:

```bash
veriedit edit \
  --input IMG \
  --prompt TEXT \
  --allow-tool stroke_paint \
  --allow-tool dust_cleanup
```

Behavior:
- if omitted, all workflow tools are allowed
- if provided, planner and executor must stay within the allowed set
- intended primarily for debugging, benchmarking, and single-tool evaluation

### Proposed result object
```json
{
  "success": true,
  "output_image": "result.png",
  "report_json": "result.report.json",
  "report_md": "result.report.md",
  "iterations": 3,
  "applied_tools": [
    "estimate_white_balance",
    "remove_small_dust",
    "wavelet_denoise",
    "unsharp_mask"
  ],
  "review_summary": "Result improved tonal balance and reduced visible dust without over-smoothing."
}
```

---

## 8. Editing Philosophy

Each edit should be framed as a sequence of explicit, inspectable operations.

Examples:
- “normalize exposure” → histogram / gamma / CLAHE operations
- “remove small dust specks” → thresholding + morphology + local repair using non-generative clone/heal rules
- “reduce color cast” → white balance transform
- “match the feel of the reference image” → transfer only allowed low-level characteristics:
  - histogram tendencies
  - contrast profile
  - saturation range
  - warmth / coolness
  - texture softness / crispness

The reference image must **not** be used to copy semantic content.

---

## 9. Tool Catalog

### Geometry tools
- crop
- resize
- rotate
- perspective correction
- de-skew

### Exposure / tone tools
- brightness
- contrast
- gamma correction
- CLAHE
- levels / curves approximation
- shadow / highlight compression

### Color tools
- white balance estimation
- channel scaling
- color cast correction
- saturation tuning
- histogram matching (bounded)

### Cleanup / restoration tools
- median filter
- bilateral filter
- non-local means denoise
- wavelet denoise
- morphology open/close
- threshold-based dust detection
- scratch candidate detection
- local retouch by clone / patch from nearby existing pixels only

### Detail tools
- unsharp mask
- edge-aware sharpening
- texture-preserving sharpen

### Analysis tools
- blur score
- noise estimate
- exposure histogram stats
- color distribution stats
- similarity to reference style descriptors
- change-mask estimation

---

## 10. Multi-Agent Workflow (AG2)

### Agent 1: Planner
Inputs:
- source image
- optional reference image
- prompt
- image diagnostics

Responsibilities:
- interpret the user request
- classify allowed vs disallowed edits
- produce an ordered edit plan
- choose bounded parameters
- define acceptance criteria

Output example:
```json
{
  "intent": "photo restoration",
  "constraints": [
    "non-generative only",
    "preserve original identity",
    "avoid over-sharpening"
  ],
  "plan": [
    {"tool": "white_balance", "params": {"method": "gray_world"}},
    {"tool": "dust_cleanup", "params": {"max_area": 18}},
    {"tool": "nlm_denoise", "params": {"strength": 6}},
    {"tool": "unsharp_mask", "params": {"radius": 1.2, "amount": 0.6}}
  ],
  "acceptance": [
    "visible dust reduced",
    "yellow cast reduced",
    "skin or key structures remain natural"
  ]
}
```

### Agent 2: Executor
Responsibilities:
- run tools in order
- save intermediate outputs
- collect metrics after each step
- support rollback when a step degrades quality

### Agent 3: Reviewer
Responsibilities:
- inspect current result against source, reference, and prompt
- use VLM reasoning for qualitative review
- combine with numerical metrics
- detect artifacts such as halos, oversmoothing, clipped highlights, crushed shadows

### Agent 4: Retry / Policy Agent
Responsibilities:
- decide whether to accept, retry, reduce intensity, or stop
- enforce policy boundaries
- limit loop count

### Closed-loop behavior
```text
plan -> execute -> review -> revise plan -> re-execute -> accept/stop
```

Maximum default iterations:
- 3 full cycles
- configurable

---

## 11. Reference Image Handling

When two images are provided:
- `input_image`: the image to be edited
- `reference_image`: a style / texture / feeling guide only

Allowed signals from the reference image:
- tonal range
- warmth / coolness
- contrast softness / punch
- grain / smoothness preference
- sharpness feel
- restoration target style

Disallowed use:
- copy objects, faces, backgrounds, text, or layout
- transplant semantic content

Implementation suggestion:
- extract low-level descriptors from reference image
- create a bounded target style profile
- use this profile only to bias tool parameters

---

## 12. Safety / Policy Rules

The package must reject or warn on prompts that imply deception, forgery, or evidence tampering.

Examples to reject:
- “make this manipulated image look original so nobody can tell”
- “remove proof of editing”
- “hide that something was added later”
- “change document evidence without detection”

The package can still support benign restoration requests such as:
- “fix scan dust”
- “remove age-related fading”
- “improve old family photo quality”

Policy layer requirements:
- prompt risk classification
- operation allowlist
- optional watermark / report note for sensitive cases
- auditable log of all steps

---

## 13. Quality Checks

### Objective checks
- Laplacian variance / blur proxy
- noise estimate before vs after
- clipping percentage in highlights / shadows
- color cast score
- SSIM / PSNR versus prior iteration where useful
- change area ratio

### Subjective checks via VLM
Reviewer asks:
- Did the result satisfy the prompt?
- Does the result remain natural?
- Are there halos or oversmoothed regions?
- Does the result preserve identity / key structure?
- Does the result reflect the reference image only at the style level?

### Acceptance rule
A result is accepted only if:
- prompt satisfaction improves
- artifact risk remains below threshold
- no disallowed semantic changes are detected

---

## 14. Public API Design

### Main API
```python
from veriedit import edit_image

result = edit_image(
    source_image="input.jpg",
    prompt="Restore this old photo, reduce yellowing, and keep it natural.",
    reference_image="target_feel.jpg",
    output_path="result.png",
    allowed_tools=["dust_cleanup", "stroke_paint"],  # optional
)
```

### Lower-level API
```python
from veriedit.workflow import VeriEditWorkflow

wf = VeriEditWorkflow()
result = wf.run(request)
```

### Request schema
```python
class EditRequest(BaseModel):
    source_image: str
    prompt: str
    reference_image: str | None = None
    output_path: str | None = None
    allowed_tools: list[str] = []
    max_iterations: int = 3
    preserve_metadata: bool = False
    save_intermediates: bool = True
    llm_model: str = "gemini-3-flash"
```

Notes:
- `output_path` in the Python API controls the filename inside the run folder
- CLI users should prefer `--output-folder` for `veriedit edit`
- `allowed_tools` is optional and intended mainly for debugging or isolated tool evaluation

---

## 15. CLI Commands

```bash
veriedit edit --input IMG --prompt TEXT [--reference IMG] [--output-folder DIR] [--allow-tool TOOL ...]
veriedit inspect --input IMG
veriedit batch --input-dir DIR --prompt TEXT
veriedit report --run-id RUN_ID
```

### Example
```bash
veriedit edit \
  --input scan.jpg \
  --reference clean_warm_photo.jpg \
  --prompt "Clean dust, gently restore contrast, and keep the result realistic." \
  --output-folder /tmp/veriedit \
  --allow-tool dust_cleanup \
  --allow-tool stroke_paint \
  --save-intermediates
```

---

## 16. Reporting Requirements

Each run should generate:

### `report.json`
- input metadata
- plan
- tool sequence
- parameters
- metrics per step
- review decisions
- final decision

### `report.md`
Human-readable summary:
- request
- key constraints
- what changed
- what was intentionally not changed
- confidence and warnings

### Intermediate artifacts
- step images
- masks
- before/after comparison board

---

## 17. Testing Strategy

### Unit tests
- each tool
- schema validation
- policy checks
- planner output format

### Integration tests
- full workflow on sample restoration tasks
- one-image mode
- two-image mode
- retry loop behavior

### Regression tests
- compare metrics and snapshots on curated fixtures
- ensure no accidental generative path is added

---

## 18. Future Extensions

- desktop UI with Gradio or Qt
- segmentation-based local editing without generation
- ONNX-based non-generative defect detectors
- RAW processing workflows
- batch restoration pipelines
- plugin system for custom tools
- provenance signing for reports

---

## 18.1 Photoshop Gap Roadmap

The project should explicitly acknowledge why a skilled human retoucher can still outperform the autonomous workflow:
- humans make high-bandwidth local judgments
- humans branch quickly and keep only the best trial
- humans work with masks, layers, opacity, and selective brushes
- humans distinguish real damage from true image structure more reliably
- humans know when to stop before an edit looks artificial

To close that gap without becoming generative, the roadmap should prioritize:

### Region-aware diagnostics
- detect and save candidate dust, scratch, border-damage, and low-contrast regions
- emit machine-readable region summaries
- persist diagnostic masks as review artifacts

### Region-first planning
- prefer targeted repairs before broad global edits
- identify which planned tools are global versus local
- describe which regions should be protected from each step

### Branch-and-compare execution
- allow the executor to try a small set of bounded variants for risky steps
- compare candidates quantitatively before choosing one
- log rejected candidates for auditability

### Layer-like execution model
- maintain a reversible sequence of masked edits instead of only flattened image states
- track per-step masks, opacity, and blend intent where applicable

### Patch-level review
- review local areas separately from full-frame tonal changes
- distinguish broad but legitimate restoration from suspicious semantic drift
- score repairs against the defect maps that motivated them

### Human-in-the-loop mode
- allow optional approval before committing ambiguous or high-impact steps
- present comparison boards and mask previews for manual selection

### Incremental delivery guidance
These features should be implemented one by one, not as a single rewrite. Each completed increment should be tracked in `CHANGE_LOG.md` with:
- what shipped
- what remains incomplete
- why any deferred items were postponed

---

## 19. MVP Definition

MVP must include:
- pip-installable package
- CLI + Python API
- one-image and two-image inputs
- AG2 multi-agent runtime loop
- Gemini-driven planner/reviewer
- OpenCV/Pillow/scikit-image execution layer
- JSON + Markdown reports
- policy guardrails against deceptive editing requests

---

## 20. Success Criteria

The project is successful if it can:
- take an old photo plus a natural-language prompt
- optionally use a second image as a tonal/style reference
- improve the photo with explicit non-generative tools
- explain exactly what it did
- review its own result and retry when needed
- avoid fabricating image content
