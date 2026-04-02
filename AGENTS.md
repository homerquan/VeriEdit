# ToolEdit / VeriEdit — `agents.md`

## 1. Purpose

This document defines the multi-agent architecture for a **non-generative image editing system** built with **LangGraph**.

The agents coordinate:
- planning
- execution
- review
- retry decisions
- policy enforcement

The system edits images using **procedural tools only**.
No agent may call a generative image model.

---

## 2. Design Principles

1. **Tool use over generation**
   - All image changes must come from explicit editing operators.
2. **Closed loop**
   - The system reviews results and can revise the plan.
3. **Auditable**
   - Every action must be logged.
4. **Bounded autonomy**
   - Agents can choose tools and parameters only within policy and tool limits.
5. **Reference-guided, not content-copying**
   - A second image can guide feeling and texture, not semantic content.

---

## 3. Agent Topology

```text
                ┌─────────────────┐
                │  Policy Agent   │
                └────────┬────────┘
                         │
                         v
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Diagnostics │-> │   Planner   │-> │  Executor   │-> │  Reviewer   │
└─────────────┘   └─────────────┘   └─────────────┘   └──────┬──────┘
                                                              │
                                                              v
                                                       ┌─────────────┐
                                                       │ Retry Agent │
                                                       └──────┬──────┘
                                                              │
                                                       accept / revise
```

---

## 4. Shared State Schema

All agents read/write a common LangGraph state.

```python
class WorkflowState(TypedDict):
    run_id: str
    source_image_path: str
    reference_image_path: str | None
    prompt: str

    policy_status: dict
    diagnostics: dict
    style_profile: dict | None

    plan: dict | None
    executed_steps: list[dict]
    current_image_path: str
    intermediate_paths: list[str]

    review: dict | None
    retry_decision: dict | None
    final_result: dict | None

    iteration: int
    max_iterations: int
    stop_reason: str | None
```

---

## 5. Agent Definitions

### 5.1 Policy Agent

#### Goal
Check whether the request is allowed before any editing happens.

#### Inputs
- prompt
- source image metadata
- optional reference image metadata

#### Responsibilities
- classify prompt as benign, caution, or reject
- block deceptive or fraudulent editing requests
- define operation constraints
- enforce “non-generative only” rule

#### Example outputs
```json
{
  "status": "allow",
  "risk_level": "low",
  "constraints": [
    "non-generative only",
    "do not alter semantic identity",
    "do not fabricate missing content"
  ]
}
```

```json
{
  "status": "reject",
  "risk_level": "high",
  "reason": "Request asks to conceal evidence of prior editing."
}
```

#### Notes
This agent should be deterministic where possible. A small rule layer should backstop the LLM classification.

---

### 5.2 Diagnostics Agent

#### Goal
Inspect the input image(s) and produce a machine-readable summary for planning.

#### Inputs
- source image
- optional reference image

#### Responsibilities
- compute image size, bit depth, and color mode
- estimate blur, noise, clipping, cast, contrast, and skew
- extract low-level style descriptors from the reference image
- identify candidate restoration issues such as dust, scratches, fade, low contrast

#### Output example
```json
{
  "source": {
    "width": 2400,
    "height": 1800,
    "blur_score": 87.2,
    "noise_score": 0.18,
    "yellow_cast": 0.42,
    "underexposed": false,
    "dust_candidates": 137
  },
  "reference": {
    "warmth": 0.61,
    "contrast": 0.48,
    "sharpness_feel": 0.37,
    "grain_level": 0.22
  }
}
```

#### Implementation guidance
This agent should be mostly non-LLM code.
Use OpenCV, NumPy, and scikit-image first.

---

### 5.3 Planner Agent

#### Goal
Translate user intent + diagnostics into a bounded edit plan.

#### Inputs
- prompt
- policy constraints
- diagnostics
- optional reference style profile

#### Responsibilities
- infer the editing objective
- choose a sequence of tools
- select conservative parameters
- define acceptance criteria
- describe what must not change

#### Output contract
```json
{
  "objective": "restore faded old photo naturally",
  "must_preserve": [
    "identity",
    "facial structure",
    "overall composition"
  ],
  "must_avoid": [
    "oversmoothing",
    "halo artifacts",
    "semantic content changes"
  ],
  "steps": [
    {
      "tool": "white_balance",
      "params": {"method": "gray_world", "strength": 0.7},
      "reason": "Reduce yellow cast"
    },
    {
      "tool": "dust_cleanup",
      "params": {"max_area": 20, "sensitivity": 0.45},
      "reason": "Remove small scan dust"
    },
    {
      "tool": "nlm_denoise",
      "params": {"h": 5},
      "reason": "Reduce sensor/scanner noise"
    },
    {
      "tool": "unsharp_mask",
      "params": {"radius": 1.0, "amount": 0.4},
      "reason": "Light sharpening only"
    }
  ],
  "acceptance": [
    "dust visibly reduced",
    "warmer/yellow cast reduced",
    "fine detail remains natural"
  ]
}
```

#### LLM choice
Use Gemini Flash for this agent because it is fast and strong enough for multimodal planning and structured output. Google’s Gemini docs show Flash as suited to agentic use cases and structured/function-driven workflows. citeturn101608search2turn101608search3

---

### 5.4 Executor Agent

#### Goal
Run the plan with real editing tools.

#### Inputs
- current image
- plan steps
- policy constraints

#### Responsibilities
- map plan steps to Python tool calls
- apply edits in sequence
- save each intermediate result
- collect metrics after each step
- rollback or reduce intensity if the step is harmful

#### Important rule
Executor can call only approved tools from the tool registry.
It cannot invent new operations.

#### Tool registry shape
```python
class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict
    safety_notes: list[str]
```

#### Example execution record
```json
{
  "step_index": 2,
  "tool": "dust_cleanup",
  "params": {"max_area": 20, "sensitivity": 0.45},
  "before_metrics": {"noise": 0.21, "dust_count": 137},
  "after_metrics": {"noise": 0.20, "dust_count": 32},
  "output_path": "runs/123/step_02.png",
  "status": "ok"
}
```

---

### 5.5 Reviewer Agent

#### Goal
Judge whether the current output is acceptable.

#### Inputs
- original image
- current image
- optional reference image
- prompt
- plan
- metrics

#### Responsibilities
- compare source vs current result
- compare current result vs reference style profile
- identify artifacts and over-editing
- decide whether the output satisfies the request

#### Review dimensions
- prompt satisfaction
- naturalness
- preservation of key structure
- no semantic fabrication
- bounded similarity to reference style only
- artifact risk

#### Output example
```json
{
  "status": "revise",
  "prompt_score": 0.74,
  "artifact_risk": 0.31,
  "findings": [
    "yellow cast improved",
    "small dust mostly removed",
    "slight haloing around high-contrast edges"
  ],
  "recommendations": [
    "reduce sharpen amount by 30%",
    "stop further denoise"
  ]
}
```

#### Review method
Use a hybrid approach:
- quantitative metrics from classical vision
- qualitative review from Gemini multimodal reasoning

---

### 5.6 Retry Agent

#### Goal
Decide what to do after review.

#### Inputs
- review result
- iteration count
- plan
- execution trace

#### Responsibilities
- accept final output
- request a revised plan
- lower parameter intensity
- terminate when diminishing returns appear

#### Output options
```json
{"decision": "accept", "reason": "Prompt satisfied and artifacts low"}
```

```json
{"decision": "retry", "reason": "Good progress but halos detected", "strategy": "reduce sharpen and rerun last two steps"}
```

```json
{"decision": "stop", "reason": "Reached max iterations"}
```

---

## 6. Tooling Rules

### Allowed tool families
- geometric transforms
- color correction
- exposure adjustment
- denoise
- sharpen
- morphology-based cleanup
- bounded histogram/style transfer
- local clone/patch from nearby existing pixels only

### Disallowed tool families
- diffusion inpainting
- generative fill
- text-to-image
- semantic object insertion
- face synthesis
- background hallucination

### Reference-image rule
A reference image may influence:
- warmth
- contrast
- texture softness
- grain level
- sharpness feel

It may not influence:
- people
- objects
- layout
- text content

---

## 7. LangGraph Node Design

Suggested node list:
- `policy_check`
- `diagnose_inputs`
- `plan_edits`
- `execute_plan`
- `review_result`
- `decide_retry`
- `finalize_report`

### Routing logic
```text
policy_check
  -> reject => finalize_report
  -> allow  => diagnose_inputs

diagnose_inputs -> plan_edits -> execute_plan -> review_result -> decide_retry

decide_retry
  -> accept => finalize_report
  -> retry  => plan_edits or execute_plan
  -> stop   => finalize_report
```

---

## 8. Prompting Contracts

### Planner system prompt requirements
- Prefer the minimal effective edit sequence
- Use only registry tools
- Keep parameters conservative
- Never propose generative fill
- Respect policy constraints

### Reviewer system prompt requirements
- Judge realism over aggression
- Penalize over-sharpening and over-smoothing
- Distinguish style influence from content copying
- Flag suspicious semantic changes

### Retry system prompt requirements
- Prefer reducing intensity before adding steps
- Stop when improvements are marginal
- Never bypass policy or tool restrictions

---

## 9. Observability and Logging

Each agent should emit structured logs.

### Minimum log fields
- `run_id`
- `agent_name`
- `iteration`
- `input_summary`
- `decision`
- `output_summary`
- `latency_ms`

### Saved artifacts
- agent messages
- structured plans
- execution trace
- intermediate images
- final report

---

## 10. Failure Modes and Handling

### Planner failures
- invalid JSON
- unavailable tool
- unsafe step requested

Handling:
- repair output once
- otherwise fallback to conservative baseline plan

### Executor failures
- tool crash
- unsupported image mode
- parameter out of bounds

Handling:
- sanitize params
- fallback to safe defaults
- mark failed step in trace

### Reviewer failures
- inconsistent review
- multimodal call unavailable

Handling:
- fall back to metrics-only review
- lower confidence

---

## 11. Suggested Class Layout

```python
class PolicyAgent:
    def run(self, state: WorkflowState) -> WorkflowState: ...

class DiagnosticsAgent:
    def run(self, state: WorkflowState) -> WorkflowState: ...

class PlannerAgent:
    def run(self, state: WorkflowState) -> WorkflowState: ...

class ExecutorAgent:
    def run(self, state: WorkflowState) -> WorkflowState: ...

class ReviewerAgent:
    def run(self, state: WorkflowState) -> WorkflowState: ...

class RetryAgent:
    def run(self, state: WorkflowState) -> WorkflowState: ...
```

---

## 12. Baseline Tool Registry

Initial tools to implement:
- `auto_white_balance`
- `histogram_balance`
- `clahe_contrast`
- `gamma_adjust`
- `non_local_means_denoise`
- `wavelet_denoise`
- `median_cleanup`
- `dust_cleanup`
- `scratch_candidate_cleanup`
- `unsharp_mask`
- `edge_preserving_sharpen`
- `deskew`
- `crop`
- `resize`
- `bounded_histogram_match_to_reference`
- `texture_softness_bias_from_reference`

Each tool should declare:
- parameter bounds
- expected effect
- likely failure modes
- reversibility notes

---

## 13. Minimal Acceptance Logic

Accept the output when all are true:
- policy status is allow
- prompt satisfaction score >= configured threshold
- artifact risk <= configured threshold
- no semantic fabrication signals
- iteration count within limit

Otherwise:
- retry if recoverable
- stop if not recoverable

---

## 14. MVP Agent Scope

MVP should ship with:
- Policy Agent
- Diagnostics Agent
- Planner Agent
- Executor Agent
- Reviewer Agent
- Retry Agent

A future version can add:
- region selector agent
- human approval agent
- provenance/signature agent
- batch scheduling agent

---

## 15. Final Notes

This system should feel like an **AI art director + QA reviewer for classical image editing tools**, not an image generator.

The strongest product value is:
- natural-language control
- reproducible tool execution
- closed-loop self-review
- non-generative trustworthiness

