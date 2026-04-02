# Stroke_Engine_SPEC.md

## Project
Closed-Loop Stroke Controller for Human-Like Sketching

## Summary
This project builds a minimal working architecture for **closed-loop stroke control**: an AI system that draws by repeatedly observing the current canvas, proposing short stroke actions, predicting their effects, partially committing the best action, and correcting in real time.

The system is not a typical tool-calling agent. It is a **visuomotor control system** combining:
- feedforward prediction
- feedback correction
- short-horizon planning
- learned stroke priors

The design goal is to approximate how humans sketch:
- choose a local intention
- make a short stroke
- inspect the result immediately
- correct before error accumulates

---

## Goals

### Primary goal
Produce a minimal digital sketching system that can:
1. draw simple contours on a raster canvas,
2. use short-horizon prediction before each stroke,
3. commit only a small prefix of a candidate stroke,
4. re-observe the canvas after each partial commit,
5. continuously correct its behavior based on visual feedback.

### Secondary goals
- create an interpretable architecture rather than an end-to-end black box,
- support future expansion into hatching, shading, style control, and robotic pen drawing,
- separate high-level planning from low-level continuous control.

### Non-goals for MVP
- photorealistic painting,
- full autonomous artistic composition,
- robot hardware integration,
- high-quality color painting,
- end-to-end large-scale RL from scratch.

---

## Core idea
The essential shift is:

**Do not represent drawing as “generate the whole image.”**  
Represent it as:

**observe -> predict -> commit small stroke segment -> observe again -> correct**

This makes the system closer to:
- model predictive control (MPC),
- visual servoing,
- model-based reinforcement learning,
- human sensorimotor drawing behavior.

---

## Architecture overview

The system has two nested loops.

### Outer loop: local objective selection
The outer loop chooses *what small region to work on next*.

Inputs:
- target image
- current canvas
- optional user constraints or anchor points

Outputs:
- active local patch
- drawing mode
- local priority mask

Responsibilities:
- choose a local region with high remaining error,
- decide whether the current task is contour-following or another supported mode,
- hand a small local goal to the inner controller.

The outer loop is low-frequency and strategic.

### Inner loop: predictive stroke control
The inner loop performs the actual drawing behavior.

Inputs:
- current local canvas patch
- local target patch
- pen state
- recent stroke history

Outputs:
- partial committed stroke segment
- updated canvas
- updated state

Responsibilities:
- propose a small set of candidate strokes,
- predict what each candidate would do,
- score candidates against the local target,
- commit only a short prefix of the best candidate,
- re-observe and replan.

The inner loop is high-frequency and reactive.

---

## MVP system modules

### 1. State representation
The state must support both prediction and correction.

#### Required state fields
```python
state = {
    "canvas": raster_canvas,
    "vector_history": list_of_strokes,
    "active_patch_bbox": [x0, y0, x1, y1],
    "canvas_patch": local_canvas_patch,
    "target_patch": local_target_patch,
    "pen_position": [x, y],
    "pen_down": bool,
    "recent_strokes": short_history,
    "mode": "contour",
}
```

#### Notes
- Keep both raster and vector representations.
- Raster is used for visual comparison.
- Vector history is used for action continuity and future editability.
- The local patch should be small enough for fast iteration.

---

### 2. Stroke action representation
A stroke action is a parameterized motion primitive.

#### Minimal action schema
```python
class StrokeAction:
    type: str              # "bezier" or "polyline"
    points: list           # control points
    width: float
    opacity: float
    pressure: float
    color: float           # grayscale for MVP
    pen_down: bool
```

#### MVP choice
Use one primitive first:
- short cubic Bézier stroke

Reason:
- expressive enough for contour curves,
- simple enough for optimization,
- compact representation.

#### Important rule
A candidate action is **not executed fully at once**.

Instead:
- propose full stroke
- commit only a prefix fraction, such as 0.25
- re-observe
- continue or replan

This is the key mechanism that turns prediction into continuous control.

---

### 3. Stroke proposer
The proposer provides feedforward guesses for likely good next motions.

#### Inputs
- local target patch
- local current canvas patch
- pen position
- recent stroke history
- active mode

#### Outputs
- K candidate `StrokeAction` objects

#### MVP strategy
Start simple:
- heuristic candidate generation
- optional small neural reranker later

#### Initial heuristics
Generate candidate strokes based on:
- local edge tangent,
- local edge curvature,
- nearby uncovered contour pixels,
- short continuation from prior direction.

Possible candidate families:
- short forward continuation line,
- gentle arc left,
- gentle arc right,
- local tangent-aligned segment,
- short correction stroke toward target contour.

#### Future upgrade
Train a proposal network that predicts candidate control points from local patch features.

---

### 4. World model / renderer
This module predicts what a stroke would do to the canvas.

#### Two options
##### Option A: explicit renderer (recommended first)
Use a direct local renderer:
- input: current patch + stroke action + commit fraction
- output: predicted next patch

This is the recommended MVP because it is:
- deterministic,
- interpretable,
- debuggable,
- easy to validate.

##### Option B: learned transition model
```python
f(current_patch, stroke_action) -> next_patch
```

Useful later for speed, but not required first.

#### MVP choice
Start with:
- local explicit renderer as source of truth

#### Renderer requirements
- anti-aliased line rendering
- local patch rendering only
- grayscale support
- prefix rendering by fraction of arc length

#### Why local rendering
Global rendering is too slow.
The controller only needs local consequences of short strokes.

---

### 5. Local critic
The critic scores the quality of a predicted next patch.

#### Inputs
- previous patch
- predicted next patch
- target patch
- stroke metadata
- recent state context

#### Outputs
- scalar cost
- optional decomposed sub-scores

#### MVP cost function
```python
L =
    w1 * edge_alignment_error
  + w2 * coverage_penalty
  + w3 * overdraw_penalty
  + w4 * curvature_mismatch
  + w5 * stroke_length_penalty
  + w6 * jitter_penalty
```

#### Suggested interpretation
- **edge_alignment_error**: distance between rendered stroke and desired contour
- **coverage_penalty**: missing target contour coverage
- **overdraw_penalty**: penalize drawing where canvas is already sufficiently dark
- **curvature_mismatch**: mismatch between local contour curvature and stroke curvature
- **stroke_length_penalty**: discourage long brittle strokes in MVP
- **jitter_penalty**: discourage unstable direction changes

#### MVP implementation
Start with hand-designed image metrics:
- edge map overlap,
- Chamfer-like contour distance,
- pixel gain on uncovered target edges,
- overdraw penalty from already dark canvas pixels.

#### Future upgrade
Train a learned critic from successful vs unsuccessful local strokes.

---

### 6. Receding-horizon controller
This is the core continuous-control module.

#### Responsibility
Given the current local state:
1. request candidate strokes,
2. simulate partial effect of each candidate,
3. score candidates,
4. commit only a small prefix of the best one,
5. re-observe,
6. repeat until local goal is satisfied.

#### Pseudocode
```python
while not local_goal_done(state):
    candidates = proposer.propose(state, target_patch, k=8)

    scored = []
    for action in candidates:
        pred_patch = renderer.simulate(
            state["canvas_patch"],
            action,
            fraction=0.25
        )
        score = critic.score(
            prev_patch=state["canvas_patch"],
            next_patch=pred_patch,
            target_patch=state["target_patch"],
            state=state,
            action=action,
        )
        scored.append((score, action))

    best_score, best_action = min(scored, key=lambda x: x[0])

    canvas = renderer.commit(
        state["canvas"],
        best_action,
        fraction=0.25
    )

    state = observe_updated_state(canvas, state)

    if should_abort_or_replan(state):
        continue
```

#### Important design principle
Use:
- **short planning horizon**
- **frequent replanning**
- **partial commitment**

Do not allow long open-loop motion.

---

### 7. Outer-loop local objective selector
The outer loop selects where the inner controller should work next.

#### Inputs
- global target image
- full current canvas
- global error map

#### Outputs
- active patch bounding box
- optional priority mask
- mode assignment

#### MVP behavior
- compute residual edge error between target and canvas,
- select a patch with high remaining contour mismatch,
- send that patch to the inner controller.

#### Future upgrades
- semantic part decomposition,
- contour order planning,
- structural region scheduling,
- human-specified focus zones.

---

## Prediction vs continuous control

### Prediction
Prediction answers:
**“If I make this stroke, what is likely to happen?”**

Implemented by:
- the proposer,
- the renderer / world model.

Prediction should be:
- local,
- fast,
- short-horizon,
- candidate-based rather than full-rollout-heavy.

### Continuous control
Continuous control answers:
**“Now that I have started moving, what should I do next given the latest observation?”**

Implemented by:
- partial stroke commitment,
- re-observation after every partial step,
- frequent replanning,
- ability to bend, stop, or abandon a prior candidate.

### Design decision
The system must support both:
- feedforward prediction for efficiency,
- feedback correction for robustness.

Human-like skill requires the combination.

---

## Experience and learning
In this system, “experience” is not only a better final policy. It appears in three places:

### 1. Stroke prior
The system learns which local motions are plausible and efficient.

Examples:
- contour-following arcs,
- small straight construction strokes,
- hatching bundles.

### 2. Critic sensitivity
The system learns which local errors matter more.

Examples:
- contour drift may matter more than slight local darkness mismatch in contour mode.

### 3. Recovery patterns
The system learns how to react after partial failure.

Examples:
- continue,
- taper,
- stop,
- re-route,
- draw nearby correction,
- undo recent micro-segment.

---

## Data strategy

### Initial data sources
Use:
- vector sketch datasets,
- synthetic contour traces from line drawings,
- contours extracted from photos,
- optionally hand-made simple geometric sketches.

### Recommended first data forms
- raster target patch
- vector stroke sequence
- local contour tangent and curvature fields
- partial canvas states during drawing

### Why synthetic data first
Synthetic contour targets allow:
- exact target geometry,
- controllable difficulty,
- cheap rollout generation,
- easier evaluation.

---

## Training plan

### Phase 0: no learning required
Build the full control loop using:
- heuristic proposer,
- explicit renderer,
- hand-crafted critic.

Deliverable:
- a working contour follower on simple images.

### Phase 1: supervised proposal learning
Train a small proposal network to suggest candidate strokes from:
- target patch,
- current canvas patch,
- recent state.

Target labels can come from:
- vector sketch datasets,
- synthetic optimal short contours.

Deliverable:
- fewer bad candidate proposals,
- smoother drawing.

### Phase 2: learned critic
Train a local critic to score predicted patch transitions.

Training signal:
- improvement in contour match,
- expert or synthetic preference labels,
- local success/failure comparisons.

Deliverable:
- better local action ranking.

### Phase 3: model-based RL fine-tuning
Use the renderer as the transition model and optimize local improvement over episodes.

Objective:
- improve stroke proposal and selection under repeated closed-loop interaction.

Important:
- do **not** start with end-to-end RL from scratch.

Reason:
- too unstable,
- too sample-inefficient,
- too hard to debug.

---

## MVP scope

### Supported
- digital raster canvas
- grayscale drawing
- contour-following only
- one stroke primitive family
- fixed local patch size
- explicit re-observation after each partial commit

### Not yet supported
- color
- complex brush texture
- shading and hatching
- semantic understanding of full scenes
- robot arm control
- tactile or force feedback

---

## Recommended technical stack

### Core language
- Python

### Numeric / model stack
- PyTorch
- NumPy

### Rendering
- custom local rasterizer or DiffVG-style renderer inspiration

### Image processing
- OpenCV or scikit-image for edge maps, curvature estimation, and contour extraction

### Experiment tracking
- simple local logs first
- optionally Weights & Biases later

### Visualization
- per-step saved canvases
- candidate overlay debug images
- residual error maps

---

## Interfaces

### Stroke proposer
```python
class StrokeProposer:
    def propose(self, state, k=8):
        ...
```

### Renderer
```python
class LocalRenderer:
    def simulate(self, canvas_patch, action, fraction=1.0):
        ...

    def commit(self, canvas, action, fraction=1.0):
        ...
```

### Critic
```python
class LocalCritic:
    def score(self, prev_patch, next_patch, target_patch, state, action):
        ...
```

### Controller
```python
class MPCController:
    def step(self, state):
        ...
```

### Outer-loop patch selector
```python
class PatchSelector:
    def select_patch(self, canvas, target):
        ...
```

---

## Observability and debugging
This project must be built to be inspectable.

### Required debug artifacts
For every control step, optionally save:
- current patch,
- target patch,
- candidate stroke overlays,
- predicted next patch for each candidate,
- chosen action,
- actual committed prefix,
- residual error map after commit.

### Reason
Closed-loop drawing failures are often subtle. Without visual debugging, it will be hard to know whether the problem is:
- proposal quality,
- rendering mismatch,
- critic design,
- control instability,
- patch selection.

---

## Evaluation

### MVP metrics
- contour coverage ratio
- average contour distance to target
- overdraw rate
- number of micro-commits per patch
- jitter / direction instability
- total stroke length normalized by target complexity

### Qualitative evaluation
- smoothness of contours
- degree of drift correction
- whether the system visibly replans after mismatch
- whether the output looks controlled rather than random

### Benchmark tasks
Start with:
- circles
- ellipses
- polygons
- cartoon outlines
- face contour subsets
- simple object silhouettes

Do not start with full natural-image shading.

---

## Milestones

### Milestone 1: deterministic contour follower
- explicit contour extraction from target
- heuristic proposer
- explicit renderer
- hand-designed critic
- partial commit loop working

### Milestone 2: stable closed-loop drawing
- better patch selection
- improved continuity between strokes
- reduced jitter
- better overdraw handling

### Milestone 3: learned proposals
- train small proposal network
- compare with heuristic baseline

### Milestone 4: learned critic
- local ranking improvement
- better recovery from drift

### Milestone 5: RL fine-tuning
- model-based fine-tuning using short-horizon episodes

### Milestone 6: expanded drawing modes
- hatching
- variable width
- optional style priors

---

## Risks and mitigations

### Risk 1: search space too large
Mitigation:
- short stroke length,
- strong priors,
- small local patch,
- limited primitive family.

### Risk 2: controller jitters
Mitigation:
- continuity penalty,
- recent-direction smoothing,
- minimum commitment threshold,
- hysteresis in candidate switching.

### Risk 3: renderer too slow
Mitigation:
- render only local patch,
- cache edge features,
- use small grayscale canvases first.

### Risk 4: critic mis-scores good strokes
Mitigation:
- decompose cost terms visibly,
- compare hand-designed vs learned critic,
- use candidate visualization during debugging.

### Risk 5: RL instability
Mitigation:
- start from working hand-designed system,
- use supervised priors first,
- only fine-tune later.

---

## Long-term extensions

### Algorithmic extensions
- learned world model for faster simulation,
- uncertainty-aware proposal scoring,
- hierarchical stroke primitives,
- semantic part decomposition,
- imitation learning from human sketch sessions.

### Product extensions
- human-in-the-loop ghost stroke suggestions,
- editable stroke timeline,
- style imitation,
- guided sketch tutoring,
- robotic pen plotter mode.

### Research extensions
- combine visual and tactile feedback,
- continuous latent motor primitives,
- multi-timescale control,
- active error-driven exploration,
- inverse modeling from human drawing demonstrations.

---

## Definition of success
The MVP is successful if it can:
1. draw simple contour-based targets on a digital canvas,
2. do so through repeated short-horizon predict-and-correct steps,
3. visibly recover from local mismatch instead of drifting blindly,
4. produce output that looks like controlled incremental sketching rather than one-shot raster synthesis.

---

## One-sentence project definition
A minimal closed-loop sketching system that uses stroke proposals, local simulation, visual scoring, and receding-horizon partial stroke commitment to approximate human-like drawing behavior.
