# tmachine

**A differentiable 3D Gaussian Splat mutation engine.**

Edit a 3DGS scene with a text prompt or a voice recording. The engine renders a viewport, sends the image through any 2D generative model, computes the pixel-level delta, and back-propagates the change into the scene's Gaussian parameters — all without touching the base scene file.

```bash
pip install tmachine
```

```python
from tmachine import ViewportRenderer, DeltaEngine, SplatMutator, camera_from_fov
import math

camera = camera_from_fov(position=(0, -1, -5), pitch=math.radians(6),
                         yaw=0, roll=0, fov_x=math.radians(60),
                         width=1920, height=1080)

renderer = ViewportRenderer("scene.ply")
mutator  = SplatMutator("scene.ply")

result = mutator.mutate(
    camera=camera,
    edited_image=my_ai_edited_tensor,       # (H, W, 3) float32, [0, 1]
    patch_path="patches/my_change.ply",
)
# result.patch_path      → tiny patch .ply (< 1 % of base scene size)
# result.hidden_indices  → base-scene rows to suppress when rendering
```

---

## Contents

- [Architecture](#architecture)
- [Module 1 — Viewport Renderer](#module-1--viewport-renderer)
- [Module 2 — Delta Engine](#module-2--delta-engine)
- [Module 3 — Splat Mutator](#module-3--splat-mutator)
- [AI Layer](#ai-layer)
  - [Voice Pipeline](#voice-pipeline)
  - [Image Editor](#image-editor)
- [Scene I/O](#scene-io)
- [Camera](#camera)
- [REST API](#rest-api)
- [Memory Layers](#memory-layers)
- [Database models](#database-models)
- [Installation](#installation)
- [Environment variables](#environment-variables)

---

## Architecture

The engine is a sequential three-module pipeline. Each module is usable in isolation.

```
.ply scene + Camera
        │
        ▼
┌───────────────────┐
│ ViewportRenderer  │  gsplat CUDA rasterization
│   (Module 1)      │  → (H, W, 3) differentiable tensor
└───────────────────┘
        │ original render
        │                   AI-edited image (any 2D model)
        │                           │
        ▼                           ▼
┌───────────────────────────────────────┐
│          DeltaEngine (Module 2)       │  pixel-level diff
│                                       │  → differentiable LossMap
└───────────────────────────────────────┘
        │ LossMap + camera
        ▼
┌───────────────────┐
│  SplatMutator     │  Adam optimizer loop
│   (Module 3)      │  back-props into SH + opacity logits
│                   │  → patch .ply + hidden_indices
└───────────────────┘
```

**Why operate in 2D?**

Every text-to-3D approach risks destroying multi-view consistency: change a splat in one view and it looks wrong from every other angle. TMachine sidesteps this by confining the generative model to the 2D domain — the space it was trained in — and using the differentiability of the renderer to pull the change back into 3D parameters.

**Why patches, not overwrites?**

The base `.ply` is never modified. Each edit is extracted as a minimal patch: only the splats that changed above a threshold, stored in a separate `.ply` file alongside a list of `hidden_indices` (base-scene rows the patch replaces). This enables layered, reversible, and even conflicting historical edits on the same scene — see [Memory Layers](#memory-layers).

---

## Module 1 — Viewport Renderer

**`tmachine/core/renderer.py`** · **`ViewportRenderer`**

Rasterizes a `GaussianCloud` from an arbitrary camera position. The render is fully differentiable via `gsplat`'s CUDA tile rasterizer — gradients flow back to every Gaussian parameter.

```python
from tmachine import ViewportRenderer, camera_from_fov
import math

renderer = ViewportRenderer("scene.ply")
image    = renderer.render(camera)   # (H, W, 3) float32 tensor, [0, 1]
```

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ply_path` | `str` | — | Path to the source `.ply` file |
| `device` | `str` | auto | `"cuda"` or `"cpu"` |

**`render()`**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera` | `Camera` | — | Camera intrinsics + extrinsics |
| `gaussians` | `GaussianCloud` | loaded scene | Override the scene to render |
| `sh_degree` | `int` | `3` | SH degree (0–3). Lower = faster |
| `background` | `tuple[float,float,float]` | `(0,0,0)` | RGB background colour |

---

## Module 2 — Delta Engine

**`tmachine/core/delta_engine.py`** · **`DeltaEngine`**

Compares the original viewport render against an AI-edited image. Returns a `LossMap` — a differentiable description of what changed and by how much.

```python
from tmachine import DeltaEngine

delta    = DeltaEngine()
loss_map = delta.compute(original_tensor, edited_tensor)

print(loss_map)
# LossMap(total=0.01823, L1=0.01654, L2=0.02840, changed=4.31%)

loss_map.total_loss.backward()   # drive the optimizer
```

**`DeltaEngine` constructor**

| Parameter | Default | Description |
|---|---|---|
| `l1_weight` | `0.8` | Weight of L1 in `total_loss` |
| `l2_weight` | `0.2` | Weight of L2 in `total_loss` |
| `change_threshold` | `0.01` | Min luminance delta to flag a pixel as changed |

**`LossMap` fields**

| Field | Shape | Description |
|---|---|---|
| `pixel_diff` | `(H, W, 3)` | Absolute per-channel difference |
| `luminance_diff` | `(H, W)` | ITU-R BT.601 weighted magnitude |
| `change_mask` | `(H, W)` bool | Pixels that changed above threshold |
| `l1_loss` | scalar | Mean absolute error |
| `l2_loss` | scalar | Mean squared error |
| `total_loss` | scalar | `0.8·L1 + 0.2·L2` — differentiable |
| `changed_pixel_ratio` | `float` | Fraction of pixels flagged as changed |

---

## Module 3 — Splat Mutator

**`tmachine/core/splat_mutator.py`** · **`SplatMutator`**

Runs an Adam optimization loop until the rendered view matches the edited target. After convergence, extracts only the changed splats into a minimal patch `.ply`.

```python
from tmachine import SplatMutator

mutator = SplatMutator("scene.ply")
result  = mutator.mutate(
    camera=camera,
    edited_image=edited_tensor,
    n_iters=300,
    patch_path="patches/my_change.ply",
)

print(result)
# MutationResult(iters=247, loss 0.02341→0.00189, patch='...', changed_splats=4821)

result.patch_path      # tiny patch .ply
result.hidden_indices  # list[int] — base-scene rows to suppress
```

**Why SH + opacity only (by default)**

Colour and appearance changes are encoded in Spherical Harmonic coefficients and opacity logits. Freezing geometry (positions, scales, rotations) preserves multi-view structural integrity. Pass `optimize_geometry=True` for structural edits.

**`mutate()` parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera` | `Camera` | — | Must match the camera used to capture the original render |
| `edited_image` | `Tensor (H,W,3)` | — | AI-edited target image |
| `n_iters` | `int` | `300` | Maximum optimizer steps |
| `patch_path` | `str` | auto (UUID) | Output path for the patch `.ply` |
| `sh_degree` | `int` | `3` | Must match the scene's training degree |
| `optimize_geometry` | `bool` | `False` | Also optimize positions, scales, rotations |
| `on_iter` | `callable` | `None` | Progress callback `fn(iter: int, loss: float)` |
| `convergence_threshold` | `float` | `1e-7` | Early-stop when loss delta falls below this |
| `change_threshold` | `float` | `1e-4` | Per-splat delta threshold for patch extraction |

**`MutationResult`**

| Field | Type | Description |
|---|---|---|
| `patch_path` | `str` | Path to the saved patch `.ply` |
| `hidden_indices` | `list[int]` | Base-scene splat rows the patch replaces |
| `changed_splat_count` | `int` | Number of splats in the patch |
| `initial_loss` | `float` | Loss before optimization |
| `final_loss` | `float` | Loss after optimization |
| `iterations_run` | `int` | Actual steps taken |
| `loss_history` | `list[float]` | Per-iteration loss curve |

---

## AI Layer

The AI layer is optional (`pip install tmachine[ai]`). Any 2D image editing pipeline can be substituted — the three core modules have no AI dependency.

### Voice Pipeline

**`tmachine/ai/voice_pipeline.py`** · **`VoicePipeline`**

Two-stage pipeline: Whisper STT → GPT-4o-mini prompt extraction.

```
Spoken:       "Well, back in my day that awning was never red —
               it was a dark hunter green."
Transcript:   (raw Whisper output)
Edit prompt:  "Change the awning color to dark hunter green"
```

```python
from tmachine.ai import VoicePipeline

pipeline = VoicePipeline()

# From a file:
result = pipeline.process_file("recording.m4a")
print(result.transcript)    # full Whisper transcription
print(result.edit_prompt)   # extracted imperative instruction
print(result.llm_used)      # True if GPT was called

# From raw bytes (e.g. HTTP multipart upload):
result = pipeline.process_bytes(audio_bytes, suffix=".webm")
```

When `OPENAI_API_KEY` is absent, a regex-based fallback cleans the transcript and returns it directly as the edit prompt.

| Env var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for LLM extraction |
| `TMACHINE_WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large` |

---

### Image Editor

**`tmachine/ai/image_editor.py`** · **`ImageEditor`**

Wraps `timbrooks/instruct-pix2pix` (InstructPix2Pix via HuggingFace `diffusers`). Lazy-loads the pipeline (~3 GB) on first call.

```python
from tmachine.ai import ImageEditor
from PIL import Image

editor     = ImageEditor()
edited_pil = editor.edit(
    image=Image.open("render_original.png"),
    prompt="change the awning to dark hunter green",
    image_guidance_scale=1.5,
    guidance_scale=7.5,
    seed=42,
)
editor.unload()   # free VRAM before running SplatMutator
```

| Parameter | Default | Description |
|---|---|---|
| `image_guidance_scale` | `1.5` | 1.0–2.5. Higher = preserve more structure |
| `guidance_scale` | `7.5` | 1.0–15.0. Higher = follow prompt more strictly |
| `num_inference_steps` | `50` | Reduce to 20–30 for faster previews |
| `seed` | `None` | Pass an int for deterministic output |

| Env var | Default | Description |
|---|---|---|
| `TMACHINE_IP2P_MODEL` | `timbrooks/instruct-pix2pix` | HuggingFace model ID or local path |
| `TMACHINE_DEVICE` | auto | `cuda` / `mps` / `cpu` |

---

## Scene I/O

**`tmachine/io/ply_handler.py`**

### `GaussianCloud`

In-memory representation. All tensors store pre-activation raw values matching the `.ply` layout.

| Tensor | Shape | Activation | Description |
|---|---|---|---|
| `means` | `(N, 3)` | — | World-space positions |
| `quats` | `(N, 4)` | normalize | Rotation quaternions (w, x, y, z) |
| `log_scales` | `(N, 3)` | `exp()` | Log-space Gaussian scales |
| `raw_opacities` | `(N,)` | `sigmoid()` | Opacity logits |
| `sh_dc` | `(N, 3)` | — | DC (degree-0) SH colour |
| `sh_rest` | `(N, 15, 3)` | — | SH bands 1–3 |

Convenience properties: `.scales` (activated), `.opacities` (activated), `.sh_all` → `(N, 16, 3)`.

### `load_ply` / `save_ply`

```python
from tmachine.io import load_ply, save_ply

cloud = load_ply("scene.ply", device="cuda")
save_ply(cloud, "scene_out.ply")
```

Files are byte-compatible with the [Inria 3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting) and all standard 3DGS viewers.

---

## Camera

**`tmachine/utils/camera.py`**

OpenCV convention (+X right, +Y down, +Z forward).

```python
from tmachine import camera_from_fov, camera_from_euler
import math

# From horizontal FoV:
cam = camera_from_fov(
    position=(0.0, -1.0, -5.0),
    pitch=math.radians(6),
    yaw=0.0, roll=0.0,
    fov_x=math.radians(60),
    width=1920, height=1080,
)

# From focal length:
cam = camera_from_euler(
    position=(0, 0, -3),
    pitch=0, yaw=0, roll=0,
    fx=1200, fy=1200,
    width=1920, height=1080,
)
```

| Property | Description |
|---|---|
| `camera.viewmat` | `(4, 4)` world-to-camera matrix |
| `camera.K` | `(3, 3)` intrinsic matrix |
| `camera.fov_x / fov_y` | Field of view in radians |

---

## REST API

A FastAPI server wraps the engine for networked use. Requires `pip install tmachine[api]`.

### Running

```bash
# API server
uvicorn tmachine.api.app:app --reload --port 8000

# Celery worker (one process per GPU)
celery -A tmachine.api.worker worker --loglevel=info --concurrency=1

# Optional job monitor
celery -A tmachine.api.worker flower --port=5555
```

---

### `GET /render`

Render a viewport image. Synchronous, ~100 ms on GPU.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scene` | string | required | Absolute path to the base `.ply` |
| `x`, `y`, `z` | float | `0,0,-5` | Camera world-space position |
| `pitch`, `yaw`, `roll` | float | `0` | Camera orientation (radians) |
| `fov_x` | float | `1.047` | Horizontal FoV (radians) |
| `width`, `height` | int | `1920, 1080` | Output dimensions |
| `active_layers` | int[] | `[]` | Memory Layer IDs to composite |

Response: `image/png`

```bash
curl "http://localhost:8000/render?scene=/data/s.ply" --output render.png
curl "http://localhost:8000/render?scene=/data/s.ply&active_layers=3&active_layers=7" --output layered.png
```

---

### `POST /proposals/prompt`

Enqueue Stage 1 (render → AI edit → save preview PNG → write `MemoryProposal`). Returns immediately.

**Form fields:** `scene`, `camera` (JSON), `prompt`

```json
{ "job_id": "abc123", "status": "PENDING" }
```

---

### `POST /proposals/voice`

Full Tap-and-Talk pipeline. Whisper + LLM run synchronously; AI generation runs in background.

**Form fields:** `audio` (file), `scene`, `camera` (JSON)

```json
{
  "job_id": "abc123",
  "status": "PENDING",
  "transcript": "Well, back in my day that awning was never red...",
  "edit_prompt": "Change the awning color to dark hunter green",
  "llm_used": true
}
```

---

### `GET /proposals`

| Parameter | Default | Description |
|---|---|---|
| `scene` | required | Filter by base `.ply` path |
| `status` | `PENDING` | `PENDING` / `APPROVED` / `REJECTED` / `ALL` |

---

### `GET /proposals/{id}/preview`

Returns the 2D AI-edited preview PNG for a proposal.

---

### `POST /proposals/{id}/vote`

**Body:** `{ "vote": "yes" | "no" }`

---

### `POST /proposals/{id}/approve`

Marks approved and enqueues Stage 2 (`bake_approved_patch` — runs `SplatMutator`, writes a `MemoryLayer` row).

**Body:** `{ "n_iters": 300 }`

---

### `POST /proposals/{id}/reject`

Marks rejected. No 3D mutation is performed.

---

### `GET /layers`

| Parameter | Description |
|---|---|
| `scene` | Filter by base `.ply` path |

---

### `GET /status/{job_id}`

```json
{
  "job_id": "abc123",
  "status": "SUCCESS",
  "result": { "proposal_id": 7, "preview_path": "...", "prompt": "..." }
}
```

Statuses: `PENDING` · `STARTED` · `SUCCESS` · `FAILURE` · `RETRY`

---

## Memory Layers

Memory Layers is the Git-style patch system that enables branching, reversible, and conflicting edits on the same base scene.

### Data model

Each approved edit produces two artefacts:

1. **Patch `.ply`** — a `GaussianCloud` containing only the changed splats (typically ≪ 1 % of the base scene).
2. **`hidden_indices`** — the integer row indices in the base scene that the patch replaces.

The base `.ply` is never modified.

### Layered rendering

When `GET /render?active_layers=3,7` is called:

1. Load the base `GaussianCloud` from the renderer cache.
2. For each active layer, zero out `raw_opacities[hidden_indices]` (≈ 0 opacity).
3. Concatenate the patch clouds onto the base.
4. Rasterize the merged cloud.

### Two-stage human review

```
User speaks / types
       │
       ▼
Stage 1: generate_memory_proposal                       (~30–60 s)
  Render → AI edit → save preview PNG → MemoryProposal(PENDING)
  ← returns immediately with job_id

       │
       ▼
  Community votes YES / NO
  Curator calls POST /proposals/{id}/approve

       │
       ▼
Stage 2: bake_approved_patch                            (~2–5 min)
  SplatMutator → patch.ply + hidden_indices → MemoryLayer row
  Base .ply untouched
```

### Handling conflicting memories

Two people may remember the same corner differently. Both proposals are stored as discrete `MemoryLayer` rows. The frontend's layer toggle allows switching between competing historical realities as a first-class feature.

---

## Database models

Requires `pip install tmachine[api]` and PostgreSQL (or SQLite for dev).

```bash
export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/tmachine"
python -c "
from tmachine.db.models import Base
from tmachine.db.session import engine
Base.metadata.create_all(engine)
"
```

### `MemoryProposal`

| Column | Type | Description |
|---|---|---|
| `id` | bigint PK | — |
| `scene` | string | Base `.ply` path (indexed) |
| `prompt` | text | Edit instruction |
| `preview_path` | string | Path to the 2D AI-edited PNG |
| `status` | enum | `PENDING` / `APPROVED` / `REJECTED` |
| `votes_yes` / `votes_no` | int | Community vote counts |
| `cam_x` … `cam_fov_x` | float | Camera parameters |
| `bake_job_id` | string | Celery task ID set on approval |

### `MemoryLayer`

| Column | Type | Description |
|---|---|---|
| `id` | bigint PK | — |
| `proposal_id` | FK → `MemoryProposal` | Source proposal |
| `scene` | string | Base `.ply` path (indexed) |
| `patch_path` | string | Path to the patch `.ply` |
| `hidden_indices` | JSON `int[]` | Base-scene rows to suppress |
| `changed_splat_count` | int | Splats in the patch |
| `initial_loss` / `final_loss` | float | Optimization quality metrics |

---

## Installation

Always install inside a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Then install the package:

```bash
# Core engine only (rendering + optimization, no AI, no API)
pip install tmachine

# With AI generation layer
pip install "tmachine[ai]"

# With REST API server
pip install "tmachine[api]"

# Everything
pip install "tmachine[full]"

# Editable install for development (includes test deps)
pip install -e ".[full,dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0. CUDA is strongly recommended — CPU mode is functional but significantly slower.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `TMACHINE_DEVICE` | auto | `cuda` / `mps` / `cpu` |
| `TMACHINE_IP2P_MODEL` | `timbrooks/instruct-pix2pix` | InstructPix2Pix model ID or local path |
| `TMACHINE_WHISPER_MODEL` | `base` | Whisper model size |
| `OPENAI_API_KEY` | — | GPT-4o-mini prompt extraction (optional) |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + result backend |
| `DATABASE_URL` | `postgresql+psycopg2://tmachine:tmachine@localhost:5432/tmachine` | PostgreSQL DSN |
| `PREVIEW_DIR` | `./previews` | Directory for proposal preview PNGs |
| `LOCK_TIMEOUT` | `300` | Seconds to wait for a `.ply` file lock |

---

## Example

```bash
# Text prompt
python examples/basic_edit.py \
    --ply /data/main_street.ply \
    --prompt "change the awning to dark hunter green"

# Voice recording
python examples/basic_edit.py \
    --ply /data/main_street.ply \
    --audio grandmas_memory.m4a
```

Each run produces:

| File | Description |
|---|---|
| `render_original.png` | Pristine viewport render (Module 1) |
| `render_edited.png` | AI-edited image |
| `loss_map.png` | Greyscale luminance change magnitude (Module 2) |
| `render_verification.png` | Re-render after mutation (Module 3) |
