"""
tmachine/api/worker.py — Celery application + task definitions.

Architecture — Memory Layer pipeline
---------------------------------------
The pipeline is split into two stages.  The approval workflow between them
is the responsibility of the calling application, not the engine.

Stage 1 ── generate_preview  (fast, ~20-60 s)
    • Render current scene (Module 1)
    • Edit rendered image with InstructPix2Pix (or accept uploaded image)
    • Save the 2D preview PNG to disk
    • Return {preview_filename, preview_path, prompt} — no DB write

    ══════════════════════════════
    ←── CALLER'S APPROVAL LOGIC ──►
    ══════════════════════════════

Stage 2 ── bake_patch  (slow, ~2-5 min)
    • Triggered by the caller after they decide to commit the preview
    • Runs SplatMutator.mutate() → produces a patch .ply + hidden_indices
    • Writes a MemoryLayer row to the database
    • Never touches the base .ply file

File-lock safety
----------------
Only Stage 2 acquires a FileLock (on the patch output path) since the base
scene is never written — each patch is a brand-new file.

Environment variables
---------------------
CELERY_BROKER_URL    – Celery broker URL.  Default: redis://localhost:6379/0
                       Examples: redis://host:6379/0  |  amqp://user:pass@host//
CELERY_RESULT_BACKEND– Celery result backend URL.  Defaults to CELERY_BROKER_URL.
                       Can differ from the broker (e.g. RabbitMQ broker + Redis store).
REDIS_URL            – Legacy: used as fallback for both broker and backend when
                       CELERY_BROKER_URL / CELERY_RESULT_BACKEND are not set.
DATABASE_URL         – Any SQLAlchemy-compatible DSN.
                       Default: postgresql+psycopg2://tmachine:tmachine@localhost:5432/tmachine
LOCK_TIMEOUT         – Max seconds to wait for a file lock.  Default: 300
PREVIEW_DIR          – Directory for storing 2D preview PNGs.  Default: ./previews
"""

from __future__ import annotations

import base64
import io
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from celery import Celery
from celery.utils.log import get_task_logger

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------

_REDIS_FALLBACK   = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_BROKER_URL       = os.environ.get("CELERY_BROKER_URL",     _REDIS_FALLBACK)
_BACKEND_URL      = os.environ.get("CELERY_RESULT_BACKEND", _BROKER_URL)
_LOCK_TIMEOUT     = int(os.environ.get("LOCK_TIMEOUT", "300"))
_PREVIEW_DIR      = Path(os.environ.get("PREVIEW_DIR", "./previews"))

# ---------------------------------------------------------------------------
# Scene LRU cache
# ---------------------------------------------------------------------------
# Avoids re-parsing the source .ply file for consecutive bake jobs on the same
# scene.  Keyed by (absolute_path, mtime) so stale entries auto-invalidate when
# the file changes on disk.  Cache is not thread-safe; safe under Celery’s
# default --concurrency=1 (single worker thread).

_SCENE_CACHE_MAX: int = int(os.environ.get("SCENE_CACHE_SIZE", "4"))
_scene_cache: dict[tuple[str, float], Any] = {}


def _load_scene_cached(path: str) -> Any:
    """Return a GaussianCloud for *path*, loading from disk only when necessary."""
    from ..io.ply_handler import load_ply
    mtime = os.path.getmtime(path)
    key = (path, mtime)
    if key not in _scene_cache:
        # Evict oldest entry (dict preserves insertion order since Python 3.7)
        while len(_scene_cache) >= _SCENE_CACHE_MAX:
            _scene_cache.pop(next(iter(_scene_cache)))
        _scene_cache[key] = load_ply(path)
    return _scene_cache[key]

celery_app = Celery(
    "tmachine",
    broker=_BROKER_URL,
    backend=_BACKEND_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    worker_prefetch_multiplier=1,
)

logger = get_task_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_from_dict(cam: dict):
    from ..utils.camera import camera_from_fov
    return camera_from_fov(
        position=(cam["x"], cam["y"], cam["z"]),
        pitch=cam["pitch"],
        yaw=cam["yaw"],
        roll=cam["roll"],
        fov_x=cam["fov_x"],
        width=cam["width"],
        height=cam["height"],
        near=cam["near"],
        far=cam["far"],
    )


def _tensor_to_pil(t):
    import numpy as np
    from PIL import Image
    arr = (t.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(img):
    import numpy as np
    import torch
    arr = np.array(img.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr)


def _save_preview(pil_img, filename: str) -> str:
    """Save a PIL image as a preview PNG; return its absolute path."""
    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = _PREVIEW_DIR / filename
    pil_img.save(str(path), format="PNG")
    return str(path)


# ---------------------------------------------------------------------------
# Stage 1 — generate_preview
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10,
                 name="generate_preview")
def generate_preview(
    self,
    scene: str,
    camera_dict: dict,
    prompt: str,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    sh_degree: int = 3,
) -> dict[str, Any]:
    """
    Stage 1 — Render the scene, apply the AI image edit, save a 2D preview.

    **No 3D modification.  No DB write.**

    Returns
    -------
    dict with keys: preview_filename, preview_path, prompt
    """
    from ..ai.image_editor import ImageEditor
    from ..core.renderer import ViewportRenderer

    camera = _camera_from_dict(camera_dict)

    # ── Module 1: Render current base scene ───────────────────────────────
    logger.info("Rendering scene for preview: %r", prompt)
    renderer        = ViewportRenderer(scene)
    original_tensor = renderer.render(camera, sh_degree=sh_degree)
    original_pil    = _tensor_to_pil(original_tensor)

    # ── AI Edit: InstructPix2Pix ──────────────────────────────────────────
    logger.info("Running ImageEditor with prompt: %r", prompt)
    editor     = ImageEditor()
    edited_pil = editor.edit(
        image=original_pil,
        prompt=prompt,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    editor.unload()

    # ── Save preview PNG ──────────────────────────────────────────────────
    preview_filename = f"preview_{uuid.uuid4().hex}.png"
    preview_path     = _save_preview(edited_pil, preview_filename)

    logger.info("Preview saved: %s", preview_path)
    return {
        "preview_filename": preview_filename,
        "preview_path":     preview_path,
        "prompt":           prompt,
    }


# ---------------------------------------------------------------------------
# Stage 2 — bake_patch
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10,
                 name="bake_patch")
def bake_patch(
    self,
    scene: str,
    edited_image_b64: str,
    camera_dict: dict,
    patch_dir: Optional[str] = None,
    n_iters: int = 300,
    sh_degree: int = 3,
    external_ref: Optional[str] = None,
) -> dict[str, Any]:
    """
    Stage 2 — Run the SplatMutator from a pre-edited image and save the patch.

    Does not load from DB — all required data is passed directly.
    Produces a patch .ply + hidden_indices, writes a ``MemoryLayer`` row.
    Never modifies the base .ply.

    Parameters
    ----------
    scene :
        Absolute path to the base .ply file.
    edited_image_b64 :
        Base64-encoded PNG/JPEG of the AI-edited image.
    camera_dict :
        Camera parameters dict (same shape as CameraParams).
    patch_dir :
        Directory for patch .ply files.  Default: same directory as the
        source .ply.
    n_iters, sh_degree :
        Forwarded to :meth:`SplatMutator.mutate`.
    external_ref :
        Opaque reference from the caller (e.g. their proposal ID).
        Stored on the MemoryLayer row; not interpreted by the engine.

    Returns
    -------
    dict with keys: layer_id, patch_path, hidden_indices, changed_splat_count,
                    initial_loss, final_loss, iterations_run
    """
    try:
        from filelock import FileLock, Timeout
    except ImportError as exc:
        raise ImportError("filelock is required: pip install 'tmachine[api]'") from exc

    from PIL import Image as PILImage

    from ..core.splat_mutator import SplatMutator
    from ..db.models import MemoryLayer
    from ..db.session import SessionLocal

    camera = _camera_from_dict(camera_dict)

    # ── Decode edited image ───────────────────────────────────────────────
    image_bytes   = base64.b64decode(edited_image_b64)
    edited_pil    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    edited_tensor = _pil_to_tensor(
        edited_pil.resize((camera_dict["width"], camera_dict["height"]))
    )

    # ── Determine patch output path ───────────────────────────────────────
    base_dir  = Path(patch_dir) if patch_dir else Path(scene).parent
    patch_out = str(base_dir / f"patch_{uuid.uuid4().hex[:8]}.ply")
    lock_path = patch_out + ".lock"

    # ── Run SplatMutator ──────────────────────────────────────────────────
    try:
        with FileLock(lock_path, timeout=_LOCK_TIMEOUT):
            logger.info("Baking patch → %s", patch_out)
            mutator = SplatMutator(_load_scene_cached(scene))
            result  = mutator.mutate(
                camera=camera,
                edited_image=edited_tensor,
                n_iters=n_iters,
                patch_path=patch_out,
                sh_degree=sh_degree,
                on_iter=lambda i, loss_val: self.update_state(
                    state="PROGRESS",
                    meta={"iter": i, "loss": loss_val, "n_iters": n_iters},
                ),
            )
            logger.info("Bake complete: %s", result)
    except Timeout as exc:
        logger.warning("Lock timeout — retrying")
        raise self.retry(exc=exc)

    # ── Write MemoryLayer to DB ────────────────────────────────────────────
    with SessionLocal() as db:
        layer = MemoryLayer(
            scene=scene,
            patch_path=result.patch_path,
            hidden_indices=result.hidden_indices,
            changed_splat_count=result.changed_splat_count,
            initial_loss=result.initial_loss,
            final_loss=result.final_loss,
            iterations_run=result.iterations_run,
            external_ref=external_ref,
        )
        db.add(layer)
        db.commit()
        db.refresh(layer)
        layer_id = layer.id

    logger.info("MemoryLayer %d created", layer_id)
    return {
        "layer_id":            layer_id,
        "patch_path":          result.patch_path,
        "hidden_indices":      result.hidden_indices,
        "changed_splat_count": result.changed_splat_count,
        "initial_loss":        result.initial_loss,
        "final_loss":          result.final_loss,
        "iterations_run":      result.iterations_run,
    }


# ---------------------------------------------------------------------------
# Legacy tasks — direct mutation without the DB / layer workflow
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10,
                 name="mutate_from_prompt")
def mutate_from_prompt(
    self,
    scene: str,
    camera_dict: dict,
    prompt: str,
    output_path: Optional[str] = None,
    n_iters: int = 300,
    sh_degree: int = 3,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Legacy direct task — render → AI edit → SplatMutator → patch file.

    Does **not** write to the database.  For the full Memory Layers workflow
    use ``generate_preview`` + ``bake_patch``.
    """
    try:
        from filelock import FileLock, Timeout
    except ImportError as exc:
        raise ImportError("filelock is required: pip install 'tmachine[api]'") from exc

    from ..ai.image_editor import ImageEditor
    from ..core.renderer import ViewportRenderer
    from ..core.splat_mutator import SplatMutator

    camera          = _camera_from_dict(camera_dict)
    renderer        = ViewportRenderer(scene)
    original_tensor = renderer.render(camera, sh_degree=sh_degree)
    original_pil    = _tensor_to_pil(original_tensor)

    editor     = ImageEditor()
    edited_pil = editor.edit(
        image=original_pil,
        prompt=prompt,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    editor.unload()

    edited_tensor = _pil_to_tensor(edited_pil)
    dest          = output_path or str(
        Path(scene).parent / f"patch_{uuid.uuid4().hex[:8]}.ply"
    )
    lock_path = dest + ".lock"

    try:
        with FileLock(lock_path, timeout=_LOCK_TIMEOUT):
            mutator = SplatMutator(scene)
            result  = mutator.mutate(
                camera=camera,
                edited_image=edited_tensor,
                n_iters=n_iters,
                patch_path=dest,
                sh_degree=sh_degree,
            )
    except Timeout as exc:
        raise self.retry(exc=exc)

    return {
        "patch_path":          result.patch_path,
        "hidden_indices":      result.hidden_indices,
        "changed_splat_count": result.changed_splat_count,
        "initial_loss":        result.initial_loss,
        "final_loss":          result.final_loss,
        "iterations_run":      result.iterations_run,
    }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10,
                 name="mutate_from_image")
def mutate_from_image(
    self,
    scene: str,
    camera_dict: dict,
    edited_image_b64: str,
    patch_path: Optional[str],
    n_iters: int,
    sh_degree: int,
) -> dict[str, Any]:
    """
    Run Module 3 directly from a base64-encoded edited image.

    This task does **not** write to the database.  It is kept for direct
    API calls where the caller manages layer tracking externally.
    Returns patch_path + hidden_indices for the caller to store.
    """
    try:
        from filelock import FileLock, Timeout
    except ImportError as exc:
        raise ImportError("filelock is required: pip install 'tmachine[api]'") from exc

    from ..core.splat_mutator import SplatMutator

    camera = _camera_from_dict(camera_dict)

    image_bytes = base64.b64decode(edited_image_b64)
    from PIL import Image as PILImage
    edited_pil    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    edited_tensor = _pil_to_tensor(edited_pil)

    dest      = patch_path or str(
        Path(scene).parent / f"patch_{uuid.uuid4().hex[:8]}.ply"
    )
    lock_path = dest + ".lock"

    try:
        with FileLock(lock_path, timeout=_LOCK_TIMEOUT):
            mutator = SplatMutator(scene)
            result  = mutator.mutate(
                camera=camera,
                edited_image=edited_tensor,
                n_iters=n_iters,
                patch_path=dest,
                sh_degree=sh_degree,
            )
    except Timeout as exc:
        raise self.retry(exc=exc)

    return {
        "patch_path":          result.patch_path,
        "hidden_indices":      result.hidden_indices,
        "changed_splat_count": result.changed_splat_count,
        "initial_loss":        result.initial_loss,
        "final_loss":          result.final_loss,
        "iterations_run":      result.iterations_run,
    }
