"""
tmachine/api/worker.py — Celery application + task definitions.

Architecture — Memory Layer pipeline
---------------------------------------
The pipeline is now split into two explicit stages with human review between them:

Stage 1 ── generate_memory_proposal  (fast, ~20-60 s)
    • Render current scene (Module 1)
    • Edit rendered image with InstructPix2Pix (or accept uploaded image)
    • Save the 2D preview PNG to disk
    • Write a MemoryProposal row (status=PENDING) to the database
    • Return immediately — no 3D modification yet

    ══════════════════════════════
    ←── HUMAN REVIEW / VOTING ──►
    ══════════════════════════════

Stage 2 ── bake_approved_patch  (slow, ~2-5 min)
    • Triggered only after a proposal is approved
    • Runs SplatMutator.mutate() → produces a patch .ply + hidden_indices
    • Writes a MemoryLayer row to the database
    • Never touches the base .ply file

File-lock safety
----------------
Only Stage 2 acquires a FileLock (on the patch output path) since the base
scene is never written — each patch is a brand-new file.

Environment variables
---------------------
REDIS_URL         – Redis connection string.  Default: redis://localhost:6379/0
DATABASE_URL      – PostgreSQL DSN.  Default: postgresql+psycopg2://...
LOCK_TIMEOUT      – Max seconds to wait for a file lock.  Default: 300
PREVIEW_DIR       – Directory for storing 2D preview PNGs.  Default: ./previews
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

_REDIS_URL    = os.environ.get("REDIS_URL",    "redis://localhost:6379/0")
_LOCK_TIMEOUT = int(os.environ.get("LOCK_TIMEOUT", "300"))
_PREVIEW_DIR  = Path(os.environ.get("PREVIEW_DIR", "./previews"))

celery_app = Celery(
    "tmachine",
    broker=_REDIS_URL,
    backend=_REDIS_URL,
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


def _save_preview(pil_img, proposal_id: int) -> str:
    """Save a PIL image as the preview PNG for a proposal; return its path."""
    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = _PREVIEW_DIR / f"proposal_{proposal_id}.png"
    pil_img.save(str(path), format="PNG")
    return str(path)


# ---------------------------------------------------------------------------
# Stage 1 — generate_memory_proposal
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10,
                 name="generate_memory_proposal")
def generate_memory_proposal(
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

    **No 3D modification.  No patch file.  No SplatMutator.**

    Creates one ``MemoryProposal`` row (status=PENDING) and returns its ID.
    The frontend can then display the preview at ``GET /proposals/{id}/preview``
    and collect votes before anything touches the 3D scene.

    Returns
    -------
    dict with keys: proposal_id, preview_path, prompt
    """
    from ..ai.image_editor import ImageEditor
    from ..core.renderer import ViewportRenderer
    from ..db.models import MemoryProposal, ProposalStatus
    from ..db.session import SessionLocal

    camera = _camera_from_dict(camera_dict)

    # ── Module 1: Render current base scene ───────────────────────────────
    logger.info("Rendering scene for proposal: %r", prompt)
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

    # ── Persist proposal to DB (get ID first) ─────────────────────────────
    with SessionLocal() as db:
        proposal = MemoryProposal(
            scene=scene,
            prompt=prompt,
            preview_path="",           # filled in after we know the ID
            status=ProposalStatus.PENDING,
            cam_x=camera_dict["x"],
            cam_y=camera_dict["y"],
            cam_z=camera_dict["z"],
            cam_pitch=camera_dict["pitch"],
            cam_yaw=camera_dict["yaw"],
            cam_roll=camera_dict["roll"],
            cam_fov_x=camera_dict["fov_x"],
            cam_width=camera_dict["width"],
            cam_height=camera_dict["height"],
        )
        db.add(proposal)
        db.commit()
        db.refresh(proposal)
        proposal_id = proposal.id

        # Save preview PNG using the DB-assigned ID
        preview_path = _save_preview(edited_pil, proposal_id)
        proposal.preview_path = preview_path
        db.commit()

    logger.info("Proposal %d created: %s", proposal_id, preview_path)
    return {"proposal_id": proposal_id, "preview_path": preview_path, "prompt": prompt}


# ---------------------------------------------------------------------------
# Stage 2 — bake_approved_patch
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10,
                 name="bake_approved_patch")
def bake_approved_patch(
    self,
    proposal_id: int,
    patch_dir: Optional[str] = None,
    n_iters: int = 300,
    sh_degree: int = 3,
) -> dict[str, Any]:
    """
    Stage 2 — Run the SplatMutator for an approved proposal and save the patch.

    Reads all required data (scene, camera, preview image) from the DB row.
    Produces a patch .ply + hidden_indices, writes a ``MemoryLayer`` row.
    Never modifies the base .ply.

    Parameters
    ----------
    proposal_id :
        Primary key of the approved ``MemoryProposal``.
    patch_dir :
        Directory for patch .ply files.  Default: same directory as the
        source .ply.
    n_iters, sh_degree :
        Forwarded to :meth:`SplatMutator.mutate`.

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
    from ..db.models import MemoryLayer, MemoryProposal, ProposalStatus
    from ..db.session import SessionLocal

    # ── Load proposal from DB ─────────────────────────────────────────────
    with SessionLocal() as db:
        proposal = db.get(MemoryProposal, proposal_id)
        if proposal is None:
            raise ValueError(f"MemoryProposal {proposal_id} not found")
        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError(
                f"Proposal {proposal_id} is not APPROVED (status={proposal.status})"
            )
        # Capture all values while session is open
        scene        = proposal.scene
        preview_path = proposal.preview_path
        cam_dict     = {
            "x":      proposal.cam_x,
            "y":      proposal.cam_y,
            "z":      proposal.cam_z,
            "pitch":  proposal.cam_pitch,
            "yaw":    proposal.cam_yaw,
            "roll":   proposal.cam_roll,
            "fov_x":  proposal.cam_fov_x,
            "width":  proposal.cam_width,
            "height": proposal.cam_height,
            "near":   0.01,
            "far":    1000.0,
        }

    camera       = _camera_from_dict(cam_dict)
    edited_pil   = PILImage.open(preview_path).convert("RGB")
    edited_tensor = _pil_to_tensor(
        edited_pil.resize((cam_dict["width"], cam_dict["height"]))
    )

    # ── Determine patch output path ───────────────────────────────────────
    base_dir  = Path(patch_dir) if patch_dir else Path(scene).parent
    patch_out = str(base_dir / f"patch_{proposal_id}_{uuid.uuid4().hex[:8]}.ply")
    lock_path = patch_out + ".lock"

    # ── Run SplatMutator ──────────────────────────────────────────────────
    try:
        with FileLock(lock_path, timeout=_LOCK_TIMEOUT):
            logger.info("Baking patch for proposal %d → %s", proposal_id, patch_out)
            mutator = SplatMutator(scene)
            result  = mutator.mutate(
                camera=camera,
                edited_image=edited_tensor,
                n_iters=n_iters,
                patch_path=patch_out,
                sh_degree=sh_degree,
            )
            logger.info("Bake complete: %s", result)
    except Timeout as exc:
        logger.warning("Lock timeout for proposal %d — retrying", proposal_id)
        raise self.retry(exc=exc)

    # ── Write MemoryLayer to DB ────────────────────────────────────────────
    with SessionLocal() as db:
        layer = MemoryLayer(
            proposal_id=proposal_id,
            scene=scene,
            patch_path=result.patch_path,
            hidden_indices=result.hidden_indices,
            changed_splat_count=result.changed_splat_count,
            initial_loss=result.initial_loss,
            final_loss=result.final_loss,
            iterations_run=result.iterations_run,
        )
        db.add(layer)
        db.commit()
        db.refresh(layer)
        layer_id = layer.id

    logger.info("MemoryLayer %d created for proposal %d", layer_id, proposal_id)
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
# Legacy tasks — direct mutation without the DB / proposal workflow
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
    use ``generate_memory_proposal`` + ``bake_approved_patch``.
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
    API calls where the caller manages proposal tracking externally.
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
