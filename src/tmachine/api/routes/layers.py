"""
tmachine/api/routes/layers.py — Preview generation and Memory Layer endpoints.

Endpoints
---------
POST  /previews/generate       Enqueue Stage 1 (render → AI edit → save preview PNG)
POST  /previews/voice          Voice recording → Whisper → LLM → Stage 1
GET   /previews/{filename}     Serve a saved preview PNG from PREVIEW_DIR
POST  /layers/bake             Enqueue Stage 2 (bake edited image → patch .ply → MemoryLayer)
GET   /layers                  List MemoryLayers for a scene
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse

from ...db import MemoryLayer, get_db
from ..models import JobResponse, VoiceJobResponse
from ..utils import validate_scene_path
from ..worker import (
    bake_patch,
    generate_preview,
    celery_app,
)

try:
    from celery.result import AsyncResult as _AsyncResult
except ImportError:  # pragma: no cover
    _AsyncResult = None  # type: ignore[assignment,misc]

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

router = APIRouter()

_PREVIEW_DIR = Path(os.environ.get("PREVIEW_DIR", "./previews"))


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class LayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:                  int
    scene:               str
    patch_path:          str
    hidden_indices:      list
    changed_splat_count: int
    initial_loss:        float
    final_loss:          float
    external_ref:        Optional[str]


# ---------------------------------------------------------------------------
# POST /previews/generate  —  Stage 1 via text prompt
# ---------------------------------------------------------------------------

@router.post(
    "/previews/generate",
    response_model=JobResponse,
    summary="Generate a 2D AI-edited preview from a text prompt.",
)
async def generate_preview_from_prompt(
    scene:                str   = Form(...),
    camera:               str   = Form(..., description="JSON-encoded CameraParams"),
    prompt:               str   = Form(...),
    image_guidance_scale: float = Form(1.5),
    guidance_scale:       float = Form(7.5),
    seed:                 Optional[int] = Form(None),
    sh_degree:            int   = Form(3),
) -> JobResponse:
    """
    Enqueue Stage 1: render → InstructPix2Pix edit → save preview PNG.
    Returns a job_id; poll /status/{job_id} to get preview_filename when done.
    The caller is responsible for any proposal/approval workflow.
    """
    scene = validate_scene_path(scene)
    from ..models import CameraParams
    try:
        camera_dict = CameraParams(**json.loads(camera)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid camera JSON: {exc}") from exc

    task = generate_preview.delay(
        scene=scene,
        camera_dict=camera_dict,
        prompt=prompt,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        seed=seed,
        sh_degree=sh_degree,
    )
    return JobResponse(job_id=task.id)


# ---------------------------------------------------------------------------
# POST /previews/voice  —  Stage 1 via voice recording
# ---------------------------------------------------------------------------

@router.post(
    "/previews/voice",
    response_model=VoiceJobResponse,
    summary="Generate a preview from a spoken edit instruction.",
)
async def generate_preview_from_voice(
    audio:     UploadFile       = File(...),
    scene:     str              = Form(...),
    camera:    str              = Form(...),
    sh_degree: int              = Form(3),
) -> VoiceJobResponse:
    """
    Whisper transcribes the audio synchronously, LLM extracts the edit prompt,
    then Stage 1 is enqueued in the background.
    Returns immediately with transcript + edit_prompt + job_id.
    """
    try:
        from ...ai.voice_pipeline import VoicePipeline
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Voice pipeline not installed") from exc

    from ..models import CameraParams

    try:
        camera_dict = CameraParams(**json.loads(camera)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid camera JSON: {exc}") from exc

    content_type_to_suffix = {
        "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/webm": ".webm", "audio/ogg": ".ogg",
        "video/webm": ".webm",
    }
    suffix      = content_type_to_suffix.get(audio.content_type or "", ".wav")
    audio_bytes = await audio.read()

    pipeline     = VoicePipeline()
    voice_result = pipeline.process_bytes(audio_bytes, suffix=suffix)

    task = generate_preview.delay(
        scene=scene,
        camera_dict=camera_dict,
        prompt=voice_result.edit_prompt,
        sh_degree=sh_degree,
    )

    return VoiceJobResponse(
        job_id=task.id,
        transcript=voice_result.transcript,
        edit_prompt=voice_result.edit_prompt,
        llm_used=voice_result.llm_used,
    )


# ---------------------------------------------------------------------------
# GET /previews/{filename}  —  serve a saved preview PNG
# ---------------------------------------------------------------------------

@router.get(
    "/previews/{filename}",
    summary="Serve a saved 2D preview PNG.",
    responses={200: {"content": {"image/png": {}}}},
)
async def serve_preview(filename: str) -> Response:
    """
    Serve a preview PNG that was saved by ``generate_preview``.
    ``filename`` must be a bare filename (no path separators).
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _PREVIEW_DIR / filename
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Preview not found")
    return Response(content=data, media_type="image/png")


# ---------------------------------------------------------------------------
# POST /layers/bake  —  Stage 2 from an uploaded edited image
# ---------------------------------------------------------------------------

@router.post(
    "/layers/bake",
    response_model=JobResponse,
    summary="Bake an edited image into the 3D scene as a new MemoryLayer.",
)
async def bake_layer(
    scene:        str              = Form(...),
    camera:       str              = Form(..., description="JSON-encoded CameraParams"),
    edited_image: Optional[UploadFile] = File(None, description="Edited image (PNG/JPEG)"),
    preview_path: Optional[str]    = Form(None, description="Absolute path to a saved preview PNG"),
    external_ref: Optional[str]    = Form(None, description="Caller-supplied reference (e.g. proposal ID)"),
    n_iters:      int              = Form(300),
    sh_degree:    int              = Form(3),
    patch_dir:    Optional[str]    = Form(None),
) -> JobResponse:
    """
    Enqueue Stage 2: bake an edited image into the 3D scene.

    Supply either ``edited_image`` (uploaded file) **or** ``preview_path``
    (absolute path to a file already on the engine's filesystem).
    """
    if edited_image is None and not preview_path:
        raise HTTPException(
            status_code=422,
            detail="Supply either 'edited_image' or 'preview_path'.",
        )

    scene = validate_scene_path(scene)

    from ..models import CameraParams
    try:
        camera_dict = CameraParams(**json.loads(camera)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid camera JSON: {exc}") from exc

    # Read image bytes from upload or from disk
    if edited_image is not None:
        if edited_image.content_type not in ("image/png", "image/jpeg", "image/webp"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image type: {edited_image.content_type}",
            )
        image_bytes = await edited_image.read()
    else:
        try:
            image_bytes = Path(preview_path).read_bytes()  # type: ignore[arg-type]
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="preview_path not found on disk")

    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    task = bake_patch.delay(
        scene=scene,
        edited_image_b64=image_b64,
        camera_dict=camera_dict,
        patch_dir=patch_dir,
        n_iters=n_iters,
        sh_degree=sh_degree,
        external_ref=external_ref,
    )
    return JobResponse(job_id=task.id)


# ---------------------------------------------------------------------------
# GET /layers  —  list MemoryLayers for a scene
# ---------------------------------------------------------------------------

@router.get(
    "/layers",
    response_model=list[LayerOut],
    summary="List all baked memory layers for a scene.",
)
async def list_layers(
    scene: str = Query(..., description="Filter by base .ply path"),
    db:    Session = Depends(get_db),
) -> list[LayerOut]:
    return (
        db.query(MemoryLayer)
        .filter(MemoryLayer.scene == scene)
        .order_by(MemoryLayer.created_at.desc())
        .all()
    )
