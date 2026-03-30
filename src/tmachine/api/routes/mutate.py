"""
tmachine/api/routes/mutate.py — Mutation + Voice endpoints.

Endpoints
---------
POST /mutate/image
    Upload an already-edited image (PNG/JPEG multipart) and trigger the
    SplatMutator.  Returns a job_id immediately.

POST /mutate/prompt
    Send a text prompt; the server renders the current scene, calls
    InstructPix2Pix, and then runs the SplatMutator.  Returns a job_id.

POST /voice-edit
    Upload a voice recording (any Whisper-supported format).  The server
    runs Whisper + LLM extraction synchronously, then enqueues the full
    prompt pipeline.  Returns job_id + transcript + edit_prompt immediately.

GET /status/{job_id}
    Poll the status of any queued job.
"""

from __future__ import annotations

import base64
import json

from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..models import (
    CameraParams,
    JobResponse,
    JobStatusResponse,
    MutateImageRequest,
    MutatePromptRequest,
    VoiceJobResponse,
)
from ..worker import celery_app, mutate_from_image, mutate_from_prompt

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /mutate/image
# ---------------------------------------------------------------------------

@router.post(
    "/mutate/image",
    response_model=JobResponse,
    summary="Enqueue a splat mutation from an uploaded edited image.",
)
async def mutate_image(
    edited_image: UploadFile           = File(..., description="Edited image file (PNG or JPEG)"),
    scene:        str                  = Form(..., description="Absolute path to the source .ply"),
    camera:       str                  = Form(..., description="JSON-encoded CameraParams"),
    output_path:  str | None           = Form(None),
    n_iters:      int                  = Form(300),
    sh_degree:    int                  = Form(3),
) -> JobResponse:
    """
    Accept a pre-edited image and run Module 3 (SplatMutator) in the background.
    Use this endpoint when you already have an AI-generated image
    (produced externally) and just want to bake it into the 3D scene.
    """
    # Validate camera JSON
    try:
        camera_dict = CameraParams(**json.loads(camera)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid camera JSON: {exc}") from exc

    # Read and base64-encode the uploaded image for Celery serialisation
    if edited_image.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {edited_image.content_type}. Use PNG or JPEG.",
        )
    image_bytes   = await edited_image.read()
    image_b64     = base64.b64encode(image_bytes).decode("ascii")

    task = mutate_from_image.delay(
        scene=scene,
        camera_dict=camera_dict,
        edited_image_b64=image_b64,
        output_path=output_path,
        n_iters=n_iters,
        sh_degree=sh_degree,
    )
    return JobResponse(job_id=task.id)


# ---------------------------------------------------------------------------
# POST /mutate/prompt
# ---------------------------------------------------------------------------

@router.post(
    "/mutate/prompt",
    response_model=JobResponse,
    summary="Enqueue the full pipeline: render → AI edit → mutate.",
)
async def mutate_prompt(req: MutatePromptRequest) -> JobResponse:
    """
    Accepts a text prompt and runs the complete pipeline in the background:

    1. Render current .ply from camera  (Module 1)
    2. Edit render with InstructPix2Pix (AI)
    3. Compute delta                    (Module 2)
    4. Back-prop mutation               (Module 3)
    5. Save updated .ply
    """
    task = mutate_from_prompt.delay(
        scene=req.scene,
        camera_dict=req.camera.model_dump(),
        prompt=req.prompt,
        output_path=req.output_path,
        n_iters=req.n_iters,
        sh_degree=req.sh_degree,
        image_guidance_scale=req.image_guidance_scale,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
    )
    return JobResponse(job_id=task.id)


# ---------------------------------------------------------------------------
# POST /voice-edit
# ---------------------------------------------------------------------------

@router.post(
    "/voice-edit",
    response_model=VoiceJobResponse,
    summary="Transcribe a voice memo and enqueue the full edit pipeline.",
)
async def voice_edit(
    audio:       UploadFile  = File(...,  description="Voice recording (mp3, m4a, wav, webm)"),
    scene:       str         = Form(...,  description="Absolute path to the source .ply"),
    camera:      str         = Form(...,  description="JSON-encoded CameraParams"),
    output_path: str | None  = Form(None),
    n_iters:     int         = Form(300),
    sh_degree:   int         = Form(3),
) -> VoiceJobResponse:
    """
    Full "Tap and Talk" pipeline:

    1. Whisper transcribes the uploaded audio  (fast, done synchronously)
    2. LLM extracts a clean edit instruction   (fast, done synchronously)
    3. Returns transcript + edit_prompt + job_id immediately
    4. Full render → AI edit → mutate runs in the background

    The client receives the edit_prompt so it can display
    "I heard: change the awning to dark hunter green" before the 3D update
    is complete.
    """
    try:
        from ...ai.voice_pipeline import VoicePipeline
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Voice pipeline not installed. Run: pip install 'tmachine[ai-voice]'",
        ) from exc

    # Validate camera JSON
    try:
        camera_dict = CameraParams(**json.loads(camera)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid camera JSON: {exc}") from exc

    # Determine audio suffix for the temp file so Whisper picks the right decoder
    content_type_to_suffix: dict[str, str] = {
        "audio/mpeg":   ".mp3",
        "audio/mp4":    ".m4a",
        "audio/wav":    ".wav",
        "audio/x-wav":  ".wav",
        "audio/webm":   ".webm",
        "audio/ogg":    ".ogg",
        "video/webm":   ".webm",  # browsers often send WebM for getUserMedia recordings
    }
    suffix = content_type_to_suffix.get(audio.content_type or "", ".wav")

    audio_bytes = await audio.read()

    # Stages 1 + 2: STT + LLM (synchronous, fast)
    pipeline    = VoicePipeline()
    voice_result = pipeline.process_bytes(audio_bytes, suffix=suffix)

    # Stage 3: Enqueue full pipeline in background
    task = mutate_from_prompt.delay(
        scene=scene,
        camera_dict=camera_dict,
        prompt=voice_result.edit_prompt,
        output_path=output_path,
        n_iters=n_iters,
        sh_degree=sh_degree,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
        seed=None,
    )

    return VoiceJobResponse(
        job_id=task.id,
        transcript=voice_result.transcript,
        edit_prompt=voice_result.edit_prompt,
        llm_used=voice_result.llm_used,
    )


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll the status of a queued mutation job.",
)
async def job_status(job_id: str) -> JobStatusResponse:
    """
    Returns the current status of a job.

    Possible statuses:
    - ``PENDING``  — queued, not yet started
    - ``STARTED``  — worker has picked it up
    - ``SUCCESS``  — completed; ``result`` contains the MutationResult dict
    - ``FAILURE``  — an unrecoverable error occurred; ``error`` has details
    - ``RETRY``    — transient failure (e.g. file-lock timeout), retrying
    """
    async_result = AsyncResult(job_id, app=celery_app)
    status = async_result.status  # PENDING | STARTED | SUCCESS | FAILURE | RETRY

    result_payload = None
    error_msg      = None

    if status == "SUCCESS":
        result_payload = async_result.result
    elif status == "FAILURE":
        error_msg = str(async_result.result)

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        result=result_payload,
        error=error_msg,
    )
