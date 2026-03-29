"""
tmachine/api/routes/proposals.py — Proposal review and layer management endpoints.

Endpoints
---------
POST  /proposals/prompt        Enqueue Stage 1 (render → AI edit → save proposal)
POST  /proposals/voice         Voice recording → Whisper → LLM → Stage 1
GET   /proposals               List PENDING proposals for a scene
GET   /proposals/{id}/preview  Serve the 2D preview PNG
POST  /proposals/{id}/vote     Cast a yes / no vote
POST  /proposals/{id}/approve  Mark approved + enqueue bake_approved_patch
POST  /proposals/{id}/reject   Mark rejected
GET   /layers                  List active MemoryLayers for a scene
GET   /status/{job_id}         Poll any Celery job
"""

from __future__ import annotations

import io
import json
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from ...db import MemoryLayer, MemoryProposal, ProposalStatus, get_db
from ..models import JobResponse, JobStatusResponse, VoiceJobResponse
from ..utils import validate_scene_path
from ..worker import (
    bake_approved_patch,
    celery_app,
    generate_memory_proposal,
    mutate_from_image,
)
try:
    from celery.result import AsyncResult as _AsyncResult  # module-level for test patching
except ImportError:  # pragma: no cover
    _AsyncResult = None  # type: ignore[assignment,misc]

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic / response schemas (inline to keep the file self-contained)
# ---------------------------------------------------------------------------

from pydantic import BaseModel, ConfigDict


class ProposalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:           int
    scene:        str
    prompt:       str
    status:       str
    votes_yes:    int
    votes_no:     int
    preview_path: str
    bake_job_id:  Optional[str]


class LayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:                  int
    proposal_id:         int
    scene:               str
    patch_path:          str
    hidden_indices:      list
    changed_splat_count: int
    initial_loss:        float
    final_loss:          float


# ---------------------------------------------------------------------------
# POST /proposals/prompt  —  Stage 1 via text prompt
# ---------------------------------------------------------------------------

@router.post(
    "/proposals/prompt",
    response_model=JobResponse,
    summary="Generate a memory proposal from a text prompt.",
)
async def propose_from_prompt(
    scene:                str   = Form(...),
    camera:               str   = Form(..., description="JSON-encoded CameraParams"),
    prompt:               str   = Form(...),
    image_guidance_scale: float = Form(1.5),
    guidance_scale:       float = Form(7.5),
    seed:                 Optional[int] = Form(None),
    sh_degree:            int   = Form(3),
) -> JobResponse:
    """
    Enqueue Stage 1: render → InstructPix2Pix edit → save PENDING proposal.
    Returns a job_id; poll /status/{job_id} to get the proposal_id when done.
    """
    scene = validate_scene_path(scene)
    from ..models import CameraParams
    try:
        camera_dict = CameraParams(**json.loads(camera)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid camera JSON: {exc}") from exc

    task = generate_memory_proposal.delay(
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
# POST /proposals/voice  —  Stage 1 via voice recording
# ---------------------------------------------------------------------------

@router.post(
    "/proposals/voice",
    response_model=VoiceJobResponse,
    summary="Generate a memory proposal from a spoken memory recording.",
)
async def propose_from_voice(
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

    task = generate_memory_proposal.delay(
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
# GET /proposals  —  list pending proposals for a scene
# ---------------------------------------------------------------------------

@router.get(
    "/proposals",
    response_model=list[ProposalOut],
    summary="List proposals for review.",
)
async def list_proposals(
    scene:  str = Query(..., description="Filter by base .ply path"),
    status: str = Query("PENDING", description="PENDING | APPROVED | REJECTED | ALL"),
    db:     Session = Depends(get_db),
) -> list[ProposalOut]:
    q = db.query(MemoryProposal).filter(MemoryProposal.scene == scene)
    if status != "ALL":
        try:
            q = q.filter(MemoryProposal.status == ProposalStatus(status))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown status: {status}")
    return q.order_by(MemoryProposal.created_at.desc()).all()


# ---------------------------------------------------------------------------
# GET /proposals/{id}/preview  —  serve the 2D preview image
# ---------------------------------------------------------------------------

@router.get(
    "/proposals/{proposal_id}/preview",
    summary="Download the AI-edited 2D preview image for a proposal.",
    responses={200: {"content": {"image/png": {}}}},
)
async def proposal_preview(
    proposal_id: int,
    db:          Session = Depends(get_db),
) -> Response:
    proposal = db.get(MemoryProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    try:
        with open(proposal.preview_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Preview image file missing on disk")
    return Response(content=data, media_type="image/png")


# ---------------------------------------------------------------------------
# POST /proposals/{id}/vote
# ---------------------------------------------------------------------------

@router.post(
    "/proposals/{proposal_id}/vote",
    summary="Cast a vote (yes or no) on a pending proposal.",
)
async def vote_on_proposal(
    proposal_id: int,
    vote:        str = Form(..., description="'yes' or 'no'"),
    db:          Session = Depends(get_db),
) -> dict:
    if vote not in ("yes", "no"):
        raise HTTPException(status_code=400, detail="vote must be 'yes' or 'no'")

    proposal = db.get(MemoryProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if proposal.status != ProposalStatus.PENDING:
        raise HTTPException(status_code=409, detail="Proposal is no longer pending")

    if vote == "yes":
        proposal.votes_yes += 1
    else:
        proposal.votes_no += 1
    db.commit()

    return {"proposal_id": proposal_id, "votes_yes": proposal.votes_yes,
            "votes_no": proposal.votes_no}


# ---------------------------------------------------------------------------
# POST /proposals/{id}/approve  —  approve + enqueue Stage 2
# ---------------------------------------------------------------------------

@router.post(
    "/proposals/{proposal_id}/approve",
    response_model=JobResponse,
    summary="Approve a proposal and enqueue the 3D patch bake.",
)
async def approve_proposal(
    proposal_id: int,
    n_iters:     int = Form(300),
    patch_dir:   Optional[str] = Form(None),
    db:          Session = Depends(get_db),
) -> JobResponse:
    """
    Marks the proposal APPROVED and enqueues ``bake_approved_patch`` (Stage 2).
    The bake task runs the SplatMutator and writes a MemoryLayer row when done.
    """
    proposal = db.get(MemoryProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if proposal.status != ProposalStatus.PENDING:
        raise HTTPException(status_code=409,
                            detail=f"Proposal is already {proposal.status.value}")

    proposal.status = ProposalStatus.APPROVED
    task = bake_approved_patch.delay(
        proposal_id=proposal_id,
        patch_dir=patch_dir,
        n_iters=n_iters,
    )
    proposal.bake_job_id = task.id
    db.commit()

    return JobResponse(job_id=task.id)


# ---------------------------------------------------------------------------
# POST /proposals/{id}/reject
# ---------------------------------------------------------------------------

@router.post(
    "/proposals/{proposal_id}/reject",
    summary="Reject a proposal. No 3D changes will be made.",
)
async def reject_proposal(
    proposal_id: int,
    db:          Session = Depends(get_db),
) -> dict:
    proposal = db.get(MemoryProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if proposal.status != ProposalStatus.PENDING:
        raise HTTPException(status_code=409,
                            detail=f"Proposal is already {proposal.status.value}")

    proposal.status = ProposalStatus.REJECTED
    db.commit()
    return {"proposal_id": proposal_id, "status": "REJECTED"}


# ---------------------------------------------------------------------------
# GET /layers  —  list active memory layers for a scene
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


# ---------------------------------------------------------------------------
# GET /status/{job_id}  —  poll Celery job status
# ---------------------------------------------------------------------------

@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll the status of any queued job (proposal generation or patch bake).",
)
async def job_status(job_id: str) -> JobStatusResponse:
    async_result = _AsyncResult(job_id, app=celery_app)
    status       = async_result.status
    result_payload = None
    error_msg      = None
    if status == "SUCCESS":
        result_payload = async_result.result
    elif status == "FAILURE":
        error_msg = str(async_result.result)
    return JobStatusResponse(
        job_id=job_id, status=status,
        result=result_payload, error=error_msg,
    )
