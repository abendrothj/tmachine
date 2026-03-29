"""
tmachine/api/models.py — Pydantic request and response schemas.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class CameraParams(BaseModel):
    """
    Complete camera specification sent by the client.
    All angles are in radians.
    """
    x:      float = Field(0.0,                    description="Camera X position (world space)")
    y:      float = Field(0.0,                    description="Camera Y position (world space)")
    z:      float = Field(-5.0,                   description="Camera Z position (world space)")
    pitch:  float = Field(0.0,                    description="Tilt angle (radians)")
    yaw:    float = Field(0.0,                    description="Pan angle (radians)")
    roll:   float = Field(0.0,                    description="Bank angle (radians)")
    fov_x:  float = Field(math.radians(60),       description="Horizontal FoV (radians)")
    width:  int   = Field(1920, ge=64, le=8192,   description="Output image width (pixels)")
    height: int   = Field(1080, ge=64, le=8192,   description="Output image height (pixels)")
    near:   float = Field(0.01,                   description="Near clipping plane")
    far:    float = Field(1000.0,                 description="Far clipping plane")

    @field_validator("fov_x")
    @classmethod
    def fov_range(cls, v: float) -> float:
        if not (0.01 < v < math.pi):
            raise ValueError("fov_x must be between 0.01 and π radians")
        return v


# ---------------------------------------------------------------------------
# Mutation via pre-edited image (Module 2 + Module 3 only)
# ---------------------------------------------------------------------------

class MutateImageRequest(BaseModel):
    """
    Trigger a splat mutation from an already-edited image.
    The edited image is uploaded as multipart form data separately;
    this model carries the metadata fields in the same form.
    """
    scene:       str            = Field(..., description="Absolute path to the source .ply file")
    camera:      CameraParams
    output_path: Optional[str]  = Field(None, description="Destination .ply path (default: overwrite source)")
    n_iters:     int            = Field(300, ge=1, le=5000)
    sh_degree:   int            = Field(3,   ge=0, le=3)


# ---------------------------------------------------------------------------
# Mutation via text prompt (Module 1 + AI edit + Module 2 + Module 3)
# ---------------------------------------------------------------------------

class MutatePromptRequest(BaseModel):
    """
    Trigger the full pipeline: render → AI edit → backprop → save .ply.
    """
    scene:       str            = Field(..., description="Absolute path to the source .ply file")
    camera:      CameraParams
    prompt:      str            = Field(..., min_length=3, max_length=500,
                                        description="Edit instruction, e.g. 'change the awning to dark hunter green'")
    output_path: Optional[str]  = Field(None)
    n_iters:     int            = Field(300, ge=1, le=5000)
    sh_degree:   int            = Field(3,   ge=0, le=3)
    image_guidance_scale: float = Field(1.5, ge=1.0, le=2.5)
    guidance_scale:       float = Field(7.5, ge=1.0, le=15.0)
    seed:        Optional[int]  = Field(None)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    """Returned immediately when a mutation job is enqueued."""
    job_id: str
    status: str = "PENDING"


class JobStatusResponse(BaseModel):
    """Returned by GET /status/{job_id}."""
    job_id:  str
    status:  str   # PENDING | STARTED | SUCCESS | FAILURE | RETRY | REVOKED
    result:  Optional[dict[str, Any]] = None
    error:   Optional[str]            = None


class VoiceJobResponse(BaseModel):
    """
    Returned immediately after the voice endpoint processes the transcript —
    before the mutation job completes.
    """
    job_id:      str
    status:      str = "PENDING"
    transcript:  str
    edit_prompt: str
    llm_used:    bool
