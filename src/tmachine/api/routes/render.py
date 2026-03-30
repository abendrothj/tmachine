"""
tmachine/api/routes/render.py — GET /render

Renders a 2D viewport image from the scene and returns it as a PNG.

Supports layered rendering via the optional ``active_layers`` parameter.
When layer IDs are provided the endpoint:
  1. Loads each MemoryLayer row from the database.
  2. Sets the opacity of the hidden_indices in the base scene to 0.
  3. Concatenates each patch GaussianCloud onto the base.
  4. Rasterises the merged cloud normally.

This enables A/B toggling of historical memory patches on the client:
    GET /render?scene=base.ply&x=0&y=0&z=-5&active_layers=3&active_layers=7

Query Parameters
----------------
scene           str    Absolute path to the base .ply file.
x,y,z           float  Camera world-space position.
pitch           float  Tilt (radians).
yaw             float  Pan (radians).
roll            float  Bank (radians).
fov_x           float  Horizontal FoV in radians (default: π/3 ≈ 60°).
width           int    Image width in pixels (default: 1920).
height          int    Image height in pixels (default: 1080).
near            float  Near clipping plane (default: 0.01).
far             float  Far clipping plane (default: 1000.0).
active_layers   int[]  Optional list of MemoryLayer IDs to composite.
"""

from __future__ import annotations

import io
import math
import os
import threading
from collections import OrderedDict
from typing import Annotated, List, Optional

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image

from ...core.renderer import ViewportRenderer
from ...io.ply_handler import GaussianCloud, load_ply
from ...utils.camera import camera_from_fov
from ..utils import validate_scene_path

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency: parse camera + layer query params
# ---------------------------------------------------------------------------

class _RenderQuery:
    def __init__(
        self,
        scene:         str          = Query(...,                     description="Absolute path to base .ply"),
        x:             float        = Query(0.0),
        y:             float        = Query(0.0),
        z:             float        = Query(-5.0),
        pitch:         float        = Query(0.0),
        yaw:           float        = Query(0.0),
        roll:          float        = Query(0.0),
        fov_x:         float        = Query(math.radians(60),        description="Horizontal FoV in radians"),
        width:         int          = Query(1920, ge=64, le=8192),
        height:        int          = Query(1080, ge=64, le=8192),
        near:          float        = Query(0.01),
        far:           float        = Query(1000.0),
        active_layers: List[int]    = Query(default=[],             description="MemoryLayer IDs to composite"),
    ):
        self.scene         = scene
        self.x             = x
        self.y             = y
        self.z             = z
        self.pitch         = pitch
        self.yaw           = yaw
        self.roll          = roll
        self.fov_x         = fov_x
        self.width         = width
        self.height        = height
        self.near          = near
        self.far           = far
        self.active_layers = active_layers


# ---------------------------------------------------------------------------
# Renderer cache — LRU, bounded, thread-safe
# ---------------------------------------------------------------------------
# Max scenes held in GPU/CPU memory at once. Override via env var.
_CACHE_MAX_SIZE = int(os.environ.get("TMACHINE_RENDERER_CACHE_SIZE", "8"))


class _RendererCache:
    """Thread-safe LRU cache for ViewportRenderer instances."""

    def __init__(self, max_size: int = _CACHE_MAX_SIZE) -> None:
        self._cache: OrderedDict[str, ViewportRenderer] = OrderedDict()
        self._lock  = threading.Lock()
        self._max   = max_size

    def get(self, scene: str) -> ViewportRenderer:
        with self._lock:
            if scene in self._cache:
                self._cache.move_to_end(scene)
                return self._cache[scene]
            renderer = ViewportRenderer(scene)
            self._cache[scene] = renderer
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)  # evict least-recently-used
            return renderer


_renderer_cache = _RendererCache()


def _get_renderer(scene: str) -> ViewportRenderer:
    return _renderer_cache.get(scene)


# ---------------------------------------------------------------------------
# Layer compositing
# ---------------------------------------------------------------------------

def _apply_layers(
    base: GaussianCloud,
    layer_ids: List[int],
    device: str,
) -> GaussianCloud:
    """
    Composite one or more MemoryLayers onto the base GaussianCloud.

    For each layer:
      1. Zero-out the opacity of the base splats it replaces (hidden_indices).
      2. Append the patch splats.

    Returns a merged GaussianCloud ready for rasterisation.
    """
    try:
        from ...db.models import MemoryLayer
        from ...db.session import SessionLocal
    except ImportError as exc:
        raise RuntimeError(
            "Database support required for layered rendering. "
            "Install with: pip install 'tmachine[api]'"
        ) from exc

    with SessionLocal() as db:
        layers = db.query(MemoryLayer).filter(MemoryLayer.id.in_(layer_ids)).all()

    if not layers:
        return base

    # Work on a clone so we don't corrupt the cached base cloud
    merged = base.clone().to(device)

    # Collect all unique hidden indices across all layers and zero their opacity
    all_hidden: list[int] = []
    for layer in layers:
        all_hidden.extend(layer.hidden_indices)

    if all_hidden:
        hidden_t = torch.tensor(all_hidden, dtype=torch.long, device=device)
        # raw_opacity → −∞ ≈ sigmoid(−∞) = 0.0
        merged.raw_opacities = merged.raw_opacities.clone()
        merged.raw_opacities[hidden_t] = -30.0

    # Append each patch cloud
    patch_clouds: list[GaussianCloud] = []
    for layer in layers:
        patch = load_ply(layer.patch_path, device=device)
        patch_clouds.append(patch)

    if not patch_clouds:
        return merged

    # Concatenate all tensors along the splat dimension
    def _cat(attr: str) -> torch.Tensor:
        tensors = [getattr(merged, attr)] + [getattr(p, attr) for p in patch_clouds]
        return torch.cat(tensors, dim=0)

    # SH-degree normalisation — patches baked at different SH degrees have
    # sh_rest shapes like (N, 15, 3) vs (M, 0, 3).  Pad smaller clouds with
    # zeros so torch.cat doesn't raise a shape-mismatch RuntimeError.
    all_clouds = [merged, *patch_clouds]
    max_sh_rest = max(c.sh_rest.shape[1] for c in all_clouds)

    def _pad_sh_rest(t: torch.Tensor) -> torch.Tensor:
        deficit = max_sh_rest - t.shape[1]
        if deficit == 0:
            return t
        pad = torch.zeros(t.shape[0], deficit, t.shape[2], dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)

    sh_rest_cat = torch.cat([_pad_sh_rest(c.sh_rest) for c in all_clouds], dim=0)

    return GaussianCloud(
        means=_cat("means"),
        quats=_cat("quats"),
        log_scales=_cat("log_scales"),
        raw_opacities=_cat("raw_opacities"),
        sh_dc=_cat("sh_dc"),
        sh_rest=sh_rest_cat,
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get(
    "/render",
    summary="Render a 2D viewport image, optionally compositing memory layers.",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG viewport image"},
        400: {"description": "Invalid parameters"},
        500: {"description": "Rendering failed"},
    },
)
async def render_view(params: Annotated[_RenderQuery, Depends()]) -> StreamingResponse:
    """
    Render the scene from the given camera.

    If ``active_layers`` is non-empty, the base scene is composited with the
    specified memory layer patches before rasterisation, enabling real-time
    A/B preview of historical edits without modifying any file on disk.
    """
    try:
        camera = camera_from_fov(
            position=(params.x, params.y, params.z),
            pitch=params.pitch,
            yaw=params.yaw,
            roll=params.roll,
            fov_x=params.fov_x,
            width=params.width,
            height=params.height,
            near=params.near,
            far=params.far,
        )
    except (ValueError, Exception) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scene = validate_scene_path(params.scene)

    try:
        renderer = _get_renderer(scene)

        if params.active_layers:
            # Composite layers onto the base cloud, then render
            base_cloud    = renderer.gaussians
            merged_cloud  = _apply_layers(base_cloud, params.active_layers, renderer.device)
            image_tensor  = renderer.render(camera, gaussians=merged_cloud)
        else:
            image_tensor = renderer.render(camera)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Scene not found: {scene}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Render failed: {exc}") from exc

    arr     = (image_tensor.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf     = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=False)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
