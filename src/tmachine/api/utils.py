"""
tmachine/api/utils.py — Shared API utilities.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Scene-path validation
# ---------------------------------------------------------------------------
# Set TMACHINE_SCENE_ROOT to an absolute directory to restrict which .ply
# files the API may open.  When unset, all absolute paths are accepted
# (suitable for single-machine dev; should always be set in production).

_SCENE_ROOT: str | None = os.environ.get("TMACHINE_SCENE_ROOT") or None


def validate_scene_path(scene: str) -> str:
    """
    Resolve *scene* and, when ``TMACHINE_SCENE_ROOT`` is configured, confirm
    it lives within that directory.

    Relative paths are resolved against ``TMACHINE_SCENE_ROOT`` when set,
    making ``livingroom.ply`` equivalent to ``<root>/livingroom.ply``.

    Raises ``HTTPException(400)`` on any suspicious path.
    Returns the resolved absolute path string.
    """
    try:
        p = Path(scene)
        if _SCENE_ROOT is not None and not p.is_absolute():
            p = Path(_SCENE_ROOT) / p
        resolved = p.resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid scene path: {exc}") from exc

    if _SCENE_ROOT is not None:
        allowed = Path(_SCENE_ROOT).resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Scene path is outside the allowed scene root. "
                    "Set TMACHINE_SCENE_ROOT to permit this path."
                ),
            )

    return str(resolved)
