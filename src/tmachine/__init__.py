"""
tmachine — agnostic 3D Gaussian Splat mutation engine.

Three-module pipeline:
    Module 1  ViewportRenderer  – .ply + camera  →  2D image
    Module 2  DeltaEngine       – original + edited  →  LossMap
    Module 3  SplatMutator      – LossMap + camera   →  updated .ply

Imports are lazy so that submodules without heavy dependencies (e.g.
tmachine.io, tmachine.utils) can be used without gsplat being installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    # Modules
    "ViewportRenderer",
    "DeltaEngine", "LossMap",
    "SplatMutator", "MutationResult",
    # Camera
    "Camera", "camera_from_euler", "camera_from_fov",
    # Scene I/O
    "GaussianCloud", "load_ply", "save_ply",
]


def __getattr__(name: str):
    """Lazy top-level imports — only load heavy dependencies when accessed."""
    if name in ("ViewportRenderer",):
        from .core.renderer import ViewportRenderer
        return ViewportRenderer
    if name in ("DeltaEngine", "LossMap"):
        from .core.delta_engine import DeltaEngine, LossMap
        return {"DeltaEngine": DeltaEngine, "LossMap": LossMap}[name]
    if name in ("SplatMutator", "MutationResult"):
        from .core.splat_mutator import SplatMutator, MutationResult
        return {"SplatMutator": SplatMutator, "MutationResult": MutationResult}[name]
    if name in ("Camera", "camera_from_euler", "camera_from_fov"):
        from .utils.camera import Camera, camera_from_euler, camera_from_fov
        return {"Camera": Camera, "camera_from_euler": camera_from_euler,
                "camera_from_fov": camera_from_fov}[name]
    if name in ("GaussianCloud", "load_ply", "save_ply"):
        from .io.ply_handler import GaussianCloud, load_ply, save_ply
        return {"GaussianCloud": GaussianCloud, "load_ply": load_ply,
                "save_ply": save_ply}[name]
    raise AttributeError(f"module 'tmachine' has no attribute {name!r}")
