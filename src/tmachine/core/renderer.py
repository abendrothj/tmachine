"""
tmachine/core/renderer.py  –  Module 1: Viewport Renderer

Responsibility
--------------
Given a 3D Gaussian Splat scene (.ply) and a camera definition, produce a
pristine 2D image.

The render is fully differentiable: gradients flow back through the
rasterisation to every Gaussian parameter.  This is the property that makes
the SplatMutator (Module 3) possible.

Backend
-------
Uses gsplat's `rasterization()` kernel, which provides:
    • CUDA-accelerated tile-based splatting
    • Differentiable alpha-compositing
    • View-dependent colour via Spherical Harmonics

Interface
---------
    renderer = ViewportRenderer("scene.ply")
    image = renderer.render(camera)            # → (H, W, 3) float tensor [0, 1]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from gsplat import rasterization

from ..io.ply_handler import GaussianCloud, load_ply
from ..utils.camera import Camera


class ViewportRenderer:
    """
    Module 1 — Viewport Renderer.

    Rasterises a GaussianCloud to a 2D image from an arbitrary camera.
    The returned tensor retains its full autograd history so back-propagation
    reaches the Gaussian parameters.

    Parameters
    ----------
    ply_path : str, optional
        If supplied, load this .ply immediately on construction.
    device : str, optional
        Torch device ('cuda' or 'cpu').  Defaults to CUDA when available.
    """

    def __init__(
        self,
        ply_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device: str = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gaussians: Optional[GaussianCloud] = None

        if ply_path is not None:
            self.load(ply_path)

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def load(self, ply_path: str) -> None:
        """Load (or reload) a Gaussian splat scene from *ply_path*."""
        self.gaussians = load_ply(ply_path, device=self.device)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        camera: Camera,
        gaussians: Optional[GaussianCloud] = None,
        sh_degree: int = 3,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        packed: bool = False,
    ) -> torch.Tensor:
        """
        Render the scene to a 2-D image.

        Parameters
        ----------
        camera :
            Camera intrinsics + extrinsics.
        gaussians :
            Cloud to render.  Falls back to the scene loaded via :meth:`load`.
        sh_degree :
            Spherical harmonics degree for view-dependent colour (0–3).
            Lower values are faster; 0 gives plain, direction-agnostic colours.
        background :
            RGB background colour, values in [0, 1].
        packed :
            Pass ``True`` to use gsplat's packed rasterisation kernel.
            Faster for scenes with high spatial sparsity (many culled Gaussians).
            Use ``False`` (default) for dense scenes or when gradients are
            needed through all Gaussian parameters.

        Returns
        -------
        torch.Tensor
            (H, W, 3) float32 image in [0, 1] on the renderer's device.
            Gradient history is preserved for back-propagation.
        """
        g = gaussians if gaussians is not None else self.gaussians
        if g is None:
            raise RuntimeError(
                "No scene loaded.  Call ViewportRenderer.load(ply_path) "
                "or pass gaussians= explicitly."
            )

        # Move all inputs to a consistent device
        dev = self.device

        # Gaussian parameters — activations happen inside these properties
        means     = g.means.to(dev)        # (N, 3)
        quats     = g.quats.to(dev)        # (N, 4)  normalised quaternions
        scales    = g.scales.to(dev)       # (N, 3)  exp(log_scales)
        opacities = g.opacities.to(dev)    # (N,)    sigmoid(raw_opacities)
        colors    = g.sh_all.to(dev)       # (N, K, 3) full SH coefficients

        # Clamp sh_degree to what's actually stored in the cloud.
        # A plain point-cloud import has sh_rest with 0 bands → degree 0.
        n_bands   = colors.shape[1]             # 1 for degree-0, 16 for degree-3
        max_deg   = int(n_bands ** 0.5) - 1    # 0, 1, 2, or 3
        sh_degree = min(sh_degree, max_deg)

        # gsplat expects batched camera inputs: (C, 4, 4) and (C, 3, 3)
        viewmat = camera.viewmat.unsqueeze(0).to(dev)   # (1, 4, 4)
        K_mat   = camera.K.unsqueeze(0).to(dev)         # (1, 3, 3)

        bg = torch.tensor(
            background, dtype=torch.float32, device=dev
        ).unsqueeze(0)  # (1, 3)

        renders, _alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K_mat,
            width=camera.width,
            height=camera.height,
            sh_degree=sh_degree,
            backgrounds=bg,
            near_plane=camera.near,
            far_plane=camera.far,
            packed=packed,
        )

        # renders: (1, H, W, 3) → (H, W, 3), clipped to [0, 1]
        return renders.squeeze(0).clamp(0.0, 1.0)
