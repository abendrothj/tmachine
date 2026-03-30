"""
tmachine/core/splat_mutator.py  –  Module 3: Splat Mutator

Responsibility
--------------
The hardest and most valuable part of the engine.

Given a camera viewpoint and a target (AI-edited) image, run a differentiable
optimisation loop that adjusts the 3D Gaussian parameters until the rendered
view matches the target.

Patch / Memory-Layer mode (default)
-------------------------------------
Instead of overwriting the base scene, the mutator extracts only the splats
that actually changed and saves a tiny "patch" .ply containing just those
splats.  Alongside the patch file, it records the *hidden_indices* — the
row indices of the corresponding original splats that should be made invisible
when the patch is composited on top of the base scene.

This is the "Git-style diff" model:

    base.ply  (never touched)
    └── patch_abc.ply   ← only changed splats
        └── hidden_indices: [1042, 8374, …]  ← suppress these in base

At render time the ViewportRenderer loads both, sets the hidden-index opacities
to 0, concatenates the patch cloud, and rasterises the merged scene.

How it works (the back-prop loop)
----------------------------------
1. Clone the source GaussianCloud so the original is never mutated in-place.
2. Expose the appearance parameters (SH / opacity) as differentiable leaves.
3. Every iteration: Forward → Loss → Backward → Adam step.
4. After convergence:
       a. Compute per-splat SH-DC delta magnitude between optimised and source.
       b. Threshold to find ``changed_indices`` (splats that moved).
       c. Extract those rows into a tiny ``GaussianCloud`` — the patch.
       d. Save the patch .ply.
       e. Return ``MutationResult`` with ``patch_path`` + ``hidden_indices``.

Interface
---------
    # From a path (loads from disk once):
    mutator = SplatMutator("scene.ply")

    # From an already-loaded GaussianCloud (skips disk I/O):
    cloud   = load_ply("scene.ply")
    mutator = SplatMutator(cloud)

    result  = mutator.mutate(camera, edited_image, n_iters=300,
                             patch_path="patches/patch_abc.ply")
    # result.patch_path       — path to the saved patch .ply
    # result.hidden_indices   — list[int] of base-scene splat indices to hide
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.optim as optim

from ..io.ply_handler import GaussianCloud, load_ply, save_ply
from ..utils.camera import Camera
from .delta_engine import DeltaEngine
from .renderer import ViewportRenderer


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MutationResult:
    """
    Summary of a completed optimisation run.

    Attributes
    ----------
    iterations_run : int
        Number of gradient steps taken.
    initial_loss : float
        Total loss at iteration 0.
    final_loss : float
        Total loss at the last iteration.
    patch_path : str
        Path to the saved patch .ply (contains only the changed splats).
    hidden_indices : list[int]
        Row indices of the *base scene* splats that the patch replaces.
        At render time, set their opacity to 0 and concatenate the patch.
    changed_splat_count : int
        Number of splats included in the patch.
    loss_history : list[float]
        Per-iteration total loss values.
    """

    iterations_run:       int
    initial_loss:         float
    final_loss:           float
    patch_path:           str
    hidden_indices:       List[int]
    changed_splat_count:  int
    loss_history:         list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"MutationResult("
            f"iters={self.iterations_run}, "
            f"loss {self.initial_loss:.5f} → {self.final_loss:.5f}, "
            f"patch='{self.patch_path}', "
            f"changed_splats={self.changed_splat_count})"
        )


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------

class SplatMutator:
    """
    Module 3 — Splat Mutator (Back-prop).

    Parameters
    ----------
    scene : str | GaussianCloud
        Source (base) scene.  Pass a ``str`` path to load from disk, or pass
        a :class:`~tmachine.io.ply_handler.GaussianCloud` that is already in
        memory to skip the disk read (useful when running multiple mutations on
        the same scene in a single process).
    device : str, optional
        Torch device.  Defaults to CUDA when available.
    lr_sh : float
        Adam learning rate for Spherical Harmonic colour coefficients.
    lr_opacity : float
        Adam learning rate for raw opacity logits.
    lr_geometry : float
        Adam learning rate for position / scale / rotation when
        ``optimize_geometry=True`` is passed to :meth:`mutate`.
    scheduler_gamma : float
        Exponential LR decay factor applied every iteration.
        Default 0.995 — halves the LR roughly every 1 000 steps.
    convergence_window : int
        Rolling window size for early-stopping.  Stop when
        ``max(window) - min(window) < convergence_threshold``.
    change_threshold : float
        Minimum L2 norm of the SH-DC delta (per splat, summed across RGB)
        for a splat to be considered changed.  Tune lower to capture subtle
        colour shifts; higher to ignore noise.
    """

    def __init__(
        self,
        scene: "str | GaussianCloud",
        device: Optional[str] = None,
        lr_sh: float = 5e-4,
        lr_opacity: float = 5e-4,
        lr_geometry: float = 1e-5,
        scheduler_gamma: float = 0.995,
        convergence_window: int = 10,
        change_threshold: float = 1e-3,
    ) -> None:
        self.device           = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_sh            = lr_sh
        self.lr_opacity       = lr_opacity
        self.lr_geometry      = lr_geometry
        self.scheduler_gamma  = scheduler_gamma
        self.convergence_window = convergence_window
        self.change_threshold = change_threshold

        self._renderer     = ViewportRenderer(device=self.device)
        self._delta_engine = DeltaEngine()

        if isinstance(scene, GaussianCloud):
            self.ply_path = None
            self._source  = scene.to(self.device)
        else:
            self.ply_path = Path(scene)
            self._source  = load_ply(str(self.ply_path), device=self.device)

    def reload(self) -> None:
        """Re-read the source .ply from disk (only valid when constructed with a path)."""
        if self.ply_path is None:
            raise RuntimeError(
                "reload() is not available when SplatMutator was constructed from a "
                "GaussianCloud directly.  Pass a ply_path string to enable reloading."
            )
        self._source = load_ply(str(self.ply_path), device=self.device)

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _extract_patch(
        self,
        source: GaussianCloud,
        final_sh_dc: torch.Tensor,
        final_sh_rest: torch.Tensor,
        final_raw_opacities: torch.Tensor,
        final_means: torch.Tensor,
        final_log_scales: torch.Tensor,
        final_quats: torch.Tensor,
    ) -> tuple[GaussianCloud, list[int]]:
        """
        Compute per-splat deltas, threshold them, and return a patch cloud
        containing only the rows that meaningfully changed.

        Returns
        -------
        (patch_cloud, hidden_indices)
        """
        # Per-splat L2 norm of SH-DC delta — shape (N,)
        dc_delta = (final_sh_dc - source.sh_dc.to(self.device)).norm(dim=-1)

        # Higher-order SH delta — captures view-dependent specular changes (N,)
        rest_delta = (final_sh_rest - source.sh_rest.to(self.device)).norm(dim=(-1, -2))

        # Include splats whose SH (DC or rest) or opacity changed past the threshold
        opacity_delta = (final_raw_opacities - source.raw_opacities.to(self.device)).abs()
        combined_delta = dc_delta + rest_delta + opacity_delta

        mask = combined_delta > self.change_threshold        # (N,) bool
        indices = torch.where(mask)[0]                       # 1-D int64 tensor

        if indices.numel() == 0:
            # Nothing changed — return an empty but valid cloud and no hidden indices
            empty = GaussianCloud(
                means=torch.zeros(0, 3, device=self.device),
                quats=torch.zeros(0, 4, device=self.device),
                log_scales=torch.zeros(0, 3, device=self.device),
                raw_opacities=torch.zeros(0, device=self.device),
                sh_dc=torch.zeros(0, 3, device=self.device),
                sh_rest=torch.zeros(0, source.sh_rest.shape[1], 3, device=self.device),
            )
            return empty, []

        patch_cloud = GaussianCloud(
            means=final_means[indices],
            quats=final_quats[indices],
            log_scales=final_log_scales[indices],
            raw_opacities=final_raw_opacities[indices],
            sh_dc=final_sh_dc[indices],
            sh_rest=final_sh_rest[indices],
        )

        hidden_indices: list[int] = indices.cpu().numpy().tolist()
        return patch_cloud, hidden_indices

    # ------------------------------------------------------------------
    # Core optimisation
    # ------------------------------------------------------------------

    def mutate(
        self,
        camera: Camera,
        edited_image: torch.Tensor,
        n_iters: int = 300,
        patch_path: Optional[str] = None,
        sh_degree: int = 3,
        optimize_geometry: bool = False,
        on_iter: Optional[Callable[[int, float], None]] = None,
        convergence_threshold: float = 1e-7,
    ) -> MutationResult:
        """
        Optimise the GaussianCloud to match *edited_image* from *camera*,
        then extract and save only the changed splats as a patch .ply.

        Parameters
        ----------
        camera :
            The exact viewpoint used to capture the original render.
        edited_image : (H, W, 3) float tensor in [0, 1]
            The AI-modified image that defines the desired appearance.
        n_iters :
            Maximum gradient steps.
        patch_path :
            Where to save the patch .ply.  Defaults to
            ``<source_stem>_patch_<hex8>.ply`` alongside the source file.
        sh_degree :
            Must match the SH degree of the scene (usually 3).
        optimize_geometry :
            If True, also optimise positions, scales, and rotations.
        on_iter :
            Progress callback fired after every iteration.
        convergence_threshold :
            Stop early when |loss[t] − loss[t−1]| < this value.

        Returns
        -------
        MutationResult
            Contains ``patch_path`` and ``hidden_indices`` — the two pieces a
            Memory Layer record needs to store.
        """
        # ── Validate inputs ────────────────────────────────────────────────
        edited_image = edited_image.to(self.device).float()
        if edited_image.dim() != 3 or edited_image.shape[2] != 3:
            raise ValueError("edited_image must be (H, W, 3).")
        if edited_image.shape[0] != camera.height or edited_image.shape[1] != camera.width:
            raise ValueError(
                f"edited_image size ({edited_image.shape[0]}×{edited_image.shape[1]}) "
                f"does not match camera ({camera.height}×{camera.width})."
            )

        # ── Default patch path ─────────────────────────────────────────────
        if patch_path is None:
            import uuid
            hex8 = uuid.uuid4().hex[:8]
            if self.ply_path is not None:
                patch_path = str(
                    self.ply_path.parent
                    / f"{self.ply_path.stem}_patch_{hex8}.ply"
                )
            else:
                import tempfile
                patch_path = str(
                    Path(tempfile.gettempdir()) / f"tmachine_patch_{hex8}.ply"
                )

        # ── Clone source — never mutate the cached original ────────────────
        g = self._source.clone().to(self.device)

        # ── Appearance parameters — these will receive gradients ───────────
        sh_dc         = g.sh_dc.detach().clone().requires_grad_(True)
        sh_rest       = g.sh_rest.detach().clone().requires_grad_(True)
        raw_opacities = g.raw_opacities.detach().clone().requires_grad_(True)

        optim_groups: list[dict] = [
            {"params": [sh_dc],         "lr": self.lr_sh},
            {"params": [sh_rest],       "lr": self.lr_sh},
            {"params": [raw_opacities], "lr": self.lr_opacity},
        ]

        # ── Optional geometry parameters ───────────────────────────────────
        if optimize_geometry:
            geo_means      = g.means.detach().clone().requires_grad_(True)
            geo_log_scales = g.log_scales.detach().clone().requires_grad_(True)
            geo_quats      = g.quats.detach().clone().requires_grad_(True)
            optim_groups += [
                {"params": [geo_means],      "lr": self.lr_geometry},
                {"params": [geo_log_scales], "lr": self.lr_geometry},
                {"params": [geo_quats],      "lr": self.lr_geometry},
            ]
        else:
            geo_means      = g.means
            geo_log_scales = g.log_scales
            geo_quats      = g.quats

        # ── Optimizer + LR schedule ────────────────────────────────────────
        optimizer = optim.Adam(optim_groups, eps=1e-8)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)

        # ── Optimisation loop ──────────────────────────────────────────────
        loss_history: list[float] = []
        initial_loss: float = 0.0

        for i in range(n_iters):
            optimizer.zero_grad()

            if optimize_geometry:
                quats_norm = geo_quats / (
                    geo_quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                )
            else:
                quats_norm = geo_quats

            working_cloud = GaussianCloud(
                means=geo_means,
                quats=quats_norm,
                log_scales=geo_log_scales,
                raw_opacities=raw_opacities,
                sh_dc=sh_dc,
                sh_rest=sh_rest,
            )

            rendered = self._renderer.render(
                camera=camera,
                gaussians=working_cloud,
                sh_degree=sh_degree,
            )

            loss_map = self._delta_engine.compute(rendered, edited_image)
            loss     = loss_map.total_loss
            loss.backward()

            # Clip gradients to prevent instability on high-contrast edits
            _grad_params = [sh_dc, sh_rest, raw_opacities]
            if optimize_geometry:
                _grad_params += [geo_means, geo_log_scales, geo_quats]
            torch.nn.utils.clip_grad_norm_(_grad_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if i == 0:
                initial_loss = loss_val

            if on_iter is not None:
                on_iter(i, loss_val)

            # Rolling-window convergence check
            if len(loss_history) >= self.convergence_window:
                window = loss_history[-self.convergence_window:]
                if max(window) - min(window) < convergence_threshold:
                    break

        # ── Finalise optimised parameters (detached) ──────────────────────
        final_sh_dc    = sh_dc.detach()
        final_sh_rest  = sh_rest.detach()
        final_opacities = raw_opacities.detach()

        if optimize_geometry:
            final_means      = geo_means.detach()
            final_log_scales = geo_log_scales.detach()
            final_quats = (
                geo_quats / geo_quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ).detach()
        else:
            final_means      = g.means
            final_log_scales = g.log_scales
            final_quats      = g.quats

        # ── Extract patch (only changed splats) ────────────────────────────
        patch_cloud, hidden_indices = self._extract_patch(
            source=self._source,
            final_sh_dc=final_sh_dc,
            final_sh_rest=final_sh_rest,
            final_raw_opacities=final_opacities,
            final_means=final_means,
            final_log_scales=final_log_scales,
            final_quats=final_quats,
        )

        # ── Persist patch to disk ──────────────────────────────────────────
        save_ply(patch_cloud, patch_path)

        return MutationResult(
            iterations_run=len(loss_history),
            initial_loss=initial_loss,
            final_loss=loss_history[-1],
            patch_path=patch_path,
            hidden_indices=hidden_indices,
            changed_splat_count=len(hidden_indices),
            loss_history=loss_history,
        )
