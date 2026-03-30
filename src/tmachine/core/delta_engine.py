"""
tmachine/core/delta_engine.py  –  Module 2: Delta Engine

Responsibility
--------------
Bridge between 2D images and the 3D optimiser.  Takes the original viewport
render and the AI-edited version, and produces a LossMap that precisely
quantifies *what* changed, *where*, and *by how much*.

The losses inside LossMap are fully differentiable (PyTorch tensors), so they
can be directly used as the objective in the Module 3 back-propagation loop.

Interface
---------
    delta = DeltaEngine()
    loss_map = delta.compute(original_render, edited_render)

    # Inspect:
    print(f"{loss_map.changed_pixel_ratio:.1%} pixels changed")

    # Drive optimisation:
    loss_map.total_loss.backward()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class LossMap:
    """
    Rich description of pixel-level changes between two renders.

    All tensor fields are on the same device as the input images and retain
    gradient history unless explicitly detached.

    Fields
    ------
    pixel_diff : (H, W, 3)
        Absolute per-channel difference, values in [0, 1].
    luminance_diff : (H, W)
        ITU-R BT.601 weighted luminance magnitude, values in [0, 1].
    change_mask : (H, W)  bool
        True for pixels whose luminance changed above ``change_threshold``.
    l1_loss : scalar
        Mean absolute error across all channels — primary optimisation signal.
    l2_loss : scalar
        Mean squared error across all channels — penalises large deviations.
    total_loss : scalar
        Weighted combination: ``l1_weight * l1_loss + l2_weight * l2_loss``.
        This is the value you call ``.backward()`` on.
    changed_pixel_ratio : float
        Python float — fraction of pixels flagged by ``change_mask``.
    """

    pixel_diff:           torch.Tensor            # (H, W, 3)
    luminance_diff:       torch.Tensor            # (H, W)
    change_mask:          torch.Tensor            # (H, W) bool
    l1_loss:              torch.Tensor            # scalar
    l2_loss:              torch.Tensor            # scalar
    lpips_loss:           Optional[torch.Tensor]  # scalar, None when LPIPS disabled
    total_loss:           torch.Tensor            # scalar  ← backward() target
    changed_pixel_ratio:  float

    def __repr__(self) -> str:
        lpips_str = (
            f", LPIPS={self.lpips_loss.item():.5f}"
            if self.lpips_loss is not None
            else ""
        )
        return (
            f"LossMap("
            f"total={self.total_loss.item():.5f}, "
            f"L1={self.l1_loss.item():.5f}, "
            f"L2={self.l2_loss.item():.5f}"
            f"{lpips_str}, "
            f"changed={self.changed_pixel_ratio:.2%})"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DeltaEngine:
    """
    Module 2 — Delta Engine.

    Computes a pixel-level LossMap between an original viewport render and an
    AI-edited version of the same frame.

    Parameters
    ----------
    l1_weight : float
        Weight for L1 (MAE) loss in ``total_loss``.  Default 0.8.
    l2_weight : float
        Weight for L2 (MSE) loss in ``total_loss``.  Default 0.2.
    lpips_weight : float
        Weight for LPIPS perceptual loss in ``total_loss``.  Default 0.0
        (disabled).  When non-zero, requires ``pip install lpips``.
        ``l1_weight + l2_weight + lpips_weight`` must equal 1.0.
    change_threshold : float
        Minimum luminance delta (in [0, 1]) to flag a pixel as changed in
        ``change_mask``.  Default 0.01 (≈ 3 intensity steps out of 255).
    """

    # ITU-R BT.601 luma weights (matches human perceptual brightness)
    _LUMA = (0.299, 0.587, 0.114)

    def __init__(
        self,
        l1_weight: float = 0.8,
        l2_weight: float = 0.2,
        lpips_weight: float = 0.0,
        change_threshold: float = 0.01,
    ) -> None:
        if abs(l1_weight + l2_weight + lpips_weight - 1.0) > 1e-6:
            raise ValueError(
                f"l1_weight + l2_weight + lpips_weight must equal 1.0, "
                f"got {l1_weight + l2_weight + lpips_weight}"
            )
        self.l1_weight        = l1_weight
        self.l2_weight        = l2_weight
        self.lpips_weight     = lpips_weight
        self.change_threshold = change_threshold
        self._lpips_fn        = None  # lazy-loaded
        self._lpips_device: str | None = None

    def compute(
        self,
        original: torch.Tensor,
        edited: torch.Tensor,
    ) -> LossMap:
        """
        Compare two images and produce a :class:`LossMap`.

        Parameters
        ----------
        original : (H, W, 3) float tensor in [0, 1]
            The pristine render produced by the Viewport Renderer (Module 1).
        edited : (H, W, 3) float tensor in [0, 1]
            The AI-modified image approved by the user.

        Returns
        -------
        LossMap
        """
        if original.shape != edited.shape:
            raise ValueError(
                f"Shape mismatch: original {tuple(original.shape)} "
                f"vs edited {tuple(edited.shape)}"
            )
        if original.dim() != 3 or original.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) images, got {tuple(original.shape)}."
            )

        # ── Pixel-level absolute difference ────────────────────────────────
        pixel_diff = (edited - original).abs()           # (H, W, 3)

        # ── Luminance-weighted magnitude ───────────────────────────────────
        luma = torch.tensor(
            self._LUMA, dtype=original.dtype, device=original.device
        )
        luminance_diff = (pixel_diff * luma).sum(dim=-1)  # (H, W)

        # ── Change mask ────────────────────────────────────────────────────
        change_mask = luminance_diff > self.change_threshold  # (H, W) bool

        # ── Differentiable losses ──────────────────────────────────────────
        l1_loss = pixel_diff.mean()
        l2_loss = ((edited - original) ** 2).mean()
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss

        # ── Optional LPIPS perceptual loss ─────────────────────────────
        lpips_loss: Optional[torch.Tensor] = None
        if self.lpips_weight > 0.0:
            lpips_loss  = self._compute_lpips(original, edited)
            total_loss  = total_loss + self.lpips_weight * lpips_loss

        return LossMap(
            pixel_diff=pixel_diff,
            luminance_diff=luminance_diff,
            change_mask=change_mask,
            l1_loss=l1_loss,
            l2_loss=l2_loss,
            lpips_loss=lpips_loss,
            total_loss=total_loss,
            changed_pixel_ratio=change_mask.float().mean().item(),
        )

    # ------------------------------------------------------------------
    # LPIPS helper
    # ------------------------------------------------------------------

    def _compute_lpips(self, original: torch.Tensor, edited: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS perceptual distance between two (H, W, 3) images.

        Lazy-loads the ``lpips`` model on the first call and caches it.
        Requires ``pip install lpips`` (or ``pip install tmachine[ai-image]``).
        """
        if self._lpips_fn is None:
            try:
                import lpips as _lpips_lib
            except ImportError as exc:
                raise ImportError(
                    "lpips is required when lpips_weight > 0. "
                    "Install with: pip install lpips"
                ) from exc
            self._lpips_fn = _lpips_lib.LPIPS(net="alex", verbose=False)
            self._lpips_fn.eval()
            self._lpips_device: str | None = None

        device = str(original.device)
        if device != self._lpips_device:
            self._lpips_fn = self._lpips_fn.to(device)  # type: ignore[assignment]
            self._lpips_device = device

        # lpips expects (N, 3, H, W) in [-1, 1]
        def _to_lpips(t: torch.Tensor) -> torch.Tensor:
            return t.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

        return self._lpips_fn(_to_lpips(original), _to_lpips(edited)).squeeze()
