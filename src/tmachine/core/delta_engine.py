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
    masked_loss : scalar
        ``total_loss`` restricted to pixels flagged by ``change_mask``.
        Better gradient signal for targeted edits — drives the optimizer to
        focus on changed regions rather than penalising the whole frame.
        Falls back to ``total_loss`` when no pixels are masked.
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
    masked_loss:          torch.Tensor            # scalar — loss restricted to changed pixels
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
    mask_dilation : int
        Radius (in pixels) of a morphological dilation applied to
        ``change_mask`` after thresholding.  Expands the masked region so
        that splats bordering the edit receive gradient signal too, reducing
        patch boundary artefacts.  Default 0 (disabled).
    """

    # ITU-R BT.601 luma weights (matches human perceptual brightness)
    # Stored as float32 regardless of input dtype — fp16 rounding on these
    # coefficients is large enough to corrupt the luminance diff meaningfully.
    _LUMA = (0.299, 0.587, 0.114)

    def __init__(
        self,
        l1_weight: float = 0.8,
        l2_weight: float = 0.2,
        lpips_weight: float = 0.0,
        change_threshold: float = 0.01,
        mask_dilation: int = 0,
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
        self.mask_dilation    = mask_dilation
        self._lpips_fn        = None  # lazy-loaded
        self._lpips_device: str | None = None
        self._luma_cache: dict[str, torch.Tensor] = {}  # device → fp32 luma tensor

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
        # Always fp32 — fp16 rounding on the BT.601 weights is large enough
        # to produce a noticeably wrong luminance mask.
        dev = str(original.device)
        if dev not in self._luma_cache:
            self._luma_cache[dev] = torch.tensor(
                self._LUMA, dtype=torch.float32, device=original.device
            )
        luma = self._luma_cache[dev]
        luminance_diff = (pixel_diff.float() * luma).sum(dim=-1)  # (H, W) fp32

        # ── Change mask ────────────────────────────────────────────────────
        change_mask = luminance_diff > self.change_threshold  # (H, W) bool
        # Optional morphological dilation — softens sharp edit boundaries
        # and gives adjacent splats gradient signal they would otherwise lack.
        if self.mask_dilation > 0:
            from torch.nn import functional as _F
            k = 2 * self.mask_dilation + 1
            dilated = _F.max_pool2d(
                change_mask.float().unsqueeze(0).unsqueeze(0),
                kernel_size=k,
                stride=1,
                padding=self.mask_dilation,
            )
            change_mask = dilated.squeeze().bool()
        # ── Differentiable losses ──────────────────────────────────────────
        l1_loss = pixel_diff.mean()
        l2_loss = ((edited - original) ** 2).mean()
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss

        # ── Masked loss — restrict signal to changed pixels ────────────────
        # This gives the optimizer a cleaner gradient: it pushes hard on the
        # region that actually changed and doesn't waste capacity on pixels
        # the AI left untouched.  Falls back to total_loss when mask is empty.
        if change_mask.any():
            mask3 = change_mask.unsqueeze(-1).expand_as(pixel_diff)  # (H, W, 3)
            masked_pixel_diff = pixel_diff[mask3].view(-1, 3)
            masked_diff_raw   = (edited - original)[mask3].view(-1, 3)
            masked_l1 = masked_pixel_diff.mean()
            masked_l2 = (masked_diff_raw ** 2).mean()
            masked_loss = self.l1_weight * masked_l1 + self.l2_weight * masked_l2
        else:
            masked_loss = total_loss

        # ── Optional LPIPS perceptual loss ─────────────────────────────
        lpips_loss: Optional[torch.Tensor] = None
        if self.lpips_weight > 0.0:
            lpips_loss  = self._compute_lpips(original, edited)
            total_loss  = total_loss  + self.lpips_weight * lpips_loss
            # LPIPS is a whole-image perceptual signal — no spatial mask exists,
            # so it is added to masked_loss verbatim (same weight as total_loss).
            masked_loss = masked_loss + self.lpips_weight * lpips_loss

        return LossMap(
            pixel_diff=pixel_diff,
            luminance_diff=luminance_diff,
            change_mask=change_mask,
            l1_loss=l1_loss,
            l2_loss=l2_loss,
            lpips_loss=lpips_loss,
            total_loss=total_loss,
            masked_loss=masked_loss,
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
