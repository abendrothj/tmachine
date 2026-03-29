"""
tmachine/io/ply_handler.py

Handles reading and writing 3D Gaussian Splat scenes stored in the Inria .ply
format.  All field names and byte layout match the original gaussian-splatting
repository so files are interchangeable with other 3DGS tooling.

Stored layout (per-splat inside the .ply vertex element):
    x, y, z              – world-space position
    nx, ny, nz           – surface normals (unused, kept for compatibility)
    f_dc_{0,1,2}         – DC spherical-harmonic coefficients (base colour)
    f_rest_{0..44}       – higher-order SH coefficients (degrees 1–3)
    opacity              – raw logit; apply sigmoid() to get [0, 1]
    scale_{0,1,2}        – log-scale; apply exp() to get linear size
    rot_{0,1,2,3}        – unit quaternion (w, x, y, z)

In memory (GaussianCloud) the tensors carry their raw / un-activated values
so that autograd can differentiate through the activations during optimisation.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GaussianCloud:
    """
    An in-memory representation of a 3DGS scene.

    All tensors store *raw* (pre-activation) values matching the .ply layout:
        • log_scales      → exp() → linear scale
        • raw_opacities   → sigmoid() → opacity in [0, 1]
        • sh_dc / sh_rest → used directly as SH coefficients

    Shape conventions
    -----------------
    means           (N, 3)
    quats           (N, 4)   normalised (w, x, y, z)
    log_scales      (N, 3)
    raw_opacities   (N,)
    sh_dc           (N, 3)
    sh_rest         (N, 15, 3)  – degrees 1-3, basis-first, then channel
    """

    means: torch.Tensor
    quats: torch.Tensor
    log_scales: torch.Tensor
    raw_opacities: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    # ------------------------------------------------------------------
    # Convenience properties (activated values, differentiable)
    # ------------------------------------------------------------------

    @property
    def scales(self) -> torch.Tensor:
        """Linear (activated) scales — (N, 3)."""
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> torch.Tensor:
        """Activated opacities in [0, 1] — (N,)."""
        return torch.sigmoid(self.raw_opacities)

    @property
    def sh_all(self) -> torch.Tensor:
        """
        Full SH coefficient array in the format expected by gsplat:
            (N, (sh_degree+1)^2, 3)  →  (N, 16, 3) for degree-3 scenes.
        DC band is prepended as the zeroth basis function.
        """
        dc = self.sh_dc[:, None, :]              # (N, 1, 3)
        return torch.cat([dc, self.sh_rest], dim=1)  # (N, 16, 3)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.means.shape[0]

    def __repr__(self) -> str:
        return (
            f"GaussianCloud({len(self):,} splats, "
            f"device={self.means.device}, "
            f"sh_rest={self.sh_rest.shape})"
        )

    def to(self, device: str | torch.device) -> GaussianCloud:
        """Return a copy with all tensors moved to *device*."""
        return GaussianCloud(
            means=self.means.to(device),
            quats=self.quats.to(device),
            log_scales=self.log_scales.to(device),
            raw_opacities=self.raw_opacities.to(device),
            sh_dc=self.sh_dc.to(device),
            sh_rest=self.sh_rest.to(device),
        )

    def clone(self) -> GaussianCloud:
        """Deep-copy all tensors (detached, no gradient history)."""
        return GaussianCloud(
            means=self.means.detach().clone(),
            quats=self.quats.detach().clone(),
            log_scales=self.log_scales.detach().clone(),
            raw_opacities=self.raw_opacities.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _is_gaussian_splat(vertex) -> bool:
    """Return True if this PLY vertex element contains 3DGS fields."""
    names = {p.name for p in vertex.properties}
    return "opacity" in names and "f_dc_0" in names


def _load_point_cloud_as_gaussians(vertex, device: str) -> GaussianCloud:
    """
    Convert a plain RGB point cloud (x, y, z, red, green, blue) into a
    GaussianCloud of tiny spherical splats.  Quality is limited (no
    view-dependent colour, fixed size) but the scene will render.
    """
    N = len(vertex["x"])

    means = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float32)  # (N, 3)

    # Identity quaternion (w=1, x=0, y=0, z=0) → upright spheres
    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    # Small uniform scale: log(0.005) ≈ -5.3 → ~5 mm splats
    log_scales = np.full((N, 3), -5.3, dtype=np.float32)

    # High opacity: sigmoid(3) ≈ 0.95
    raw_opacities = np.full(N, 3.0, dtype=np.float32)

    # Convert uint8 RGB [0..255] → SH DC coefficient
    # gsplat converts SH DC to colour as: colour = 0.5 + SH_0 * C0  (C0 ≈ 0.2821)
    # So: SH_0 = (colour - 0.5) / C0
    C0 = 0.28209479177387814
    r = np.asarray(vertex["red"],   dtype=np.float32) / 255.0
    g = np.asarray(vertex["green"], dtype=np.float32) / 255.0
    b = np.asarray(vertex["blue"],  dtype=np.float32) / 255.0
    sh_dc = np.stack(
        [(r - 0.5) / C0, (g - 0.5) / C0, (b - 0.5) / C0], axis=1
    )  # (N, 3)

    # No higher-order SH (degree 0 — direction-agnostic colour)
    sh_rest = np.zeros((N, 0, 3), dtype=np.float32)

    return GaussianCloud(
        means=torch.from_numpy(means).to(device),
        quats=torch.from_numpy(quats).to(device),
        log_scales=torch.from_numpy(log_scales).to(device),
        raw_opacities=torch.from_numpy(raw_opacities).to(device),
        sh_dc=torch.from_numpy(sh_dc).to(device),
        sh_rest=torch.from_numpy(sh_rest).to(device),
    )


def load_ply(path: str | Path, device: str = "cpu") -> GaussianCloud:
    """
    Load a 3DGS scene from an Inria-format .ply file, or an RGB point cloud.

    Accepts both:
    - Full 3DGS .ply files (f_dc_, f_rest_, opacity, scale_, rot_ fields)
    - Plain RGB point clouds (x, y, z, red, green, blue) — converted to
      naïve spherical Gaussians for basic rendering.

    Parameters
    ----------
    path:   Path to the .ply file.
    device: Torch device for the returned tensors.

    Returns
    -------
    GaussianCloud with raw (pre-activation) tensor values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    plydata = PlyData.read(str(path))
    vertex = plydata.elements[0]

    if not _is_gaussian_splat(vertex):
        return _load_point_cloud_as_gaussians(vertex, device)

    # ── Positions ──────────────────────────────────────────────────────────
    means = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float32)  # (N, 3)

    # ── Quaternions (w, x, y, z) ──────────────────────────────────────────
    quats = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
        axis=1,
    ).astype(np.float32)  # (N, 4)
    # Normalise for numerical safety
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.where(norms > 0, norms, 1.0)

    # ── Log-scales ────────────────────────────────────────────────────────
    log_scales = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
    ).astype(np.float32)  # (N, 3)

    # ── Opacity logits ────────────────────────────────────────────────────
    raw_opacities = np.asarray(vertex["opacity"], dtype=np.float32)  # (N,)

    # ── SH DC coefficients ────────────────────────────────────────────────
    sh_dc = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
    ).astype(np.float32)  # (N, 3)

    # ── SH rest coefficients ──────────────────────────────────────────────
    # Inria save order: _features_rest (N, 15, 3)
    #   → .transpose(1,2) → (N, 3, 15)
    #   → .flatten(1)     → (N, 45)   (f_rest_0 … f_rest_44)
    # Reading back: (N, 45) → reshape(N, 3, 15) → transpose axes 1,2 → (N, 15, 3)
    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda n: int(n.split("_")[-1]),
    )
    if rest_names:
        sh_rest_flat = np.column_stack(
            [np.asarray(vertex[n], dtype=np.float32) for n in rest_names]
        )  # (N, 45)
        n_rest = sh_rest_flat.shape[1]           # 45 for degree-3
        n_per_channel = n_rest // 3              # 15
        # (N, 45) → (N, 3, 15) → (N, 15, 3)
        sh_rest = (
            sh_rest_flat.reshape(-1, 3, n_per_channel)
            .transpose(0, 2, 1)
            .astype(np.float32)
        )
    else:
        # No rest coefficients — scene was trained at SH degree 0.
        sh_rest = np.zeros((len(means), 0, 3), dtype=np.float32)

    return GaussianCloud(
        means=torch.from_numpy(means).to(device),
        quats=torch.from_numpy(quats).to(device),
        log_scales=torch.from_numpy(log_scales).to(device),
        raw_opacities=torch.from_numpy(raw_opacities).to(device),
        sh_dc=torch.from_numpy(sh_dc).to(device),
        sh_rest=torch.from_numpy(sh_rest).to(device),
    )


def save_ply(gaussians: GaussianCloud, path: str | Path) -> None:
    """
    Write a GaussianCloud back to an Inria-format .ply file.

    Tensors are automatically detached and moved to CPU before serialisation.
    The output is byte-for-byte compatible with the original 3DGS tooling.

    Parameters
    ----------
    gaussians:  The cloud to serialise.
    path:       Destination path (parent directories are created as needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().float().numpy()

    means = _np(gaussians.means)            # (N, 3)
    quats = _np(gaussians.quats)            # (N, 4)
    log_scales = _np(gaussians.log_scales)  # (N, 3)
    raw_opacities = _np(gaussians.raw_opacities)  # (N,)
    sh_dc = _np(gaussians.sh_dc)            # (N, 3)
    sh_rest_nhc = _np(gaussians.sh_rest)    # (N, 15, 3)

    N = means.shape[0]

    # Convert sh_rest back to Inria storage layout:
    # (N, 15, 3) → transpose(0,2,1) → (N, 3, 15) → reshape → (N, 45)
    sh_rest_flat = sh_rest_nhc.transpose(0, 2, 1).reshape(N, -1)  # (N, 45)
    n_rest = sh_rest_flat.shape[1]

    dtype_fields = (
        [("x", "f4"), ("y", "f4"), ("z", "f4")]
        + [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        + [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
        + [(f"f_rest_{i}", "f4") for i in range(n_rest)]
        + [("opacity", "f4")]
        + [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
        + [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    )

    vertex_data = np.zeros(N, dtype=dtype_fields)

    vertex_data["x"]       = means[:, 0]
    vertex_data["y"]       = means[:, 1]
    vertex_data["z"]       = means[:, 2]
    vertex_data["nx"]      = 0.0
    vertex_data["ny"]      = 0.0
    vertex_data["nz"]      = 0.0
    vertex_data["f_dc_0"]  = sh_dc[:, 0]
    vertex_data["f_dc_1"]  = sh_dc[:, 1]
    vertex_data["f_dc_2"]  = sh_dc[:, 2]
    for i in range(n_rest):
        vertex_data[f"f_rest_{i}"] = sh_rest_flat[:, i]
    vertex_data["opacity"] = raw_opacities
    vertex_data["scale_0"] = log_scales[:, 0]
    vertex_data["scale_1"] = log_scales[:, 1]
    vertex_data["scale_2"] = log_scales[:, 2]
    vertex_data["rot_0"]   = quats[:, 0]
    vertex_data["rot_1"]   = quats[:, 1]
    vertex_data["rot_2"]   = quats[:, 2]
    vertex_data["rot_3"]   = quats[:, 3]

    el = PlyElement.describe(vertex_data, "vertex")

    # Write atomically: persist to a sibling temp file then rename.
    # This prevents a crash or OOM during the write from leaving a
    # partially-written (and permanently corrupted) .ply file.
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + "_", suffix=".ply.tmp"
    )
    try:
        os.close(tmp_fd)
        PlyData([el]).write(tmp_path)
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up the temp file if anything went wrong before the rename.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
