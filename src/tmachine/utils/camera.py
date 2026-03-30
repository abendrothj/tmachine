"""
tmachine/utils/camera.py

Pinhole camera model in OpenCV convention (x → right, y → down, z → forward).

Coordinate system notes
-----------------------
The 3DGS rasteriser (gsplat) uses the OpenCV camera convention, which is also
the standard in computer vision:
    +X  right
    +Y  down
    +Z  into the scene (forward)

Euler angle convention used here (applied right-to-left: Ry first, then Rx, then Rz):
    yaw   – rotation around the world Y-axis (pan left / right)
    pitch – rotation around the local X-axis (tilt up / down)
    roll  – rotation around the local Z-axis (bank clockwise / counter-clockwise)

This is the same "aircraft / camera" convention used by most NeRF & 3DGS tools.
All angles are in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Camera definition
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """
    Immutable pinhole camera: extrinsics + intrinsics.

    Attributes
    ----------
    position : (3,) float32 ndarray
        Camera centre in world space.
    R : (3, 3) float32 ndarray
        World-to-camera rotation matrix.
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels (usually image centre).
    width, height : int
        Image dimensions in pixels.
    near, far : float
        Clipping plane distances.
    """

    position: np.ndarray   # (3,)
    R: np.ndarray          # (3, 3) world-to-camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.01
    far: float = 1_000.0

    # ------------------------------------------------------------------
    # Derived matrices (returned as torch tensors for gsplat)
    # ------------------------------------------------------------------

    @property
    def viewmat(self) -> torch.Tensor:
        """
        4×4 world-to-camera view matrix.

        Follows the OpenCV convention:
            [R  | -R @ position]
            [0  |       1      ]
        """
        t = -(self.R @ self.position)       # translation vector (3,)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = self.R
        mat[:3, 3]  = t
        return torch.from_numpy(mat)        # (4, 4)

    @property
    def K(self) -> torch.Tensor:
        """3×3 intrinsic (calibration) matrix."""
        return torch.tensor(
            [
                [self.fx,    0.0,  self.cx],
                [  0.0,   self.fy, self.cy],
                [  0.0,     0.0,     1.0 ],
            ],
            dtype=torch.float32,
        )

    @property
    def fov_x(self) -> float:
        """Horizontal field of view in radians."""
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def fov_y(self) -> float:
        """Vertical field of view in radians."""
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    def __repr__(self) -> str:
        return (
            f"Camera({self.width}×{self.height}px  "
            f"fx={self.fx:.1f} fy={self.fy:.1f}  "
            f"pos=({self.position[0]:.2f},{self.position[1]:.2f},{self.position[2]:.2f}))"
        )


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _Rx(a: float) -> np.ndarray:
    """Rotation matrix about the X axis by angle *a* (radians)."""
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[1, 0,  0],
         [0, c, -s],
         [0, s,  c]], dtype=np.float32
    )


def _Ry(a: float) -> np.ndarray:
    """Rotation matrix about the Y axis by angle *a* (radians)."""
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[ c, 0, s],
         [ 0, 1, 0],
         [-s, 0, c]], dtype=np.float32
    )


def _Rz(a: float) -> np.ndarray:
    """Rotation matrix about the Z axis by angle *a* (radians)."""
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, 0],
         [s,  c, 0],
         [0,  0, 1]], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Camera factory
# ---------------------------------------------------------------------------

def camera_from_euler(
    position: Tuple[float, float, float],
    pitch: float,
    yaw: float,
    roll: float,
    fx: float,
    fy: float,
    width: int,
    height: int,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    near: float = 0.01,
    far: float = 1_000.0,
) -> Camera:
    """
    Construct a Camera from world-space position and Euler angles.

    Parameters
    ----------
    position : (x, y, z)
        Camera centre in world space.
    pitch : float
        Tilt angle around X (radians).  Positive = nose down.
    yaw : float
        Pan angle around Y (radians).   Positive = turn right.
    roll : float
        Bank angle around Z (radians).  Positive = clockwise.
    fx, fy : float
        Focal lengths in pixels.  For a square sensor with known FoV:
            fx = width  / (2 * tan(fov_x / 2))
    width, height : int
        Output image dimensions in pixels.
    cx, cy : float, optional
        Principal point.  Defaults to the image centre.
    near, far : float
        Clipping plane distances.

    Returns
    -------
    Camera
    """
    # World-to-camera rotation:
    #   1. Ry(yaw)   – orient camera in the horizontal plane
    #   2. Rx(pitch) – tilt
    #   3. Rz(roll)  – bank
    R_world_to_cam: np.ndarray = _Rz(roll) @ _Rx(pitch) @ _Ry(yaw)

    return Camera(
        position=np.array(position, dtype=np.float32),
        R=R_world_to_cam,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx) if cx is not None else width  / 2.0,
        cy=float(cy) if cy is not None else height / 2.0,
        width=int(width),
        height=int(height),
        near=float(near),
        far=float(far),
    )


def camera_from_fov(
    position: Tuple[float, float, float],
    pitch: float,
    yaw: float,
    roll: float,
    fov_x: float,
    width: int,
    height: int,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    near: float = 0.01,
    far: float = 1_000.0,
) -> Camera:
    """
    Convenience wrapper: specify horizontal FoV instead of focal length.

    Parameters
    ----------
    fov_x : float
        Horizontal field of view in radians (e.g. math.radians(60)).
    All other parameters as per :func:`camera_from_euler`.

    Notes
    -----
    For a standard square-pixel pinhole camera ``fx == fy``.  The vertical
    FoV is not an independent degree of freedom — it is determined by
    ``fov_x`` and the image aspect ratio:

        fov_y = 2 * atan(tan(fov_x / 2) * height / width)

    This is encoded automatically via the ``fov_y`` property.  If your
    capture equipment has non-square pixels, use :func:`camera_from_euler`
    directly and supply separate ``fx`` and ``fy`` values.
    """
    fx = width / (2.0 * math.tan(fov_x / 2.0))
    return camera_from_euler(
        position=position,
        pitch=pitch, yaw=yaw, roll=roll,
        fx=fx, fy=fx,  # square pixels: fx == fy
        width=width, height=height,
        cx=cx, cy=cy,
        near=near, far=far,
    )


def camera_from_colmap(
    R: np.ndarray,
    t: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 1_000.0,
) -> Camera:
    """
    Construct a Camera from COLMAP-style world-to-camera extrinsics.

    COLMAP exports rotation matrix ``R`` and translation ``t`` where
    ``t = R @ (−position_world)``, so ``position_world = −R.T @ t``.

    Parameters
    ----------
    R : (3, 3) float ndarray
        World-to-camera rotation matrix (orthonormal, row-major).
    t : (3,) float ndarray
        World-to-camera translation vector.
    fx, fy : float
        Focal lengths in pixels from the COLMAP Camera model.
    cx, cy : float
        Principal point in pixels.
    width, height : int
        Image dimensions in pixels.
    near, far : float
        Clipping plane distances.

    Returns
    -------
    Camera
    """
    R_arr = np.asarray(R, dtype=np.float32)
    t_arr = np.asarray(t, dtype=np.float32)
    position = -(R_arr.T @ t_arr)   # world-space camera centre
    return Camera(
        position=position,
        R=R_arr,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        width=int(width),
        height=int(height),
        near=float(near),
        far=float(far),
    )


def auto_consistency_cameras(
    primary: Camera,
    count: int = 4,
    yaw_step_deg: float = 30.0,
) -> "list[Camera]":
    """
    Generate consistency cameras by rotating the primary camera's yaw.

    Cameras are generated at ±yaw_step_deg, ±2×yaw_step_deg, … alternating
    left and right.  The primary camera's position, pitch, and roll are
    preserved — only the horizontal viewing direction changes.

    Use these cameras with :meth:`~tmachine.core.splat_mutator.SplatMutator.mutate`
    ``consistency_cameras`` to coarsely guard against multi-view drift without
    requiring a supplied camera trajectory.

    Parameters
    ----------
    primary : Camera
        The bake-viewpoint camera.
    count : int
        Total number of consistency cameras to generate.
    yaw_step_deg : float
        Angular step between successive cameras (degrees).

    Returns
    -------
    list[Camera]
        *count* cameras, alternating left and right of the primary yaw.
    """
    cameras: list[Camera] = []
    for k in range(1, count + 1):
        sign      = +1 if k % 2 == 1 else -1
        magnitude = (k + 1) // 2
        delta     = math.radians(sign * magnitude * yaw_step_deg)
        # Rotate the world by −delta before applying primary R, which is
        # equivalent to rotating the camera yaw by +delta in world space.
        R_new = (primary.R @ _Ry(-delta)).astype(np.float32)
        cameras.append(Camera(
            position=primary.position.copy(),
            R=R_new,
            fx=primary.fx, fy=primary.fy,
            cx=primary.cx, cy=primary.cy,
            width=primary.width, height=primary.height,
            near=primary.near, far=primary.far,
        ))
    return cameras
