"""
tests/test_camera.py — Camera model unit tests

CPU-only, no GPU or real scene required.

Covers
------
- camera_from_fov / camera_from_euler: output shapes and types
- viewmat: correct OpenCV-convention world-to-camera matrix
- K matrix: intrinsic matrix values match constructor values
- fov_x / fov_y properties: round-trip with constructor fov_x
- Camera repr: smoke test
- Identity camera: position at origin, zero angles → R == I
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from tmachine.utils.camera import (
    Camera,
    camera_from_euler,
    camera_from_fov,
    camera_from_colmap,
    auto_consistency_cameras,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cam(**kwargs) -> Camera:
    defaults = dict(
        position=(0.0, 0.0, -5.0),
        pitch=0.0, yaw=0.0, roll=0.0,
        fov_x=math.radians(60),
        width=640, height=480,
    )
    defaults.update(kwargs)
    return camera_from_fov(**defaults)


# ---------------------------------------------------------------------------
# camera_from_fov
# ---------------------------------------------------------------------------

class TestCameraFromFov(unittest.TestCase):

    def test_returns_camera(self):
        cam = _default_cam()
        self.assertIsInstance(cam, Camera)

    def test_width_height_preserved(self):
        cam = _default_cam(width=1920, height=1080)
        self.assertEqual(cam.width, 1920)
        self.assertEqual(cam.height, 1080)

    def test_near_far_defaults(self):
        cam = _default_cam()
        self.assertAlmostEqual(cam.near, 0.01)
        self.assertAlmostEqual(cam.far, 1000.0)

    def test_cx_cy_default_to_image_centre(self):
        cam = _default_cam(width=640, height=480)
        self.assertAlmostEqual(cam.cx, 320.0)
        self.assertAlmostEqual(cam.cy, 240.0)

    def test_fov_x_round_trip(self):
        fov = math.radians(75)
        cam = camera_from_fov(
            position=(0, 0, -3), pitch=0, yaw=0, roll=0,
            fov_x=fov, width=800, height=600,
        )
        self.assertAlmostEqual(cam.fov_x, fov, places=5)

    def test_fx_fy_equal_square_pixels(self):
        """camera_from_fov assumes square pixels: fx must equal fy."""
        cam = _default_cam(width=1920, height=1080)  # 16:9 — still fx==fy
        self.assertAlmostEqual(cam.fx, cam.fy, places=6,
                               msg="camera_from_fov should set fx==fy (square pixels)")

    def test_fov_y_derived_from_aspect_ratio(self):
        """
        For a non-square sensor fov_y must follow from fov_x and aspect ratio.
        With square pixels: fov_y = 2*atan(tan(fov_x/2) * height/width).
        """
        fov_x = math.radians(60)
        width, height = 1920, 1080
        cam = camera_from_fov(
            position=(0, 0, -5), pitch=0, yaw=0, roll=0,
            fov_x=fov_x, width=width, height=height,
        )
        expected_fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * height / width)
        self.assertAlmostEqual(cam.fov_y, expected_fov_y, places=5)
        # fov_y for a 16:9 sensor with 60° horizontal FoV is ~34°, not 60°
        self.assertLess(cam.fov_y, fov_x)

    def test_fx_from_fov_formula(self):
        """fx = width / (2 * tan(fov_x / 2))."""
        fov_x = math.radians(60)
        width = 640
        expected_fx = width / (2.0 * math.tan(fov_x / 2.0))
        cam = camera_from_fov(
            position=(0, 0, -5), pitch=0, yaw=0, roll=0,
            fov_x=fov_x, width=width, height=480,
        )
        self.assertAlmostEqual(cam.fx, expected_fx, places=4)


# ---------------------------------------------------------------------------
# camera_from_euler
# ---------------------------------------------------------------------------

class TestCameraFromEuler(unittest.TestCase):

    def test_identity_rotation_at_zero_angles(self):
        """Zero yaw/pitch/roll → R should be identity."""
        cam = camera_from_euler(
            position=(0, 0, 0), pitch=0, yaw=0, roll=0,
            fx=500, fy=500, width=640, height=480,
        )
        np.testing.assert_allclose(cam.R, np.eye(3, dtype=np.float32), atol=1e-6)

    def test_position_stored_correctly(self):
        cam = camera_from_euler(
            position=(1.5, -2.0, 3.0), pitch=0, yaw=0, roll=0,
            fx=400, fy=400, width=640, height=480,
        )
        np.testing.assert_allclose(cam.position, [1.5, -2.0, 3.0], atol=1e-6)

    def test_custom_cx_cy(self):
        cam = camera_from_euler(
            position=(0, 0, 0), pitch=0, yaw=0, roll=0,
            fx=500, fy=500, width=640, height=480,
            cx=310.0, cy=235.0,
        )
        self.assertAlmostEqual(cam.cx, 310.0)
        self.assertAlmostEqual(cam.cy, 235.0)


# ---------------------------------------------------------------------------
# viewmat
# ---------------------------------------------------------------------------

class TestViewmat(unittest.TestCase):

    def test_viewmat_shape(self):
        cam = _default_cam()
        vm  = cam.viewmat
        self.assertEqual(vm.shape, (4, 4))
        self.assertIsInstance(vm, torch.Tensor)

    def test_viewmat_dtype(self):
        self.assertEqual(_default_cam().viewmat.dtype, torch.float32)

    def test_viewmat_bottom_row(self):
        """Last row of a valid view matrix must be [0, 0, 0, 1]."""
        vm = _default_cam().viewmat
        torch.testing.assert_close(vm[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def test_viewmat_identity_rotation_at_origin(self):
        """Camera at origin with zero angles: top-left 3x3 is identity."""
        cam = camera_from_euler(
            position=(0, 0, 0), pitch=0, yaw=0, roll=0,
            fx=500, fy=500, width=640, height=480,
        )
        vm = cam.viewmat
        np.testing.assert_allclose(
            vm[:3, :3].numpy(), np.eye(3, dtype=np.float32), atol=1e-6
        )

    def test_viewmat_translation_encodes_position(self):
        """t = -R @ pos; for R=I, t = -pos."""
        pos = (3.0, -1.0, 5.0)
        cam = camera_from_euler(
            position=pos, pitch=0, yaw=0, roll=0,
            fx=500, fy=500, width=640, height=480,
        )
        vm = cam.viewmat
        expected_t = torch.tensor([-3.0, 1.0, -5.0])
        torch.testing.assert_close(vm[:3, 3], expected_t, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# K matrix
# ---------------------------------------------------------------------------

class TestKMatrix(unittest.TestCase):

    def test_K_shape(self):
        self.assertEqual(_default_cam().K.shape, (3, 3))

    def test_K_dtype(self):
        self.assertEqual(_default_cam().K.dtype, torch.float32)

    def test_K_values(self):
        cam = camera_from_euler(
            position=(0, 0, 0), pitch=0, yaw=0, roll=0,
            fx=800.0, fy=750.0, cx=320.0, cy=240.0,
            width=640, height=480,
        )
        K = cam.K
        self.assertAlmostEqual(K[0, 0].item(), 800.0, places=4)
        self.assertAlmostEqual(K[1, 1].item(), 750.0, places=4)
        self.assertAlmostEqual(K[0, 2].item(), 320.0, places=4)
        self.assertAlmostEqual(K[1, 2].item(), 240.0, places=4)
        self.assertAlmostEqual(K[2, 2].item(), 1.0,   places=4)
        # Off-diagonal entries in the standard 3x3 K matrix are zero
        self.assertAlmostEqual(K[0, 1].item(), 0.0, places=4)

    def test_K_bottom_row(self):
        K = _default_cam().K
        torch.testing.assert_close(K[2], torch.tensor([0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# fov_y property
# ---------------------------------------------------------------------------

class TestFovProperties(unittest.TestCase):

    def test_fov_y_smaller_for_landscape(self):
        """In a landscape image (width > height), fov_y < fov_x."""
        cam = _default_cam(
            fov_x=math.radians(60), width=1920, height=1080
        )
        self.assertLess(cam.fov_y, cam.fov_x)

    def test_fov_y_formula(self):
        """fov_y = 2 * atan(height / (2 * fy))."""
        cam = camera_from_euler(
            position=(0, 0, 0), pitch=0, yaw=0, roll=0,
            fx=500, fy=400, width=640, height=480,
        )
        expected_fov_y = 2.0 * math.atan(480 / (2.0 * 400))
        self.assertAlmostEqual(cam.fov_y, expected_fov_y, places=6)


# ---------------------------------------------------------------------------
# Camera repr
# ---------------------------------------------------------------------------

class TestCameraRepr(unittest.TestCase):

    def test_repr_is_string(self):
        cam = _default_cam()
        s   = repr(cam)
        self.assertIsInstance(s, str)
        self.assertIn("640", s)
        self.assertIn("480", s)


# ---------------------------------------------------------------------------
# camera_from_colmap
# ---------------------------------------------------------------------------

class TestCameraFromColmap(unittest.TestCase):

    def test_identity_rotation_zero_translation(self):
        """R=I, t=0 → position at world origin."""
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        cam = camera_from_colmap(R=R, t=t, fx=500, fy=500, cx=320, cy=240,
                                 width=640, height=480)
        np.testing.assert_allclose(cam.position, [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(cam.R, R, atol=1e-6)

    def test_translation_recovers_position(self):
        """t = -(R @ pos) → camera_from_colmap must recover pos from t."""
        R   = np.eye(3, dtype=np.float32)
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t   = -(R @ pos)
        cam = camera_from_colmap(R=R, t=t, fx=500, fy=500, cx=320, cy=240,
                                 width=640, height=480)
        np.testing.assert_allclose(cam.position, pos, atol=1e-6)

    def test_non_identity_rotation(self):
        """Rotation matrix is stored as-is (world-to-camera convention)."""
        R = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [-1, 0, 0]], dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        cam = camera_from_colmap(R=R, t=t, fx=500, fy=500, cx=320, cy=240,
                                 width=640, height=480)
        np.testing.assert_allclose(cam.R, R, atol=1e-6)

    def test_intrinsics_preserved(self):
        cam = camera_from_colmap(R=np.eye(3), t=np.zeros(3),
                                 fx=800, fy=750, cx=310, cy=235,
                                 width=640, height=480)
        self.assertAlmostEqual(cam.fx,     800.0)
        self.assertAlmostEqual(cam.fy,     750.0)
        self.assertAlmostEqual(cam.cx,     310.0)
        self.assertAlmostEqual(cam.cy,     235.0)
        self.assertEqual(cam.width,  640)
        self.assertEqual(cam.height, 480)


# ---------------------------------------------------------------------------
# auto_consistency_cameras
# ---------------------------------------------------------------------------

class TestAutoConsistencyCameras(unittest.TestCase):

    def test_returns_correct_count(self):
        primary = _default_cam()
        cams = auto_consistency_cameras(primary, count=4)
        self.assertEqual(len(cams), 4)

    def test_single_camera(self):
        primary = _default_cam()
        cams = auto_consistency_cameras(primary, count=1)
        self.assertEqual(len(cams), 1)

    def test_intrinsics_match_primary(self):
        primary = _default_cam(width=800, height=600)
        for cam in auto_consistency_cameras(primary, count=4):
            self.assertEqual(cam.width,  primary.width)
            self.assertEqual(cam.height, primary.height)
            self.assertAlmostEqual(cam.fx, primary.fx)
            self.assertAlmostEqual(cam.fy, primary.fy)

    def test_position_matches_primary(self):
        """Consistency cameras rotate orientation, not position."""
        primary = _default_cam()
        for cam in auto_consistency_cameras(primary, count=4):
            np.testing.assert_allclose(cam.position, primary.position, atol=1e-6)

    def test_rotation_differs_from_primary(self):
        """Each consistency camera must have a different yaw than the primary."""
        primary = _default_cam()
        for cam in auto_consistency_cameras(primary, count=4):
            self.assertFalse(
                np.allclose(cam.R, primary.R, atol=1e-5),
                msg="Consistency camera should have a different rotation from primary.",
            )

    def test_cameras_alternate_left_right(self):
        """With a 30° step the first two cameras should differ by ~60°."""
        primary = _default_cam()
        cams = auto_consistency_cameras(primary, count=2, yaw_step_deg=30.0)
        # Both cameras should differ from each other (opposite yaw signs)
        self.assertFalse(
            np.allclose(cams[0].R, cams[1].R, atol=1e-5),
            msg="First two consistency cameras should not be identical.",
        )


if __name__ == "__main__":
    unittest.main()
