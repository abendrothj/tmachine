"""
tests/test_splat_mutator.py  –  Tier 2: core optimisation loop

Proves that gradients flow correctly through the Spherical Harmonics and that
the Adam optimizer actually reduces loss over three iterations.

Design principles
-----------------
- Uses a tiny 50-splat synthetic cloud (no real .ply needed).
- Creates a synthetic "target" image directly as a tensor (no AI model needed).
- Runs exactly 3 iterations — fast enough for CI, yet proves the math.
- Device-agnostic: runs on CPU when CUDA is unavailable.
- The only assertion that matters: result.final_loss < result.initial_loss
  This proves the backward pass through gsplat → SH coefficients is wired correctly.

Note on gsplat availability
---------------------------
gsplat requires a compiled CUDA extension.  If it is not installed in the test
environment, these tests are skipped rather than errored so that a basic
CPU-only CI pipeline can still run Tier-1 tests without failing here.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np
import torch

# --------------------------------------------------------------------------
# Skip the entire module if CUDA is unavailable.
# gsplat can be *imported* without CUDA, but its rasterization kernel
# requires compiled CUDA extensions which are absent on CPU-only machines.
# --------------------------------------------------------------------------
_GSPLAT_AVAILABLE = torch.cuda.is_available()
_SKIP_REASON = "CUDA not available — gsplat rasterization requires compiled CUDA extensions"


def _make_camera(width: int = 64, height: int = 64):
    """Tiny camera suitable for the 50-splat synthetic scene."""
    from tmachine.utils.camera import camera_from_fov
    return camera_from_fov(
        position=(0.0, 0.0, -3.0),
        pitch=0.0, yaw=0.0, roll=0.0,
        fov_x=math.radians(60),
        width=width,
        height=height,
    )


def _make_cloud(n: int = 50, device: str = "cpu"):
    """
    Build a minimal synthetic GaussianCloud that gsplat can rasterize.

    Splats are placed in a small cluster in front of the camera (z ≈ 0),
    with mild opacities and tiny scales so the scene renders a visible but
    low-complexity image.
    """
    from tmachine.io.ply_handler import GaussianCloud

    rng = np.random.default_rng(seed=99)

    means = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
    means[:, 2] = 0.0   # place all splats on the z=0 plane

    # Unit quaternions (w=1, xyz=0) — axis-aligned Gaussians
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    log_scales    = np.full((n, 3), fill_value=-3.5, dtype=np.float32)  # small splats
    raw_opacities = np.full(n, fill_value=1.0, dtype=np.float32)        # high opacity
    sh_dc         = rng.uniform(0.1, 0.5, (n, 3)).astype(np.float32)
    sh_rest       = np.zeros((n, 15, 3), dtype=np.float32)              # degree-3

    return GaussianCloud(
        means=torch.from_numpy(means).to(device),
        quats=torch.from_numpy(quats).to(device),
        log_scales=torch.from_numpy(log_scales).to(device),
        raw_opacities=torch.from_numpy(raw_opacities).to(device),
        sh_dc=torch.from_numpy(sh_dc).to(device),
        sh_rest=torch.from_numpy(sh_rest).to(device),
    )


@unittest.skipUnless(_GSPLAT_AVAILABLE, _SKIP_REASON)
class TestSplatMutatorGradientFlow(unittest.TestCase):
    """The optimizer must reduce loss within 3 iterations."""

    def setUp(self):
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.camera  = _make_camera()
        self.cloud   = _make_cloud(device=self.device)

    def _save_cloud_to_temp(self) -> str:
        from tmachine.io.ply_handler import save_ply
        fd, path = tempfile.mkstemp(suffix=".ply")
        os.close(fd)
        save_ply(self.cloud, path)
        return path

    def _make_target_image(self) -> torch.Tensor:
        """
        Render the scene once, then slightly brighten it to create a target.
        We avoid calling ImageEditor so no AI model is required.
        """
        from tmachine.core.renderer import ViewportRenderer
        renderer = ViewportRenderer(device=self.device)
        original = renderer.render(self.camera, gaussians=self.cloud, sh_degree=3)
        # Small perturbation: push brightness up by 0.05
        return (original + 0.05).clamp(0.0, 1.0).detach()

    def test_loss_decreases_after_3_iters(self):
        """Same test, with proper camera argument."""
        from tmachine.core.splat_mutator import SplatMutator

        ply_path     = self._save_cloud_to_temp()
        target_image = self._make_target_image()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            patch_path = f.name

        try:
            mutator = SplatMutator(ply_path, device=self.device)
            result  = mutator.mutate(
                camera=self.camera,
                edited_image=target_image,
                n_iters=3,
                patch_path=patch_path,
                sh_degree=3,
                convergence_threshold=0.0,
            )
        finally:
            os.unlink(ply_path)
            if os.path.exists(patch_path):
                os.unlink(patch_path)

        self.assertLess(
            result.final_loss,
            result.initial_loss,
            msg=(
                f"Loss did not decrease: initial={result.initial_loss:.6f}, "
                f"final={result.final_loss:.6f}. "
                "Gradients are not flowing through the SH coefficients."
            ),
        )

    def test_iterations_run_equals_requested(self):
        """With convergence_threshold=0, all 3 steps must execute."""
        from tmachine.core.splat_mutator import SplatMutator

        ply_path     = self._save_cloud_to_temp()
        target_image = self._make_target_image()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            patch_path = f.name

        try:
            mutator = SplatMutator(ply_path, device=self.device)
            result  = mutator.mutate(
                camera=self.camera,
                edited_image=target_image,
                n_iters=3,
                patch_path=patch_path,
                convergence_threshold=0.0,
            )
        finally:
            os.unlink(ply_path)
            if os.path.exists(patch_path):
                os.unlink(patch_path)

        self.assertEqual(result.iterations_run, 3)

    def test_patch_file_is_valid_ply(self):
        """The patch .ply written by mutate() must be loadable."""
        from tmachine.core.splat_mutator import SplatMutator
        from tmachine.io.ply_handler import load_ply

        ply_path     = self._save_cloud_to_temp()
        target_image = self._make_target_image()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            patch_path = f.name

        try:
            mutator = SplatMutator(ply_path, device=self.device)
            result  = mutator.mutate(
                camera=self.camera,
                edited_image=target_image,
                n_iters=3,
                patch_path=patch_path,
                convergence_threshold=0.0,
            )
            loaded = load_ply(result.patch_path, device="cpu")
            self.assertEqual(loaded.sh_rest.shape[0], result.changed_splat_count)
        finally:
            os.unlink(ply_path)
            if os.path.exists(patch_path):
                os.unlink(patch_path)

    def test_loss_history_length(self):
        """loss_history must have exactly iterations_run entries."""
        from tmachine.core.splat_mutator import SplatMutator

        ply_path     = self._save_cloud_to_temp()
        target_image = self._make_target_image()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            patch_path = f.name

        try:
            mutator = SplatMutator(ply_path, device=self.device)
            result  = mutator.mutate(
                camera=self.camera,
                edited_image=target_image,
                n_iters=3,
                patch_path=patch_path,
                convergence_threshold=0.0,
            )
        finally:
            os.unlink(ply_path)
            if os.path.exists(patch_path):
                os.unlink(patch_path)

        self.assertEqual(len(result.loss_history), result.iterations_run)

    def test_no_nan_in_loss_history(self):
        """
        Vulnerability C fix: eps=1e-8 must prevent NaN losses.
        If any loss value is NaN, the fix was not applied or is insufficient.
        """
        from tmachine.core.splat_mutator import SplatMutator

        ply_path     = self._save_cloud_to_temp()
        target_image = self._make_target_image()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            patch_path = f.name

        try:
            mutator = SplatMutator(ply_path, device=self.device)
            result  = mutator.mutate(
                camera=self.camera,
                edited_image=target_image,
                n_iters=20,      # more iterations to surface potential NaN instability
                patch_path=patch_path,
            )
        finally:
            os.unlink(ply_path)
            if os.path.exists(patch_path):
                os.unlink(patch_path)

        nan_iters = [i for i, v in enumerate(result.loss_history) if math.isnan(v)]
        self.assertEqual(
            nan_iters, [],
            msg=f"NaN loss at iterations {nan_iters}. Adam eps may be too small.",
        )


@unittest.skipUnless(_GSPLAT_AVAILABLE, _SKIP_REASON)
class TestSplatMutatorGeometryOptimization(unittest.TestCase):
    """optimize_geometry=True: positions/scales/rotations receive gradients."""

    def setUp(self):
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.camera  = _make_camera()
        self.cloud   = _make_cloud(device=self.device)

    def _save_cloud_to_temp(self) -> str:
        from tmachine.io.ply_handler import save_ply
        fd, path = tempfile.mkstemp(suffix=".ply")
        os.close(fd)
        save_ply(self.cloud, path)
        return path

    def _make_target_image(self) -> torch.Tensor:
        from tmachine.core.renderer import ViewportRenderer
        renderer = ViewportRenderer(device=self.device)
        original = renderer.render(self.camera, gaussians=self.cloud, sh_degree=3)
        return (original + 0.05).clamp(0.0, 1.0).detach()

    def test_loss_decreases_with_geometry_optimization(self):
        """Loss must still decrease when geometry parameters are also optimised."""
        from tmachine.core.splat_mutator import SplatMutator

        ply_path     = self._save_cloud_to_temp()
        target_image = self._make_target_image()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            patch_path = f.name

        try:
            mutator = SplatMutator(ply_path, device=self.device)
            result  = mutator.mutate(
                camera=self.camera,
                edited_image=target_image,
                n_iters=3,
                patch_path=patch_path,
                sh_degree=3,
                optimize_geometry=True,
                convergence_threshold=0.0,
            )
        finally:
            os.unlink(ply_path)
            if os.path.exists(patch_path):
                os.unlink(patch_path)

        self.assertLess(
            result.final_loss,
            result.initial_loss,
            msg=(
                f"Loss did not decrease with optimize_geometry=True: "
                f"initial={result.initial_loss:.6f}, final={result.final_loss:.6f}"
            ),
        )

    def test_quaternion_normalization_preserved(self):
        """
        With optimize_geometry=True, the patch splats must have unit quaternions.
        The normalization applied during the loop must survive into the saved patch.
        """
        from tmachine.core.splat_mutator import SplatMutator
        from tmachine.io.ply_handler import load_ply

        ply_path     = self._save_cloud_to_temp()
        target_image = self._make_target_image()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            patch_path = f.name

        try:
            mutator = SplatMutator(ply_path, device=self.device)
            result  = mutator.mutate(
                camera=self.camera,
                edited_image=target_image,
                n_iters=3,
                patch_path=patch_path,
                optimize_geometry=True,
                convergence_threshold=0.0,
            )
            if result.changed_splat_count > 0:
                patch = load_ply(result.patch_path, device="cpu")
                norms = patch.quats.norm(dim=-1)
                max_dev = (norms - 1.0).abs().max().item()
                self.assertAlmostEqual(
                    max_dev, 0.0, places=5,
                    msg=f"Quaternions not unit-length after geometry optimization (max dev={max_dev:.2e})",
                )
        finally:
            os.unlink(ply_path)
            if os.path.exists(patch_path):
                os.unlink(patch_path)

    def test_no_nan_in_geometry_loss_history(self):
        """Gradient clipping must prevent NaN when geometry is unlocked."""
        from tmachine.core.splat_mutator import SplatMutator

        ply_path     = self._save_cloud_to_temp()
        target_image = self._make_target_image()

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            patch_path = f.name

        try:
            mutator = SplatMutator(ply_path, device=self.device)
            result  = mutator.mutate(
                camera=self.camera,
                edited_image=target_image,
                n_iters=20,
                patch_path=patch_path,
                optimize_geometry=True,
            )
        finally:
            os.unlink(ply_path)
            if os.path.exists(patch_path):
                os.unlink(patch_path)

        nan_iters = [i for i, v in enumerate(result.loss_history) if math.isnan(v)]
        self.assertEqual(nan_iters, [], msg=f"NaN loss at iterations {nan_iters}")


if __name__ == "__main__":
    unittest.main()
