"""
tests/test_ply_handler.py  –  Tier 1: deterministic I/O tests

Exercises load_ply / save_ply with synthetic in-memory clouds.
No CUDA or real scene files required.

Covers
------
- Round-trip fidelity (all SH degrees: 0, 1, 2, 3)
- Dynamic SH degree inference (Vulnerability A fix)
- Atomic write: temp file is cleaned up; output only appears on success
  (Vulnerability B fix)
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

from tmachine.io.ply_handler import GaussianCloud, load_ply, save_ply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cloud(n: int, sh_degree: int, device: str = "cpu") -> GaussianCloud:
    """Create a deterministic synthetic GaussianCloud with *n* splats."""
    rng = np.random.default_rng(seed=42)

    # Number of rest SH basis functions: (deg+1)^2 - 1
    n_rest = (sh_degree + 1) ** 2 - 1

    raw_quats = rng.standard_normal((n, 4)).astype(np.float32)
    norms     = np.linalg.norm(raw_quats, axis=1, keepdims=True)
    quats     = raw_quats / np.where(norms > 0, norms, 1.0)   # unit quaternions

    return GaussianCloud(
        means=torch.from_numpy(rng.standard_normal((n, 3)).astype(np.float32)).to(device),
        quats=torch.from_numpy(quats).to(device),
        log_scales=torch.from_numpy(rng.standard_normal((n, 3)).astype(np.float32)).to(device),
        raw_opacities=torch.from_numpy(rng.standard_normal(n).astype(np.float32)).to(device),
        sh_dc=torch.from_numpy(rng.standard_normal((n, 3)).astype(np.float32)).to(device),
        sh_rest=torch.from_numpy(rng.standard_normal((n, n_rest, 3)).astype(np.float32)).to(device),
    )


def _round_trip(cloud: GaussianCloud) -> GaussianCloud:
    """Save *cloud* to a temp file and load it back."""
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = f.name
    try:
        save_ply(cloud, tmp)
        return load_ply(tmp, device=str(cloud.means.device))
    finally:
        os.unlink(tmp)
        # Also clean up any leftover .tmp file from a failed atomic write
        for leftover in Path(tmp).parent.glob(Path(tmp).stem + "_*.ply.tmp"):
            leftover.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Round-trip tests (one per SH degree)
# ---------------------------------------------------------------------------

class TestPlyRoundTrip(unittest.TestCase):

    TOLERANCE = 1e-6

    def _assert_tensors_close(self, a: torch.Tensor, b: torch.Tensor, name: str) -> None:
        self.assertEqual(a.shape, b.shape, msg=f"{name}: shape mismatch {a.shape} vs {b.shape}")
        if a.numel() == 0:
            return   # empty tensors (e.g. sh_rest for degree-0) — nothing to compare
        max_err = (a - b).abs().max().item()
        self.assertLess(
            max_err, self.TOLERANCE,
            msg=f"{name}: max absolute error {max_err:.2e} exceeds tolerance {self.TOLERANCE}",
        )

    def _round_trip_assert(self, sh_degree: int) -> None:
        original = _make_cloud(10, sh_degree)
        loaded   = _round_trip(original)

        self._assert_tensors_close(original.means,         loaded.means,         "means")
        self._assert_tensors_close(original.quats,         loaded.quats,         "quats")
        self._assert_tensors_close(original.log_scales,    loaded.log_scales,    "log_scales")
        self._assert_tensors_close(original.raw_opacities, loaded.raw_opacities, "raw_opacities")
        self._assert_tensors_close(original.sh_dc,         loaded.sh_dc,         "sh_dc")
        self._assert_tensors_close(original.sh_rest,       loaded.sh_rest,       "sh_rest")

    def test_round_trip_sh_degree_0(self):
        """Degree-0 cloud has no rest coefficients; sh_rest shape must survive as (N, 0, 3)."""
        self._round_trip_assert(sh_degree=0)

    def test_round_trip_sh_degree_1(self):
        self._round_trip_assert(sh_degree=1)

    def test_round_trip_sh_degree_2(self):
        self._round_trip_assert(sh_degree=2)

    def test_round_trip_sh_degree_3(self):
        """Standard Inria-trained scene with 15 rest coefficients per splat."""
        self._round_trip_assert(sh_degree=3)


# ---------------------------------------------------------------------------
# Vulnerability A: dynamic SH degree inference
# ---------------------------------------------------------------------------

class TestSHDegreeInference(unittest.TestCase):

    def test_degree_1_preserves_n_rest(self):
        """Saving a degree-1 scene (3 rest SH) and loading must NOT inflate to 15."""
        cloud  = _make_cloud(5, sh_degree=1)
        loaded = _round_trip(cloud)
        self.assertEqual(
            loaded.sh_rest.shape,
            (5, 3, 3),  # (N, n_rest=3, 3 channels)
            msg=f"Expected sh_rest shape (5,3,3) but got {loaded.sh_rest.shape}. "
                "load_ply may be hardcoding degree-3.",
        )

    def test_degree_0_preserves_empty_rest(self):
        """Degree-0 cloud must come back with sh_rest shape (N, 0, 3), not (N, 15, 3)."""
        cloud  = _make_cloud(4, sh_degree=0)
        loaded = _round_trip(cloud)
        self.assertEqual(
            loaded.sh_rest.shape,
            (4, 0, 3),
            msg=f"Expected sh_rest shape (4,0,3) but got {loaded.sh_rest.shape}. "
                "load_ply is incorrectly defaulting to degree-3 zeros.",
        )


# ---------------------------------------------------------------------------
# Vulnerability B: atomic write behaviour
# ---------------------------------------------------------------------------

class TestAtomicWrite(unittest.TestCase):

    def test_no_partial_file_on_normal_write(self):
        """After a successful save_ply, no .tmp sibling file should remain."""
        cloud = _make_cloud(5, sh_degree=3)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "scene.ply")
            save_ply(cloud, out)
            tmp_files = list(Path(d).glob("*.tmp"))
            self.assertEqual(
                tmp_files, [],
                msg=f"Leftover temp files after successful save: {tmp_files}",
            )
            self.assertTrue(os.path.exists(out), "Output .ply not written")

    def test_output_is_valid_after_write(self):
        """File written by save_ply must be loadable immediately afterwards."""
        cloud  = _make_cloud(8, sh_degree=3)
        loaded = _round_trip(cloud)
        self.assertEqual(len(loaded), 8)

    def test_concurrent_writes_to_different_paths(self):
        """Two threads writing different patch files must not corrupt each other."""
        errors: list[Exception] = []

        def _write(idx: int, d: str) -> None:
            try:
                cloud = _make_cloud(5, sh_degree=3)
                save_ply(cloud, os.path.join(d, f"patch_{idx}.ply"))
            except Exception as exc:
                errors.append(exc)

        with tempfile.TemporaryDirectory() as d:
            threads = [threading.Thread(target=_write, args=(i, d)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(errors, [], msg=f"Concurrent write errors: {errors}")


# ---------------------------------------------------------------------------
# GaussianCloud convenience properties
# ---------------------------------------------------------------------------

class TestGaussianCloudProperties(unittest.TestCase):

    def test_scales_are_activated(self):
        cloud = _make_cloud(6, sh_degree=3)
        self.assertTrue(
            (cloud.scales > 0).all(),
            "scales property should be exp(log_scales), always positive",
        )

    def test_opacities_in_unit_range(self):
        cloud = _make_cloud(6, sh_degree=3)
        ops = cloud.opacities
        self.assertTrue(
            ((ops > 0) & (ops < 1)).all(),
            "opacities property should be sigmoid(raw_opacities) ∈ (0, 1)",
        )

    def test_sh_all_shape_degree_3(self):
        cloud = _make_cloud(6, sh_degree=3)
        # (deg+1)^2 = 16 total basis functions
        self.assertEqual(cloud.sh_all.shape, (6, 16, 3))

    def test_sh_all_shape_degree_1(self):
        cloud = _make_cloud(6, sh_degree=1)
        # (1+1)^2 = 4 total basis functions
        self.assertEqual(cloud.sh_all.shape, (6, 4, 3))


if __name__ == "__main__":
    unittest.main()
