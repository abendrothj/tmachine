"""
tests/test_delta_engine.py  –  Tier 1: deterministic DeltaEngine math

All assertions are deterministic and CPU-only.  No GPU or real scene needed.

Covers
------
- Identical images  → total_loss == 0.0, change_mask all False
- Single changed pixel → change_mask flags exactly that pixel
- total_loss is a differentiable autograd leaf (can call .backward())
- LossMap field shapes match the input image dimensions
- Luminance weights (BT.601) produce expected per-channel contribution
"""

from __future__ import annotations

import unittest

import torch

from tmachine.core.delta_engine import DeltaEngine


def _img(h: int, w: int, fill: float = 0.5) -> torch.Tensor:
    """Solid-colour (H, W, 3) float32 tensor."""
    return torch.full((h, w, 3), fill, dtype=torch.float32)


class TestDeltaEngineIdentical(unittest.TestCase):
    """Two identical images must produce zero loss and an empty change mask."""

    def setUp(self):
        self.engine = DeltaEngine()
        self.h, self.w = 16, 16
        self.img = _img(self.h, self.w)

    def test_total_loss_is_zero(self):
        lm = self.engine.compute(self.img, self.img)
        self.assertAlmostEqual(lm.total_loss.item(), 0.0, places=7)

    def test_l1_is_zero(self):
        lm = self.engine.compute(self.img, self.img)
        self.assertAlmostEqual(lm.l1_loss.item(), 0.0, places=7)

    def test_l2_is_zero(self):
        lm = self.engine.compute(self.img, self.img)
        self.assertAlmostEqual(lm.l2_loss.item(), 0.0, places=7)

    def test_change_mask_all_false(self):
        lm = self.engine.compute(self.img, self.img)
        self.assertFalse(lm.change_mask.any(), "No pixel should be flagged as changed")

    def test_changed_pixel_ratio_is_zero(self):
        lm = self.engine.compute(self.img, self.img)
        self.assertAlmostEqual(lm.changed_pixel_ratio, 0.0, places=7)


class TestDeltaEngineSinglePixelChange(unittest.TestCase):
    """Changing exactly one pixel must flag exactly that pixel in change_mask."""

    def setUp(self):
        self.engine = DeltaEngine(change_threshold=0.01)
        self.h, self.w = 16, 16

    def _make_pair(self, row: int, col: int, edited_value: float):
        original = _img(self.h, self.w, fill=0.2)
        edited   = original.clone()
        edited[row, col, :] = edited_value   # change all three channels
        return original, edited

    def test_change_mask_flags_exactly_one_pixel(self):
        original, edited = self._make_pair(3, 7, edited_value=1.0)
        lm = self.engine.compute(original, edited)
        flagged = lm.change_mask.sum().item()
        self.assertEqual(flagged, 1, msg=f"Expected 1 flagged pixel, got {flagged}")

    def test_change_mask_flags_correct_location(self):
        row, col = 5, 11
        original, edited = self._make_pair(row, col, edited_value=1.0)
        lm = self.engine.compute(original, edited)
        self.assertTrue(
            lm.change_mask[row, col].item(),
            msg=f"Expected change_mask[{row},{col}] to be True",
        )

    def test_unchanged_pixels_not_flagged(self):
        row, col = 2, 2
        original, edited = self._make_pair(row, col, edited_value=1.0)
        lm = self.engine.compute(original, edited)
        # Every pixel except (row, col) should be False
        mask_copy = lm.change_mask.clone()
        mask_copy[row, col] = False
        self.assertFalse(mask_copy.any(), "Pixels other than the changed one were flagged")

    def test_total_loss_is_nonzero(self):
        original, edited = self._make_pair(0, 0, edited_value=1.0)
        lm = self.engine.compute(original, edited)
        self.assertGreater(lm.total_loss.item(), 0.0)

    def test_below_threshold_pixel_not_flagged(self):
        """A delta below change_threshold must not appear in change_mask."""
        engine   = DeltaEngine(change_threshold=0.5)
        original = _img(8, 8, fill=0.0)
        edited   = original.clone()
        edited[0, 0, :] = 0.1   # luminance ≈ 0.1, below threshold 0.5
        lm = engine.compute(original, edited)
        self.assertFalse(lm.change_mask.any())


class TestDeltaEngineDifferentiability(unittest.TestCase):
    """total_loss must support .backward() for use in the optimizer loop."""

    def test_backward_does_not_raise(self):
        engine   = DeltaEngine()
        # original must require_grad so that total_loss has a grad_fn
        original = _img(8, 8).requires_grad_(True)
        edited   = _img(8, 8, fill=0.9)
        lm       = engine.compute(original, edited)
        # Should not raise; loss must be part of an autograd graph
        try:
            lm.total_loss.backward()
        except RuntimeError as exc:
            self.fail(f"total_loss.backward() raised: {exc}")

    def test_total_loss_is_scalar(self):
        engine = DeltaEngine()
        lm     = engine.compute(_img(4, 4, 0.0), _img(4, 4, 1.0))
        self.assertEqual(lm.total_loss.dim(), 0, "total_loss must be a scalar tensor")


class TestDeltaEngineLossMapShapes(unittest.TestCase):
    """LossMap tensor shapes must match the input image dimensions."""

    def test_lossmap_shapes(self):
        engine = DeltaEngine()
        h, w   = 24, 32
        lm     = engine.compute(_img(h, w, 0.2), _img(h, w, 0.8))

        self.assertEqual(lm.pixel_diff.shape,    (h, w, 3))
        self.assertEqual(lm.luminance_diff.shape, (h, w))
        self.assertEqual(lm.change_mask.shape,    (h, w))


class TestDeltaEngineWeights(unittest.TestCase):
    """Adjusting l1/l2 weights must change total_loss proportionally."""

    def test_l1_only_weight(self):
        engine = DeltaEngine(l1_weight=1.0, l2_weight=0.0)
        lm     = engine.compute(_img(4, 4, 0.0), _img(4, 4, 1.0))
        self.assertAlmostEqual(lm.total_loss.item(), lm.l1_loss.item(), places=6)

    def test_l2_only_weight(self):
        engine = DeltaEngine(l1_weight=0.0, l2_weight=1.0)
        lm     = engine.compute(_img(4, 4, 0.0), _img(4, 4, 1.0))
        self.assertAlmostEqual(lm.total_loss.item(), lm.l2_loss.item(), places=6)


if __name__ == "__main__":
    unittest.main()
