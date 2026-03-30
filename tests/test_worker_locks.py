"""
tests/test_worker_locks.py  –  Tier 3b: FileLock + Celery task safety tests

Tests the file-locking and retry behaviour of the bake_patch Celery task
without requiring a live Redis broker or database.

Strategy
--------
All external I/O is monkeypatched:
  - Database session is replaced with an in-memory SQLite session.
  - SplatMutator is replaced with a no-op stub.
  - Celery is invoked via .apply() (synchronous, in-process, no broker needed).

Covers
------
- Lock timeout triggers self.retry() (file-lock contract)
- bake_patch writes a MemoryLayer row on success (happy path stub)
- generate_preview saves a preview file and returns preview_filename (happy path stub)
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

# --------------------------------------------------------------------------
# Skip if Celery / SQLAlchemy / filelock are not installed
# --------------------------------------------------------------------------
try:
    from celery import Celery
    from filelock import FileLock
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

_SKIP_REASON = "celery/filelock/sqlalchemy not installed (pip install tmachine[api])"


def _make_in_memory_session():
    """Return a SQLAlchemy session factory backed by a shared in-memory SQLite DB."""
    from sqlalchemy.pool import StaticPool
    from tmachine.db.models import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestFileLockTimeout(unittest.TestCase):
    """
    Holding a lock on the patch file while bake_patch runs must trigger
    self.retry() rather than silently proceeding with a corrupt write.
    """

    def test_lock_timeout_triggers_retry(self):
        from filelock import Timeout as _LockTimeout
        from tmachine.api.worker import bake_patch

        import base64, io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (64, 64)).save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        cam = {
            "x": 0, "y": 0, "z": -5,
            "pitch": 0, "yaw": 0, "roll": 0,
            "fov_x": math.radians(60),
            "width": 64, "height": 64,
            "near": 0.01, "far": 1000.0,
        }

        Session = _make_in_memory_session()

        with patch("tmachine.db.session.SessionLocal", Session):
            with patch("filelock.FileLock") as MockLock:
                MockLock.return_value.__enter__.side_effect = _LockTimeout("/fake/lock")
                result = bake_patch.apply(args=["/fake/scene.ply", image_b64, cam])

        self.assertNotEqual(
            result.state, "SUCCESS",
            msg="Task returned SUCCESS even though every FileLock attempt timed out.",
        )


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestBakePatchHappyPath(unittest.TestCase):
    """
    On success, bake_patch must write a MemoryLayer row with correct fields.
    SplatMutator is fully stubbed — no GPU / real .ply needed.
    """

    def setUp(self):
        self.Session = _make_in_memory_session()

    def test_memory_layer_written_on_success(self):
        from tmachine.db.models import MemoryLayer
        from tmachine.api.worker import bake_patch

        import base64, io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (64, 64)).save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        cam = {
            "x": 0, "y": 0, "z": -5,
            "pitch": 0, "yaw": 0, "roll": 0,
            "fov_x": math.radians(60),
            "width": 64, "height": 64,
            "near": 0.01, "far": 1000.0,
        }

        fake_result = MagicMock()
        fake_result.patch_path           = "/fake/patch.ply"
        fake_result.hidden_indices       = [1, 2, 3]
        fake_result.changed_splat_count  = 3
        fake_result.initial_loss         = 0.05
        fake_result.final_loss           = 0.01
        fake_result.iterations_run       = 3

        with patch("tmachine.db.session.SessionLocal", self.Session):
            with patch("tmachine.core.splat_mutator.SplatMutator") as MockMutator:
                MockMutator.return_value.mutate.return_value = fake_result
                with patch("filelock.FileLock"):
                    with patch("tmachine.api.worker._LOCK_TIMEOUT", 10):
                        result = bake_patch.apply(
                            args=["/fake/scene.ply", image_b64, cam],
                            kwargs={"external_ref": "proposal-99"},
                        ).get()

        self.assertEqual(result["patch_path"], "/fake/patch.ply")
        self.assertEqual(result["changed_splat_count"], 3)

        with self.Session() as db:
            layers = db.query(MemoryLayer).all()
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0].patch_path, "/fake/patch.ply")
            self.assertEqual(layers[0].hidden_indices, [1, 2, 3])
            self.assertEqual(layers[0].external_ref, "proposal-99")


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestGeneratePreviewHappyPath(unittest.TestCase):
    """
    generate_preview must save a PNG file and return preview_filename + prompt.
    ImageEditor and ViewportRenderer are fully stubbed.
    """

    def test_preview_file_saved(self):
        from tmachine.api.worker import generate_preview

        cam = {
            "x": 0, "y": 0, "z": -5,
            "pitch": 0, "yaw": 0, "roll": 0,
            "fov_x": math.radians(60),
            "width": 64, "height": 64,
            "near": 0.01, "far": 1000.0,
        }

        mock_render = torch.ones(64, 64, 3)

        from PIL import Image as PILImage
        fake_pil = PILImage.new("RGB", (64, 64), color=(128, 128, 128))

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("tmachine.api.worker._PREVIEW_DIR",
                       __import__("pathlib").Path(tmp_dir)):
                with patch("tmachine.core.renderer.ViewportRenderer") as MockRenderer:
                    MockRenderer.return_value.render.return_value = mock_render
                    with patch("tmachine.ai.image_editor.ImageEditor") as MockEditor:
                        MockEditor.return_value.edit.return_value  = fake_pil
                        MockEditor.return_value.unload.return_value = None
                        result = generate_preview.apply(
                            args=["/fake/scene.ply", cam, "change awning to green"]
                        ).get()

        self.assertIn("preview_filename", result)
        self.assertIn("preview_path", result)
        self.assertEqual(result["prompt"], "change awning to green")
        self.assertTrue(result["preview_filename"].endswith(".png"))


if __name__ == "__main__":
    unittest.main()
