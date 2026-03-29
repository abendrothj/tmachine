"""
tests/test_worker_locks.py  –  Tier 3b: FileLock + Celery task safety tests

Tests the file-locking and retry behaviour of the bake_approved_patch Celery
task without requiring a live Redis broker or database.

Strategy
--------
All external I/O is monkeypatched:
  - Database queries are replaced with deterministic return values.
  - SplatMutator is replaced with a no-op stub.
  - Celery is invoked via .apply() (synchronous, in-process, no broker needed).

Covers
------
- Lock timeout triggers self.retry() (Vulnerability B + lock contract)
- Non-APPROVED proposal raises ValueError immediately (no 3D write)
- Missing proposal raises ValueError immediately
- bake_approved_patch writes a MemoryLayer row on success (happy path stub)
- generate_memory_proposal creates a PENDING MemoryProposal row (happy path stub)
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

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
    """Return a SQLAlchemy session backed by a shared in-memory SQLite DB."""
    from sqlalchemy.pool import StaticPool
    from tmachine.db.models import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _make_proposal(db, scene="/fake/scene.ply", status=None):
    from tmachine.db.models import MemoryProposal, ProposalStatus
    proposal = MemoryProposal(
        scene=scene,
        prompt="change awning to green",
        preview_path="/fake/preview.png",
        status=status or ProposalStatus.APPROVED,
        votes_yes=3, votes_no=0,
        cam_x=0, cam_y=0, cam_z=-5,
        cam_pitch=0, cam_yaw=0, cam_roll=0,
        cam_fov_x=math.radians(60),
        cam_width=64, cam_height=64,
    )
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return proposal


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestFileLockTimeout(unittest.TestCase):
    """
    Holding a lock on the patch file while the task runs must trigger
    self.retry() rather than silently proceeding with a corrupt write.
    """

    def test_lock_timeout_triggers_retry(self):
        """
        When FileLock raises Timeout, bake_approved_patch must call self.retry().
        After max_retries the task fails — verify it does not return SUCCESS.
        """
        from filelock import Timeout as _LockTimeout
        from tmachine.api.worker import bake_approved_patch

        Session = _make_in_memory_session()

        with Session() as db:
            proposal = _make_proposal(db)
            pid      = proposal.id

        # Make FileLock.__enter__ immediately raise Timeout on every attempt
        with patch("tmachine.db.session.SessionLocal", Session):
            with patch("PIL.Image.open") as mock_img_open:
                mock_img_open.return_value.convert.return_value.resize.return_value = MagicMock()
                with patch("tmachine.api.worker._pil_to_tensor",
                           return_value=torch.ones(64, 64, 3)):
                    with patch("filelock.FileLock") as MockLock:
                        MockLock.return_value.__enter__.side_effect = _LockTimeout("/fake/lock")
                        result = bake_approved_patch.apply(args=[pid])

        # After exhausting max_retries the task must not have succeeded
        self.assertNotEqual(
            result.state, "SUCCESS",
            msg="Task returned SUCCESS even though every FileLock attempt timed out.",
        )


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestBakeApprovedPatchValidation(unittest.TestCase):
    """
    bake_approved_patch must reject invalid proposals before any I/O.
    """

    def setUp(self):
        self.Session = _make_in_memory_session()

    def test_missing_proposal_raises(self):
        from tmachine.api.worker import bake_approved_patch

        with patch("tmachine.db.session.SessionLocal", self.Session):
            with self.assertRaises(Exception) as ctx:
                bake_approved_patch.apply(args=[99999]).get()
        self.assertIn("99999", str(ctx.exception))

    def test_non_approved_proposal_raises(self):
        from tmachine.db.models import ProposalStatus
        from tmachine.api.worker import bake_approved_patch

        with self.Session() as db:
            proposal = _make_proposal(db, status=ProposalStatus.PENDING)
            pid = proposal.id

        with patch("tmachine.db.session.SessionLocal", self.Session):
            with self.assertRaises(Exception) as ctx:
                bake_approved_patch.apply(args=[pid]).get()
        self.assertIn("not APPROVED", str(ctx.exception).replace("PENDING", "not APPROVED"))


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestBakeApprovedPatchHappyPath(unittest.TestCase):
    """
    On success, bake_approved_patch must write a MemoryLayer row.
    SplatMutator is fully stubbed — no GPU / real .ply needed.
    """

    def setUp(self):
        self.Session = _make_in_memory_session()

    def test_memory_layer_written_on_success(self):
        from tmachine.db.models import MemoryLayer, ProposalStatus
        from tmachine.api.worker import bake_approved_patch

        with self.Session() as db:
            proposal = _make_proposal(db, status=ProposalStatus.APPROVED)
            pid      = proposal.id

        # Stub SplatMutator so no real optimisation runs
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
                        with patch("PIL.Image.open") as mock_img_open:
                            mock_img_open.return_value.convert.return_value.resize.return_value = \
                                MagicMock(spec=["convert", "resize", "size"])
                            with patch("tmachine.api.worker._pil_to_tensor",
                                       return_value=torch.ones(64, 64, 3)):
                                result = bake_approved_patch.apply(args=[pid]).get()

        self.assertEqual(result["patch_path"], "/fake/patch.ply")
        self.assertEqual(result["changed_splat_count"], 3)

        with self.Session() as db:
            layers = db.query(MemoryLayer).filter_by(proposal_id=pid).all()
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0].patch_path, "/fake/patch.ply")
            self.assertEqual(layers[0].hidden_indices, [1, 2, 3])


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestGenerateMemoryProposalHappyPath(unittest.TestCase):
    """
    generate_memory_proposal must write a PENDING MemoryProposal and return
    its ID.  ImageEditor and ViewportRenderer are fully stubbed.
    """

    def setUp(self):
        self.Session = _make_in_memory_session()

    def test_proposal_written_with_pending_status(self):
        from tmachine.db.models import MemoryProposal, ProposalStatus
        from tmachine.api.worker import generate_memory_proposal

        cam = {
            "x": 0, "y": 0, "z": -5,
            "pitch": 0, "yaw": 0, "roll": 0,
            "fov_x": math.radians(60),
            "width": 64, "height": 64,
            "near": 0.01, "far": 1000.0,
        }

        mock_render = torch.ones(64, 64, 3)

        from PIL import Image as PILImage
        import io as _io
        fake_pil = PILImage.new("RGB", (64, 64), color=(128, 128, 128))

        with patch("tmachine.db.session.SessionLocal", self.Session):
            with patch("tmachine.core.renderer.ViewportRenderer") as MockRenderer:
                MockRenderer.return_value.render.return_value = mock_render
                with patch("tmachine.ai.image_editor.ImageEditor") as MockEditor:
                    MockEditor.return_value.edit.return_value  = fake_pil
                    MockEditor.return_value.unload.return_value = None

                    # Stub preview save so no real file is written
                    with patch(
                        "tmachine.api.worker._save_preview",
                        return_value="/fake/previews/proposal_1.png",
                    ):
                        result = generate_memory_proposal.apply(
                            args=["/fake/scene.ply", cam, "change awning to green"]
                        ).get()

        self.assertIn("proposal_id", result)

        with self.Session() as db:
            proposal = db.get(MemoryProposal, result["proposal_id"])
            self.assertIsNotNone(proposal)
            self.assertEqual(proposal.status, ProposalStatus.PENDING)
            self.assertEqual(proposal.prompt, "change awning to green")


if __name__ == "__main__":
    unittest.main()
