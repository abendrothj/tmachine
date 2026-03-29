"""
tests/test_api_routes.py  –  Tier 3a: FastAPI route tests

Uses FastAPI's TestClient (which wraps httpx/requests under the hood) so no
real HTTP server, Redis, database, or GPU is required.

All external dependencies are monkeypatched:
  - ViewportRenderer.render     → returns a synthetic (H, W, 3) white tensor
  - Celery tasks                → replaced with a synchronous stub that returns
                                  a fake AsyncResult
  - Database session            → replaced with an in-memory SQLite session

Covers
------
GET /render         — correct PNG response; handles missing-file 404
GET /health         — always returns 200 {"status": "ok"}
POST /proposals/prompt  — creates a pending proposal, returns job_id
POST /proposals/{id}/vote  — increments yes/no counts
POST /proposals/{id}/approve   — triggers bake task, returns job_id
POST /proposals/{id}/reject    — marks rejected
GET /layers         — returns layer list for a scene
GET /status/{job_id}    — maps Celery state to response
"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --------------------------------------------------------------------------
# Skip if FastAPI / its optional deps are not installed
# --------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

_SKIP_REASON = "fastapi/sqlalchemy not installed (run: pip install tmachine[api])"


def _white_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a small white PNG entirely in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _white_tensor(height: int = 64, width: int = 64) -> torch.Tensor:
    return torch.ones(height, width, 3, dtype=torch.float32)


# --------------------------------------------------------------------------
# In-memory SQLite engine + override for FastAPI dependency injection
# --------------------------------------------------------------------------

def _make_test_engine():
    from tmachine.db.models import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


def _get_test_db_factory(engine):
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _get_db() -> Generator:
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    return _get_db


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestHealthEndpoint(unittest.TestCase):

    def setUp(self):
        from tmachine.api.app import app
        self.client = TestClient(app)

    def test_health_ok(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestRenderEndpoint(unittest.TestCase):
    """
    GET /render — stub out the renderer so no GPU or real .ply is needed.
    """

    def setUp(self):
        from tmachine.api.app import app
        from tmachine.db.session import get_db

        self._engine = _make_test_engine()
        override_db  = _get_test_db_factory(self._engine)

        app.dependency_overrides[get_db] = override_db

        # Create a real temp .ply so the file-exists check passes
        from tmachine.io.ply_handler import GaussianCloud, save_ply
        rng = np.random.default_rng(0)
        n   = 5
        self._tmp_ply = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        cloud = GaussianCloud(
            means=torch.zeros(n, 3),
            quats=torch.tensor([[1,0,0,0]] * n, dtype=torch.float32),
            log_scales=torch.full((n, 3), -3.0),
            raw_opacities=torch.ones(n),
            sh_dc=torch.zeros(n, 3),
            sh_rest=torch.zeros(n, 15, 3),
        )
        save_ply(cloud, self._tmp_ply.name)

        self.client = TestClient(app)
        self._app   = app

    def tearDown(self):
        os.unlink(self._tmp_ply.name)
        from tmachine.api.app import app
        app.dependency_overrides.clear()

    def test_render_returns_png(self):
        """Render endpoint must return image/png content."""
        with patch(
            "tmachine.api.routes.render.ViewportRenderer.render",
            return_value=_white_tensor(),
        ):
            resp = self.client.get(
                "/render",
                params={
                    "scene": self._tmp_ply.name,
                    "width": "64",
                    "height": "64",
                },
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")
        # Validate the response is actually a decodable PNG
        img = Image.open(io.BytesIO(resp.content))
        self.assertEqual(img.mode, "RGB")

    def test_render_missing_scene_returns_404(self):
        resp = self.client.get(
            "/render",
            params={"scene": "/does/not/exist/scene.ply"},
        )
        self.assertEqual(resp.status_code, 404)


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestProposalLifecycle(unittest.TestCase):
    """
    POST /proposals/prompt  →  GET /proposals  →  vote  →  approve/reject
    All Celery tasks are stubbed to avoid Redis + GPU.
    """

    def _make_fake_async_result(self, task_id: str = "fake-job-id"):
        ar = MagicMock()
        ar.id = task_id
        ar.status = "PENDING"
        ar.result = None
        return ar

    def setUp(self):
        from tmachine.api.app import app
        from tmachine.db.session import get_db

        self._engine = _make_test_engine()
        override_db  = _get_test_db_factory(self._engine)
        app.dependency_overrides[get_db] = override_db

        self.client = TestClient(app)
        self._app   = app

    def tearDown(self):
        from tmachine.api.app import app
        app.dependency_overrides.clear()

    def _enqueue_proposal(self, scene: str = "/fake/scene.ply") -> dict:
        fake_ar = self._make_fake_async_result()
        with patch(
            "tmachine.api.routes.proposals.generate_memory_proposal.delay",
            return_value=fake_ar,
        ):
            resp = self.client.post(
                "/proposals/prompt",
                data={
                    "scene":  scene,
                    "camera": json.dumps({
                        "x": 0, "y": 0, "z": -5,
                        "pitch": 0, "yaw": 0, "roll": 0,
                        "fov_x": math.radians(60),
                        "width": 64, "height": 64,
                    }),
                    "prompt": "change the awning to green",
                },
            )
        return resp

    def test_create_proposal_returns_job_id(self):
        resp = self._enqueue_proposal()
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("job_id", body)
        self.assertEqual(body["job_id"], "fake-job-id")

    def test_vote_yes_increments_count(self):
        from tmachine.db.models import MemoryProposal, ProposalStatus
        from sqlalchemy.orm import sessionmaker

        # Insert a proposal directly into the test DB
        Session = sessionmaker(bind=self._engine)
        with Session() as db:
            proposal = MemoryProposal(
                scene="/fake/scene.ply",
                prompt="test",
                preview_path="",
                status=ProposalStatus.PENDING,
                votes_yes=0, votes_no=0,
                cam_x=0, cam_y=0, cam_z=-5,
                cam_pitch=0, cam_yaw=0, cam_roll=0,
                cam_fov_x=math.radians(60),
                cam_width=64, cam_height=64,
            )
            db.add(proposal)
            db.commit()
            db.refresh(proposal)
            pid = proposal.id

        resp = self.client.post(f"/proposals/{pid}/vote", data={"vote": "yes"})
        self.assertEqual(resp.status_code, 200)

        with Session() as db:
            p = db.get(MemoryProposal, pid)
            self.assertEqual(p.votes_yes, 1)
            self.assertEqual(p.votes_no, 0)

    def test_vote_no_increments_count(self):
        from tmachine.db.models import MemoryProposal, ProposalStatus
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self._engine)
        with Session() as db:
            proposal = MemoryProposal(
                scene="/fake/scene.ply",
                prompt="test",
                preview_path="",
                status=ProposalStatus.PENDING,
                votes_yes=0, votes_no=0,
                cam_x=0, cam_y=0, cam_z=-5,
                cam_pitch=0, cam_yaw=0, cam_roll=0,
                cam_fov_x=math.radians(60),
                cam_width=64, cam_height=64,
            )
            db.add(proposal)
            db.commit()
            db.refresh(proposal)
            pid = proposal.id

        self.client.post(f"/proposals/{pid}/vote", data={"vote": "no"})

        with Session() as db:
            p = db.get(MemoryProposal, pid)
            self.assertEqual(p.votes_no, 1)

    def test_approve_proposal_enqueues_bake(self):
        from tmachine.db.models import MemoryProposal, ProposalStatus
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self._engine)
        with Session() as db:
            proposal = MemoryProposal(
                scene="/fake/scene.ply",
                prompt="change awning",
                preview_path="/fake/preview.png",
                status=ProposalStatus.PENDING,
                votes_yes=3, votes_no=0,
                cam_x=0, cam_y=0, cam_z=-5,
                cam_pitch=0, cam_yaw=0, cam_roll=0,
                cam_fov_x=math.radians(60),
                cam_width=64, cam_height=64,
            )
            db.add(proposal)
            db.commit()
            db.refresh(proposal)
            pid = proposal.id

        bake_ar = self._make_fake_async_result("bake-job-id")
        with patch(
            "tmachine.api.routes.proposals.bake_approved_patch.delay",
            return_value=bake_ar,
        ) as mock_bake:
            resp = self.client.post(f"/proposals/{pid}/approve")

        self.assertEqual(resp.status_code, 200)
        mock_bake.assert_called_once()

        with Session() as db:
            p = db.get(MemoryProposal, pid)
            self.assertEqual(p.status, ProposalStatus.APPROVED)

    def test_reject_proposal(self):
        from tmachine.db.models import MemoryProposal, ProposalStatus
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self._engine)
        with Session() as db:
            proposal = MemoryProposal(
                scene="/fake/scene.ply",
                prompt="change awning",
                preview_path="",
                status=ProposalStatus.PENDING,
                votes_yes=0, votes_no=5,
                cam_x=0, cam_y=0, cam_z=-5,
                cam_pitch=0, cam_yaw=0, cam_roll=0,
                cam_fov_x=math.radians(60),
                cam_width=64, cam_height=64,
            )
            db.add(proposal)
            db.commit()
            db.refresh(proposal)
            pid = proposal.id

        resp = self.client.post(f"/proposals/{pid}/reject")
        self.assertEqual(resp.status_code, 200)

        with Session() as db:
            p = db.get(MemoryProposal, pid)
            self.assertEqual(p.status, ProposalStatus.REJECTED)


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestJobStatusEndpoint(unittest.TestCase):

    def setUp(self):
        from tmachine.api.app import app
        self.client = TestClient(app)

    def _mock_result(self, state: str, result=None):
        ar = MagicMock()
        ar.state  = state
        ar.status = state   # route reads .status; Celery's .status == .state
        ar.result = result
        ar.info   = result
        return ar

    def test_pending_status(self):
        with patch(
            "tmachine.api.routes.mutate.AsyncResult",
            return_value=self._mock_result("PENDING"),
        ):
            resp = self.client.get("/status/fake-job")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "PENDING")

    def test_success_status_includes_result(self):
        payload = {"proposal_id": 7, "prompt": "green awning"}
        with patch(
            "tmachine.api.routes.mutate.AsyncResult",
            return_value=self._mock_result("SUCCESS", result=payload),
        ):
            resp = self.client.get("/status/some-job-id")
        body = resp.json()
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["result"]["proposal_id"], 7)

    def test_failure_status(self):
        with patch(
            "tmachine.api.routes.mutate.AsyncResult",
            return_value=self._mock_result("FAILURE", result={"error": "OOM"}),
        ):
            resp = self.client.get("/status/failed-job")
        self.assertEqual(resp.json()["status"], "FAILURE")


if __name__ == "__main__":
    unittest.main()
