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
GET /render             — correct PNG response; handles missing-file 404
GET /health             — always returns 200 {"status": "ok"}
POST /previews/generate — enqueues generate_preview, returns job_id
POST /layers/bake       — enqueues bake_patch from uploaded image, returns job_id
GET  /layers            — returns layer list for a scene
GET  /previews/{file}   — serves a saved preview PNG
GET  /status/{job_id}   — maps Celery state to response
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
        img = Image.open(io.BytesIO(resp.content))
        self.assertEqual(img.mode, "RGB")

    def test_render_missing_scene_returns_404(self):
        resp = self.client.get(
            "/render",
            params={"scene": "/does/not/exist/scene.ply"},
        )
        self.assertEqual(resp.status_code, 404)


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestPreviewGenerateEndpoint(unittest.TestCase):
    """
    POST /previews/generate — enqueues generate_preview, returns job_id.
    """

    def _make_fake_async_result(self, task_id: str = "fake-job-id"):
        ar = MagicMock()
        ar.id = task_id
        ar.status = "PENDING"
        ar.result = None
        return ar

    def setUp(self):
        from tmachine.api.app import app
        self.client = TestClient(app)

    def test_generate_preview_returns_job_id(self):
        fake_ar = self._make_fake_async_result()
        with patch(
            "tmachine.api.routes.layers.generate_preview.delay",
            return_value=fake_ar,
        ):
            resp = self.client.post(
                "/previews/generate",
                data={
                    "scene":  "/fake/scene.ply",
                    "camera": json.dumps({
                        "x": 0, "y": 0, "z": -5,
                        "pitch": 0, "yaw": 0, "roll": 0,
                        "fov_x": math.radians(60),
                        "width": 64, "height": 64,
                    }),
                    "prompt": "change the awning to green",
                },
            )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("job_id", resp.json())
        self.assertEqual(resp.json()["job_id"], "fake-job-id")


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestLayersBakeEndpoint(unittest.TestCase):
    """
    POST /layers/bake — enqueues bake_patch, returns job_id.
    """

    def _make_fake_async_result(self, task_id: str = "bake-job-id"):
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

        # Need a real temp .ply for scene path validation
        from tmachine.io.ply_handler import GaussianCloud, save_ply
        n = 5
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

    def tearDown(self):
        os.unlink(self._tmp_ply.name)
        from tmachine.api.app import app
        app.dependency_overrides.clear()

    def test_bake_with_uploaded_image_returns_job_id(self):
        fake_ar = self._make_fake_async_result()
        with patch(
            "tmachine.api.routes.layers.bake_patch.delay",
            return_value=fake_ar,
        ):
            resp = self.client.post(
                "/layers/bake",
                data={
                    "scene":        self._tmp_ply.name,
                    "camera":       json.dumps({
                        "x": 0, "y": 0, "z": -5,
                        "pitch": 0, "yaw": 0, "roll": 0,
                        "fov_x": math.radians(60),
                        "width": 64, "height": 64,
                    }),
                    "external_ref": "proposal-42",
                },
                files={"edited_image": ("edit.png", _white_png_bytes(), "image/png")},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["job_id"], "bake-job-id")

    def test_bake_requires_image_or_path(self):
        resp = self.client.post(
            "/layers/bake",
            data={
                "scene":  self._tmp_ply.name,
                "camera": json.dumps({
                    "x": 0, "y": 0, "z": -5,
                    "pitch": 0, "yaw": 0, "roll": 0,
                    "fov_x": math.radians(60),
                    "width": 64, "height": 64,
                }),
            },
        )
        self.assertEqual(resp.status_code, 422)


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestPreviewServeEndpoint(unittest.TestCase):
    """GET /previews/{filename} — serves a saved PNG."""

    def setUp(self):
        from tmachine.api.app import app
        self.client = TestClient(app)
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_serves_existing_preview(self):
        # Write a real PNG to a temp preview dir
        png_bytes = _white_png_bytes()
        preview_file = os.path.join(self._tmp_dir, "preview_test.png")
        with open(preview_file, "wb") as f:
            f.write(png_bytes)

        with patch("tmachine.api.routes.layers._PREVIEW_DIR",
                   __import__("pathlib").Path(self._tmp_dir)):
            resp = self.client.get("/previews/preview_test.png")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")

    def test_rejects_path_traversal(self):
        resp = self.client.get("/previews/../etc/passwd")
        self.assertIn(resp.status_code, (400, 422))

    def test_missing_preview_returns_404(self):
        with patch("tmachine.api.routes.layers._PREVIEW_DIR",
                   __import__("pathlib").Path(self._tmp_dir)):
            resp = self.client.get("/previews/does_not_exist.png")
        self.assertEqual(resp.status_code, 404)


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestLayersListEndpoint(unittest.TestCase):
    """GET /layers — returns MemoryLayer list for a scene."""

    def setUp(self):
        from tmachine.api.app import app
        from tmachine.db.session import get_db

        self._engine = _make_test_engine()
        override_db  = _get_test_db_factory(self._engine)
        app.dependency_overrides[get_db] = override_db
        self.client = TestClient(app)
        self._Session = sessionmaker(bind=self._engine)

    def tearDown(self):
        from tmachine.api.app import app
        app.dependency_overrides.clear()

    def test_empty_list_for_unknown_scene(self):
        resp = self.client.get("/layers", params={"scene": "/no/scene.ply"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    def test_returns_layer_for_scene(self):
        from tmachine.db.models import MemoryLayer
        with self._Session() as db:
            layer = MemoryLayer(
                scene="/fake/scene.ply",
                patch_path="/fake/patch.ply",
                hidden_indices=[1, 2, 3],
                changed_splat_count=3,
                initial_loss=0.1,
                final_loss=0.01,
                iterations_run=300,
                external_ref="proposal-1",
            )
            db.add(layer)
            db.commit()

        resp = self.client.get("/layers", params={"scene": "/fake/scene.ply"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["external_ref"], "proposal-1")


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class TestJobStatusEndpoint(unittest.TestCase):

    def setUp(self):
        from tmachine.api.app import app
        self.client = TestClient(app)

    def _mock_result(self, state: str, result=None):
        ar = MagicMock()
        ar.state  = state
        ar.status = state
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
        payload = {"preview_filename": "preview_abc.png", "prompt": "green awning"}
        with patch(
            "tmachine.api.routes.mutate.AsyncResult",
            return_value=self._mock_result("SUCCESS", result=payload),
        ):
            resp = self.client.get("/status/some-job-id")
        body = resp.json()
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["result"]["preview_filename"], "preview_abc.png")

    def test_failure_status(self):
        with patch(
            "tmachine.api.routes.mutate.AsyncResult",
            return_value=self._mock_result("FAILURE", result={"error": "OOM"}),
        ):
            resp = self.client.get("/status/failed-job")
        self.assertEqual(resp.json()["status"], "FAILURE")


if __name__ == "__main__":
    unittest.main()
