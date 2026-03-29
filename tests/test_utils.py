"""Tests for tmachine.api.utils.validate_scene_path."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from fastapi import HTTPException


def _load_utils(scene_root: str | None):
    """Reload utils with a fresh TMACHINE_SCENE_ROOT value."""
    if scene_root is None:
        os.environ.pop("TMACHINE_SCENE_ROOT", None)
    else:
        os.environ["TMACHINE_SCENE_ROOT"] = scene_root

    import tmachine.api.utils as m
    importlib.reload(m)
    return m


@pytest.fixture(autouse=True)
def _clean_env():
    """Restore env var after every test."""
    yield
    os.environ.pop("TMACHINE_SCENE_ROOT", None)


@pytest.fixture()
def scene_root(tmp_path: Path) -> str:
    (tmp_path / "scene.ply").touch()
    return str(tmp_path)


class TestValidateScenePath:
    def test_relative_path_resolved_against_root(self, scene_root):
        u = _load_utils(scene_root)
        result = u.validate_scene_path("scene.ply")
        assert result == str(Path(scene_root) / "scene.ply")

    def test_absolute_path_inside_root_accepted(self, scene_root):
        u = _load_utils(scene_root)
        abs_path = str(Path(scene_root) / "scene.ply")
        assert u.validate_scene_path(abs_path) == abs_path

    def test_traversal_relative_blocked(self, scene_root):
        u = _load_utils(scene_root)
        with pytest.raises(HTTPException) as exc_info:
            u.validate_scene_path("../etc/passwd")
        assert exc_info.value.status_code == 400

    def test_absolute_outside_root_blocked(self, scene_root):
        u = _load_utils(scene_root)
        with pytest.raises(HTTPException) as exc_info:
            u.validate_scene_path("/etc/passwd")
        assert exc_info.value.status_code == 400

    def test_no_root_absolute_path_accepted(self):
        u = _load_utils(None)
        result = u.validate_scene_path("/home/ja/scenes/livingroom.ply")
        assert result == str(Path("/home/ja/scenes/livingroom.ply").resolve())

    def test_empty_string_root_treated_as_unset(self):
        """TMACHINE_SCENE_ROOT='' must not restrict paths."""
        os.environ["TMACHINE_SCENE_ROOT"] = ""
        import tmachine.api.utils as m
        importlib.reload(m)
        # should not raise
        result = m.validate_scene_path("/home/ja/scenes/livingroom.ply")
        assert result == str(Path("/home/ja/scenes/livingroom.ply").resolve())
