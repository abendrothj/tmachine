"""
tests/test_db_integration.py — Tier 4: Postgres ORM integration tests

Requires a live PostgreSQL database.  Skipped automatically when
DATABASE_URL is not set (e.g. local runs without Docker/Postgres).

In CI, DATABASE_URL is injected as an env var and ``alembic upgrade head``
runs before pytest, so the table structure will match the current model.

Covers
------
- MemoryLayer INSERT / SELECT round-trip
- hidden_indices JSON array survives the full ORM round-trip without mutation
- external_ref indexed column is queryable by value
- created_at is stored and returned as a timezone-aware datetime
- Schema smoke test: memory_layers has the expected column set
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime

# ---------------------------------------------------------------------------
# Connection guard — skip the entire module on machines without Postgres
# ---------------------------------------------------------------------------
_DB_URL = os.getenv("DATABASE_URL")
_SKIP = "DATABASE_URL not set — skipping Postgres integration tests."


def _make_session(db_url: str):
    """Return (sessionmaker, engine) bound to *db_url*."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url, future=True, pool_pre_ping=True)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return factory, engine


@unittest.skipUnless(_DB_URL, _SKIP)
class TestMemoryLayerPostgres(unittest.TestCase):
    """Direct ORM tests against real Postgres using DATABASE_URL."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from sqlalchemy import create_engine  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("sqlalchemy not installed")

        from tmachine.db.models import MemoryLayer
        cls._MemoryLayer = MemoryLayer
        cls.Session, cls.engine = _make_session(_DB_URL)

    def tearDown(self) -> None:
        """Remove all rows written by this test class (external_ref starts with pytest-)."""
        with self.Session() as db:
            db.query(self._MemoryLayer).filter(
                self._MemoryLayer.external_ref.like("pytest-%")
            ).delete(synchronize_session=False)
            db.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _insert(
        self,
        external_ref: str,
        hidden_indices: list[int] | None = None,
    ) -> int:
        with self.Session() as db:
            layer = self._MemoryLayer(
                scene="/test/scene.ply",
                patch_path="/test/patch.ply",
                hidden_indices=hidden_indices if hidden_indices is not None else [10, 20, 30],
                changed_splat_count=3,
                initial_loss=0.05,
                final_loss=0.01,
                iterations_run=100,
                external_ref=external_ref,
            )
            db.add(layer)
            db.commit()
            db.refresh(layer)
            return layer.id

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_insert_and_select(self) -> None:
        """Basic INSERT + SELECT — scalar columns survive the round-trip."""
        row_id = self._insert("pytest-basic")

        with self.Session() as db:
            layer = db.get(self._MemoryLayer, row_id)

        self.assertIsNotNone(layer)
        self.assertEqual(layer.scene, "/test/scene.ply")
        self.assertEqual(layer.patch_path, "/test/patch.ply")
        self.assertEqual(layer.changed_splat_count, 3)
        self.assertAlmostEqual(layer.initial_loss, 0.05, places=4)
        self.assertAlmostEqual(layer.final_loss, 0.01, places=4)
        self.assertEqual(layer.iterations_run, 100)

    def test_hidden_indices_json_roundtrip(self) -> None:
        """JSON array must survive write → read without ordering or type change."""
        indices = [0, 42, 1337, 99999]
        row_id = self._insert("pytest-json", hidden_indices=indices)

        with self.Session() as db:
            layer = db.get(self._MemoryLayer, row_id)

        self.assertEqual(layer.hidden_indices, indices)

    def test_empty_hidden_indices(self) -> None:
        """An empty list is a valid JSON value and must round-trip as []."""
        row_id = self._insert("pytest-empty-indices", hidden_indices=[])

        with self.Session() as db:
            layer = db.get(self._MemoryLayer, row_id)

        self.assertEqual(layer.hidden_indices, [])

    def test_external_ref_query(self) -> None:
        """external_ref is indexed — point-lookup must return exactly one row."""
        self._insert("pytest-ref-lookup")

        with self.Session() as db:
            results = (
                db.query(self._MemoryLayer)
                .filter(self._MemoryLayer.external_ref == "pytest-ref-lookup")
                .all()
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].external_ref, "pytest-ref-lookup")

    def test_external_ref_none_allowed(self) -> None:
        """external_ref is nullable — NULL must be storable and round-trip as None."""
        with self.Session() as db:
            layer = self._MemoryLayer(
                scene="/test/scene.ply",
                patch_path="/test/patch.ply",
                hidden_indices=[],
                changed_splat_count=0,
                initial_loss=0.0,
                final_loss=0.0,
                iterations_run=0,
                external_ref=None,
            )
            db.add(layer)
            db.commit()
            db.refresh(layer)
            row_id = layer.id

        with self.Session() as db:
            layer = db.get(self._MemoryLayer, row_id)

        self.assertIsNone(layer.external_ref)

        # Cleanup (tearDown only removes rows with external_ref LIKE 'pytest-%')
        with self.Session() as db:
            db.delete(db.get(self._MemoryLayer, row_id))
            db.commit()

    def test_created_at_timezone_aware(self) -> None:
        """created_at must be returned as a timezone-aware datetime."""
        row_id = self._insert("pytest-tz")

        with self.Session() as db:
            layer = db.get(self._MemoryLayer, row_id)

        self.assertIsInstance(layer.created_at, datetime)
        self.assertIsNotNone(
            layer.created_at.tzinfo,
            "created_at must be timezone-aware (DateTime(timezone=True))",
        )

    def test_table_schema_columns(self) -> None:
        """Smoke test: memory_layers must have all expected columns after alembic upgrade."""
        from sqlalchemy import inspect

        insp = inspect(self.engine)
        actual_cols = {c["name"] for c in insp.get_columns("memory_layers")}
        required = {
            "id",
            "scene",
            "patch_path",
            "hidden_indices",
            "changed_splat_count",
            "initial_loss",
            "final_loss",
            "iterations_run",
            "external_ref",
            "created_at",
        }
        missing = required - actual_cols
        self.assertFalse(missing, f"Columns missing from memory_layers: {missing}")

    def test_multiple_rows_same_scene(self) -> None:
        """Multiple layers for the same scene path must all be queryable by scene."""
        for i in range(3):
            self._insert(f"pytest-multi-{i}")

        with self.Session() as db:
            count = (
                db.query(self._MemoryLayer)
                .filter(self._MemoryLayer.scene == "/test/scene.ply")
                .filter(self._MemoryLayer.external_ref.like("pytest-multi-%"))
                .count()
            )

        self.assertEqual(count, 3)


if __name__ == "__main__":
    unittest.main()
