"""
tmachine/db/models.py — SQLAlchemy ORM models.

memory_layers
    Stores merged (approved and baked) 3D patches.  Each row holds:
      • patch_path     — path to the tiny patch .ply file
      • hidden_indices — JSON array of base-scene splat row indices to hide
      • external_ref   — opaque reference set by the caller (e.g. a proposal ID
                         managed by an upstream service)

At render time, active layer IDs are resolved to their patch files and
hidden-index arrays, and the scene is composited on the fly.

Setup
-----
    from tmachine.db.session import engine
    from tmachine.db.models import Base
    Base.metadata.create_all(engine)

Or use the Alembic migration provided in alembic/:
    alembic upgrade head
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
)

# BigInteger works for PostgreSQL; SQLite requires Integer for ROWID autoincrement.
_PK_TYPE = BigInteger().with_variant(Integer, "sqlite")
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# memory_layers
# ---------------------------------------------------------------------------

class MemoryLayer(Base):
    """
    A baked 3D memory patch.

    Created by the ``bake_patch`` Celery task after a caller decides to
    commit an AI-edited preview into the 3D scene.  The upstream approval
    workflow (proposals, voting, etc.) is the caller's responsibility and is
    not modelled here.

    When the render endpoint receives ``active_layers=[id1, id2, …]``:
        1. Fetch each MemoryLayer row.
        2. For each layer, set ``hidden_indices`` opacities → 0 in the base.
        3. Concatenate patch GaussianClouds.
        4. Rasterise the merged scene.
    """

    __tablename__ = "memory_layers"

    id:             Mapped[int]        = mapped_column(_PK_TYPE, primary_key=True, autoincrement=True)
    scene:          Mapped[str]        = mapped_column(String(1024), nullable=False, index=True,
                                                       comment="Absolute path to the base .ply file")
    patch_path:     Mapped[str]        = mapped_column(String(1024), nullable=False,
                                                       comment="Absolute path to the patch .ply")
    hidden_indices: Mapped[list]       = mapped_column(JSON, nullable=False,
                                                       comment="Base-scene row indices to hide (int[])")
    changed_splat_count: Mapped[int]   = mapped_column(Integer, nullable=False, default=0)
    initial_loss:   Mapped[float]      = mapped_column(Float, nullable=False)
    final_loss:     Mapped[float]      = mapped_column(Float, nullable=False)
    iterations_run: Mapped[int]        = mapped_column(Integer, nullable=False)

    # Opaque reference supplied by the caller — e.g. a proposal ID managed by
    # an upstream service.  Not interpreted by the engine.
    external_ref:   Mapped[str | None] = mapped_column(String(256), nullable=True, index=True,
                                                        comment="Caller-supplied opaque reference")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryLayer id={self.id} splats={self.changed_splat_count} "
            f"patch={self.patch_path!r}>"
        )
