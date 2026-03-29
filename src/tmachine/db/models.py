"""
tmachine/db/models.py — SQLAlchemy ORM models.

Two tables drive the Memory Layers system:

memory_proposals
    Stores AI-generated 2D edit candidates waiting for community approval.
    Each row captures the prompt, source scene, camera, and the path to the
    AI-generated preview image.  Status moves PENDING → APPROVED | REJECTED.

memory_layers
    Stores merged (approved and baked) 3D patches.  Each row holds:
      • patch_path     — path to the tiny patch .ply file
      • hidden_indices — JSON array of base-scene splat row indices to hide

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

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)

# BigInteger works for PostgreSQL; SQLite requires Integer for ROWID autoincrement.
_PK_TYPE = BigInteger().with_variant(Integer, "sqlite")
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProposalStatus(str, enum.Enum):
    PENDING  = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# memory_proposals
# ---------------------------------------------------------------------------

class MemoryProposal(Base):
    """
    A pending 2D AI-edit awaiting community vote.

    Lifecycle
    ---------
    1. Created by the ``generate_memory_proposal`` Celery task.
    2. Served to the Review UI by GET /proposals.
    3. Users vote via POST /proposals/{id}/vote.
    4. When approval threshold is met, POST /proposals/{id}/approve enqueues
       ``bake_approved_patch`` and sets status → APPROVED.
    5. On rejection, status → REJECTED  (patch never baked, no 3D change).
    """

    __tablename__ = "memory_proposals"

    id:            Mapped[int]      = mapped_column(_PK_TYPE, primary_key=True, autoincrement=True)
    scene:         Mapped[str]      = mapped_column(String(1024), nullable=False,
                                                    index=True,
                                                    comment="Absolute path to the base .ply file")
    prompt:        Mapped[str]      = mapped_column(Text, nullable=False,
                                                    comment="Edit instruction sent to InstructPix2Pix")
    preview_path:  Mapped[str]      = mapped_column(String(1024), nullable=False,
                                                    comment="Path to the 2D AI-edited PNG preview image")
    status:        Mapped[ProposalStatus] = mapped_column(
                                        SAEnum(ProposalStatus), nullable=False,
                                        default=ProposalStatus.PENDING, index=True)
    votes_yes:     Mapped[int]      = mapped_column(Integer, nullable=False, default=0)
    votes_no:      Mapped[int]      = mapped_column(Integer, nullable=False, default=0)

    # Camera extrinsics — stored flat so the backend can reconstruct a Camera
    cam_x:     Mapped[float] = mapped_column(Float, nullable=False)
    cam_y:     Mapped[float] = mapped_column(Float, nullable=False)
    cam_z:     Mapped[float] = mapped_column(Float, nullable=False)
    cam_pitch: Mapped[float] = mapped_column(Float, nullable=False)
    cam_yaw:   Mapped[float] = mapped_column(Float, nullable=False)
    cam_roll:  Mapped[float] = mapped_column(Float, nullable=False)
    cam_fov_x: Mapped[float] = mapped_column(Float, nullable=False)
    cam_width: Mapped[int]   = mapped_column(Integer, nullable=False)
    cam_height:Mapped[int]   = mapped_column(Integer, nullable=False)

    # Optional: Celery job ID for the bake task, set when proposal is approved
    bake_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # One proposal → at most one baked layer
    layer: Mapped["MemoryLayer | None"] = relationship(
        "MemoryLayer", back_populates="proposal", uselist=False
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryProposal id={self.id} status={self.status.value!r} "
            f"prompt={self.prompt[:40]!r}>"
        )


# ---------------------------------------------------------------------------
# memory_layers
# ---------------------------------------------------------------------------

class MemoryLayer(Base):
    """
    A merged (approved + baked) 3D memory patch.

    Each row corresponds to one approved MemoryProposal that has been baked
    into a patch .ply by the ``bake_approved_patch`` Celery task.

    When the render endpoint receives ``active_layers=[id1, id2, …]``:
        1. Fetch each MemoryLayer row.
        2. For each layer, set ``hidden_indices`` opacities → 0 in the base.
        3. Concatenate patch GaussianClouds.
        4. Rasterise the merged scene.
    """

    __tablename__ = "memory_layers"

    id:             Mapped[int]   = mapped_column(_PK_TYPE, primary_key=True, autoincrement=True)
    proposal_id:    Mapped[int]   = mapped_column(
                                        ForeignKey("memory_proposals.id"),
                                        nullable=False, unique=True, index=True)
    scene:          Mapped[str]   = mapped_column(String(1024), nullable=False, index=True,
                                                  comment="Absolute path to the base .ply file")
    patch_path:     Mapped[str]   = mapped_column(String(1024), nullable=False,
                                                  comment="Absolute path to the patch .ply")
    hidden_indices: Mapped[list]  = mapped_column(JSON, nullable=False,
                                                  comment="Base-scene row indices to hide (int[])")
    changed_splat_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    initial_loss:   Mapped[float] = mapped_column(Float, nullable=False)
    final_loss:     Mapped[float] = mapped_column(Float, nullable=False)
    iterations_run: Mapped[int]   = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    proposal: Mapped["MemoryProposal"] = relationship(
        "MemoryProposal", back_populates="layer"
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryLayer id={self.id} proposal={self.proposal_id} "
            f"splats={self.changed_splat_count} patch={self.patch_path!r}>"
        )
