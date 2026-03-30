"""create memory_layers table

Revision ID: 0001
Revises:
Create Date: 2026-03-30
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_layers",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer, "sqlite"),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column("scene", sa.String(1024), nullable=False,
                  comment="Absolute path to the base .ply file"),
        sa.Column("patch_path", sa.String(1024), nullable=False,
                  comment="Absolute path to the patch .ply"),
        sa.Column("hidden_indices", sa.JSON, nullable=False,
                  comment="Base-scene row indices to hide (int[])"),
        sa.Column("changed_splat_count", sa.Integer, nullable=False,
                  server_default="0"),
        sa.Column("initial_loss", sa.Float, nullable=False),
        sa.Column("final_loss", sa.Float, nullable=False),
        sa.Column("iterations_run", sa.Integer, nullable=False),
        sa.Column("external_ref", sa.String(256), nullable=True,
                  comment="Caller-supplied opaque reference"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_memory_layers_scene", "memory_layers", ["scene"])
    op.create_index("ix_memory_layers_external_ref", "memory_layers", ["external_ref"])


def downgrade() -> None:
    op.drop_index("ix_memory_layers_external_ref", table_name="memory_layers")
    op.drop_index("ix_memory_layers_scene", table_name="memory_layers")
    op.drop_table("memory_layers")
