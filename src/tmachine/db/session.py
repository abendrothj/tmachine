"""
tmachine/db/session.py — SQLAlchemy engine + session factory.

Environment variables
---------------------
DATABASE_URL  PostgreSQL DSN.
              Default: postgresql+psycopg2://tmachine:tmachine@localhost:5432/tmachine

              For SQLite during local dev / testing:
              DATABASE_URL=sqlite:///./tmachine.db

Usage
-----
    from tmachine.db.session import get_db

    # In a FastAPI dependency:
    async def my_route(db: Session = Depends(get_db)):
        proposals = db.query(MemoryProposal).all()

    # In a Celery task (sync):
    with SessionLocal() as db:
        db.add(layer)
        db.commit()
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://tmachine:tmachine@localhost:5432/tmachine",
)

engine = create_engine(
    _DATABASE_URL,
    pool_pre_ping=True,       # recover stale connections automatically
    pool_size=5,
    max_overflow=10,
    connect_args=(
        {}
        if _DATABASE_URL.startswith("postgresql")
        else {"check_same_thread": False}  # SQLite only
    ),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
