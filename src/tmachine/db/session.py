"""
tmachine/db/session.py — SQLAlchemy engine + session factory.

Environment variables
---------------------
DATABASE_URL  Database DSN — any SQLAlchemy-compatible URL.
              Default: postgresql+psycopg2://tmachine:tmachine@localhost:5432/tmachine

              Examples:
                postgresql+psycopg2://user:pass@host/db   (Postgres, sync)
                postgresql+asyncpg://user:pass@host/db    (Postgres, async driver)
                postgres://user:pass@host/db              (Heroku-style)
                sqlite:///./tmachine.db                   (SQLite, local dev)

Usage
-----
    from tmachine.db.session import get_db

    # In a FastAPI dependency:
    async def my_route(db: Session = Depends(get_db)):
        layers = db.query(MemoryLayer).all()

    # In a Celery task (sync):
    with SessionLocal() as db:
        db.add(layer)
        db.commit()
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://tmachine:tmachine@localhost:5432/tmachine",
)

# Normalise Heroku-style "postgres://" to "postgresql+psycopg2://"
if _DATABASE_URL.startswith("postgres://"):
    _DATABASE_URL = _DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

_dialect = make_url(_DATABASE_URL).get_dialect().name  # e.g. "postgresql", "sqlite"

_engine_kwargs: dict = {"pool_pre_ping": True}
if _dialect == "sqlite":
    # SQLite is single-file: thread-check disabled, no pool needed
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Server-side databases benefit from a connection pool
    _engine_kwargs["pool_size"]    = 5
    _engine_kwargs["max_overflow"] = 10

engine = create_engine(_DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
