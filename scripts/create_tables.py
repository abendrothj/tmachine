#!/usr/bin/env python3
"""
scripts/create_tables.py — Create all database tables.

Run this once after setting up your database (Neon or local Postgres).
It is safe to run multiple times — existing tables are never dropped.

Usage
-----
    # From the tmachine/ directory with the venv active:
    python scripts/create_tables.py

    # Or with an explicit DATABASE_URL:
    DATABASE_URL="postgresql+psycopg2://..." python scripts/create_tables.py

Environment variables
---------------------
DATABASE_URL  PostgreSQL DSN.  Reads from .env if python-dotenv is installed,
              otherwise falls back to the default in db/session.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env if present (python-dotenv is optional)
# ---------------------------------------------------------------------------
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
        print(f"Loaded environment from {_env_file}")
    except ImportError:
        print("tip: pip install python-dotenv to auto-load .env")

# ---------------------------------------------------------------------------
# Create tables
# ---------------------------------------------------------------------------
try:
    from tmachine.db.session import engine
    from tmachine.db.models import Base
except ImportError as exc:
    print(f"Import error: {exc}")
    print("Make sure you have run: pip install -e '.[api]'")
    sys.exit(1)

print(f"Connecting to: {engine.url}")
print("Creating tables…")

try:
    Base.metadata.create_all(engine)
except Exception as exc:
    print(f"Failed: {exc}")
    sys.exit(1)

print("Done. Tables created:")
for table in Base.metadata.sorted_tables:
    print(f"  ✓ {table.name}")
