# Contributing to tmachine

Thank you for your interest in contributing.

---

## Dev setup

```bash
# 1. Fork and clone
git clone https://github.com/yourorg/tmachine.git
cd tmachine

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with all extras
pip install -e ".[ai-image,ai-voice,server,dev]"

# 4. Copy and fill in environment variables
cp .env.example .env
```

### Services (for API + worker tests)

The API routes and worker tests require Redis and a database.  The easiest way
to spin them up locally is with Docker:

```bash
docker run -d -p 6379:6379 redis:7-alpine
# PostgreSQL (or set DATABASE_URL=sqlite:///./tmachine_test.db in .env)
docker run -d -p 5432:5432 -e POSTGRES_USER=tmachine \
  -e POSTGRES_PASSWORD=tmachine -e POSTGRES_DB=tmachine postgres:16-alpine
```

Then apply migrations:

```bash
alembic upgrade head
```

---

## Running tests

```bash
# All tests (CPU-only, no GPU required)
pytest

# Skip GPU tests
pytest -m "not gpu"

# A single module
pytest tests/test_ply_handler.py -v

# With coverage
pytest --cov=tmachine --cov-report=term-missing
```

### Test tiers

| File | What it covers | Needs services |
| --- | --- | --- |
| `test_ply_handler.py` | PLY I/O, round-trip fidelity | No |
| `test_delta_engine.py` | Loss computation, LPIPS | No |
| `test_splat_mutator.py` | End-to-end optimisation | No (CPU) |
| `test_utils.py` | Camera helpers | No |
| `test_api_routes.py` | FastAPI endpoints | Redis + DB |
| `test_worker_locks.py` | Celery task + file locks | Redis + DB |

---

## Code style

- **Formatter**: `ruff format` (line length 100)
- **Linter**: `ruff check`
- **Type checking**: `mypy src/`

```bash
pip install ruff mypy
ruff format .
ruff check .
mypy src/
```

CI enforces `ruff check` and `ruff format --check`.  Fix lint errors before
opening a PR or the workflow will fail.

---

## PR process

1. **Branch** off `main`: `git checkout -b feat/your-feature`
2. **Write tests** for new behaviour.  Aim to keep coverage above the current
   baseline (`pytest --cov` will report it).
3. **Run the full test suite** locally before pushing.
4. **Open a PR** against `main`.  The PR description should explain *what* and
   *why*; the diff explains *how*.
5. A maintainer will review within a few days.  Please respond to review
   comments; stale PRs (no activity for 30 days) may be closed.

### Commit messages

Use the imperative mood in the subject line and keep it under 72 characters:

```text
Add configurable system_prompt to VoicePipeline
Fix off-by-one in convergence window check
```

---

## Reporting bugs

Open an issue at <https://github.com/yourorg/tmachine/issues> and include:

- tmachine version (`pip show tmachine`)
- Python version and OS
- Minimal reproduction script
- Full traceback
