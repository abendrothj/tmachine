"""
tmachine/api/app.py — FastAPI application entrypoint.

Starting the server
-------------------
    # Development (auto-reload):
    uvicorn tmachine.api.app:app --reload --port 8000

    # Production (4 workers, GPU machine with 1 GPU — keep at 1 for GPU memory):
    uvicorn tmachine.api.app:app --workers 1 --port 8000

Starting the Celery worker (separate terminal)
----------------------------------------------
    celery -A tmachine.api.worker worker --loglevel=info --concurrency=1

    # Monitor jobs in the browser:
    celery -A tmachine.api.worker flower --port=5555

Environment variables (all optional)
--------------------------------------
CELERY_BROKER_URL      Celery broker URL.  Default: redis://localhost:6379/0 (via REDIS_URL fallback)
CELERY_RESULT_BACKEND  Celery result backend.  Defaults to CELERY_BROKER_URL.
REDIS_URL              Legacy fallback for CELERY_BROKER_URL / CELERY_RESULT_BACKEND.
TMACHINE_IP2P_MODEL    InstructPix2Pix model ID or local path.
TMACHINE_WHISPER_MODEL Whisper model size (tiny|base|small|medium|large).
OPENAI_API_KEY         Required for LLM prompt extraction in the voice pipeline.
LOCK_TIMEOUT           Max seconds to wait for a .ply file lock.  Default: 300
TMACHINE_CORS_ORIGINS  Comma-separated list of allowed CORS origins.
                       Default: * (open, credentials disabled).
                       Example: https://app.example.com,https://preview.example.com
                       Note: allow_credentials is only enabled when explicit origins are set.

API overview
------------
GET  /render                  → PNG image (Module 1)
POST /previews/generate       → {job_id}  (Stage 1 — render + AI edit → preview PNG)
POST /previews/voice          → {job_id, transcript, edit_prompt}
GET  /previews/{filename}     → PNG image (serve saved preview)
POST /layers/bake             → {job_id}  (Stage 2 — bake preview into patch .ply)
GET  /layers                  → [{layer}]
POST /mutate/image            → {job_id}  (Module 3 only — upload edited image)
POST /mutate/prompt           → {job_id}  (full pipeline — text prompt)
POST /voice-edit              → {job_id, transcript, edit_prompt}
GET  /status/{job_id}         → {status, result?, error?}
GET  /health                  → {"status": "ok"}
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env from the project root if present (no-op if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.render import router as render_router
from .routes.mutate import router as mutate_router
from .routes.layers import router as layers_router

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TMachine API",
    description=(
        "Agnostic 3D Gaussian Splat mutation engine.\n\n"
        "**Module 1** — Viewport Renderer: `.ply` + camera → 2D image\n\n"
        "**Module 2** — Delta Engine: original + edited → loss map\n\n"
        "**Module 3** — Splat Mutator: loss map + camera → updated `.ply`\n\n"
        "AI layer: InstructPix2Pix for image editing; Whisper + GPT for voice."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_cors_env = os.environ.get("TMACHINE_CORS_ORIGINS", "")
_cors_origins: list[str] = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["*"]
)
# allow_credentials=True requires explicit origins (not "*") per the CORS spec.
# Browsers will reject credentialed requests when the origin is a wildcard.
_allow_credentials = _cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(render_router,  tags=["Rendering"])
app.include_router(layers_router,  tags=["Memory Layers"])
app.include_router(mutate_router,  tags=["Mutation"])

# ---------------------------------------------------------------------------
# Health check — used by load balancers / container orchestrators
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"], summary="Liveness probe.")
async def health_check() -> dict:
    return {"status": "ok"}
