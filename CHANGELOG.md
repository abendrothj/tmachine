# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **Proposal/voting workflow removed from the engine.** The engine no longer owns the approval lifecycle. Callers are responsible for managing proposals, votes, and the decision to bake.
- `bake_approved_patch` Celery task replaced by `bake_patch`. It now accepts all parameters directly (`scene`, `edited_image_b64`, `camera_dict`, etc.) instead of reading from a `MemoryProposal` DB row.
- `generate_memory_proposal` Celery task replaced by `generate_preview`. It saves the preview PNG and returns `{preview_filename, preview_path, prompt}` without writing any DB record.
- `MemoryLayer` no longer has a `proposal_id` foreign key. Added optional `external_ref` (string) so callers can link a layer back to their own upstream ID.
- `POST /proposals/*` routes replaced by `POST /previews/generate`, `POST /previews/voice`, `GET /previews/{filename}`, and `POST /layers/bake`.

### Removed

- `MemoryProposal` SQLAlchemy model and `memory_proposals` table.
- `ProposalStatus` enum.
- `tmachine/api/routes/proposals.py`.

### Added

- `tmachine/api/routes/layers.py` — preview generation, preview serving, layer bake, and layer list endpoints.
- `GET /previews/{filename}` — serves saved preview PNGs from `PREVIEW_DIR`.
- `POST /layers/bake` — triggers Stage 2 from an uploaded image or a saved `preview_path`; accepts `external_ref`.

---

## [0.1.0] — 2026-03-28

- `ViewportRenderer` — differentiable 3DGS rasteriser with per-pixel gradient output
- `DeltaEngine` — pixel-diff → splat gradient backprop with `LossMap` dataclass
- `SplatMutator` — end-to-end render → edit → optimise → patch pipeline
- `GaussianCloud` — typed dataclass representing a loaded .ply scene
- `load_ply` / `save_ply` — atomic PLY I/O with SH-degree inference
- `Camera` helpers — `camera_from_euler`, `camera_from_fov`
- AI layer: `ImageEditor` (InstructPix2Pix / SDXL) and `VoicePipeline` (Whisper + LLM)
- FastAPI REST server with Celery background jobs and Redis broker
- SQLAlchemy model: `MemoryLayer`
- Five-tier test suite (PLY I/O, delta engine, splat mutator, API routes, worker locks)
- `src/` layout for clean editable install isolation
