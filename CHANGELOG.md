# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-03-28

### Added
- `ViewportRenderer` — differentiable 3DGS rasteriser with per-pixel gradient output
- `DeltaEngine` — pixel-diff → splat gradient backprop with `LossMap` dataclass
- `SplatMutator` — end-to-end render → edit → optimise → patch pipeline
- `GaussianCloud` — typed dataclass representing a loaded .ply scene
- `load_ply` / `save_ply` — atomic PLY I/O with SH-degree inference
- `Camera` helpers — `camera_from_euler`, `camera_from_fov`
- AI layer: `ImageEditor` (InstructPix2Pix / SDXL) and `VoicePipeline` (Whisper + LLM)
- FastAPI REST server with Celery background jobs and Redis broker
- SQLAlchemy models: `MemoryProposal`, `MemoryLayer`
- Five-tier test suite (PLY I/O, delta engine, splat mutator, API routes, worker locks)
- `src/` layout for clean editable install isolation
