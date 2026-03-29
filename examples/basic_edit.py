"""
examples/basic_edit.py

Full end-to-end demonstration of the TMachine pipeline.

Steps
-----
1. [Module 1]  Load a .ply scene, render it from a fixed camera.
2. [AI]        Pass the render + a text prompt through InstructPix2Pix.
               With --audio, the voice pipeline (Whisper + LLM) generates
               the prompt automatically from a spoken memory.
3. [Module 2]  Run the Delta Engine to compute the LossMap.
4. [Module 3]  Run the Splat Mutator to bake the edit into the 3D scene.

Usage
-----
    # Text-prompt mode:
    python examples/basic_edit.py --ply scene.ply \\
        --prompt "change the awning to dark hunter green"

    # Voice mode (Whisper + GPT extraction, requires OPENAI_API_KEY):
    python examples/basic_edit.py --ply scene.ply --audio recording.m4a

    # Skip AI generation (supply a pre-edited image):
    python examples/basic_edit.py --ply scene.ply --edited my_edit.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

import tmachine
from tmachine.ai import ImageEditor, VoicePipeline


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)  # (H, W, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TMachine end-to-end edit demo")
    parser.add_argument("--ply",    required=True,              help="Source .ply file")
    parser.add_argument("--out",    default="scene_edited.ply", help="Output .ply file")
    parser.add_argument("--width",  type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--iters",  type=int, default=300,      help="Optimisation iterations")

    # AI edit source (pick exactly one):
    edit_group = parser.add_mutually_exclusive_group(required=True)
    edit_group.add_argument("--prompt", help="Text edit instruction for InstructPix2Pix")
    edit_group.add_argument("--audio",  help="Voice recording file (Whisper + LLM → prompt)")
    edit_group.add_argument("--edited", help="Pre-edited image path (skip AI generation)")

    # Camera
    parser.add_argument("--x",     type=float, default=0.0)
    parser.add_argument("--y",     type=float, default=-1.0)
    parser.add_argument("--z",     type=float, default=-5.0)
    parser.add_argument("--pitch", type=float, default=math.radians(6),
                        help="Tilt in radians (default: 6°)")
    parser.add_argument("--yaw",   type=float, default=0.0)
    parser.add_argument("--roll",  type=float, default=0.0)
    parser.add_argument("--fov",   type=float, default=60.0,
                        help="Horizontal FoV in degrees (default: 60)")
    args = parser.parse_args()

    # ── Camera ──────────────────────────────────────────────────────────────
    camera = tmachine.camera_from_fov(
        position=(args.x, args.y, args.z),
        pitch=args.pitch,
        yaw=args.yaw,
        roll=args.roll,
        fov_x=math.radians(args.fov),
        width=args.width,
        height=args.height,
    )
    print(f"Camera: {camera}")

    # ── Module 1: Render the current scene ──────────────────────────────────
    print("\n[M1] Rendering original scene…")
    renderer = tmachine.ViewportRenderer(args.ply)
    original = renderer.render(camera)  # (H, W, 3)
    tensor_to_pil(original).save("render_original.png")
    print(f"     → render_original.png  ({original.shape[1]}×{original.shape[0]}px)")

    # ── AI Edit ─────────────────────────────────────────────────────────────
    edited_tensor: Optional[torch.Tensor] = None
    effective_prompt: str = ""

    if args.edited:
        # ── Path A: Pre-edited image supplied — skip AI generation ──────────
        print(f"\n[edit] Loading pre-edited image from {args.edited!r}…")
        edited_pil    = Image.open(args.edited).convert("RGB")
        edited_tensor = pil_to_tensor(edited_pil.resize((args.width, args.height)))
        effective_prompt = "(external image)"

    elif args.audio:
        # ── Path B: Voice recording → Whisper → LLM → prompt ───────────────
        print(f"\n[voice] Transcribing {args.audio!r}…")
        voice_pipeline = VoicePipeline()
        voice_result   = voice_pipeline.process_file(args.audio)
        print(f"         transcript : {voice_result.transcript!r}")
        print(f"         edit prompt: {voice_result.edit_prompt!r}")
        print(f"         llm used   : {voice_result.llm_used}")
        effective_prompt = voice_result.edit_prompt
        args.prompt      = effective_prompt  # fall through to AI edit below

    if args.prompt and edited_tensor is None:
        # ── Path C: Text prompt → InstructPix2Pix ───────────────────────────
        effective_prompt = args.prompt
        print(f"\n[AI] Running InstructPix2Pix: {effective_prompt!r}…")
        original_pil = tensor_to_pil(original)
        editor       = ImageEditor()
        edited_pil   = editor.edit(image=original_pil, prompt=effective_prompt)
        edited_tensor = pil_to_tensor(
            edited_pil.resize((args.width, args.height), Image.LANCZOS)
        )
        editor.unload()

    assert edited_tensor is not None, "No edited image produced — check arguments."
    tensor_to_pil(edited_tensor).save("render_edited.png")
    print("     → render_edited.png")

    # ── Module 2: Delta Engine ───────────────────────────────────────────────
    print("\n[M2] Computing loss map…")
    delta    = tmachine.DeltaEngine()
    loss_map = delta.compute(original, edited_tensor)
    print(f"     {loss_map}")
    luma_vis = loss_map.luminance_diff.unsqueeze(-1).expand(-1, -1, 3)
    tensor_to_pil(luma_vis).save("loss_map.png")
    print("     → loss_map.png")

    # ── Module 3: Splat Mutator (back-prop into the 3D scene) ───────────────
    print(f"\n[M3] Mutating splats ({args.iters} max iters)…")

    def on_iter(i: int, loss: float) -> None:
        if i % 50 == 0:
            print(f"     iter {i:>4d}  loss={loss:.6f}")

    mutator = tmachine.SplatMutator(args.ply)
    result  = mutator.mutate(
        camera=camera,
        edited_image=edited_tensor,
        n_iters=args.iters,
        output_path=args.out,
        on_iter=on_iter,
    )
    print(f"\n[M3] Done — {result}")

    # ── Verify: re-render the mutated scene ─────────────────────────────────
    print("\n[verify] Re-rendering mutated scene…")
    renderer_new = tmachine.ViewportRenderer(args.out)
    verification = renderer_new.render(camera)
    tensor_to_pil(verification).save("render_verification.png")
    print("     → render_verification.png")

    print(f"\nEdit prompt applied: {effective_prompt!r}")


if __name__ == "__main__":
    main()
