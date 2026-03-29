"""
tmachine/ai/image_editor.py — Generative AI Image Editor

Wraps InstructPix2Pix (timbrooks/instruct-pix2pix) for text-driven
image-to-image editing.  The pipeline is heavy (~3 GB), so it is lazy-loaded
on the first call and cached for the lifetime of the process.

Usage
-----
    editor = ImageEditor()
    edited_pil = editor.edit(original_pil, "change the awning to dark hunter green")

Backends
--------
InstructPix2Pix is the default and recommended backend because it requires no
mask — the model interprets the instruction directly against the full image.

Environment variables
---------------------
TMACHINE_IP2P_MODEL    – HuggingFace model ID or local path.
                         Default: timbrooks/instruct-pix2pix
TMACHINE_DEVICE        – 'cuda' | 'cpu' | 'mps'.
                         Defaults to CUDA → MPS → CPU in that order.

Notes on reproducibility
------------------------
Pass seed= for deterministic output.  The same seed produces the same
edit given the same model, image, and prompt.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image


_DEFAULT_MODEL = "timbrooks/instruct-pix2pix"


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ImageEditor:
    """
    Text-driven image editor backed by InstructPix2Pix.

    Parameters
    ----------
    model_id : str, optional
        HuggingFace model ID or local directory.  Overrides
        ``TMACHINE_IP2P_MODEL``.
    device : str, optional
        Torch device string.  Overrides ``TMACHINE_DEVICE``.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = (
            model_id
            or os.environ.get("TMACHINE_IP2P_MODEL")
            or _DEFAULT_MODEL
        )
        self.device = (
            device
            or os.environ.get("TMACHINE_DEVICE")
            or _best_device()
        )
        self._pipe = None  # lazy

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._pipe is not None:
            return

        # Import here so the rest of the package works without diffusers
        try:
            from diffusers import (
                StableDiffusionInstructPix2PixPipeline,
                EulerAncestralDiscreteScheduler,
            )
        except ImportError as exc:
            raise ImportError(
                "diffusers is required for the AI image editor.\n"
                "Install with: pip install 'tmachine[ai]'"
            ) from exc

        dtype = (
            torch.float16
            if self.device in ("cuda", "mps")
            else torch.float32
        )

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,  # disabled — scenes are architectural, not harmful
        )
        # EulerAncestralDiscrete is the scheduler recommended by the IP2P authors
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        pipe = pipe.to(self.device)

        self._pipe = pipe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, artefacts",
        num_inference_steps: int = 50,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Apply a text edit instruction to *image*.

        Parameters
        ----------
        image :
            PIL Image in RGB mode.  Should be the pristine viewport render
            produced by Module 1 (exactly what the user sees on screen).
        prompt :
            Edit instruction in imperative form, e.g.
            "change the awning to dark hunter green".
        negative_prompt :
            Things to avoid in the output.
        num_inference_steps :
            Diffusion steps — trade-off between quality and speed.
            50 is a good default; reduce to 20-30 for fast previews.
        image_guidance_scale :
            How closely the output follows the *input image* (1.0–2.5).
            Higher keeps structure; lower allows more creative freedom.
        guidance_scale :
            How closely the output follows the *text prompt* (1.0–15.0).
        seed :
            RNG seed for reproducible outputs.

        Returns
        -------
        PIL.Image.Image
            Edited image, same size as the input.
        """
        self._load()

        original_size = image.size  # (W, H)
        # IP2P works best at 512×512; we resize, edit, then resize back
        working = image.convert("RGB").resize((512, 512), Image.LANCZOS)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=working,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        edited = result.images[0]  # PIL Image at 512×512

        # Restore original resolution
        return edited.resize(original_size, Image.LANCZOS)

    def unload(self) -> None:
        """Release GPU memory by unloading the pipeline."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
