"""
tmachine/ai/voice_pipeline.py — Voice-to-Prompt Pipeline

Converts a user's spoken memory into a precise, machine-readable image-editing
instruction consumable by InstructPix2Pix.

Two-stage pipeline
------------------
1. STT  — OpenAI Whisper transcribes the audio to raw text.
2. LLM  — A small GPT call (or regex fallback) extracts a single, clean
           imperative edit instruction from the transcript.

Example
-------
    Spoken:    "Well, back in my day that awning was never red, it was more of
                a dark hunter green, almost forest-coloured."
    Transcript: (raw Whisper output)
    Output prompt: "Change the awning color to dark hunter green"

Usage
-----
    pipeline = VoicePipeline()

    # With a custom domain-specific system prompt:
    pipeline = VoicePipeline(system_prompt="You are an assistant that ...")

    # From a file on disk:
    result = pipeline.process_file("recording.m4a")
    print(result.transcript)     # full Whisper text
    print(result.edit_prompt)    # extracted instruction

    # From raw bytes (e.g. an HTTP upload):
    result = pipeline.process_bytes(audio_bytes, suffix=".webm")

Environment variables
---------------------
OPENAI_API_KEY     — Used for the LLM extraction step.  If absent, the raw
                     transcript is returned as the edit prompt (usable but
                     verbose).
TMACHINE_WHISPER_MODEL — Whisper model size: tiny | base | small | medium |
                         large.  Default: base  (balances speed and accuracy).

Notes
-----
Whisper is also lazy-loaded; the first call takes a few seconds to download
and initialise the model weights.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VoiceResult:
    """
    Outcome of a voice pipeline run.

    Attributes
    ----------
    transcript : str
        Raw text produced by Whisper.
    edit_prompt : str
        Concise instructional prompt extracted from the transcript, ready
        to feed directly to :class:`~tmachine.ai.image_editor.ImageEditor`.
    model_used : str
        Whisper model size that was used.
    llm_used : bool
        True if an LLM was used for extraction; False = transcript forwarded.
    """

    transcript:   str
    edit_prompt:  str
    model_used:   str
    llm_used:     bool

    def __repr__(self) -> str:
        return (
            f"VoiceResult(\n"
            f"  transcript = {self.transcript!r}\n"
            f"  edit_prompt = {self.edit_prompt!r}\n"
            f"  llm_used={self.llm_used}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """
You are a precise image editing assistant.

Your task is to convert a user's spoken description into a single, clean image editing
instruction suitable for an AI image editor (InstructPix2Pix / Stable Diffusion).

Rules:
1. Output ONLY the edit instruction — no preamble, no explanation, no quotes.
2. Use imperative form: "Change X to Y", "Replace X with Y", "Make X look like Y".
3. Be specific about colours, materials, and visual features.
4. If the user mentions multiple changes, pick only the single most prominent one.
5. If the transcript is already a clean instruction, return it unchanged.

Examples:
  Input:  "The awning was a dark hunter green, not red."
  Output: Change the awning color to dark hunter green

  Input:  "Those concrete columns used to be white marble."
  Output: Replace the concrete columns with white marble columns

  Input:  "The facade was warm terracotta, not this pale beige."
  Output: Change the building facade color from pale beige to warm terracotta
""".strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VoicePipeline:
    """
    Two-stage voice-to-prompt pipeline: Whisper STT → LLM extraction.

    Parameters
    ----------
    whisper_model : str, optional
        Whisper model size.  Overrides ``TMACHINE_WHISPER_MODEL``.
        Smaller is faster; larger is more accurate for accented speech.
        Options: tiny | base | small | medium | large  (default: base)
    openai_api_key : str, optional
        Override ``OPENAI_API_KEY`` env var for LLM extraction.
        Ignored when ``llm_fn`` is provided.
    llm_model : str
        OpenAI model to use for prompt extraction.  Default: gpt-4o-mini.
        Ignored when ``llm_fn`` is provided.
    system_prompt : str, optional
        System prompt sent to the LLM extraction step.  Defaults to a
        domain-agnostic prompt.  Override to specialise the pipeline for
        your own application domain.  Ignored when ``llm_fn`` is provided.
    llm_fn : Callable[[str], str], optional
        Drop-in replacement for the entire LLM extraction step.  Called
        with the raw transcript and must return the edit prompt string.
        Use this to plug in any provider (Anthropic, local Ollama, a
        fine-tuned model, a simple regex, etc.) without subclassing::

            pipeline = VoicePipeline(
                llm_fn=lambda transcript: my_anthropic_client.extract(transcript)
            )

        When provided, ``openai_api_key``, ``llm_model``, and
        ``system_prompt`` are ignored.
    """

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.whisper_model_name = (
            whisper_model
            or os.environ.get("TMACHINE_WHISPER_MODEL", "base")
        )
        self._openai_api_key = (
            openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        )
        self.llm_model = llm_model
        self.system_prompt = system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT
        self.llm_fn = llm_fn
        self._whisper_model = None  # lazy

    # ------------------------------------------------------------------
    # Stage 1: STT
    # ------------------------------------------------------------------

    def _load_whisper(self):
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required for the voice pipeline.\n"
                "Install with: pip install 'tmachine[ai-voice]'"
            ) from exc
        self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe an audio file to text using Whisper.

        Parameters
        ----------
        audio_path :
            Any format Whisper supports: mp3, mp4, m4a, wav, webm, ogg.

        Returns
        -------
        str
            Raw transcript, stripped of leading/trailing whitespace.
        """
        model = self._load_whisper()
        result = model.transcribe(str(audio_path), fp16=False)
        return result["text"].strip()

    # ------------------------------------------------------------------
    # Stage 2: LLM extraction
    # ------------------------------------------------------------------

    def extract_edit_prompt(self, transcript: str) -> tuple[str, bool]:
        """
        Extract a clean edit instruction from *transcript*.

        Returns
        -------
        (edit_prompt, llm_used)
        """
        # ── Pluggable provider ────────────────────────────────────────────
        if self.llm_fn is not None:
            return self.llm_fn(transcript), True

        # ── No API key — regex fallback ───────────────────────────────────
        if not self._openai_api_key:
            return self._regex_clean(transcript), False

        # ── Default: OpenAI ───────────────────────────────────────────────
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for LLM extraction.\n"
                "Install with: pip install 'tmachine[ai-voice]'  or  pip install openai"
            ) from exc

        client = OpenAI(api_key=self._openai_api_key)
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": transcript},
            ],
            max_tokens=80,
            temperature=0.2,  # low temperature for consistent, format-faithful output
        )
        edit_prompt = response.choices[0].message.content.strip()
        return edit_prompt, True

    @staticmethod
    def _regex_clean(transcript: str) -> str:
        """
        Minimal cleaning when no LLM is available.

        Strips common filler phrases and trims whitespace.  The result is
        still usable by InstructPix2Pix; it's just more verbose.
        """
        fillers = [
            r"\bwell[,]?\b",
            r"\buh+\b",
            r"\bum+\b",
            r"\byou know[,]?\b",
            r"\bi mean[,]?\b",
            r"\bback in my day[,]?\b",
            r"\back then[,]?\b",
        ]
        cleaned = transcript
        for pattern in fillers:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        # Collapse multiple spaces
        cleaned = re.sub(r"  +", " ", cleaned).strip(" ,.")
        return cleaned if cleaned else transcript

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def process_file(self, audio_path: str | Path) -> VoiceResult:
        """
        Full pipeline from an audio file on disk.

        Parameters
        ----------
        audio_path :
            Path to the audio file (mp3, m4a, wav, webm, ogg, etc.).

        Returns
        -------
        VoiceResult
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        transcript  = self.transcribe(audio_path)
        edit_prompt, llm_used = self.extract_edit_prompt(transcript)

        return VoiceResult(
            transcript=transcript,
            edit_prompt=edit_prompt,
            model_used=self.whisper_model_name,
            llm_used=llm_used,
        )

    def process_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".wav",
    ) -> VoiceResult:
        """
        Full pipeline from raw audio bytes (e.g. an HTTP upload).

        The bytes are written to a secure temporary file, processed, then
        immediately deleted.

        Parameters
        ----------
        audio_bytes :
            Raw audio data.
        suffix :
            File extension for the temp file so Whisper recognises the format.
            E.g. '.wav', '.mp3', '.m4a', '.webm'.

        Returns
        -------
        VoiceResult
        """
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self.process_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
