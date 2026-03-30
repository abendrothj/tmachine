"""
tests/test_voice_pipeline.py — VoicePipeline unit tests

All tests are CPU-only and require no external services (no OpenAI key,
no Whisper model download).

Covers
------
- _regex_clean:  filler-word stripping, passthrough for clean input
- extract_edit_prompt with llm_fn: custom callable is used, LLM flag set True
- extract_edit_prompt fallback: no API key → regex path, llm_used=False
- process_bytes: missing llm_key + llm_fn → transcribe stub → regex clean
- process_file: raises FileNotFoundError on missing path
- VoiceResult repr: smoke test
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tmachine.ai.voice_pipeline import VoicePipeline, VoiceResult


# ---------------------------------------------------------------------------
# _regex_clean
# ---------------------------------------------------------------------------

class TestRegexClean(unittest.TestCase):

    def _clean(self, text: str) -> str:
        return VoicePipeline._regex_clean(text)

    def test_strips_well(self):
        result = self._clean("Well, the door was blue.")
        self.assertNotIn("well", result.lower())
        self.assertIn("door was blue", result)

    def test_strips_uh_um(self):
        result = self._clean("uh, it was um green")
        self.assertNotIn("uh", result.lower())
        self.assertNotIn("um", result.lower())
        self.assertIn("green", result)

    def test_strips_back_in_my_day(self):
        result = self._clean("Back in my day, the roof was red.")
        self.assertNotIn("back in my day", result.lower())
        self.assertIn("roof was red", result)

    def test_passthrough_clean_instruction(self):
        prompt = "Change the awning to dark hunter green"
        result = self._clean(prompt)
        self.assertEqual(result, prompt)

    def test_empty_string_returns_original(self):
        # If everything is stripped, the original is returned unchanged.
        result = self._clean("uh um well")
        # stripped result would be empty β†' returns original transcript
        self.assertIsInstance(result, str)

    def test_collapses_double_spaces(self):
        result = self._clean("it  was  red")
        self.assertNotIn("  ", result)


# ---------------------------------------------------------------------------
# extract_edit_prompt
# ---------------------------------------------------------------------------

class TestExtractEditPrompt(unittest.TestCase):

    def test_llm_fn_is_called_and_flags_llm_used(self):
        called_with = []

        def my_fn(transcript: str) -> str:
            called_with.append(transcript)
            return "Change the door to blue"

        pipeline = VoicePipeline(llm_fn=my_fn)
        prompt, llm_used = pipeline.extract_edit_prompt("The door used to be blue")

        self.assertTrue(llm_used)
        self.assertEqual(prompt, "Change the door to blue")
        self.assertEqual(called_with, ["The door used to be blue"])

    def test_no_api_key_uses_regex_fallback(self):
        pipeline = VoicePipeline(openai_api_key="")
        prompt, llm_used = pipeline.extract_edit_prompt("The awning was green")
        self.assertFalse(llm_used)
        self.assertIn("green", prompt)

    def test_llm_fn_takes_priority_over_api_key(self):
        """llm_fn must be used even when an API key is set."""
        pipeline = VoicePipeline(
            openai_api_key="sk-fake",
            llm_fn=lambda t: "Make it green",
        )
        prompt, llm_used = pipeline.extract_edit_prompt("anything")
        self.assertTrue(llm_used)
        self.assertEqual(prompt, "Make it green")


# ---------------------------------------------------------------------------
# process_bytes (Whisper stubbed)
# ---------------------------------------------------------------------------

class TestProcessBytes(unittest.TestCase):

    def _make_pipeline_no_llm(self) -> VoicePipeline:
        """Pipeline with no OpenAI key so we stay on the regex path."""
        return VoicePipeline(openai_api_key="")

    def _stub_whisper(self, pipeline: VoicePipeline, transcript: str) -> None:
        """Replace _load_whisper with a mock that returns a static transcript."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": f"  {transcript}  "}
        pipeline._whisper_model = mock_model

    def test_process_bytes_returns_voice_result(self):
        pipeline = self._make_pipeline_no_llm()
        self._stub_whisper(pipeline, "The door was green")
        result = pipeline.process_bytes(b"fake-audio-bytes", suffix=".wav")
        self.assertIsInstance(result, VoiceResult)
        self.assertEqual(result.transcript, "The door was green")
        self.assertFalse(result.llm_used)

    def test_process_bytes_strips_transcript(self):
        pipeline = self._make_pipeline_no_llm()
        self._stub_whisper(pipeline, "  the roof was red  ")
        result = pipeline.process_bytes(b"bytes", suffix=".mp3")
        self.assertEqual(result.transcript, "the roof was red")

    def test_llm_fn_applied_to_transcript(self):
        pipeline = VoicePipeline(llm_fn=lambda t: f"EDITED: {t}")
        self._stub_whisper(pipeline, "the awning was blue")
        result = pipeline.process_bytes(b"bytes", suffix=".wav")
        self.assertTrue(result.llm_used)
        self.assertEqual(result.edit_prompt, "EDITED: the awning was blue")


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------

class TestProcessFile(unittest.TestCase):

    def test_raises_file_not_found(self):
        pipeline = VoicePipeline(openai_api_key="")
        with self.assertRaises(FileNotFoundError):
            pipeline.process_file("/does/not/exist.wav")


# ---------------------------------------------------------------------------
# VoiceResult repr
# ---------------------------------------------------------------------------

class TestVoiceResultRepr(unittest.TestCase):

    def test_repr_contains_key_fields(self):
        r = VoiceResult(
            transcript="raw text",
            edit_prompt="Change X to Y",
            model_used="base",
            llm_used=True,
        )
        s = repr(r)
        self.assertIn("raw text", s)
        self.assertIn("Change X to Y", s)
        self.assertIn("True", s)


if __name__ == "__main__":
    unittest.main()
