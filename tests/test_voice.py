"""Tests for the voice interface (Phase 14).

These cover the pure, dependency-free pieces: sentence chunking, speech text
cleanup, and the threaded :class:`SentenceStreamer` pipeline (with audio
playback monkeypatched out so no real sound device is required).
"""

from __future__ import annotations

import pytest

from src.interface.voice import (
    SentenceAccumulator,
    SentenceStreamer,
    clean_for_speech,
    extract_sentences,
)


# ---------------------------------------------------------------------------
# extract_sentences
# ---------------------------------------------------------------------------


def test_extract_sentences_basic():
    sentences, remainder = extract_sentences("Hello there. How are you? ")
    assert sentences == ["Hello there.", "How are you?"]
    assert remainder.strip() == ""


def test_extract_sentences_keeps_incomplete_remainder():
    sentences, remainder = extract_sentences("The orc snarls and lunges")
    assert sentences == []
    assert remainder == "The orc snarls and lunges"


def test_extract_sentences_waits_for_whitespace_after_punctuation():
    """Punctuation at the very end (no trailing space) is held back."""
    sentences, remainder = extract_sentences("A sudden noise.")
    assert sentences == []
    assert remainder == "A sudden noise."


def test_extract_sentences_paragraph_break():
    sentences, remainder = extract_sentences("First line\n\nSecond part keeps going")
    assert sentences == ["First line"]
    assert remainder == "Second part keeps going"


def test_extract_sentences_handles_quotes_and_brackets():
    # A closing quote after terminal punctuation is treated as a boundary, so a
    # quoted exclamation becomes its own spoken chunk — fine for TTS pacing.
    sentences, remainder = extract_sentences('"Run!" she shouted. Then silence ')
    assert sentences == ['"Run!"', "she shouted."]
    assert remainder.strip() == "Then silence"


# ---------------------------------------------------------------------------
# SentenceAccumulator — incremental streaming
# ---------------------------------------------------------------------------


def test_accumulator_streams_across_chunks():
    acc = SentenceAccumulator()
    out: list[str] = []
    for chunk in ["The torch ", "flickers. ", "Shadows ", "dance on the wall. "]:
        out.extend(acc.push(chunk))
    assert out == ["The torch flickers.", "Shadows dance on the wall."]
    # No trailing partial sentence remains.
    assert acc.flush() == []


def test_accumulator_flush_returns_trailing_partial():
    acc = SentenceAccumulator()
    assert acc.push("A lingering thought") == []
    assert acc.flush() == ["A lingering thought"]


def test_accumulator_sentence_split_across_chunk_boundary():
    acc = SentenceAccumulator()
    assert acc.push("The torch flickers") == []
    # Punctuation arrives with no trailing whitespace yet — held back.
    assert acc.push(".") == []
    # Whitespace in the next chunk completes the sentence.
    completed = acc.push(" Then darkness.")
    assert completed == ["The torch flickers."]
    assert acc.flush() == ["Then darkness."]


# ---------------------------------------------------------------------------
# clean_for_speech
# ---------------------------------------------------------------------------


def test_clean_for_speech_strips_markdown():
    assert clean_for_speech("**Bold** and _italic_ and `code`") == "Bold and italic and code"


def test_clean_for_speech_strips_emoji_and_collapses_space():
    assert clean_for_speech("✨ You rolled   a   19 🎲") == "You rolled a 19"


def test_clean_for_speech_empty_when_only_decoration():
    assert clean_for_speech("***") == ""


# ---------------------------------------------------------------------------
# SentenceStreamer — threaded pipeline (audio mocked out)
# ---------------------------------------------------------------------------


class _FakeTTS:
    """Records synthesised text and returns a tiny silent waveform."""

    sample_rate = 24_000

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str):
        import numpy as np

        self.calls.append(text)
        return np.zeros(240, dtype="float32"), self.sample_rate


def _silent_streamer(tts, speak=True):
    """A SentenceStreamer with audio playback and text reveal neutralised."""
    streamer = SentenceStreamer(tts, speak=speak, display=False)
    streamer._play_audio = lambda samples, sr: None  # type: ignore[assignment]
    streamer._wait_audio = lambda: None  # type: ignore[assignment]
    return streamer


def test_streamer_dispatches_sentences_in_order():
    tts = _FakeTTS()
    streamer = _silent_streamer(tts)
    for chunk in ["The door opens. ", "A guard ", "steps forward. "]:
        streamer.feed(chunk)
    streamer.finish()

    assert streamer.dispatched == ["The door opens.", "A guard steps forward."]
    # Each dispatched sentence is synthesised (markdown-cleaned).
    assert tts.calls == ["The door opens.", "A guard steps forward."]


def test_streamer_flushes_trailing_partial_on_finish():
    tts = _FakeTTS()
    streamer = _silent_streamer(tts)
    streamer.feed("No terminal punctuation here")
    streamer.finish()
    assert streamer.dispatched == ["No terminal punctuation here"]


def test_streamer_does_not_synthesize_when_muted():
    tts = _FakeTTS()
    streamer = SentenceStreamer(tts, speak=False, display=False)
    streamer.feed("Silent narration here. ")
    streamer.finish()
    assert streamer.dispatched == ["Silent narration here."]
    assert tts.calls == []
