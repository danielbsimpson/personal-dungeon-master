"""
Voice interface for Personal Dungeon Master (Phase 14).

Adds speech-to-text (microphone → text) and text-to-speech (DM narration →
audio) so the player can *speak* to the Dungeon Master and *hear* the replies.

Seamless streaming design
-------------------------
The DM streams its response token by token (``DungeonMaster.respond_stream``).
Tokens are accumulated into a :class:`SentenceAccumulator`; the moment a full
sentence is complete it is dispatched to a two-stage audio pipeline:

    sentence text ──▶ synth worker ──▶ (text, audio) ──▶ play worker

The synth worker calls Kokoro TTS while the *previous* sentence is still
playing, so audio for the next sentence is ready by the time the current one
finishes — eliminating gaps between sentences.

The play worker reveals the sentence text character by character over the
duration of its audio clip (a "karaoke" effect) so the on-screen text appears
in sync with the spoken words.

Heavy dependencies (``kokoro``, ``faster-whisper``, ``sounddevice``) are
imported lazily so this module can be imported — and its pure helpers tested —
without them installed.  Install with::

    pip install -r requirements-voice.txt
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.rule import Rule

if TYPE_CHECKING:
    import numpy as np

    from src.config import Settings
    from src.dm.dungeon_master import DungeonMaster
    from src.dm.memory.manager import MemoryManager
    from src.rules.loader import RulesReference

log = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Sentence chunking (pure helpers — fully unit-testable)
# ---------------------------------------------------------------------------

# Sentence-terminating punctuation, optionally followed by closing quotes/brackets.
_SENTENCE_END = set(".!?")
_TRAILING = set("\"')]}")
# Markdown / decoration we don't want the TTS engine to read aloud.
_MARKDOWN_RE = re.compile(r"[*_`#>]+")
_ROLL_DECOR_RE = re.compile(r"[✨🎲🎯⚔️🛡️]")
_WHITESPACE_RE = re.compile(r"\s+")


def extract_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Split *buffer* into completed sentences and a trailing remainder.

    A sentence is considered complete when a run of ``.``/``!``/``?`` (plus any
    closing quotes or brackets) is followed by whitespace, or when a blank line
    (``\\n\\n``) is encountered.  Punctuation at the very end of *buffer* with no
    following whitespace is treated as incomplete and kept in the remainder so a
    later chunk can finish it.

    Parameters
    ----------
    buffer:
        The accumulated text streamed so far.

    Returns
    -------
    tuple[list[str], str]
        ``(completed_sentences, remainder)``.  Each completed sentence is
        stripped of surrounding whitespace.
    """
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(buffer)

    while i < n:
        ch = buffer[i]

        # Hard break on a blank line (paragraph boundary).
        if ch == "\n" and i + 1 < n and buffer[i + 1] == "\n":
            sentence = buffer[start : i + 1].strip()
            if sentence:
                sentences.append(sentence)
            # Skip the run of newlines.
            j = i
            while j < n and buffer[j] == "\n":
                j += 1
            start = j
            i = j
            continue

        if ch in _SENTENCE_END:
            j = i + 1
            while j < n and buffer[j] in _SENTENCE_END:
                j += 1
            while j < n and buffer[j] in _TRAILING:
                j += 1
            if j >= n:
                # Punctuation at the end with no following char yet — wait.
                break
            if buffer[j].isspace():
                sentence = buffer[start:j].strip()
                if sentence:
                    sentences.append(sentence)
                start = j
                i = j
                continue
        i += 1

    return sentences, buffer[start:]


def clean_for_speech(text: str) -> str:
    """
    Strip markdown decoration and emoji so the TTS engine reads clean prose.

    Removes ``* _ ` # >`` markdown characters and dice/result emoji, then
    collapses repeated whitespace.  Returns an empty string if nothing speakable
    remains.
    """
    text = _MARKDOWN_RE.sub("", text)
    text = _ROLL_DECOR_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


class SentenceAccumulator:
    """
    Incrementally buffers streamed text and emits complete sentences.

    Feed token chunks via :meth:`push`; it returns any newly-completed
    sentences.  Call :meth:`flush` at the end of the stream to retrieve any
    trailing partial sentence.
    """

    def __init__(self) -> None:
        self._buf = ""

    def push(self, chunk: str) -> list[str]:
        """Add *chunk* and return any sentences completed by it."""
        self._buf += chunk
        sentences, self._buf = extract_sentences(self._buf)
        return sentences

    def flush(self) -> list[str]:
        """Return the remaining buffered text (if any) as a final sentence."""
        rest = self._buf.strip()
        self._buf = ""
        return [rest] if rest else []


# ---------------------------------------------------------------------------
# Text-to-speech (Kokoro) and speech-to-text (faster-whisper)
# ---------------------------------------------------------------------------


class KokoroTTS:
    """
    Text-to-speech backed by the Kokoro model.

    Kokoro is fast and runs locally.  ``synthesize`` returns a mono ``float32``
    waveform and its sample rate (Kokoro outputs 24 kHz).
    """

    sample_rate: int = 24_000

    def __init__(
        self, voice: str = "af_heart", lang_code: str = "a", speed: float = 1.0
    ) -> None:
        try:
            from kokoro import KPipeline  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Kokoro is not installed. Install voice dependencies with:\n"
                "  pip install -r requirements-voice.txt"
            ) from exc

        self._voice = voice
        self._speed = speed
        self._pipeline = KPipeline(lang_code=lang_code)
        log.info("KokoroTTS ready: voice='%s' lang='%s' speed=%s", voice, lang_code, speed)

    def synthesize(self, text: str) -> "tuple[np.ndarray, int]":
        """Render *text* to a waveform.  Returns ``(samples, sample_rate)``."""
        import numpy as np  # noqa: PLC0415

        chunks: list[np.ndarray] = []
        for result in self._pipeline(text, voice=self._voice, speed=self._speed):
            audio = result[-1] if isinstance(result, tuple) else getattr(result, "audio", result)
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype="float32").flatten())

        if not chunks:
            return np.zeros(0, dtype="float32"), self.sample_rate
        return np.concatenate(chunks).astype("float32"), self.sample_rate


class WhisperSTT:
    """Speech-to-text backed by faster-whisper (runs locally)."""

    def __init__(
        self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "faster-whisper is not installed. Install voice dependencies with:\n"
                "  pip install -r requirements-voice.txt"
            ) from exc

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log.info(
            "WhisperSTT ready: model='%s' device='%s' compute='%s'",
            model_size,
            device,
            compute_type,
        )

    def transcribe(self, audio: "np.ndarray") -> str:
        """Transcribe a mono 16 kHz ``float32`` waveform to text."""
        if audio is None or len(audio) == 0:
            return ""
        segments, _info = self._model.transcribe(audio, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()


def build_voice_engines(settings: "Settings") -> tuple[WhisperSTT, KokoroTTS]:
    """
    Construct the STT and TTS engines from application settings.

    Raises a clear :class:`RuntimeError` if the optional voice dependencies are
    not installed.
    """
    tts = KokoroTTS(
        voice=settings.tts_voice,
        lang_code=settings.tts_lang_code,
        speed=settings.tts_speed,
    )
    stt = WhisperSTT(
        model_size=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
    )
    return stt, tts


# ---------------------------------------------------------------------------
# Microphone capture (push-to-talk)
# ---------------------------------------------------------------------------


def record_until_enter(sample_rate: int = 16_000) -> "np.ndarray":
    """
    Capture microphone audio until the player presses Enter (push-to-talk).

    Returns a mono ``float32`` waveform at *sample_rate*.  Returns an empty
    array if nothing was captured.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import sounddevice as sd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "sounddevice is not installed. Install voice dependencies with:\n"
            "  pip install -r requirements-voice.txt"
        ) from exc

    frames: list[np.ndarray] = []

    def _callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
        if status:
            log.debug("Audio input status: %s", status)
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32", callback=_callback
    )
    console.print(
        "[bold red]● Recording…[/bold red] [dim]speak now, then press Enter to stop[/dim]"
    )
    stream.start()
    try:
        input()
    finally:
        stream.stop()
        stream.close()

    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


# ---------------------------------------------------------------------------
# Streaming TTS playback pipeline
# ---------------------------------------------------------------------------


class SentenceStreamer:
    """
    Speaks streamed DM narration sentence by sentence with synced text reveal.

    Feed token chunks via :meth:`feed`; completed sentences are queued to a
    background synthesis worker (Kokoro) and a playback worker (sounddevice).
    Synthesis of the next sentence overlaps playback of the current one, so the
    DM speaks without gaps.  Call :meth:`finish` to flush the final partial
    sentence and block until all audio has played.

    Parameters
    ----------
    tts:
        A text-to-speech engine exposing ``synthesize(text) -> (samples, sr)``.
    speak:
        When ``True`` (default) audio is synthesised and played.  When ``False``
        only the on-screen text is shown (useful for testing or muted play).
    display:
        When ``True`` (default) the sentence text is revealed on screen.
    """

    def __init__(
        self,
        tts: Optional[KokoroTTS],
        speak: bool = True,
        display: bool = True,
    ) -> None:
        self._tts = tts
        self._speak = speak and tts is not None
        self._display = display
        self._acc = SentenceAccumulator()
        self.dispatched: list[str] = []

        self._synth_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._play_q: "queue.Queue[Optional[tuple]]" = queue.Queue()

        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    # -- producer side ----------------------------------------------------

    def feed(self, chunk: str) -> None:
        """Add a streamed token chunk; dispatch any newly-completed sentences."""
        for sentence in self._acc.push(chunk):
            self._dispatch(sentence)

    def finish(self) -> None:
        """Flush the trailing sentence and block until playback completes."""
        for sentence in self._acc.flush():
            self._dispatch(sentence)
        self._synth_q.put(None)  # signal end-of-stream to the synth worker
        self._play_thread.join()

    def _dispatch(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        self.dispatched.append(sentence)
        self._synth_q.put(sentence)

    # -- worker threads ---------------------------------------------------

    def _synth_worker(self) -> None:
        while True:
            sentence = self._synth_q.get()
            if sentence is None:
                self._play_q.put(None)  # propagate end-of-stream
                return
            samples = None
            sample_rate = 0
            if self._speak:
                speech = clean_for_speech(sentence)
                if speech:
                    try:
                        samples, sample_rate = self._tts.synthesize(speech)  # type: ignore[union-attr]
                    except Exception as exc:  # noqa: BLE001
                        log.warning("TTS synthesis failed: %s", exc)
                        samples, sample_rate = None, 0
            self._play_q.put((sentence, samples, sample_rate))

    def _play_worker(self) -> None:
        while True:
            item = self._play_q.get()
            if item is None:
                return
            sentence, samples, sample_rate = item
            duration = 0.0
            if samples is not None and sample_rate > 0 and len(samples) > 0:
                duration = len(samples) / float(sample_rate)
                self._play_audio(samples, sample_rate)

            if self._display:
                self._reveal_text(sentence, duration)

            if duration > 0:
                self._wait_audio()

    # -- audio / display primitives (overridable for tests) ---------------

    def _play_audio(self, samples, sample_rate: int) -> None:  # noqa: ANN001
        import sounddevice as sd  # noqa: PLC0415

        sd.play(samples, sample_rate)

    def _wait_audio(self) -> None:
        import sounddevice as sd  # noqa: PLC0415

        sd.wait()

    def _reveal_text(self, sentence: str, duration: float) -> None:
        """Print *sentence* character by character over *duration* seconds."""
        if duration <= 0 or not self._speak:
            console.print(sentence, style="green", markup=False, highlight=False)
            return

        delay = duration / max(1, len(sentence))
        for ch in sentence:
            console.print(ch, end="", style="green", markup=False, highlight=False)
            time.sleep(delay)
        console.print(" ", end="", markup=False, highlight=False)


# ---------------------------------------------------------------------------
# Voice session loop
# ---------------------------------------------------------------------------


async def run_voice_session(
    dm: "DungeonMaster",
    memory: "MemoryManager",
    stt: WhisperSTT,
    tts: KokoroTTS,
    rules: "Optional[RulesReference]" = None,
) -> None:
    """
    Run an interactive *spoken* DM session until the player quits.

    The player presses Enter to speak (push-to-talk) or types text directly.
    Slash commands (``/quit``, ``/status``, ``/journal`` …) are handled exactly
    as in the text CLI.  Normal turns stream the DM response, speaking each
    sentence as it completes while revealing the text in sync.
    """
    # Reuse the rich CLI helpers for banners, panels and commands.
    from src.interface import cli  # noqa: PLC0415

    rules = rules or dm.rules
    character = dm.campaign.character

    console.print()
    console.print(
        Rule(
            f"[dim]The adventure begins… [/dim]"
            f"[dim italic]DM: {dm.personality.name} · 🎙 voice mode[/dim italic]",
            style="dim green",
        )
    )
    console.print(
        "[dim]Press [bold]Enter[/bold] to speak, or type your action / a /command.[/dim]"
    )

    # Opening narration — spoken.
    opening = await dm.start_campaign()
    cli.print_roll_results(dm.last_roll_results)
    _speak_text(opening, tts)

    try:
        while True:
            console.print()
            typed = console.input(
                "[bold cyan]You[/bold cyan] [dim](Enter = speak)[/dim]: "
            ).strip()

            if typed:
                raw = typed
            else:
                audio = record_until_enter(sample_rate=dm._settings.voice_sample_rate)
                with console.status("[cyan]Transcribing…[/cyan]", spinner="dots"):
                    raw = stt.transcribe(audio).strip()
                if not raw:
                    console.print("[dim]Didn't catch that — try again.[/dim]")
                    continue
                console.print(f"[bold cyan]You said:[/bold cyan] {raw}")

            lower = raw.lower()

            if lower in ("/quit", "/exit"):
                console.print("[dim]Saving and exiting…[/dim]")
                break
            if lower == "/help":
                cli._cmd_help()
                continue
            if lower == "/status":
                cli._cmd_status(character)
                continue
            if lower == "/journal":
                with console.status("[magenta]Querying knowledge graph…[/magenta]", spinner="dots"):
                    await cli._cmd_journal(memory)
                continue
            if lower.startswith("/graph "):
                with console.status("[magenta]Searching graph…[/magenta]", spinner="dots"):
                    await cli._cmd_graph(memory, raw[7:].strip())
                continue
            if lower.startswith("/rules"):
                cli._cmd_rules(rules, raw[6:].strip())
                continue
            if lower.startswith("/roll "):
                cli._cmd_roll(raw[6:].strip())
                continue
            if lower == "/personality":
                cli._cmd_personality(dm)
                continue
            if lower == "/save":
                console.print("[green]Session saved.[/green]")
                continue
            if lower == "/reset":
                if cli._confirm("Wipe the session window? (Knowledge graph is preserved)"):
                    memory.reset_session()
                    console.print("[yellow]Session window cleared.[/yellow]")
                continue
            if lower == "/fullreset":
                if cli._confirm(
                    "Wipe BOTH the session window and the knowledge graph? "
                    "This restarts the campaign from scratch."
                ):
                    with console.status("[red]Resetting…[/red]", spinner="dots"):
                        await memory.full_reset()
                    console.print("[red]Full reset complete.[/red]")
                continue

            # --- Normal turn: stream + speak sentence by sentence ----------
            await _stream_and_speak(dm, raw, tts)

    except KeyboardInterrupt:
        console.print()
        console.print("[dim]Interrupted — saving session and exiting.[/dim]")

    with console.status("[dim]Saving session memory…[/dim]", spinner="dots"):
        await dm.end_session()

    console.print()
    console.print(Rule("[dim]Session ended.[/dim]", style="dim"))


async def _stream_and_speak(dm: "DungeonMaster", player_input: str, tts: KokoroTTS) -> None:
    """Stream the DM response, speaking each sentence as it completes."""
    console.print()
    console.print("[bold green]Dungeon Master[/bold green]")
    streamer = SentenceStreamer(tts, speak=True, display=True)
    try:
        async for chunk in dm.respond_stream(player_input):
            streamer.feed(chunk)
    finally:
        streamer.finish()
        console.print()

    cli_rolls = dm.last_roll_results
    if cli_rolls:
        from src.interface import cli  # noqa: PLC0415

        cli.print_roll_results(cli_rolls)


def _speak_text(text: str, tts: KokoroTTS) -> None:
    """Speak a complete block of text (e.g. the opening narration)."""
    console.print()
    console.print("[bold green]Dungeon Master[/bold green]")
    streamer = SentenceStreamer(tts, speak=True, display=True)
    streamer.feed(text)
    streamer.finish()
    console.print()
