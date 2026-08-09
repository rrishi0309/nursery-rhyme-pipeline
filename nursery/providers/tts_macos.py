"""Narration via the macOS `say` binary.

Zero download and effectively instant, which makes it the right default while
a real singing model is still being evaluated. Crucially, synthesizing one WAV
per lyric line gives exact per-line durations for free - so the pipeline gets
its scene timings without running a forced aligner at all.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import wave
from pathlib import Path

from nursery.providers.base import WordTiming

DEFAULT_VOICE = "Samantha"


class MacSayUnavailableError(RuntimeError):
    pass


def _say_binary() -> str:
    found = shutil.which("say")
    if found is None:
        raise MacSayUnavailableError("the macOS `say` binary was not found on PATH")
    return found


def wav_duration_s(path: Path) -> float:
    with contextlib.closing(wave.open(str(path))) as wf:
        return wf.getnframes() / wf.getframerate()


class MacTTSProvider:
    """Renders text to a 44.1 kHz mono WAV using the system voice."""

    name = "macos-say"

    def __init__(self, voice: str = DEFAULT_VOICE, rate_wpm: int = 140):
        self.voice = voice
        self.rate_wpm = rate_wpm

    def speak(self, text: str, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        aiff = out.with_suffix(".aiff")
        subprocess.run(
            [_say_binary(), "-v", self.voice, "-r", str(self.rate_wpm), "-o", str(aiff), text],
            check=True,
            capture_output=True,
        )
        # `say` only writes AIFF/CAF; normalise to the WAV the rest of the
        # pipeline expects.
        from nursery.video.ffmpeg import _binary, run_ffmpeg

        run_ffmpeg([
            _binary(), "-y", "-v", "error", "-i", str(aiff),
            "-ac", "1", "-ar", "44100", str(out),
        ])
        aiff.unlink(missing_ok=True)
        return out

    def speak_lines(self, lines: list[str], out_dir: Path) -> list[tuple[Path, float]]:
        """Render one WAV per line, returning (path, duration) in order."""
        results = []
        for i, line in enumerate(lines):
            path = self.speak(line, out_dir / f"line_{i:02d}.wav")
            results.append((path, wav_duration_s(path)))
        return results


def distribute_words(text: str, start_s: float, end_s: float) -> list[WordTiming]:
    """Spread a line's words evenly across its measured span.

    `say` exposes no word-level timing, so within a line this is an even split.
    The line boundaries themselves are exact, which is what keeps captions in
    step with the narration.
    """
    words = text.split()
    if not words:
        return []
    step = (end_s - start_s) / len(words)
    return [
        WordTiming(word=w, start_s=start_s + i * step, end_s=start_s + (i + 1) * step)
        for i, w in enumerate(words)
    ]
