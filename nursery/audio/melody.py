"""Synthesizes a simple music-box style melody from a note sequence.

The tunes in the catalog are public domain, and rendering them here means no
existing recording or arrangement is ever copied. The voice is a decayed sine
with a couple of harmonics, which reads as a toy glockenspiel - close enough to
the nursery-rhyme idiom without needing a sampled instrument.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 44100

_SEMITONES = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11,
}


def note_to_freq(note: str) -> float:
    """Convert scientific pitch notation (e.g. "C4", "F#5") to Hz."""
    body = note.strip().upper()
    octave_start = next(i for i, ch in enumerate(body) if ch.isdigit())
    name, octave = body[:octave_start], int(body[octave_start:])
    if name not in _SEMITONES:
        raise ValueError(f"unknown note name {name!r} in {note!r}")
    midi = (octave + 1) * 12 + _SEMITONES[name]
    return 440.0 * (2 ** ((midi - 69) / 12))


def parse_sequence(tokens: list[str], tempo_bpm: float) -> list[tuple[float | None, float]]:
    """Turn ["C4:1", "r:0.5"] into [(freq_or_None, seconds), ...]."""
    beat_s = 60.0 / tempo_bpm
    out: list[tuple[float | None, float]] = []
    for token in tokens:
        name, _, beats = token.partition(":")
        seconds = float(beats or 1) * beat_s
        out.append((None if name.strip().lower() == "r" else note_to_freq(name), seconds))
    return out


def _render_note(freq: float | None, seconds: float, amplitude: float) -> list[float]:
    n = int(SAMPLE_RATE * seconds)
    if freq is None:
        return [0.0] * n

    # Struck-then-decaying envelope, with a short fade-in so it never clicks.
    attack = max(1, int(SAMPLE_RATE * 0.006))
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        decay = math.exp(-3.2 * t / max(seconds, 1e-6))
        env = decay * (i / attack if i < attack else 1.0)
        value = (
            math.sin(2 * math.pi * freq * t)
            + 0.32 * math.sin(2 * math.pi * freq * 2 * t)
            + 0.12 * math.sin(2 * math.pi * freq * 3 * t)
        )
        samples.append(amplitude * env * value / 1.44)
    return samples


def render_melody(
    tokens: list[str], tempo_bpm: float, out: Path, amplitude: float = 0.28
) -> Path:
    """Render the note sequence to a mono 16-bit WAV."""
    samples: list[float] = []
    for freq, seconds in parse_sequence(tokens, tempo_bpm):
        samples.extend(_render_note(freq, seconds, amplitude))

    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
        )
    return out


def melody_duration_s(tokens: list[str], tempo_bpm: float) -> float:
    return sum(seconds for _, seconds in parse_sequence(tokens, tempo_bpm))
