"""Deterministic fake providers.

These produce real, valid files - a decodable PNG, a readable WAV - so that
downstream stages including ffmpeg exercise their true code paths without a
model or a GPU.
"""

from __future__ import annotations

import math
import struct
import wave
import zlib
from pathlib import Path

from nursery.providers.base import WordTiming

_SAMPLE_RATE = 44100


def _png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


class FakeImageProvider:
    """Writes a solid-colour PNG whose colour is a pure function of the seed."""

    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height

    def generate(self, prompt: str, refs: list[Path], seed: int, out: Path) -> Path:
        rgb = ((seed * 53) % 256, (seed * 97) % 256, (seed * 151) % 256)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_png_bytes(self.width, self.height, rgb))
        return out


def _write_tone(out: Path, duration_s: float, freq: float) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = int(_SAMPLE_RATE * duration_s)
    with wave.open(str(out), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(
            b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * n / _SAMPLE_RATE)))
                for n in range(frames)
            )
        )
    return out


class FakeSongProvider:
    def __init__(self, duration_s: float = 8.0):
        self.duration_s = duration_s

    def render(self, lyrics: str, style: str, out: Path) -> Path:
        return _write_tone(out, self.duration_s, freq=440.0)


class FakeTTSProvider:
    def __init__(self, duration_s: float = 8.0):
        self.duration_s = duration_s

    def speak(self, text: str, out: Path) -> Path:
        return _write_tone(out, self.duration_s, freq=220.0)


class FakeLLMProvider:
    def __init__(self, response: str = "{}"):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str, schema: dict | None = None) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeAlignProvider:
    """Spreads words evenly across the duration - contiguous, increasing."""

    def __init__(self, duration_s: float = 8.0):
        self.duration_s = duration_s

    def align(self, audio: Path, text: str) -> list[WordTiming]:
        words = text.split()
        if not words:
            return []
        step = self.duration_s / len(words)
        return [
            WordTiming(word=w, start_s=i * step, end_s=(i + 1) * step)
            for i, w in enumerate(words)
        ]


_FAKES = {
    "image": FakeImageProvider,
    "song": FakeSongProvider,
    "tts": FakeTTSProvider,
    "llm": FakeLLMProvider,
    "align": FakeAlignProvider,
}


def get_provider(kind: str, name: str) -> object:
    if name != "fake":
        raise KeyError(f"provider {name!r} for {kind!r} is not available until Plan 2")
    return _FAKES[kind]()
