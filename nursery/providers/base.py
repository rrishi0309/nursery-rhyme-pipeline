"""Narrow interfaces between stages and models.

Stages depend only on these protocols, never on a model library. That is what
lets the pipeline run in CI with fakes and swap backends in one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_s: float
    end_s: float


@runtime_checkable
class ImageProvider(Protocol):
    def generate(self, prompt: str, refs: list[Path], seed: int, out: Path) -> Path: ...


@runtime_checkable
class SongProvider(Protocol):
    def render(self, lyrics: str, style: str, out: Path) -> Path: ...


@runtime_checkable
class TTSProvider(Protocol):
    def speak(self, text: str, out: Path) -> Path: ...


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, prompt: str, schema: dict | None = None) -> str: ...


@runtime_checkable
class AlignProvider(Protocol):
    def align(self, audio: Path, text: str) -> list[WordTiming]: ...
