"""Typed pipeline configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crossfade_s: float = 0.5
    zoom_rate: float = 0.0008
    max_zoom: float = 1.18


class Config(BaseModel):
    out_dir: Path = Path("out")
    assets_dir: Path = Path("assets")
    review_gate: bool = True
    video: VideoConfig = Field(default_factory=VideoConfig)
    providers: dict[str, str] = Field(
        default_factory=lambda: {
            "llm": "fake",
            "song": "fake",
            "tts": "fake",
            "image": "fake",
            "align": "fake",
        }
    )

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        if path is not None and path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return cls.model_validate(data)
        return cls()

    def video_dir(self, video_id: str) -> Path:
        return self.out_dir / video_id
