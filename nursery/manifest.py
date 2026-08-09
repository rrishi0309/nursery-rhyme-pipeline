"""Per-video manifest: the single source of truth for a pipeline run."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

StageStatus = Literal["pending", "running", "ok", "failed"]


class Scene(BaseModel):
    index: int
    text: str
    visual_prompt: str
    characters: list[str] = Field(default_factory=list)
    image_path: str | None = None
    start_s: float | None = None
    end_s: float | None = None

    @property
    def duration_s(self) -> float:
        if self.start_s is None or self.end_s is None:
            raise ValueError(f"scene {self.index} has no timing yet")
        return self.end_s - self.start_s


class StageRecord(BaseModel):
    status: StageStatus = "pending"
    input_hash: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AudioSpec(BaseModel):
    mix_path: str
    duration_s: float
    provider: str
    degraded: bool = False


class VideoSpec(BaseModel):
    final_path: str
    thumbnail_path: str
    duration_s: float


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    video_id: str
    created_at: datetime
    source: Literal["classic", "original"]
    title: str
    scenes: list[Scene] = Field(default_factory=list)
    audio: AudioSpec | None = None
    video: VideoSpec | None = None
    stages: dict[str, StageRecord] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = path.read_text(encoding="utf-8")
        obj = cls.model_validate_json(data)
        if obj.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {obj.schema_version}, expected {SCHEMA_VERSION}"
            )
        return obj

    def save(self, path: Path) -> None:
        """Serialize atomically so an interrupted run never corrupts the manifest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def record(self, stage: str) -> StageRecord:
        return self.stages.setdefault(stage, StageRecord())
