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


class AnimateConfig(BaseModel):
    """LTX image-to-video settings and the gate that rejects bad clips.

    Dimensions must be divisible by 64; 1024x576 is native 16:9 and upscales
    cleanly to 1080p in the assemble graph.
    """

    # Must be LTX-2, not LTX-2.3. mlx-video expects a diffusers-style layout
    # (vae/encoder, vae/decoder, transformer/config.json, audio_vae, vocoder);
    # the LTX-2.3 repo is 13 flat safetensors files with no subdirectories and
    # fails at VAE loading.
    #
    # Licensing: both are under the LTX-2 Community License, which permits
    # commercial use only below $10M annual revenue. That is weaker than
    # FLUX.1-schnell's unconditional Apache-2.0.
    model_repo: str = "Lightricks/LTX-2"
    pipeline: str = "distilled"
    width: int = 1024
    height: int = 576
    cfg_scale: float | None = None
    image_strength: float | None = None

    # Interpreter to run mlx-video under. mlx-video pulls in librosa/numba,
    # so it can be kept in a separate venv rather than the pipeline's own.
    python: str | None = None

    # A clip's first frame should resemble the still it was conditioned on.
    # Lenient by default: the still is a different resolution and LTX restyles
    # slightly even when it behaves.
    min_source_ssim: float = 0.55

    # First frame vs last frame. Above the ceiling nothing moved; below the
    # floor the subject drifted into mush.
    min_motion_ssim: float = 0.35
    max_motion_ssim: float = 0.995


class Config(BaseModel):
    out_dir: Path = Path("out")
    assets_dir: Path = Path("assets")
    review_gate: bool = True
    video: VideoConfig = Field(default_factory=VideoConfig)
    animate: AnimateConfig = Field(default_factory=AnimateConfig)
    providers: dict[str, str] = Field(
        default_factory=lambda: {
            "llm": "fake",
            "song": "fake",
            "tts": "fake",
            "image": "fake",
            "align": "fake",
            "animate": "none",
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
