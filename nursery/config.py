"""Typed pipeline configuration.

Every model in this pipeline runs as a subprocess against a CLI - there are no
in-process model APIs. The fields below are exactly the flags/TOML keys each
tool's `--help` (captured in `docs/tool-help/`) exposes, plus a `python`/`bin`
interpreter path for the tools that need their own venv (ACE-Step 1.5,
whisperx-mlx, and ltx-2-mlx all require Python <3.13 or an untested-above-3.13
interpreter; mlx-lm and mflux run fine from the project's own venv).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_TOOLS_CACHE = Path.home() / ".cache" / "nursery-tools"


class VideoConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crossfade_s: float = 0.5
    zoom_rate: float = 0.0008
    max_zoom: float = 1.18


class LyricsConfig(BaseModel):
    """Qwen via `mlx_lm.generate`. Runs from the project's own venv."""

    model_repo: str = "mlx-community/Qwen3.5-4B-MLX-4bit"
    max_tokens: int = 2048
    temperature: float = 0.8
    max_retries: int = 3


class SongConfig(BaseModel):
    """ACE-Step 1.5, driven via a TOML config file (`cli.py --config`).

    ACE-Step's CLI has no direct generate flags - `cli.py` is wizard/config
    only, so the song provider writes a TOML file rather than a command line.
    """

    variant: str = "ACE-Step/acestep-v15-xl-turbo"
    inference_steps: int = 8
    guidance_scale: float = 7.0
    audio_format: str = "wav"
    duration: float = -1.0  # -1.0 = auto (let ACE-Step pick from the lyrics)
    backend: str = "mlx"

    # ACE-Step 1.5 requires Python <3.13; it lives in its own venv.
    python: str = str(_TOOLS_CACHE / "ACE-Step-1.5" / ".venv" / "bin" / "python")
    repo_dir: str = str(_TOOLS_CACHE / "ACE-Step-1.5")


class AlignConfig(BaseModel):
    """whisperx-mlx word-level alignment for recovering line timings."""

    model: str = "small"
    # Plain "mlx" may not emit word timestamps; --word_timestamps requires
    # backend='lightning' per whisperx-mlx's --help.
    backend: str = "lightning"
    max_drift_s: float = 0.75

    # Not on PyPI; lives in its own venv (Python <3.13).
    bin: str = str(_TOOLS_CACHE / "whisperx-mlx" / ".venv" / "bin" / "whisperx")


class ImageConfig(BaseModel):
    """FLUX.2 Klein via `mflux-generate-flux2`. Runs from the project's own venv."""

    model: str = "flux2-klein-4b"
    steps: int = 4
    width: int = 1344
    height: int = 768
    quantize: int = 4
    guidance: float = 3.5

    # img2img strength used when conditioning a scene on the cast sheet via
    # `--image cast.png <strength>`. mflux's only image-conditioning flag for
    # FLUX.2 Klein is this init-image/img2img path, not a dedicated
    # identity-preserving reference-conditioning mechanism.
    cast_strength: float = 0.35


class AnimateConfig(BaseModel):
    """LTX-2.3 image-to-video (`ltx-2-mlx generate`) and the gate that rejects
    bad clips.

    width/height are ltx-2-mlx's own defaults (704x480); the old 1024x576 was
    sized for a different tool (mlx-video/LTX-2). `nursery/video/kenburns.py`
    and the assemble stage still render at `VideoConfig.width/height`
    (1920x1080) - LTX clips are upscaled to match.
    """

    model_repo: str = "dgrauet/ltx-2.3-mlx-q4"
    width: int = 704
    height: int = 480
    # Mandatory: ltx-2-mlx has no default and warns that values far from 24
    # (what LTX-2.3 was trained at) drift out of distribution.
    frame_rate: int = 24
    steps: int = 8
    two_stage: bool = False
    low_ram: bool = True

    # Not on PyPI; lives in its own venv (Python <3.13, untested above).
    bin: str = str(_TOOLS_CACHE / "ltx-2-mlx" / ".venv" / "bin" / "ltx-2-mlx")

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
    lyrics: LyricsConfig = Field(default_factory=LyricsConfig)
    song: SongConfig = Field(default_factory=SongConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    animate: AnimateConfig = Field(default_factory=AnimateConfig)
    providers: dict[str, str] = Field(
        default_factory=lambda: {
            "lyrics": "qwen",
            "song": "acestep",
            "align": "whisperx",
            "image": "flux2",
            "animate": "ltx",
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
