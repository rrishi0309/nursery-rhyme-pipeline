from datetime import UTC, datetime
from pathlib import Path

import pytest

from nursery.config import Config
from nursery.manifest import AudioSpec, Manifest, Scene
from nursery.providers.fake import FakeImageProvider, FakeSongProvider


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(out_dir=tmp_path / "out", assets_dir=tmp_path / "assets")


@pytest.fixture
def ready_manifest(cfg: Config) -> Manifest:
    """A manifest with real images, real audio, and timings already filled in."""
    video_dir = cfg.video_dir("v1")
    images = FakeImageProvider(width=1280, height=720)

    scenes = []
    for i, text in enumerate(["twinkle twinkle little star", "how i wonder what you are"]):
        path = video_dir / "images" / f"scene_{i:02d}.png"
        images.generate(text, [], seed=i, out=path)
        scenes.append(
            Scene(index=i, text=text, visual_prompt=text, characters=[],
                  image_path=str(path), start_s=i * 2.0, end_s=(i + 1) * 2.0)
        )

    audio_path = video_dir / "audio" / "mix.wav"
    FakeSongProvider(duration_s=4.0).render("la", "cheerful", audio_path)

    return Manifest(
        video_id="v1", created_at=datetime(2026, 8, 8, tzinfo=UTC),
        source="classic", title="Twinkle", scenes=scenes,
        audio=AudioSpec(mix_path=str(audio_path), duration_s=4.0,
                        provider="fake", degraded=False),
    )
