from pathlib import Path

from nursery.config import Config


def test_defaults_when_no_file(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.video.width == 1920
    assert cfg.video.height == 1080
    assert cfg.video.fps == 30
    assert cfg.review_gate is True


def test_yaml_overrides_defaults(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("video:\n  fps: 24\nreview_gate: false\n")

    cfg = Config.load(path)

    assert cfg.video.fps == 24
    assert cfg.video.width == 1920  # untouched default
    assert cfg.review_gate is False


def test_paths_are_path_objects(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert isinstance(cfg.out_dir, Path)
    assert isinstance(cfg.assets_dir, Path)


def test_video_dir_is_derived_from_out_dir(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.video_dir("abc-01") == cfg.out_dir / "abc-01"


def test_default_providers_are_the_real_backends(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.providers == {
        "lyrics": "qwen",
        "song": "acestep",
        "align": "whisperx",
        "image": "flux2",
        "animate": "ltx",
    }


def test_lyrics_defaults(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.lyrics.model_repo == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert cfg.lyrics.max_retries == 3


def test_song_defaults(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.song.audio_format == "wav"
    assert cfg.song.duration == -1.0
    assert cfg.song.backend == "mlx"
    assert cfg.song.python.endswith("ACE-Step-1.5/.venv/bin/python")
    assert cfg.song.repo_dir.endswith("ACE-Step-1.5")


def test_align_defaults(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.align.model == "small"
    assert cfg.align.backend == "lightning"
    assert cfg.align.max_drift_s == 0.75
    assert cfg.align.bin.endswith("whisperx-mlx/.venv/bin/whisperx")


def test_image_defaults(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.image.model == "flux2-klein-4b"
    assert cfg.image.width == 1344
    assert cfg.image.height == 768
    assert cfg.image.quantize == 4
    assert cfg.image.guidance == 3.5
    assert cfg.image.cast_strength == 0.35


def test_animate_defaults(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.animate.model_repo == "dgrauet/ltx-2.3-mlx-q4"
    assert cfg.animate.width == 704
    assert cfg.animate.height == 480
    assert cfg.animate.frame_rate == 24
    assert cfg.animate.steps == 8
    assert cfg.animate.two_stage is False
    assert cfg.animate.low_ram is True
    assert cfg.animate.bin.endswith("ltx-2-mlx/.venv/bin/ltx-2-mlx")
    assert cfg.animate.min_source_ssim == 0.55
    assert cfg.animate.min_motion_ssim == 0.35
    assert cfg.animate.max_motion_ssim == 0.995


def test_repo_config_yaml_loads(tmp_path: Path):
    """The checked-in config.yaml must stay valid against the current schema."""
    repo_config = Path(__file__).parent.parent / "config.yaml"

    cfg = Config.load(repo_config)

    assert cfg.providers["animate"] == "ltx"
    assert cfg.animate.bin.endswith("ltx-2-mlx/.venv/bin/ltx-2-mlx")
