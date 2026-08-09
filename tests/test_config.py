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
