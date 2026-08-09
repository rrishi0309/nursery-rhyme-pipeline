import shutil
import subprocess
from pathlib import Path

import pytest

from nursery.manifest import StageRecord
from nursery.stages.assemble import AssembleStage

_FFPROBE_FULL = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffprobe")


def _ffprobe_binary() -> str:
    """Prefer the libass-capable ffprobe if present; else whatever's on PATH.

    Either binary can probe a plain MP4 - only encoding (subtitles burn-in)
    needs the libass-enabled build.
    """
    if _FFPROBE_FULL.exists():
        return str(_FFPROBE_FULL)
    found = shutil.which("ffprobe")
    if found is None:
        raise RuntimeError("ffprobe not found; install it with `brew install ffmpeg`")
    return found


def probe(path: Path, stream: str, entries: str) -> str:
    out = subprocess.run(
        [_ffprobe_binary(), "-v", "error", "-select_streams", stream,
         "-show_entries", entries, "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_produces_a_playable_mp4(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    final = Path(result.video.final_path)
    assert final.exists()
    assert final.stat().st_size > 10_000


def test_video_has_expected_resolution(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    assert probe(Path(result.video.final_path), "v:0", "stream=width,height") == "1920,1080"


def test_video_has_an_audio_stream(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    assert probe(Path(result.video.final_path), "a:0", "stream=codec_name") == "aac"


def test_duration_roughly_matches_audio(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    duration = float(probe(Path(result.video.final_path), "v:0", "format=duration"))
    assert 3.0 < duration < 5.5


def test_thumbnail_is_written(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    assert Path(result.video.thumbnail_path).exists()


def test_scene_without_timing_raises(ready_manifest, cfg):
    ready_manifest.scenes[0].start_s = None

    with pytest.raises(ValueError, match="timing"):
        AssembleStage().execute(ready_manifest, cfg)


def test_scene_without_image_raises(ready_manifest, cfg):
    ready_manifest.scenes[1].image_path = None

    with pytest.raises(ValueError, match="image"):
        AssembleStage().execute(ready_manifest, cfg)


def test_is_satisfied_after_a_successful_run(ready_manifest, cfg):
    stage = AssembleStage()
    m = stage.execute(ready_manifest, cfg)
    m.stages["assemble"] = StageRecord(status="ok", input_hash=stage.current_hash(m, cfg))

    assert stage.is_satisfied(m, cfg) is True


def test_changing_a_scene_image_invalidates_satisfaction(ready_manifest, cfg):
    stage = AssembleStage()
    m = stage.execute(ready_manifest, cfg)
    m.stages["assemble"] = StageRecord(status="ok", input_hash=stage.current_hash(m, cfg))

    m.scenes[0].image_path = "/tmp/different.png"

    assert stage.is_satisfied(m, cfg) is False
