import subprocess
from pathlib import Path

import pytest

import nursery.video.ffmpeg as ffmpeg_module
from nursery.config import VideoConfig
from nursery.video.ffmpeg import FFmpegError, build_command, has_subtitles_filter, run_ffmpeg
from nursery.video.kenburns import SceneClip


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    """The capability probe is cached at module level; isolate tests from each other."""
    ffmpeg_module._CAPABILITY_CACHE.clear()
    yield
    ffmpeg_module._CAPABILITY_CACHE.clear()


def clips(n: int) -> list[SceneClip]:
    return [SceneClip(image=Path(f"/tmp/s{i}.png"), duration_s=2.0, direction="in")
            for i in range(n)]


def test_each_image_becomes_a_looped_input():
    cmd = build_command(clips(3), Path("/tmp/a.wav"), "graph", Path("/tmp/o.mp4"),
                        VideoConfig())

    assert cmd.count("-loop") == 3
    assert cmd.count("-i") == 4  # three images plus the audio


def test_audio_is_the_last_input():
    cmd = build_command(clips(2), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())
    idx = cmd.index("/tmp/a.wav")

    assert cmd[idx - 1] == "-i"
    assert "/tmp/s1.png" in cmd[: idx - 1]


def test_maps_graph_output_and_audio_stream():
    cmd = build_command(clips(2), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())

    assert "[vout]" in cmd
    assert "2:a" in cmd


def test_encodes_h264_yuv420p_and_aac():
    cmd = build_command(clips(1), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())

    assert "libx264" in cmd
    assert "yuv420p" in cmd
    assert "aac" in cmd


def test_uses_shortest_so_video_matches_audio():
    cmd = build_command(clips(1), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())

    assert "-shortest" in cmd


def test_overwrites_without_prompting():
    cmd = build_command(clips(1), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())

    assert "-y" in cmd


def test_run_ffmpeg_raises_with_stderr_on_failure():
    with pytest.raises(FFmpegError) as exc:
        run_ffmpeg(["ffmpeg", "-i", "/nonexistent/nope.mp4", "-f", "null", "-"])

    assert exc.value.stderr


def test_extract_thumbnail_uses_update_flag_for_a_single_image(tmp_path, monkeypatch):
    from nursery.video.ffmpeg import extract_thumbnail

    video = tmp_path / "in.mp4"
    video.write_bytes(b"not a real video")
    calls = []
    monkeypatch.setattr(ffmpeg_module, "run_ffmpeg", lambda cmd: calls.append(cmd))

    extract_thumbnail(video, at_s=0.0, out=tmp_path / "thumb.png")

    cmd = calls[0]
    idx = cmd.index("-frames:v")
    assert cmd[idx + 1] == "1"
    assert "-update" in cmd
    assert cmd[cmd.index("-update") + 1] == "1"


class TestFfmpegResolution:
    """Resolution order for the ffmpeg binary: env var, then ffmpeg-full if it
    has libass, then whatever `ffmpeg` is on PATH."""

    def test_env_var_wins_when_set(self, monkeypatch):
        monkeypatch.setenv("NURSERY_FFMPEG", "/custom/path/to/ffmpeg")

        assert ffmpeg_module._binary() == "/custom/path/to/ffmpeg"

    def test_env_var_wins_even_over_ffmpeg_full(self, monkeypatch):
        monkeypatch.setenv("NURSERY_FFMPEG", "/custom/path/to/ffmpeg")
        full = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
        monkeypatch.setattr(ffmpeg_module, "_FFMPEG_FULL", full)

        assert ffmpeg_module._binary() == "/custom/path/to/ffmpeg"

    def test_capability_probe_result_is_cached(self, monkeypatch, tmp_path):
        fake_binary = tmp_path / "fake-ffmpeg"
        fake_binary.write_text("#!/bin/sh\necho ' T.. subtitles V->V desc'\n")
        fake_binary.chmod(0o755)

        calls = []
        real_run = subprocess.run

        def counting_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", counting_run)

        assert has_subtitles_filter(str(fake_binary)) is True
        assert has_subtitles_filter(str(fake_binary)) is True
        assert len(calls) == 1

    def test_no_binary_available_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("NURSERY_FFMPEG", raising=False)
        missing = Path("/nonexistent/ffmpeg-full/bin/ffmpeg")
        monkeypatch.setattr(ffmpeg_module, "_FFMPEG_FULL", missing)
        monkeypatch.setattr("shutil.which", lambda name: None)

        with pytest.raises(FFmpegError, match="ffmpeg"):
            ffmpeg_module._binary()

    def test_requesting_subtitles_without_libass_raises_actionable_error(
        self, monkeypatch, tmp_path
    ):
        fake_binary = tmp_path / "fake-ffmpeg-no-libass"
        fake_binary.write_text("#!/bin/sh\necho ' .. scale V->V desc'\n")
        fake_binary.chmod(0o755)
        monkeypatch.setenv("NURSERY_FFMPEG", str(fake_binary))

        with pytest.raises(FFmpegError, match="ffmpeg-full"):
            build_command(clips(1), Path("/tmp/a.wav"), "[v0]subtitles='x.ass'[vout]",
                          Path("/tmp/o.mp4"), VideoConfig())

    @pytest.mark.skipif(
        not Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg").exists(),
        reason="ffmpeg-full not installed on this machine",
    )
    def test_picks_up_ffmpeg_full_when_it_supports_subtitles(self, monkeypatch):
        monkeypatch.delenv("NURSERY_FFMPEG", raising=False)

        resolved = ffmpeg_module._binary()

        assert resolved == str(Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"))
