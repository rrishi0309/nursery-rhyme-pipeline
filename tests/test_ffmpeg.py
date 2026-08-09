import shutil
import subprocess
from pathlib import Path

import pytest

import nursery.video.ffmpeg as ffmpeg_module
from nursery.config import VideoConfig
from nursery.providers.fake import FakeImageProvider, FakeSongProvider
from nursery.video.ffmpeg import FFmpegError, build_command, has_subtitles_filter, run_ffmpeg
from nursery.video.kenburns import SceneClip, build_filter_graph


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    """The capability probe is cached at module level; isolate tests from each other."""
    ffmpeg_module._CAPABILITY_CACHE.clear()
    yield
    ffmpeg_module._CAPABILITY_CACHE.clear()


def clips(n: int) -> list[SceneClip]:
    return [SceneClip(image=Path(f"/tmp/s{i}.png"), start_s=i * 2.0, end_s=(i + 1) * 2.0,
                      direction="in")
            for i in range(n)]


def _resolve_ffprobe() -> str:
    """Resolve ffprobe the same way ffmpeg.py resolves ffmpeg: prefer the
    binary living next to whichever ffmpeg was picked (so ffmpeg-full's
    ffprobe is used when ffmpeg-full's ffmpeg was), falling back to PATH.
    """
    ffmpeg_bin = Path(ffmpeg_module._binary())
    sibling = ffmpeg_bin.with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    found = shutil.which("ffprobe")
    if found is None:
        raise RuntimeError("ffprobe not found; install it with `brew install ffmpeg`")
    return found


def test_each_image_is_a_single_still_input_with_no_loop_or_duration_flags():
    """A PNG is already exactly one frame; zoompan expands it to `d` frames on
    its own. `-loop 1 -t <dur>` instead hands zoompan a full stream of
    duplicated frames, so it emits `d` frames *per input frame* it receives -
    producing a video stream many times longer than the scene timeline, only
    masked by `-shortest`. See fix-round-1-findings.md finding 2.
    """
    cmd = build_command(clips(3), Path("/tmp/a.wav"), "graph", Path("/tmp/o.mp4"),
                        VideoConfig())

    assert "-loop" not in cmd
    assert "-t" not in cmd
    assert cmd.count("-i") == 4  # three images plus the audio


def test_stream_duration_matches_scene_timeline_when_actually_encoded(tmp_path):
    """The real-encode regression test for finding 2: build a genuine 4-scene
    video through the fake providers and assert the *video stream* duration
    (not the container's `format=duration`, which -shortest can mask) matches
    the scene timeline within one frame.
    """
    cfg = VideoConfig(fps=30, crossfade_s=0.5, width=320, height=180)
    images = FakeImageProvider(width=320, height=180)

    scene_clips = []
    for i in range(4):
        img = tmp_path / f"scene_{i}.png"
        images.generate("x", [], seed=i, out=img)
        scene_clips.append(
            SceneClip(image=img, start_s=i * 4.0, end_s=(i + 1) * 4.0, direction="in")
        )

    audio = tmp_path / "audio.wav"
    FakeSongProvider(duration_s=16.0).render("la", "cheerful", audio)

    graph = build_filter_graph(scene_clips, cfg, subtitles=None)
    out = tmp_path / "out.mp4"
    run_ffmpeg(build_command(scene_clips, audio, graph, out, cfg))

    probe = subprocess.run(
        [_resolve_ffprobe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    stream_duration = float(probe.stdout.strip())

    assert abs(stream_duration - 16.0) < (1 / cfg.fps)


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
