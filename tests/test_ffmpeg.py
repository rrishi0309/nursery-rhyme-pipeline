from pathlib import Path

import pytest

from nursery.config import VideoConfig
from nursery.video.ffmpeg import FFmpegError, build_command, run_ffmpeg
from nursery.video.kenburns import SceneClip


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
