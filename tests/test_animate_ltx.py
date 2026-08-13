import subprocess
from pathlib import Path

import pytest

from nursery.providers import animate_ltx
from nursery.providers.animate_ltx import (
    FRAME_QUANTUM,
    SIZE_QUANTUM,
    LTXAnimateProvider,
    LTXUnavailableError,
    frames_for,
    snap_size,
)


def _fake_binary(monkeypatch, path: str = "/fake/bin/ltx-2-mlx"):
    monkeypatch.setattr(animate_ltx.shutil, "which", lambda name: path)


def _capture_run(monkeypatch) -> list[list[str]]:
    """Records commands and, like a real successful ltx-2-mlx run, writes the
    requested --output path so the provider's post-run existence check passes.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(cmd)
        out_path = cmd[cmd.index("--output") + 1]
        Path(out_path).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(animate_ltx.subprocess, "run", fake_run)
    return calls


# --- frames_for / snap_size: unchanged behaviour, still covered ---


def test_frames_for_rounds_up_to_8k_plus_1():
    assert frames_for(4.0, 24) == 97  # ceil(96/8)=12 -> 12*8+1=97
    assert frames_for(0.1, 24) == 9  # need=3 -> k=1 -> 9


def test_frames_for_is_minimum_9():
    assert frames_for(0.0, 24) == 9


def test_snap_size_rounds_up_to_64():
    assert snap_size(704) == 704
    assert snap_size(700) == 704
    assert snap_size(1) == SIZE_QUANTUM


def test_quantum_constants_unchanged():
    assert FRAME_QUANTUM == 8
    assert SIZE_QUANTUM == 64


# --- command construction against ltx-2-mlx generate ---


def test_binary_missing_raises_ltx_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(animate_ltx.shutil, "which", lambda name: None)
    provider = LTXAnimateProvider(bin="/nonexistent/ltx-2-mlx")
    with pytest.raises(LTXUnavailableError):
        provider.animate(
            image=tmp_path / "still.png",
            prompt="a girl waves",
            duration_s=4.0,
            seed=1,
            out=tmp_path / "clip.mp4",
        )


def test_command_shape_has_required_flags(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)
    calls = _capture_run(monkeypatch)

    provider = LTXAnimateProvider(
        bin="/fake/bin/ltx-2-mlx",
        model_repo="dgrauet/ltx-2.3-mlx-q4",
        width=704,
        height=480,
        frame_rate=24,
        steps=8,
        two_stage=False,
        low_ram=True,
    )
    still = tmp_path / "still.png"
    out = tmp_path / "clip.mp4"
    provider.animate(image=still, prompt="a girl waves", duration_s=4.0, seed=42, out=out)

    assert len(calls) == 1
    cmd = calls[0]

    assert cmd[0] == "/fake/bin/ltx-2-mlx"
    assert cmd[1] == "generate"

    assert "--prompt" in cmd
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "a girl waves" in prompt
    assert animate_ltx.MOTION_SUFFIX in prompt

    assert "--image" in cmd and cmd[cmd.index("--image") + 1] == str(still)
    assert "--output" in cmd and cmd[cmd.index("--output") + 1] == str(out)

    assert "--frames" in cmd
    assert cmd[cmd.index("--frames") + 1] == str(frames_for(4.0, 24))

    assert "--width" in cmd and cmd[cmd.index("--width") + 1] == "704"
    # ltx-2-mlx's own default height (480) is not itself a multiple of 64, so
    # snap_size rounds it up to 512 - this asserts that snapping still fires.
    assert "--height" in cmd and cmd[cmd.index("--height") + 1] == "512"

    # --frame-rate is mandatory on ltx-2-mlx generate (no tool default).
    assert "--frame-rate" in cmd
    assert cmd[cmd.index("--frame-rate") + 1] == "24"

    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "42"
    assert "--steps" in cmd and cmd[cmd.index("--steps") + 1] == "8"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "dgrauet/ltx-2.3-mlx-q4"

    # low_ram=True, two_stage=False in this construction.
    assert "--low-ram" in cmd
    assert "--two-stage" not in cmd

    # No audio-skip flag exists on ltx-2-mlx generate - must never be invented.
    assert "--no-audio" not in cmd
    assert "--an" not in cmd


def test_two_stage_and_low_ram_flags_are_conditional(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)
    calls = _capture_run(monkeypatch)

    provider = LTXAnimateProvider(bin="/fake/bin/ltx-2-mlx", two_stage=True, low_ram=False)
    provider.animate(
        image=tmp_path / "still.png",
        prompt="p",
        duration_s=2.0,
        seed=1,
        out=tmp_path / "clip.mp4",
    )

    cmd = calls[0]
    assert "--two-stage" in cmd
    assert "--low-ram" not in cmd


def test_width_and_height_are_snapped_to_64(monkeypatch):
    _fake_binary(monkeypatch)
    provider = LTXAnimateProvider(bin="/fake/bin/ltx-2-mlx", width=700, height=500)
    assert provider.width == 704
    assert provider.height == 512


def test_nonzero_returncode_raises_runtime_error(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="ltx blew up")

    monkeypatch.setattr(animate_ltx.subprocess, "run", fake_run)

    provider = LTXAnimateProvider(bin="/fake/bin/ltx-2-mlx")
    with pytest.raises(RuntimeError, match="ltx blew up"):
        provider.animate(
            image=tmp_path / "still.png",
            prompt="p",
            duration_s=2.0,
            seed=1,
            out=tmp_path / "clip.mp4",
        )


def test_missing_output_file_raises_runtime_error(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)

    def fake_run(cmd, capture_output, text, check):
        # Reports success but (unlike a real run) writes nothing to --output.
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(animate_ltx.subprocess, "run", fake_run)

    provider = LTXAnimateProvider(bin="/fake/bin/ltx-2-mlx")
    out = tmp_path / "clip.mp4"
    with pytest.raises(RuntimeError, match="wrote no file"):
        provider.animate(
            image=tmp_path / "still.png", prompt="p", duration_s=2.0, seed=1, out=out
        )
