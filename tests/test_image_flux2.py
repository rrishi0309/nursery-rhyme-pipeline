import subprocess

import pytest

from nursery.providers import image_flux2
from nursery.providers.image_flux2 import Flux2ImageProvider, Flux2UnavailableError


def _fake_binary(monkeypatch):
    monkeypatch.setattr(image_flux2.shutil, "which", lambda name: "/fake/bin/mflux-generate-flux2")


def _capture_run(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(image_flux2.subprocess, "run", fake_run)
    return calls


def test_binary_missing_raises_flux2_unavailable(monkeypatch):
    monkeypatch.setattr(image_flux2.shutil, "which", lambda name: None)
    with pytest.raises(Flux2UnavailableError):
        image_flux2._binary()


def test_generate_with_reference_passes_low_strength_image_flag(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)
    calls = _capture_run(monkeypatch)

    provider = Flux2ImageProvider(cast_strength=0.35, use_reference=True)
    ref = tmp_path / "cast.png"
    out = tmp_path / "scene_00.png"
    provider.generate(prompt="a girl waves", reference=ref, seed=7, out=out)

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "/fake/bin/mflux-generate-flux2"
    assert "--image" in cmd
    img_i = cmd.index("--image")
    assert cmd[img_i + 1] == str(ref)
    assert cmd[img_i + 2] == "0.35"
    assert "--prompt" in cmd and "a girl waves" in cmd
    assert "--seed" in cmd and "7" in cmd
    assert "--output" in cmd and str(out) in cmd
    assert "--negative-prompt" in cmd


def test_generate_without_reference_omits_image_flag(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)
    calls = _capture_run(monkeypatch)

    provider = Flux2ImageProvider(use_reference=True)
    out = tmp_path / "scene_01.png"
    provider.generate(prompt="a boy jumps", reference=None, seed=3, out=out)

    assert "--image" not in calls[0]


def test_use_reference_false_omits_image_flag_even_with_reference(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)
    calls = _capture_run(monkeypatch)

    provider = Flux2ImageProvider(use_reference=False)
    ref = tmp_path / "cast.png"
    out = tmp_path / "scene_02.png"
    provider.generate(prompt="a dog runs", reference=ref, seed=1, out=out)

    assert "--image" not in calls[0]


def test_cast_sheet_builds_prompt_from_names_and_descriptions(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)
    calls = _capture_run(monkeypatch)

    provider = Flux2ImageProvider()
    cast = [
        {"name": "Luna", "description": "a small girl in a blue coat"},
        {"name": "Bramble", "description": "a shaggy brown dog"},
    ]
    out = tmp_path / "cast.png"
    provider.cast_sheet(cast=cast, style="storybook watercolor", seed=99, out=out)

    cmd = calls[0]
    prompt = cmd[cmd.index("--prompt") + 1]
    assert "Luna" in prompt
    assert "a small girl in a blue coat" in prompt
    assert "Bramble" in prompt
    assert "a shaggy brown dog" in prompt
    assert "storybook watercolor" in prompt
    assert "--output" in cmd and str(out) in cmd
    assert "--seed" in cmd and "99" in cmd


def test_nonzero_returncode_raises_runtime_error(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(image_flux2.subprocess, "run", fake_run)

    provider = Flux2ImageProvider()
    out = tmp_path / "scene_03.png"
    with pytest.raises(RuntimeError, match="boom"):
        provider.generate(prompt="x", reference=None, seed=1, out=out)


def test_generate_creates_output_parent_dir(tmp_path, monkeypatch):
    _fake_binary(monkeypatch)
    _capture_run(monkeypatch)

    provider = Flux2ImageProvider()
    out = tmp_path / "nested" / "scene_04.png"
    provider.generate(prompt="x", reference=None, seed=1, out=out)

    assert out.parent.exists()
