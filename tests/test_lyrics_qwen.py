import json
import subprocess

import pytest

from nursery.providers import lyrics_qwen
from nursery.providers.lyrics_qwen import LyricsError, LyricsProvider

VALID_JSON = json.dumps(
    {
        "title": "The Shining Star",
        "style": "gentle lullaby, acoustic guitar",
        "cast": [{"name": "Luna", "description": "a small girl in a blue coat"}],
        "lines": [
            {"text": "look up and see the shining star", "visual": "a girl points at the sky"},
            {"text": "she wonders just how bright by far", "visual": "the girl smiles"},
        ],
    }
)

# Line 2 doesn't rhyme with line 1's "ar" key, so validate.check will reject
# this on the first attempt.
INVALID_JSON = json.dumps(
    {
        "title": "The Shining Star",
        "style": "gentle lullaby",
        "cast": [],
        "lines": [
            {"text": "look up and see the shining star", "visual": "a girl points at the sky"},
            {"text": "she wonders how the day will end", "visual": "the girl smiles"},
        ],
    }
)


def test_generate_retries_after_a_validation_failure_then_succeeds(monkeypatch):
    """One monkeypatched subprocess.run exercises both the retry path (first
    attempt returns lyrics that fail rhyme validation) and the success path
    (second attempt returns valid lyrics), per the brief's request for a
    single provider test.
    """
    monkeypatch.setattr(lyrics_qwen.shutil, "which", lambda name: "/fake/bin/mlx_lm.generate")

    calls: list[list[str]] = []
    responses = [INVALID_JSON, VALID_JSON]

    def fake_run(cmd, capture_output, text, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=responses.pop(0), stderr="")

    monkeypatch.setattr(lyrics_qwen.subprocess, "run", fake_run)

    provider = LyricsProvider(model_repo="mlx-community/Qwen3.5-4B-MLX-4bit", max_retries=3)
    result = provider.generate(
        slug_hint="a star at night", target_lines=2, syllables=8, scheme="AA", seed=42
    )

    assert result["title"] == "The Shining Star"
    assert len(result["lines"]) == 2
    assert len(calls) == 2

    # The retry prompt must carry the previous attempt's violation forward.
    first_cmd, second_cmd = calls
    assert "--model" in first_cmd and "mlx-community/Qwen3.5-4B-MLX-4bit" in first_cmd
    second_prompt = second_cmd[second_cmd.index("--prompt") + 1]
    assert "previous attempt was invalid" in second_prompt


def test_generate_raises_lyrics_error_when_retries_exhausted(monkeypatch):
    monkeypatch.setattr(lyrics_qwen.shutil, "which", lambda name: "/fake/bin/mlx_lm.generate")

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=INVALID_JSON, stderr="")

    monkeypatch.setattr(lyrics_qwen.subprocess, "run", fake_run)

    provider = LyricsProvider(max_retries=2)
    with pytest.raises(LyricsError):
        provider.generate(
            slug_hint="a star at night", target_lines=2, syllables=8, scheme="AA", seed=42
        )
