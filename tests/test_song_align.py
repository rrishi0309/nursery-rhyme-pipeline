import json
import subprocess
import tomllib
import wave
from pathlib import Path

import pytest

from nursery.providers import align_whisperx, song_acestep
from nursery.providers.align_whisperx import WhisperXAlignProvider
from nursery.providers.song_acestep import AceStepSongProvider, SongError

# ---------------------------------------------------------------------------
# song_acestep: flat-TOML invariant
# ---------------------------------------------------------------------------


def test_dumps_flat_round_trips_through_tomllib_as_a_flat_table():
    config = {
        "save_dir": "/tmp/out",
        "audio_format": "wav",
        "caption": "gentle lullaby, acoustic guitar",
        "lyrics": "twinkle twinkle little star\nhow i wonder what you are",
        "duration": -1.0,
        "instrumental": False,
        "task_type": "text2music",
        "inference_steps": 8,
        "seed": 42,
        "guidance_scale": 7.0,
        "backend": "mlx",
    }
    text = song_acestep._dumps_flat(config)
    parsed = tomllib.loads(text)

    assert parsed == config
    assert all(not isinstance(v, dict) for v in parsed.values())


def test_dumps_flat_rejects_a_nested_section_before_it_can_silently_reset_defaults():
    # This is exactly the bug the module docstring warns about: a
    # dict-valued key would render as a `[section]` header, and cli.py's
    # `setattr(args, key, value)` loop would set one attribute literally
    # named "generation" while every real key silently kept its default.
    with pytest.raises(TypeError, match="section"):
        song_acestep._dumps_flat({"generation": {"seed": 42}})


def test_dumps_flat_escapes_lyrics_containing_quotes_and_newlines():
    tricky = 'she said "hello"\nline two'
    text = song_acestep._dumps_flat({"lyrics": tricky})
    assert tomllib.loads(text) == {"lyrics": tricky}


def test_toml_scalar_renders_bool_before_falling_into_the_int_branch():
    # bool is an int subclass in Python; True/False must not come out as 1/0.
    assert song_acestep._toml_scalar(True) == "true"
    assert song_acestep._toml_scalar(False) == "false"
    assert song_acestep._toml_scalar(8) == "8"
    assert song_acestep._toml_scalar(-1.0) == "-1.0"


# ---------------------------------------------------------------------------
# song_acestep: provider wiring (subprocess mocked, no real ACE-Step)
# ---------------------------------------------------------------------------


def test_generate_writes_flat_toml_runs_with_repo_cwd_and_moves_output(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, cwd, capture_output, text, check):
        calls.append({"cmd": cmd, "cwd": cwd})
        # Simulate ACE-Step writing a UUID-named file into save_dir - the
        # config path is always the last element of the command.
        config_path = Path(cmd[cmd.index("--config") + 1])
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        assert all(not isinstance(v, dict) for v in config.values())
        save_dir = Path(config["save_dir"])
        (save_dir / "3f9c-uuid.wav").write_bytes(b"RIFF....WAVEfmt ")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(song_acestep.subprocess, "run", fake_run)

    provider = AceStepSongProvider(python="/fake/venv/bin/python", repo_dir="/fake/ACE-Step-1.5")
    out = tmp_path / "song.wav"
    result = provider.generate(
        lyrics=["twinkle twinkle little star", "how i wonder what you are"],
        style="gentle lullaby",
        seed=42,
        out=out,
    )

    assert result == out
    assert out.exists()
    assert len(calls) == 1
    assert calls[0]["cmd"][0] == "/fake/venv/bin/python"
    assert calls[0]["cmd"][1] == "cli.py"
    assert calls[0]["cwd"] == "/fake/ACE-Step-1.5"


def test_generate_raises_song_error_with_stderr_tail_on_nonzero_exit(monkeypatch, tmp_path):
    def fake_run(cmd, cwd, capture_output, text, check):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="line1\nline2\nboom: out of memory"
        )

    monkeypatch.setattr(song_acestep.subprocess, "run", fake_run)

    provider = AceStepSongProvider(python="/fake/venv/bin/python", repo_dir="/fake/ACE-Step-1.5")
    with pytest.raises(SongError, match="out of memory"):
        provider.generate(lyrics=["one line"], style="cheerful", seed=1, out=tmp_path / "song.wav")


def test_generate_raises_song_error_when_no_output_file_appears(monkeypatch, tmp_path):
    def fake_run(cmd, cwd, capture_output, text, check):
        # "Succeeds" but writes nothing to save_dir.
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(song_acestep.subprocess, "run", fake_run)

    provider = AceStepSongProvider(python="/fake/venv/bin/python", repo_dir="/fake/ACE-Step-1.5")
    with pytest.raises(SongError, match="wrote no"):
        provider.generate(lyrics=["one line"], style="cheerful", seed=1, out=tmp_path / "song.wav")


# ---------------------------------------------------------------------------
# align_whisperx: word-to-line mapping (pure logic, canned data)
# ---------------------------------------------------------------------------


def _word(text, start, end):
    return {"word": text, "start": start, "end": end}


def test_extract_words_prefers_top_level_word_segments():
    payload = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hi", "words": [_word("wrong", 9, 9.5)]}],
        "word_segments": [_word("Twinkle,", 0.0, 0.5), _word("twinkle!", 0.5, 1.0)],
    }
    words = align_whisperx._extract_words(payload)
    assert [w["norm"] for w in words] == ["twinkle", "twinkle"]
    assert words[0]["start"] == 0.0 and words[0]["end"] == 0.5


def test_extract_words_falls_back_to_per_segment_words_list():
    payload = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hi there", "words": [_word("Hi", 0.0, 0.4)]},
        ]
    }
    words = align_whisperx._extract_words(payload)
    assert [w["norm"] for w in words] == ["hi"]


def test_extract_words_returns_empty_when_no_word_timestamps_present():
    # The plain-`mlx`-backend case the module docstring describes:
    # segment-level text/timing only, no "words" key anywhere.
    payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi there"}]}
    assert align_whisperx._extract_words(payload) == []


def test_map_words_to_lines_matches_in_order_and_tolerates_dropped_words():
    # ASR dropped "little" from line one entirely and misheard nothing else -
    # the mapper must not get stuck waiting for it.
    words = [
        _word("twinkle", 0.0, 0.5),
        _word("twinkle", 0.5, 1.0),
        _word("star", 1.5, 2.0),
        _word("how", 2.2, 2.5),
        _word("i", 2.5, 2.6),
        _word("wonder", 2.6, 3.0),
    ]
    norm_words = align_whisperx._extract_words(
        {"word_segments": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words]}
    )
    lines = ["twinkle twinkle little star", "how i wonder"]
    spans = align_whisperx._map_words_to_lines(norm_words, lines)

    assert spans[0] == (0.0, 2.0)  # min start .. max end of matched tokens
    assert spans[1] == (2.2, 3.0)


def test_map_words_to_lines_returns_none_for_a_line_that_matches_nothing():
    norm_words = align_whisperx._extract_words(
        {"word_segments": [_word("completely", 0.0, 0.5), _word("unrelated", 0.5, 1.0)]}
    )
    spans = align_whisperx._map_words_to_lines(norm_words, ["twinkle twinkle little star"])
    assert spans == [None]


def test_validate_spans_accepts_monotonic_nonzero_spans():
    spans = [(0.0, 1.0), (1.0, 2.5), (2.5, 4.0)]
    assert align_whisperx._validate_spans(spans) == spans


def test_validate_spans_rejects_any_none():
    assert align_whisperx._validate_spans([(0.0, 1.0), None]) is None


def test_validate_spans_rejects_zero_length_span():
    assert align_whisperx._validate_spans([(1.0, 1.0)]) is None


def test_validate_spans_rejects_overlapping_or_backwards_spans():
    assert align_whisperx._validate_spans([(0.0, 2.0), (1.0, 3.0)]) is None


# ---------------------------------------------------------------------------
# align_whisperx: even-split fallback (canned JSON / mocked subprocess)
# ---------------------------------------------------------------------------


def test_even_split_produces_contiguous_monotonic_spans():
    spans = align_whisperx._even_split(10.0, 4)
    assert spans == [(0.0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]


def test_even_split_never_returns_zero_length_spans_for_unknown_duration():
    spans = align_whisperx._even_split(0.0, 3)
    assert all(end > start for start, end in spans)


def test_even_split_handles_zero_lines_without_raising():
    assert align_whisperx._even_split(10.0, 0) == []


def _write_wav(path: Path, duration_s: float, rate: int = 8000) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(duration_s * rate))


def test_align_falls_back_to_even_split_when_whisperx_exits_nonzero(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="segfault")

    monkeypatch.setattr(align_whisperx.subprocess, "run", fake_run)

    audio = tmp_path / "song.wav"
    _write_wav(audio, duration_s=6.0)
    provider = WhisperXAlignProvider(bin="/fake/venv/bin/whisperx")
    spans = provider.align(audio, ["line one", "line two", "line three"], tmp_path / "align_out")

    assert spans == align_whisperx._even_split(6.0, 3)


def test_align_falls_back_to_even_split_when_word_timestamps_are_empty(monkeypatch, tmp_path):
    out_dir = tmp_path / "align_out"

    def fake_run(cmd, capture_output, text, check):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Plain-mlx-backend case: segments with no word-level timing at all.
        payload = {"segments": [{"start": 0.0, "end": 5.0, "text": "twinkle twinkle little star"}]}
        (out_dir / "song.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(align_whisperx.subprocess, "run", fake_run)

    audio = tmp_path / "song.wav"
    _write_wav(audio, duration_s=5.0)
    provider = WhisperXAlignProvider(bin="/fake/venv/bin/whisperx", backend="mlx")
    spans = provider.align(audio, ["line one", "line two"], out_dir)

    assert spans == align_whisperx._even_split(5.0, 2)


def test_align_falls_back_to_even_split_when_a_line_fails_validation(monkeypatch, tmp_path):
    out_dir = tmp_path / "align_out"

    def fake_run(cmd, capture_output, text, check):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Word timestamps exist, but only cover the first line - the second
        # line matches nothing, which must invalidate the whole result.
        payload = {
            "word_segments": [
                _word("twinkle", 0.0, 0.5),
                _word("twinkle", 0.5, 1.0),
                _word("star", 1.5, 2.0),
            ]
        }
        (out_dir / "song.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(align_whisperx.subprocess, "run", fake_run)

    audio = tmp_path / "song.wav"
    _write_wav(audio, duration_s=8.0)
    provider = WhisperXAlignProvider(bin="/fake/venv/bin/whisperx")
    spans = provider.align(audio, ["twinkle twinkle star", "completely unrelated line"], out_dir)

    assert spans == align_whisperx._even_split(8.0, 2)


def test_align_uses_real_word_timings_when_alignment_succeeds(monkeypatch, tmp_path):
    out_dir = tmp_path / "align_out"

    def fake_run(cmd, capture_output, text, check):
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "word_segments": [
                _word("twinkle", 0.0, 0.5),
                _word("twinkle", 0.5, 1.0),
                _word("little", 1.0, 1.4),
                _word("star", 1.5, 2.0),
                _word("how", 2.2, 2.5),
                _word("i", 2.5, 2.6),
                _word("wonder", 2.6, 3.0),
            ]
        }
        (out_dir / "song.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(align_whisperx.subprocess, "run", fake_run)

    audio = tmp_path / "song.wav"
    _write_wav(audio, duration_s=3.0)
    provider = WhisperXAlignProvider(bin="/fake/venv/bin/whisperx")
    spans = provider.align(audio, ["twinkle twinkle little star", "how i wonder"], out_dir)

    assert spans == [(0.0, 2.0), (2.2, 3.0)]


def test_align_returns_empty_list_for_no_lines_without_raising(tmp_path):
    provider = WhisperXAlignProvider(bin="/fake/venv/bin/whisperx")
    assert provider.align(tmp_path / "song.wav", [], tmp_path / "align_out") == []


def test_align_never_raises_when_binary_is_missing(tmp_path):
    audio = tmp_path / "song.wav"
    _write_wav(audio, duration_s=4.0)
    provider = WhisperXAlignProvider(bin="/definitely/not/a/real/binary")
    spans = provider.align(audio, ["line one", "line two"], tmp_path / "align_out")
    assert spans == align_whisperx._even_split(4.0, 2)


# ---------------------------------------------------------------------------
# align_whisperx: audio duration probing
# ---------------------------------------------------------------------------


def test_audio_duration_s_reads_a_real_wav_file(tmp_path):
    audio = tmp_path / "song.wav"
    _write_wav(audio, duration_s=2.5, rate=8000)
    assert align_whisperx._audio_duration_s(audio) == pytest.approx(2.5)


def test_audio_duration_s_returns_zero_for_missing_file():
    assert align_whisperx._audio_duration_s(Path("/no/such/file.wav")) == 0.0
