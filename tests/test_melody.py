import wave

import pytest

from nursery.audio.melody import (
    melody_duration_s,
    note_to_freq,
    parse_sequence,
    render_melody,
)


def test_a4_is_concert_pitch():
    assert note_to_freq("A4") == pytest.approx(440.0)


def test_octave_up_doubles_frequency():
    assert note_to_freq("A5") == pytest.approx(2 * note_to_freq("A4"))


def test_middle_c_is_correct():
    assert note_to_freq("C4") == pytest.approx(261.626, abs=0.01)


def test_sharps_and_flats_are_enharmonic():
    assert note_to_freq("C#4") == pytest.approx(note_to_freq("Db4"))


def test_unknown_note_name_raises():
    with pytest.raises(ValueError, match="unknown note name"):
        note_to_freq("H4")


def test_tempo_controls_note_length():
    at_60 = parse_sequence(["C4:1"], tempo_bpm=60)[0][1]
    at_120 = parse_sequence(["C4:1"], tempo_bpm=120)[0][1]

    assert at_60 == pytest.approx(1.0)
    assert at_120 == pytest.approx(0.5)


def test_rest_has_no_frequency():
    freq, seconds = parse_sequence(["r:2"], tempo_bpm=60)[0]

    assert freq is None
    assert seconds == pytest.approx(2.0)


def test_duration_sums_all_beats():
    # 1 + 1 + 2 beats at 60bpm = 4 seconds
    assert melody_duration_s(["C4:1", "D4:1", "E4:2"], tempo_bpm=60) == pytest.approx(4.0)


def test_render_writes_wav_of_expected_length(tmp_path):
    out = render_melody(["C4:1", "G4:1"], tempo_bpm=60, out=tmp_path / "m.wav")

    with wave.open(str(out)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 44100
        assert wf.getnframes() / 44100 == pytest.approx(2.0, abs=0.01)


def test_rendered_audio_is_not_silence(tmp_path):
    out = render_melody(["C4:1"], tempo_bpm=60, out=tmp_path / "m.wav")

    with wave.open(str(out)) as wf:
        frames = wf.readframes(wf.getnframes())

    assert max(frames) > 0
