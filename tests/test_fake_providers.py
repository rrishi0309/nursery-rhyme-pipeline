import wave
from itertools import pairwise
from pathlib import Path

from nursery.providers.fake import (
    FakeAlignProvider,
    FakeImageProvider,
    FakeSongProvider,
)


def test_fake_image_writes_a_real_png(tmp_path: Path):
    out = tmp_path / "scene.png"
    FakeImageProvider().generate("a bunny", [], seed=7, out=out)

    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_fake_image_is_deterministic_for_a_seed(tmp_path: Path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    FakeImageProvider().generate("x", [], seed=3, out=a)
    FakeImageProvider().generate("x", [], seed=3, out=b)

    assert a.read_bytes() == b.read_bytes()


def test_fake_image_differs_across_seeds(tmp_path: Path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    FakeImageProvider().generate("x", [], seed=1, out=a)
    FakeImageProvider().generate("x", [], seed=2, out=b)

    assert a.read_bytes() != b.read_bytes()


def test_fake_song_writes_a_readable_wav(tmp_path: Path):
    out = tmp_path / "mix.wav"
    FakeSongProvider(duration_s=2.0).render("la la", "cheerful", out)

    with wave.open(str(out)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 44100
        assert abs(wf.getnframes() / 44100 - 2.0) < 0.01


def test_fake_align_spreads_words_across_duration():
    timings = FakeAlignProvider(duration_s=4.0).align(Path("ignored.wav"), "one two three four")

    assert [t.word for t in timings] == ["one", "two", "three", "four"]
    assert timings[0].start_s == 0.0
    assert abs(timings[-1].end_s - 4.0) < 1e-6


def test_fake_align_timings_are_contiguous_and_increasing():
    timings = FakeAlignProvider(duration_s=3.0).align(Path("x.wav"), "a b c")

    for earlier, later in pairwise(timings):
        assert earlier.end_s == later.start_s
        assert later.start_s < later.end_s
