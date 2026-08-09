import pytest

from nursery.providers.base import WordTiming
from nursery.video.captions import (
    CaptionLine,
    build_ass,
    group_words_into_lines,
    write_ass,
)


def timings(*pairs) -> list[WordTiming]:
    return [WordTiming(word=w, start_s=s, end_s=e) for w, s, e in pairs]


def test_groups_words_to_matching_lyric_lines():
    words = timings(("twinkle", 0.0, 0.5), ("little", 0.5, 1.0), ("star", 1.0, 1.5))

    lines = group_words_into_lines(words, ["twinkle little", "star"])

    assert [w.word for w in lines[0].words] == ["twinkle", "little"]
    assert [w.word for w in lines[1].words] == ["star"]


def test_line_span_comes_from_its_words():
    line = CaptionLine(words=timings(("a", 1.0, 1.5), ("b", 1.5, 2.25)))

    assert line.start_s == 1.0
    assert line.end_s == 2.25


def test_mismatched_word_count_raises():
    with pytest.raises(ValueError, match="word count"):
        group_words_into_lines(timings(("a", 0.0, 1.0)), ["a b c"])


def test_ass_has_required_sections():
    ass = build_ass([CaptionLine(words=timings(("hi", 0.0, 1.0)))], 1920, 1080)

    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass


def test_ass_declares_the_video_resolution():
    ass = build_ass([CaptionLine(words=timings(("hi", 0.0, 1.0)))], 1920, 1080)

    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass


def test_karaoke_durations_are_centiseconds():
    # 0.75s -> \k75
    ass = build_ass([CaptionLine(words=timings(("hi", 0.0, 0.75)))], 1920, 1080)

    assert r"{\k75}hi" in ass


def test_dialogue_timestamps_are_ass_formatted():
    ass = build_ass([CaptionLine(words=timings(("hi", 61.5, 62.0)))], 1920, 1080)

    assert "0:01:01.50" in ass


def test_each_line_becomes_one_dialogue_event():
    lines = [
        CaptionLine(words=timings(("a", 0.0, 1.0))),
        CaptionLine(words=timings(("b", 1.0, 2.0))),
    ]

    assert build_ass(lines, 1920, 1080).count("Dialogue:") == 2


def test_write_ass_creates_utf8_file(tmp_path):
    out = write_ass([CaptionLine(words=timings(("hi", 0.0, 1.0)))], 1920, 1080,
                    tmp_path / "c.ass")

    assert out.exists()
    assert "[Events]" in out.read_text(encoding="utf-8")
