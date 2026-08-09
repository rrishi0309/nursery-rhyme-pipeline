r"""Generates ASS subtitles with karaoke (\k) timing from word timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nursery.providers.base import WordTiming

_STYLE = (
    "Style: Karaoke,Arial Rounded MT Bold,84,"
    # PrimaryColour is the *sung* colour (what \k fills a word to) and
    # SecondaryColour is the *unsung* colour - so the highlight amber goes
    # first, plain white second, matching the usual sing-along convention of
    # words lighting up amber as they are sung.
    "&H0000D7FF,&H00FFFFFF,&H00202020,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,6,2,2,120,120,90,1"
)


@dataclass
class CaptionLine:
    words: list[WordTiming]

    @property
    def start_s(self) -> float:
        return self.words[0].start_s

    @property
    def end_s(self) -> float:
        return self.words[-1].end_s

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def group_words_into_lines(
    timings: list[WordTiming], lyric_lines: list[str]
) -> list[CaptionLine]:
    expected = sum(len(line.split()) for line in lyric_lines)
    if expected != len(timings):
        raise ValueError(
            f"word count mismatch: lyrics have {expected}, alignment has {len(timings)}"
        )

    lines: list[CaptionLine] = []
    cursor = 0
    for line in lyric_lines:
        n = len(line.split())
        lines.append(CaptionLine(words=timings[cursor : cursor + n]))
        cursor += n
    return lines


def _ass_time(seconds: float) -> str:
    hours, rem = divmod(max(0.0, seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


def build_ass(lines: list[CaptionLine], width: int, height: int) -> str:
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"
        ),
        _STYLE,
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    events = []
    for line in lines:
        karaoke = "".join(
            f"{{\\k{round((w.end_s - w.start_s) * 100)}}}{w.word} " for w in line.words
        ).strip()
        events.append(
            f"Dialogue: 0,{_ass_time(line.start_s)},{_ass_time(line.end_s)},"
            f"Karaoke,,0,0,0,,{karaoke}"
        )

    return "\n".join(header + events) + "\n"


def write_ass(lines: list[CaptionLine], width: int, height: int, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_ass(lines, width, height), encoding="utf-8")
    return out
