"""Builds the ffmpeg filter graph that turns stills into a moving video.

zoompan quantises its zoom factor per frame, which makes a 1080p source visibly
judder. Upscaling the source first pushes the quantisation below one output
pixel, which is the standard fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PanDirection = Literal["in", "out", "left", "right"]

_CYCLE: tuple[PanDirection, ...] = ("in", "left", "out", "right")
_UPSCALE = 4


@dataclass(frozen=True)
class SceneClip:
    image: Path
    duration_s: float
    direction: PanDirection


def direction_for_index(i: int) -> PanDirection:
    return _CYCLE[i % len(_CYCLE)]


def _zoom_expr(direction: PanDirection, cfg, frames: int) -> tuple[str, str, str]:
    """Return (zoom, x, y) expressions for the given move."""
    zmax = cfg.max_zoom
    rate = cfg.zoom_rate
    centre_x = "iw/2-(iw/zoom/2)"
    centre_y = "ih/2-(ih/zoom/2)"

    if direction == "in":
        return f"min(zoom+{rate},{zmax})", centre_x, centre_y
    if direction == "out":
        return f"max({zmax}-on*{rate},1.0)", centre_x, centre_y
    if direction == "left":
        return f"{zmax}", f"(iw-iw/zoom)*(1-on/{frames})", centre_y
    return f"{zmax}", f"(iw-iw/zoom)*(on/{frames})", centre_y


def build_filter_graph(clips: list[SceneClip], cfg, subtitles: Path | None) -> str:
    if not clips:
        raise ValueError("need at least one clip to build a filter graph")

    parts: list[str] = []

    for i, clip in enumerate(clips):
        frames = max(1, round(clip.duration_s * cfg.fps))
        zoom, x, y = _zoom_expr(clip.direction, cfg, frames)
        parts.append(
            f"[{i}:v]"
            f"scale={cfg.width * _UPSCALE}:{cfg.height * _UPSCALE},"
            f"setsar=1,"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}"
            f":s={cfg.width}x{cfg.height}:fps={cfg.fps},"
            f"format=yuv420p"
            f"[v{i}]"
        )

    last = "v0"
    offset = clips[0].duration_s - cfg.crossfade_s
    for i in range(1, len(clips)):
        label = f"x{i}"
        parts.append(
            f"[{last}][v{i}]"
            f"xfade=transition=fade:duration={cfg.crossfade_s}:offset={offset:.3f}"
            f"[{label}]"
        )
        last = label
        offset += clips[i].duration_s - cfg.crossfade_s

    if subtitles is not None:
        parts.append(f"[{last}]subtitles='{subtitles}'[vout]")
    else:
        parts.append(f"[{last}]null[vout]")

    return ";".join(parts)
