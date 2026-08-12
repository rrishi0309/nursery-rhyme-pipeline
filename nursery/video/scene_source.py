"""Normalises heterogeneous scene inputs into one crossfade chain.

A scene is backed either by an LTX clip or by a still with Ken Burns pan/zoom,
and the two can be mixed freely within a video - a scene whose clip failed the
quality gate falls back to its still without affecting its neighbours.

Each source knows how to emit a filter fragment that yields a stream at exactly
`cfg.width x cfg.height @ cfg.fps`. Once every scene is normalised that way the
xfade chain that stitches them is identical regardless of what backed each
scene, which is the whole point of the split.

`kenburns.py` still owns the zoompan expressions; this module owns composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nursery.video.kenburns import _UPSCALE, PanDirection, _zoom_expr


@dataclass(frozen=True)
class StillSource:
    """A scene backed by a still image, animated with Ken Burns pan/zoom."""

    image: Path
    start_s: float
    end_s: float
    direction: PanDirection

    @property
    def path(self) -> Path:
        return self.image

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def input_args(self) -> list[str]:
        # No -loop/-t: a still is one frame and zoompan expands it to `d`
        # frames itself. Looping would make it emit `d` frames per input frame.
        return ["-i", str(self.image)]

    def fragment(self, i: int, cfg, pad: float) -> str:
        frames = max(1, round((self.duration_s + pad) * cfg.fps))
        zoom, x, y = _zoom_expr(self.direction, cfg, frames)
        return (
            f"[{i}:v]"
            f"scale={cfg.width * _UPSCALE}:{cfg.height * _UPSCALE},"
            f"setsar=1,"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}"
            f":s={cfg.width}x{cfg.height}:fps={cfg.fps},"
            f"format=yuv420p"
            f"[v{i}]"
        )


@dataclass(frozen=True)
class ClipSource:
    """A scene backed by a generated video clip."""

    clip: Path
    start_s: float
    end_s: float

    @property
    def path(self) -> Path:
        return self.clip

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def input_args(self) -> list[str]:
        return ["-i", str(self.clip)]

    def fragment(self, i: int, cfg, pad: float) -> str:
        need = self.duration_s + pad
        # LTX renders at 8k+1 frames, so the clip is a little longer than the
        # scene; trim brings it back to the timing forced alignment demands.
        # tpad covers the opposite case - a clip shorter than its scene holds
        # its last frame rather than leaving a gap in the chain.
        return (
            f"[{i}:v]"
            f"scale={cfg.width}:{cfg.height}:flags=lanczos,"
            f"setsar=1,"
            f"fps={cfg.fps},"
            f"tpad=stop_mode=clone:stop_duration={need:.3f},"
            f"trim=duration={need:.3f},"
            f"setpts=PTS-STARTPTS,"
            f"format=yuv420p"
            f"[v{i}]"
        )


SceneSource = StillSource | ClipSource


def build_scene_graph(sources: list[SceneSource], cfg, subtitles: Path | None) -> str:
    if not sources:
        raise ValueError("need at least one scene source to build a filter graph")

    parts: list[str] = []

    for i, src in enumerate(sources):
        # Source 0 is only ever the background (first) input to an xfade, and
        # xfade drops that stream once the first blend ends, so it needs
        # exactly its nominal duration. Every later source is the incoming
        # input of the xfade into it, and keeps being read for crossfade_s
        # past its nominal duration.
        pad = 0.0 if i == 0 else cfg.crossfade_s
        parts.append(src.fragment(i, cfg, pad))

    last = "v0"
    for i in range(1, len(sources)):
        label = f"x{i}"
        # Scene timings are absolute positions from forced alignment, and in a
        # left-folded xfade chain the offset is already absolute output time.
        offset = sources[i].start_s - cfg.crossfade_s
        parts.append(
            f"[{last}][v{i}]"
            f"xfade=transition=fade:duration={cfg.crossfade_s}:offset={offset:.3f}"
            f"[{label}]"
        )
        last = label

    if subtitles is not None:
        parts.append(f"[{last}]subtitles='{subtitles}'[vout]")
    else:
        parts.append(f"[{last}]null[vout]")

    return ";".join(parts)
