"""Cheap clip quality checks, measured with ffmpeg rather than a model.

LTX image-to-video fails in two recognisable ways, and both show up as SSIM
against a reference frame:

- It ignores the conditioning image and invents something else. The clip's
  first frame then diverges from the source still.
- It produces a dead clip (no motion) or a morphing one (the subject drifts
  into mush). Both show up as first-frame vs last-frame similarity, at
  opposite ends: too high means nothing moved, too low means it fell apart.

Neither check is clever. They are here to catch the obvious failures cheaply
enough to run on every clip, so a bad scene falls back to Ken Burns instead of
reaching the final render.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from nursery.video.ffmpeg import _binary, run_ffmpeg

log = logging.getLogger(__name__)

_ALL = re.compile(r"All:\s*([0-9.]+)")


def extract_frame(video: Path, at_s: float, out: Path) -> Path:
    """Write a single frame of `video` to `out` as PNG."""
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        _binary(), "-y", "-ss", f"{at_s:.3f}", "-i", str(video),
        "-frames:v", "1", "-update", "1", str(out),
    ])
    return out


def ssim(a: Path, b: Path, width: int = 512, height: int = 288) -> float:
    """Structural similarity of two images, both scaled to a common size.

    Scaling down is deliberate: the comparison is against a differently sized
    source still, and fine detail is noise for the question being asked.
    """
    graph = (
        f"[0:v]scale={width}:{height},setsar=1,format=gray[a];"
        f"[1:v]scale={width}:{height},setsar=1,format=gray[b];"
        f"[a][b]ssim=stats_file=-"
    )
    proc = subprocess.run(
        [_binary(), "-hide_banner", "-i", str(a), "-i", str(b),
         "-lavfi", graph, "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        log.warning("ssim failed for %s vs %s; treating as unknown", a.name, b.name)
        return float("nan")

    matches = _ALL.findall(proc.stdout + proc.stderr)
    if not matches:
        log.warning("ssim produced no score for %s vs %s", a.name, b.name)
        return float("nan")
    return float(matches[-1])
