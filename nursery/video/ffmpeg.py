"""Thin wrapper around the ffmpeg binary.

The `ffmpeg` on a plain `brew install ffmpeg` is commonly built without
libass, which means it has no `subtitles` filter - caption burn-in fails with
"No such filter: 'subtitles'". `_binary()` prefers a libass-capable build
(`ffmpeg-full`) when one is available, so captions work by default without
the caller having to know about the distinction.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from nursery.video.kenburns import SceneClip

log = logging.getLogger(__name__)

_FFMPEG_FULL = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")

# Capability probes are slow (they spawn ffmpeg) and their answer never
# changes for a given binary within a process, so cache by binary path.
_CAPABILITY_CACHE: dict[str, bool] = {}


class FFmpegError(RuntimeError):
    def __init__(self, message: str, stderr: str):
        super().__init__(message)
        self.stderr = stderr


def has_subtitles_filter(binary: str) -> bool:
    """Whether `binary` reports a `subtitles` entry in `-filters` (libass)."""
    if binary in _CAPABILITY_CACHE:
        return _CAPABILITY_CACHE[binary]

    try:
        proc = subprocess.run(
            [binary, "-hide_banner", "-filters"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        result = False
    else:
        result = any(
            len(fields := line.split()) > 1 and fields[1] == "subtitles"
            for line in proc.stdout.splitlines()
        )

    _CAPABILITY_CACHE[binary] = result
    return result


def _binary() -> str:
    """Resolve an ffmpeg binary: env var, then ffmpeg-full, then PATH."""
    env = os.environ.get("NURSERY_FFMPEG")
    if env:
        return env

    if _FFMPEG_FULL.exists() and has_subtitles_filter(str(_FFMPEG_FULL)):
        return str(_FFMPEG_FULL)

    found = shutil.which("ffmpeg")
    if found is None:
        raise FFmpegError(
            "ffmpeg not found. Install it (e.g. `brew install ffmpeg-full` for "
            "a build with caption support) or set NURSERY_FFMPEG to a binary.",
            "",
        )
    return found


def _check_subtitles(binary: str, graph: str) -> None:
    if "subtitles=" in graph and not has_subtitles_filter(binary):
        raise FFmpegError(
            f"resolved ffmpeg ({binary}) has no 'subtitles' filter (no libass), "
            "so captions cannot be burned in. Install a libass-enabled build "
            "with `brew install ffmpeg-full`, or set NURSERY_FFMPEG to point "
            "at one.",
            "",
        )


def build_source_command(
    sources, audio: Path, graph: str, out: Path, cfg
) -> list[str]:
    """Encode command for a mixed set of scene sources (clips and stills).

    Each source supplies its own input arguments, since a generated clip is
    read as a normal video stream while a still relies on zoompan to expand
    its single frame.
    """
    binary = _binary()
    _check_subtitles(binary, graph)

    cmd = [binary, "-y"]
    for src in sources:
        cmd += src.input_args()

    cmd += ["-i", str(audio)]
    cmd += _encode_args(len(sources), graph, out, cfg)
    return cmd


def build_command(
    clips: list[SceneClip], audio: Path, graph: str, out: Path, cfg
) -> list[str]:
    binary = _binary()
    _check_subtitles(binary, graph)
    cmd = [binary, "-y"]

    for clip in clips:
        # A still image is already exactly one frame; zoompan expands it to
        # `d` frames on its own. `-loop 1 -t <dur>` would instead hand
        # zoompan a full stream of duplicated frames, so it would emit `d`
        # frames *per input frame* it receives rather than `d` frames total.
        cmd += ["-i", str(clip.image)]

    cmd += ["-i", str(audio)]
    cmd += _encode_args(len(clips), graph, out, cfg)
    return cmd


def _encode_args(n_inputs: int, graph: str, out: Path, cfg) -> list[str]:
    return [
        "-filter_complex", graph,
        "-map", "[vout]",
        "-map", f"{n_inputs}:a",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(cfg.fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(out),
    ]


def run_ffmpeg(cmd: list[str]) -> None:
    log.debug("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
        raise FFmpegError(f"ffmpeg exited {proc.returncode}", tail)


def extract_thumbnail(video: Path, at_s: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        _binary(), "-y", "-ss", f"{at_s:.3f}", "-i", str(video),
        "-frames:v", "1", "-update", "1", "-q:v", "2", str(out),
    ])
    return out
