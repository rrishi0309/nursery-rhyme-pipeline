"""Thin wrapper around the ffmpeg binary."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from nursery.video.kenburns import SceneClip

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    def __init__(self, message: str, stderr: str):
        super().__init__(message)
        self.stderr = stderr


def _binary() -> str:
    found = shutil.which("ffmpeg")
    if found is None:
        raise FFmpegError("ffmpeg not found on PATH", "")
    return found


def build_command(
    clips: list[SceneClip], audio: Path, graph: str, out: Path, cfg
) -> list[str]:
    cmd = [_binary(), "-y"]

    for clip in clips:
        cmd += ["-loop", "1", "-t", f"{clip.duration_s:.3f}", "-i", str(clip.image)]

    cmd += ["-i", str(audio)]
    cmd += [
        "-filter_complex", graph,
        "-map", "[vout]",
        "-map", f"{len(clips)}:a",
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
    return cmd


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
        "-frames:v", "1", "-q:v", "2", str(out),
    ])
    return out
