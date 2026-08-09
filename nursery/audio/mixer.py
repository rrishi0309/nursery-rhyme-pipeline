"""Mixes narration over a melody bed into the final track.

Each lyric line is placed at the start of its musical phrase, so the narration
lands on the tune rather than drifting against it. The melody keeps playing
underneath and fills the tail of each phrase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nursery.video.ffmpeg import _binary, run_ffmpeg


@dataclass(frozen=True)
class NarrationClip:
    path: Path
    start_s: float


def mix_narration_over_bed(
    bed: Path,
    clips: list[NarrationClip],
    out: Path,
    bed_gain: float = 0.42,
    narration_gain: float = 1.6,
) -> Path:
    """Overlay narration clips onto the melody bed at their start times.

    The bed is ducked well under the voice so the words stay intelligible,
    which matters more than the music in a sing-along.
    """
    if not clips:
        raise ValueError("need at least one narration clip")

    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [_binary(), "-y", "-v", "error", "-i", str(bed)]
    for clip in clips:
        cmd += ["-i", str(clip.path)]

    parts = [f"[0:a]volume={bed_gain}[bed]"]
    labels = ["bed"]
    for i, clip in enumerate(clips, start=1):
        delay_ms = round(clip.start_s * 1000)
        parts.append(
            f"[{i}:a]adelay={delay_ms}|{delay_ms},volume={narration_gain}[n{i}]"
        )
        labels.append(f"n{i}")

    joined = "".join(f"[{label}]" for label in labels)
    parts.append(
        f"{joined}amix=inputs={len(labels)}:duration=longest:normalize=0,"
        # level=disabled: alimiter defaults to auto-normalising output back up
        # near 0dB regardless of input gain, which is what was pushing every
        # mix to the ceiling. limit=0.891 (~-1dBTP) leaves headroom for the
        # inter-sample overshoot that lossy AAC encoding commonly introduces,
        # instead of relying on the encoder not to clip.
        f"alimiter=limit=0.891:level=disabled,aresample=44100[out]"
    )

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[out]",
        "-ac", "1", "-ar", "44100",
        str(out),
    ]
    run_ffmpeg(cmd)
    return out
