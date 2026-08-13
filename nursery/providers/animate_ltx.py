"""Scene animation via LTX-2.3 image-to-video (MLX-native, Apple Silicon).

Runs image-to-video rather than text-to-video on purpose. The `images` stage
already produces a canonical still per scene, seeded and steered by
`cast.yaml`, and conditioning LTX on that still is what keeps a character
recognisable across ~20 scenes. Text-to-video would re-roll the character
every scene.

Rendering shells out to the `ltx-2-mlx generate` CLI. That tool is not
installed into this project's own venv - it lives in its own `uv`-managed
venv under `~/.cache/nursery-tools/ltx-2-mlx/.venv/` (see
`docs/tool-help/SUMMARY.md` section 5), so `self.bin` is a full path to that
venv's `ltx-2-mlx` console script, not a bare command looked up on the
project's PATH.

Two hard constraints from LTX, both encoded here:

- Frame count must be `8k+1`. Scene durations are not, so `frames_for` rounds
  *up* to the next valid count and the assemble stage trims back down to the
  exact scene duration. Generating long and trimming is what keeps scene
  timings locked to the audio alignment.
- Width and height must be divisible by 64. `ltx-2-mlx`'s own defaults are
  704x480; the assemble stage upscales clips to match `VideoConfig`'s
  1920x1080 output canvas.

`--frame-rate` is mandatory on `ltx-2-mlx generate` - it has no tool default,
and omitting it is an immediate argparse error. LTX-2.3 was trained at 24fps,
so `frames_for` is driven by `self.frame_rate`, not the pipeline's overall
video fps; the assemble stage reconciles the two.

There is no audio-skip flag anywhere on `generate` - every clip it writes has
a baked-in stereo audio track. That is stripped downstream with ffmpeg `-an`
in the assemble stage; this provider does not attempt to suppress it.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

FRAME_QUANTUM = 8
SIZE_QUANTUM = 64

MOTION_SUFFIX = (
    "gentle storybook motion, subtle camera drift, characters stay on model, "
    "consistent art style, no scene change"
)


class LTXUnavailableError(RuntimeError):
    pass


def frames_for(duration_s: float, fps: int) -> int:
    """Smallest valid LTX frame count (8k+1) covering `duration_s`."""
    need = max(1, math.ceil(duration_s * fps))
    k = max(1, math.ceil((need - 1) / FRAME_QUANTUM))
    return k * FRAME_QUANTUM + 1


def snap_size(value: int) -> int:
    """Round a dimension up to LTX's 64-pixel quantum."""
    return max(SIZE_QUANTUM, math.ceil(value / SIZE_QUANTUM) * SIZE_QUANTUM)


def _binary(bin_path: str) -> str:
    found = shutil.which(bin_path)
    if found is None:
        raise LTXUnavailableError(
            f"{bin_path} not found. ltx-2-mlx lives in its own venv, not the project's - "
            "see docs/tool-help/SUMMARY.md section 5 for the clone + `uv sync --all-extras` "
            "steps, then point AnimateConfig.bin at that venv's `ltx-2-mlx` binary."
        )
    return found


class LTXAnimateProvider:
    """Animates one still per scene with LTX-2.3 image-to-video."""

    name = "ltx-2.3-i2v"

    def __init__(
        self,
        bin: str,
        model_repo: str = "dgrauet/ltx-2.3-mlx-q4",
        width: int = 704,
        height: int = 480,
        frame_rate: int = 24,
        steps: int = 8,
        two_stage: bool = False,
        low_ram: bool = True,
    ):
        self.bin = bin
        self.model_repo = model_repo
        self.width = snap_size(width)
        self.height = snap_size(height)
        # Mandatory on ltx-2-mlx generate; LTX-2.3 was trained at 24fps, so
        # this (not the pipeline's overall video fps) drives frame counts.
        self.frame_rate = frame_rate
        self.steps = steps
        self.two_stage = two_stage
        self.low_ram = low_ram

    def _command(self, image: Path, prompt: str, frames: int, seed: int, out: Path) -> list[str]:
        cmd = [
            _binary(self.bin), "generate",
            "--prompt", prompt,
            # PATH alone -> FRAME_IDX=0 STRENGTH=1.0, i.e. anchor the whole
            # clip on this still. --image is repeatable on this tool (unlike
            # mflux's), but this provider only ever anchors frame 0.
            "--image", str(image),
            "--output", str(out),
            "--frames", str(frames),
            "--width", str(self.width),
            "--height", str(self.height),
            "--frame-rate", str(self.frame_rate),
            "--seed", str(seed),
            "--steps", str(self.steps),
            "--model", self.model_repo,
        ]
        if self.two_stage:
            cmd.append("--two-stage")
        if self.low_ram:
            cmd.append("--low-ram")
        return cmd

    def animate(
        self, image: Path, prompt: str, duration_s: float, seed: int, out: Path
    ) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = frames_for(duration_s, self.frame_rate)
        full = f"{prompt}, {MOTION_SUFFIX}"

        cmd = self._command(image, full, frames, seed, out)
        log.info("animating %s -> %s (%d frames)", image.name, out.name, frames)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
            raise RuntimeError(f"ltx-2-mlx exited {proc.returncode}:\n{tail}")

        if not out.exists():
            raise RuntimeError(f"ltx-2-mlx reported success but wrote no file at {out}")
        return out
