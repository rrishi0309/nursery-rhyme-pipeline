"""Scene animation via LTX-2.3 image-to-video (MLX-native, Apple Silicon).

Runs image-to-video rather than text-to-video on purpose. The `images` stage
already produces a canonical still per scene, seeded and steered by
`cast.yaml`, and conditioning LTX on that still is what keeps a character
recognisable across ~20 scenes. Text-to-video would re-roll the character
every scene.

Rendering shells out to the `mlx_video.models.ltx_2.generate` CLI, the same
pattern this codebase already uses for `mflux-generate` and `ffmpeg`. Note the
module path: mlx-video's own `--help` examples say `python -m
mlx_video.generate`, but no such module exists in the installed package.

Two hard constraints from LTX, both encoded here:

- Frame count must be `8k+1`. Scene durations are not, so `frames_for` rounds
  *up* to the next valid count and the assemble stage trims back down to the
  exact scene duration. Generating long and trimming is what keeps scene
  timings locked to the audio alignment.
- Width and height must be divisible by 64. 1024x576 is native 16:9
  (16*64 by 9*64) and upscales cleanly to 1920x1080 in the assemble graph.

The flag set below is taken from `--help` on an installed build. Two flags are
pipeline-sensitive: `--cfg-scale` and `--negative-prompt` apply only to the
`dev` pipelines, since `distilled` runs without CFG at all, so passing them
alongside `distilled` is silently meaningless.
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

MODULE = "mlx_video.models.ltx_2.generate"

FRAME_QUANTUM = 8
SIZE_QUANTUM = 64

MOTION_SUFFIX = (
    "gentle storybook motion, subtle camera drift, characters stay on model, "
    "consistent art style, no scene change"
)

NEGATIVE = (
    "text, letters, words, watermark, blurry, deformed, ugly, photorealistic, "
    "horror, scary, dark, gore, extra limbs, distorted faces, flicker, morphing"
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


class LTXAnimateProvider:
    """Animates one still per scene with LTX-2.3 image-to-video."""

    name = "ltx-2.3-i2v"

    def __init__(
        self,
        pipeline: str = "distilled",
        width: int = 1024,
        height: int = 576,
        fps: int = 30,
        cfg_scale: float | None = None,
        image_strength: float | None = None,
        python: str | None = None,
        model_repo: str | None = None,
    ):
        self.model_repo = model_repo
        self.pipeline = pipeline
        self.width = snap_size(width)
        self.height = snap_size(height)
        self.fps = fps
        self.cfg_scale = cfg_scale
        self.image_strength = image_strength
        # mlx-video drags in librosa/numba, so it may live in its own
        # interpreter rather than the pipeline's.
        self.python = python or sys.executable

    def _command(self, image: Path, prompt: str, frames: int, seed: int, out: Path) -> list[str]:
        cmd = [
            self.python, "-m", MODULE,
            "--image", str(image),
            "--prompt", prompt,
            "--num-frames", str(frames),
            "--width", str(self.width),
            "--height", str(self.height),
            "--fps", str(self.fps),
            "--seed", str(seed),
            "--pipeline", self.pipeline,
            "--output-path", str(out),
            # Documented as more stable than plain CFG for image-to-video,
            # which is the only mode this provider runs.
            "--apg",
        ]
        # mlx-video defaults to Lightricks/LTX-2 (19B). Pin explicitly rather
        # than inheriting that default, so which weights ran is recorded.
        if self.model_repo:
            cmd += ["--model-repo", self.model_repo]
        if self.image_strength is not None:
            cmd += ["--image-strength", str(self.image_strength)]

        # CFG and negative prompts exist only on the dev pipelines; the
        # distilled one runs without guidance.
        if self.pipeline != "distilled":
            cmd += ["--negative-prompt", NEGATIVE]
            if self.cfg_scale is not None:
                cmd += ["--cfg-scale", str(self.cfg_scale)]
        return cmd

    def animate(
        self, image: Path, prompt: str, duration_s: float, seed: int, out: Path
    ) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = frames_for(duration_s, self.fps)
        full = f"{prompt}, {MOTION_SUFFIX}"

        cmd = self._command(image, full, frames, seed, out)
        log.info("animating %s -> %s (%d frames)", image.name, out.name, frames)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
            if "No module named" in proc.stderr:
                raise LTXUnavailableError(
                    f"{MODULE} not importable under {self.python}. Install with "
                    "`uv add 'mlx-video @ git+https://github.com/Blaizzy/mlx-video.git' "
                    "'numba>=0.62'` - the numba floor is required, or uv "
                    "backtracks to a release that refuses to build on Python 3.12+."
                )
            raise RuntimeError(f"LTX exited {proc.returncode}:\n{tail}")

        if not out.exists():
            raise RuntimeError(f"LTX reported success but wrote no file at {out}")
        return out
