"""Scene illustration via FLUX.1-schnell (MLX-native, Apple Silicon).

FLUX.1-schnell is Apache-2.0, so it is commercially usable. Unlike
FLUX.1-dev, which is gated behind a non-commercial licence, schnell only needs
HF licence acceptance to download - it carries no usage restriction once
downloaded. Rendering happens via the `mflux-generate` CLI, the same shell-out
pattern this codebase already uses for ffmpeg, rather than mflux's internal
Python API.

Roughly 2x faster than the SDXL provider per image on an M4 Max (turbo/2-step
schedule vs. SDXL's 28-step default), with comparable or better illustration
quality in informal comparison.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

NEGATIVE = (
    "text, letters, words, watermark, signature, blurry, deformed, ugly, "
    "photorealistic, horror, scary, dark, gore, extra limbs, distorted faces"
)


class MFluxUnavailableError(RuntimeError):
    pass


def _binary() -> str:
    found = shutil.which("mflux-generate")
    if found is None:
        raise MFluxUnavailableError(
            "mflux-generate not found on PATH. Install with `uv add mflux`, "
            "accept the FLUX.1-schnell licence on Hugging Face, and run "
            "`uv run hf auth login`."
        )
    return found


class FluxImageProvider:
    """Renders one illustration per scene using FLUX.1-schnell, turbo schedule."""

    name = "flux-schnell"

    def __init__(
        self,
        style: str = "",
        steps: int = 2,
        width: int = 1344,
        height: int = 768,
        quantize: int = 4,
    ):
        self.style = style
        self.steps = steps
        self.width = width
        self.height = height
        self.quantize = quantize

    def generate(self, prompt: str, refs: list[Path], seed: int, out: Path) -> Path:
        full = f"{prompt}, {self.style}" if self.style else prompt
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            _binary(),
            "--model", "schnell",
            "-q", str(self.quantize),
            "--steps", str(self.steps),
            "--height", str(self.height),
            "--width", str(self.width),
            "--seed", str(seed),
            "--prompt", full,
            "--negative-prompt", NEGATIVE,
            "--output", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
            raise RuntimeError(f"mflux-generate exited {proc.returncode}:\n{tail}")
        return out
