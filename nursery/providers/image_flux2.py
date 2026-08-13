"""Cast sheet + scene stills via FLUX.2 Klein (MLX-native, Apple Silicon).

Rendering shells out to the `mflux-generate-flux2` CLI, the same shell-out
pattern this codebase already uses for `mflux-generate`
(`nursery/providers/image_flux.py`) and ffmpeg, rather than mflux's internal
Python API.

**Character-consistency design, and why it looks the way it does:**
`mflux-generate-flux2 --image PATH [STRENGTH ...]` is FLUX.2 Klein's *only*
image-conditioning flag. Reading `mflux/cli/parser/parsers.py` confirms it is
declared `nargs="+"` - a single, non-repeatable init-image/img2img control,
not a multi-reference identity adapter. Passing a cast sheet at high strength
would copy that sheet's *composition* (poses, camera angle, layout) into
every scene, which is wrong - a scene still should compose freely around the
prompt, not clone the cast sheet's framing.

So the consistency strategy is split in two, with the textual half doing
the real work:

  1. **Primary: text.** Callers (the images stage) are expected to append
     each on-screen character's name + description to the scene prompt
     before calling `generate()`, and to pass the same style string and a
     stable per-video base seed across every render. This alone is what
     keeps a character drawn recognisably the same way scene to scene.
  2. **Secondary: a low-strength img2img nudge.** `generate()` additionally
     passes `--image <cast_sheet> <cast_strength>` (default 0.35, low on
     purpose) when a reference image is supplied, as a soft style/palette
     nudge rather than a structural copy. This path is wholesale
     switchable via `use_reference=False` in case it turns out to hurt more
     than it helps.
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


class Flux2UnavailableError(RuntimeError):
    pass


def _binary() -> str:
    found = shutil.which("mflux-generate-flux2")
    if found is None:
        raise Flux2UnavailableError(
            "mflux-generate-flux2 not found on PATH. Install with `uv add mflux`, "
            "accept the FLUX.2 licence on Hugging Face, and run `uv run hf auth login`."
        )
    return found


def _cast_text(cast: list[dict]) -> str:
    """Render `[{"name": ..., "description": ...}, ...]` as prompt text."""
    parts = []
    for member in cast:
        name = str(member.get("name", "")).strip()
        description = str(member.get("description", "")).strip()
        if name and description:
            parts.append(f"{name}: {description}")
        elif name or description:
            parts.append(name or description)
    return ", ".join(parts)


class Flux2ImageProvider:
    """Renders a cast sheet plus one still per scene using FLUX.2 Klein."""

    name = "flux2-klein"

    def __init__(
        self,
        style: str = "",
        model: str = "flux2-klein-4b",
        steps: int = 4,
        width: int = 1344,
        height: int = 768,
        quantize: int = 4,
        guidance: float = 3.5,
        cast_strength: float = 0.35,
        use_reference: bool = True,
    ):
        self.style = style
        self.model = model
        self.steps = steps
        self.width = width
        self.height = height
        self.quantize = quantize
        self.guidance = guidance
        self.cast_strength = cast_strength
        # Wholesale kill switch for the img2img nudge (see module docstring)
        # if it turns out to hurt scene composition more than it helps.
        self.use_reference = use_reference

    def _base_command(self, prompt: str, seed: int, out: Path) -> list[str]:
        return [
            _binary(),
            "--model", self.model,
            "-q", str(self.quantize),
            "--steps", str(self.steps),
            "--height", str(self.height),
            "--width", str(self.width),
            "--guidance", str(self.guidance),
            "--seed", str(seed),
            "--prompt", prompt,
            "--negative-prompt", NEGATIVE,
            "--output", str(out),
        ]

    def _run(self, cmd: list[str]) -> None:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
            raise RuntimeError(f"mflux-generate-flux2 exited {proc.returncode}:\n{tail}")

    def cast_sheet(self, cast: list[dict], style: str, seed: int, out: Path) -> Path:
        """Render every character together in one reference image.

        This image is a soft secondary nudge for `generate()`, not the main
        consistency mechanism - see module docstring.
        """
        out.parent.mkdir(parents=True, exist_ok=True)
        cast_text = _cast_text(cast)
        pieces = ["character cast sheet, full-body reference poses, neutral background"]
        if cast_text:
            pieces.append(cast_text)
        if style:
            pieces.append(style)
        prompt = ", ".join(pieces)

        cmd = self._base_command(prompt, seed, out)
        self._run(cmd)
        return out

    def generate(self, prompt: str, reference: Path | None, seed: int, out: Path) -> Path:
        """Render one scene still.

        `prompt` is expected to already carry the on-screen characters' name +
        description text (the primary consistency mechanism); this method
        does not append cast text itself, since it has no cast to draw from.
        `reference`, when given, is the cast sheet path used for the
        secondary low-strength img2img nudge - pass `None`, or construct with
        `use_reference=False`, to skip it entirely.
        """
        out.parent.mkdir(parents=True, exist_ok=True)
        full = f"{prompt}, {self.style}" if self.style else prompt

        cmd = self._base_command(full, seed, out)
        if self.use_reference and reference is not None:
            cmd += ["--image", str(reference), str(self.cast_strength)]
        self._run(cmd)
        return out
