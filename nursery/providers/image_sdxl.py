"""Scene illustration via SDXL base 1.0 on Apple Silicon (MPS).

SDXL is used rather than FLUX.1-schnell because the FLUX repo is gated on
Hugging Face and needs per-user license acceptance, whereas SDXL is ungated and
its CreativeML OpenRAIL++-M licence permits commercial use.

The pipeline is loaded once and reused across every scene in a video - loading
costs far more than a render, so re-instantiating per scene would dominate.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

NEGATIVE = (
    "text, letters, words, watermark, signature, blurry, deformed, ugly, "
    "photorealistic, horror, scary, dark, gore, extra limbs, distorted faces"
)


class SDXLImageProvider:
    """Renders one illustration per scene, with a fixed seed for reproducibility."""

    name = "sdxl-base-1.0"

    def __init__(
        self,
        style: str = "",
        steps: int = 28,
        guidance: float = 6.0,
        width: int = 1344,
        height: int = 768,
    ):
        self.style = style
        self.steps = steps
        self.guidance = guidance
        self.width = width
        self.height = height
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from diffusers import StableDiffusionXLPipeline

            log.info("loading %s (first run downloads weights)", MODEL_ID)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            pipe.to("mps")
            pipe.set_progress_bar_config(disable=True)
            self._pipe = pipe
        return self._pipe

    def generate(self, prompt: str, refs: list[Path], seed: int, out: Path) -> Path:
        import torch

        full = f"{prompt}, {self.style}" if self.style else prompt
        image = self._pipeline()(
            prompt=full,
            negative_prompt=NEGATIVE,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            height=self.height,
            width=self.width,
            generator=torch.Generator("mps").manual_seed(seed),
        ).images[0]

        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        return out
