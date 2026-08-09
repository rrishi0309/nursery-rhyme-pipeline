"""End-to-end build of a real nursery rhyme video from the classics catalog.

Timing model: each lyric line owns one musical phrase of the melody, so scene
boundaries fall on musical bars. Narration is placed at the head of its phrase
and the tune carries the rest. Because every line's audio is rendered
separately, the line boundaries are exact and no forced aligner is needed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import yaml

from nursery.audio.melody import melody_duration_s, render_melody
from nursery.audio.mixer import NarrationClip, mix_narration_over_bed
from nursery.config import Config
from nursery.manifest import AudioSpec, Manifest, Scene
from nursery.providers.tts_macos import MacTTSProvider
from nursery.stages.assemble import AssembleStage

log = logging.getLogger(__name__)

CATALOG = Path(__file__).parent / "catalog" / "classics.yaml"


def load_catalog() -> dict:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


def _default_image_provider(style: str):
    """FLUX.1-schnell if available (faster, Apache-2.0), else SDXL."""
    import shutil

    if shutil.which("mflux-generate") is not None:
        log.info("using FLUX.1-schnell for image generation")
        from nursery.providers.image_flux import FluxImageProvider

        return FluxImageProvider(style=style)

    log.info("mflux not available, falling back to SDXL")
    from nursery.providers.image_sdxl import SDXLImageProvider

    return SDXLImageProvider(style=style)


def build(
    slug: str,
    cfg: Config,
    video_id: str | None = None,
    voice: str = "Samantha",
    image_provider=None,
) -> Manifest:
    entry = load_catalog()[slug]
    lines = entry["lines"]
    video_id = video_id or f"{datetime.now(UTC):%Y-%m-%d}-{slug}-01"
    out_dir = cfg.video_dir(video_id)

    # One musical phrase per lyric line.
    total_s = melody_duration_s(entry["melody"], entry["tempo_bpm"])
    phrase_s = total_s / len(lines)
    log.info("%s: %d lines, %.2fs per phrase, %.1fs total", slug, len(lines), phrase_s, total_s)

    log.info("rendering melody")
    bed = render_melody(entry["melody"], entry["tempo_bpm"], out_dir / "audio" / "melody.wav")

    log.info("rendering narration")
    tts = MacTTSProvider(voice=voice)
    spoken = tts.speak_lines([ln["text"] for ln in lines], out_dir / "audio" / "lines")

    clips = [NarrationClip(path=p, start_s=i * phrase_s) for i, (p, _) in enumerate(spoken)]
    mix = mix_narration_over_bed(bed, clips, out_dir / "audio" / "mix.wav")

    log.info("rendering %d scene images", len(lines))
    provider = image_provider
    if provider is None:
        provider = _default_image_provider(entry["style"])

    scenes = []
    for i, line in enumerate(lines):
        image = provider.generate(
            line["visual"], [], seed=1000 + i, out=out_dir / "images" / f"scene_{i:02d}.png"
        )
        scenes.append(
            Scene(
                index=i,
                text=line["text"],
                visual_prompt=line["visual"],
                image_path=str(image),
                start_s=i * phrase_s,
                end_s=(i + 1) * phrase_s,
            )
        )
        log.info("  scene %d/%d", i + 1, len(lines))

    manifest = Manifest(
        video_id=video_id,
        created_at=datetime.now(UTC),
        source="classic",
        title=entry["title"],
        scenes=scenes,
        audio=AudioSpec(
            mix_path=str(mix), duration_s=total_s, provider=f"melody+{tts.name}", degraded=False
        ),
    )
    manifest.save(out_dir / "manifest.json")

    log.info("assembling video")
    manifest = AssembleStage().execute(manifest, cfg)
    manifest.save(out_dir / "manifest.json")
    return manifest
