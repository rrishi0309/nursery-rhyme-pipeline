"""Animates each scene still into a clip with LTX-2.3 image-to-video.

Failure here is never fatal. A scene whose clip fails to render, or fails the
quality gate, keeps `animation="kenburns"` and the assemble stage pans its
still instead. That is what lets an unattended run finish even when a few
scenes come out warped - which, at roughly 2.5 minutes per clip, is much
cheaper than retrying.

The stage declares a report file rather than the clips as its outputs, because
"no clip for this scene" is a legitimate, recorded result and would otherwise
look like unfinished work to `is_satisfied`.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from nursery.config import Config
from nursery.manifest import Manifest
from nursery.stages.base import Stage, register
from nursery.video.quality import extract_frame, ssim

log = logging.getLogger(__name__)


def _judge(clip: Path, still: Path, scratch: Path, duration_s: float, cfg) -> tuple[bool, str]:
    """Accept or reject a clip on first-frame fidelity and motion."""
    first = extract_frame(clip, 0.0, scratch / "first.png")
    last = extract_frame(clip, max(0.0, duration_s - 0.1), scratch / "last.png")

    fidelity = ssim(first, still)
    motion = ssim(first, last)

    # An unmeasurable score (NaN) is not evidence of a bad clip, so it passes.
    if not math.isnan(fidelity) and fidelity < cfg.min_source_ssim:
        return False, f"ignored source still (ssim {fidelity:.3f} < {cfg.min_source_ssim})"
    if not math.isnan(motion):
        if motion > cfg.max_motion_ssim:
            return False, f"no motion (ssim {motion:.3f} > {cfg.max_motion_ssim})"
        if motion < cfg.min_motion_ssim:
            return False, f"drifted (ssim {motion:.3f} < {cfg.min_motion_ssim})"

    return True, f"ok (fidelity {fidelity:.3f}, motion {motion:.3f})"


def animate_scenes(m: Manifest, cfg: Config, provider) -> Manifest:
    """Render and gate one clip per scene, in place on the manifest."""
    out_dir = cfg.video_dir(m.video_id)
    clip_dir = out_dir / "clips"
    scratch = clip_dir / ".scratch"
    clip_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    report = []
    for scene in m.scenes:
        if scene.image_path is None:
            scene.animation, scene.animation_note = "kenburns", "no still to animate"
            continue

        still = Path(scene.image_path)
        target = clip_dir / f"scene_{scene.index:02d}.mp4"
        log.info("scene %d/%d: animating", scene.index + 1, len(m.scenes))

        try:
            provider.animate(
                image=still,
                prompt=scene.visual_prompt,
                duration_s=scene.duration_s,
                seed=2000 + scene.index,
                out=target,
            )
        except Exception as exc:  # noqa: BLE001 - a bad clip must not sink the video
            log.warning("scene %d: render failed (%s)", scene.index, exc)
            scene.clip_path, scene.animation = None, "kenburns"
            scene.animation_note = f"render failed: {exc}"
            report.append({"scene": scene.index, "ok": False, "why": scene.animation_note})
            continue

        ok, why = _judge(target, still, scratch, scene.duration_s, cfg.animate)
        if ok:
            scene.clip_path, scene.animation, scene.animation_note = str(target), "clip", why
        else:
            log.warning("scene %d: rejected - %s", scene.index, why)
            target.unlink(missing_ok=True)
            scene.clip_path, scene.animation, scene.animation_note = None, "kenburns", why

        report.append({"scene": scene.index, "ok": ok, "why": why})

    (clip_dir / "animate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    kept = sum(1 for s in m.scenes if s.animation == "clip")
    log.info("animate: %d/%d scenes are clips, rest fall back to Ken Burns", kept, len(m.scenes))
    return m


def build_provider(cfg: Config):
    """Resolve the configured animation backend, or None to skip animating."""
    which = cfg.providers.get("animate", "none")
    if which in ("none", "", None):
        return None
    if which == "ltx":
        from nursery.providers.animate_ltx import LTXAnimateProvider

        return LTXAnimateProvider(
            model_repo=cfg.animate.model_repo,
            pipeline=cfg.animate.pipeline,
            width=cfg.animate.width,
            height=cfg.animate.height,
            fps=cfg.video.fps,
            cfg_scale=cfg.animate.cfg_scale,
            image_strength=cfg.animate.image_strength,
            python=cfg.animate.python,
        )
    raise KeyError(f"unknown animate provider {which!r}")


class AnimateStage(Stage):
    name = "animate"

    def input_payload(self, m: Manifest, cfg: Config) -> object:
        return {
            "scenes": [
                {"i": s.index, "img": s.image_path, "p": s.visual_prompt,
                 "a": s.start_s, "b": s.end_s}
                for s in m.scenes
            ],
            "animate": cfg.animate.model_dump(),
            "provider": cfg.providers.get("animate", "none"),
            "fps": cfg.video.fps,
        }

    def outputs(self, m: Manifest, cfg: Config) -> list[Path]:
        return [cfg.video_dir(m.video_id) / "clips" / "animate.json"]

    def execute(self, m: Manifest, cfg: Config) -> Manifest:
        provider = build_provider(cfg)
        if provider is None:
            log.info("animate provider is 'none'; every scene stays on Ken Burns")
            for scene in m.scenes:
                scene.animation = scene.animation or "kenburns"
            clip_dir = cfg.video_dir(m.video_id) / "clips"
            clip_dir.mkdir(parents=True, exist_ok=True)
            (clip_dir / "animate.json").write_text("[]", encoding="utf-8")
            return m
        return animate_scenes(m, cfg, provider)


register(AnimateStage())
