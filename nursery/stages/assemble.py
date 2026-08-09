"""Turns scene images plus aligned audio into the finished MP4.

Pure and deterministic: same manifest in, same video out. That is what lets it
be iterated on against cached artifacts without re-running any model.
"""

from __future__ import annotations

from pathlib import Path

from nursery.config import Config
from nursery.manifest import Manifest, VideoSpec
from nursery.providers.base import WordTiming
from nursery.stages.base import Stage, register
from nursery.video.captions import group_words_into_lines, write_ass
from nursery.video.ffmpeg import build_command, extract_thumbnail, run_ffmpeg
from nursery.video.kenburns import SceneClip, build_filter_graph, direction_for_index


class AssembleStage(Stage):
    name = "assemble"

    def input_payload(self, m: Manifest, cfg: Config) -> object:
        return {
            "scenes": [
                {"i": s.index, "img": s.image_path, "a": s.start_s, "b": s.end_s}
                for s in m.scenes
            ],
            "audio": m.audio.mix_path if m.audio else None,
            "video": cfg.video.model_dump(),
        }

    def outputs(self, m: Manifest, cfg: Config) -> list[Path]:
        d = cfg.video_dir(m.video_id) / "video"
        return [d / "final.mp4", d / "thumbnail.png"]

    def execute(self, m: Manifest, cfg: Config) -> Manifest:
        if m.audio is None:
            raise ValueError("assemble requires audio; run the audio stage first")
        if not m.scenes:
            raise ValueError("assemble requires at least one scene")

        clips = []
        for scene in m.scenes:
            if scene.image_path is None:
                raise ValueError(f"scene {scene.index} has no image")
            if scene.start_s is None or scene.end_s is None:
                raise ValueError(f"scene {scene.index} has no timing")
            clips.append(
                SceneClip(
                    image=Path(scene.image_path),
                    duration_s=scene.end_s - scene.start_s,
                    direction=direction_for_index(scene.index),
                )
            )

        video_dir = cfg.video_dir(m.video_id) / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        subtitles = self._write_captions(m, cfg, video_dir)
        graph = build_filter_graph(clips, cfg.video, subtitles)
        final = video_dir / "final.mp4"

        run_ffmpeg(build_command(clips, Path(m.audio.mix_path), graph, final, cfg.video))

        thumbnail = extract_thumbnail(final, at_s=min(1.0, m.audio.duration_s / 2),
                                      out=video_dir / "thumbnail.png")

        m.video = VideoSpec(
            final_path=str(final),
            thumbnail_path=str(thumbnail),
            duration_s=m.audio.duration_s,
        )
        return m

    def _write_captions(self, m: Manifest, cfg: Config, video_dir: Path) -> Path | None:
        """Derive per-scene karaoke captions from scene timings.

        Word-level timings from the align stage land on the scene; until that
        stage exists, distribute a scene's words evenly across its span.
        """
        timings: list[WordTiming] = []
        lyric_lines: list[str] = []

        for scene in m.scenes:
            words = scene.text.split()
            if not words:
                continue
            lyric_lines.append(scene.text)
            step = (scene.end_s - scene.start_s) / len(words)
            timings += [
                WordTiming(
                    word=w,
                    start_s=scene.start_s + i * step,
                    end_s=scene.start_s + (i + 1) * step,
                )
                for i, w in enumerate(words)
            ]

        if not timings:
            return None

        lines = group_words_into_lines(timings, lyric_lines)
        return write_ass(lines, cfg.video.width, cfg.video.height, video_dir / "captions.ass")


register(AssembleStage())
