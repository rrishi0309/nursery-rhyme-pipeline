from itertools import pairwise
from pathlib import Path

import pytest

from nursery.config import VideoConfig
from nursery.video.kenburns import SceneClip, build_filter_graph, direction_for_index


def clips(n: int) -> list[SceneClip]:
    return [
        SceneClip(
            image=Path(f"/tmp/s{i}.png"),
            start_s=i * 3.0,
            end_s=(i + 1) * 3.0,
            direction=direction_for_index(i),
        )
        for i in range(n)
    ]


def test_directions_cycle_without_adjacent_repeats():
    seq = [direction_for_index(i) for i in range(8)]
    for a, b in pairwise(seq):
        assert a != b


def test_single_clip_graph_has_no_xfade():
    graph = build_filter_graph(clips(1), VideoConfig(), subtitles=None)

    assert "xfade" not in graph
    assert "zoompan" in graph


def test_two_clips_produce_one_xfade():
    graph = build_filter_graph(clips(2), VideoConfig(), subtitles=None)

    assert graph.count("xfade") == 1


def test_three_clips_produce_two_xfades():
    assert build_filter_graph(clips(3), VideoConfig(), subtitles=None).count("xfade") == 2


def test_graph_upscales_before_zoompan_to_avoid_jitter():
    graph = build_filter_graph(clips(1), VideoConfig(), subtitles=None)

    scale_pos = graph.index("scale=")
    zoom_pos = graph.index("zoompan")
    assert scale_pos < zoom_pos


def test_zoompan_duration_is_frames_not_seconds():
    cfg = VideoConfig(fps=30)
    graph = build_filter_graph(
        [SceneClip(image=Path("/tmp/a.png"), start_s=0.0, end_s=2.0, direction="in")], cfg, None
    )

    assert "d=60" in graph


def test_output_resolution_matches_config():
    cfg = VideoConfig(width=1280, height=720)
    graph = build_filter_graph(clips(1), cfg, None)

    assert "s=1280x720" in graph


def test_subtitles_filter_appended_when_provided(tmp_path):
    subs = tmp_path / "captions.ass"
    graph = build_filter_graph(clips(2), VideoConfig(), subtitles=subs)

    assert f"subtitles='{subs}'" in graph
    assert graph.strip().endswith("[vout]")


def test_no_subtitles_filter_when_absent():
    assert "subtitles=" not in build_filter_graph(clips(2), VideoConfig(), None)


def test_final_label_is_vout():
    assert build_filter_graph(clips(3), VideoConfig(), None).strip().endswith("[vout]")


def test_empty_clip_list_raises():
    with pytest.raises(ValueError, match="at least one"):
        build_filter_graph([], VideoConfig(), None)


def test_xfade_offsets_are_absolute_scene_starts_not_accumulated_durations():
    """Scene timings are absolute, authoritative positions from forced alignment.

    The transition into scene i must land at start_s_i - crossfade_s in output
    time. A left-folded chain that instead accumulates (duration - crossfade)
    per step compresses the timeline and drifts every image earlier by
    crossfade_s per scene - see fix-round-1-findings.md finding 1.
    """
    scenes = [
        SceneClip(
            image=Path(f"/tmp/s{i}.png"), start_s=i * 4.0, end_s=(i + 1) * 4.0,
            direction=direction_for_index(i),
        )
        for i in range(4)
    ]
    cfg = VideoConfig(crossfade_s=0.5)

    graph = build_filter_graph(scenes, cfg, subtitles=None)

    assert "offset=3.500" in graph
    assert "offset=7.500" in graph
    assert "offset=11.500" in graph


def test_non_first_clip_zoompan_length_includes_crossfade_overlap():
    """Every clip after the first must supply crossfade_s of extra frames.

    Those frames are what the transition into the *next* scene (or, for the
    last scene, the tail of the output) blends against; the first clip never
    needs them because xfade drops its content once the first blend ends.
    """
    cfg = VideoConfig(fps=30, crossfade_s=0.5)
    scenes = [
        SceneClip(image=Path("/tmp/a.png"), start_s=0.0, end_s=4.0, direction="in"),
        SceneClip(image=Path("/tmp/b.png"), start_s=4.0, end_s=8.0, direction="out"),
    ]

    graph = build_filter_graph(scenes, cfg, subtitles=None)

    assert "d=120" in graph  # clip 0: 4.0s * 30fps, no padding
    assert "d=135" in graph  # clip 1: (4.0 + 0.5)s * 30fps
