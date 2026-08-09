from datetime import UTC, datetime
from pathlib import Path

import pytest

from nursery.config import Config
from nursery.manifest import Manifest
from nursery.orchestrator import StageFailure, run_pipeline
from nursery.stages.base import Stage, register


class RecordingStage(Stage):
    def __init__(self, name: str, out: Path, fail: bool = False):
        self.name = name
        self.out = out
        self.fail = fail
        self.calls = 0

    def input_payload(self, m, cfg):
        return {"title": m.title}

    def outputs(self, m, cfg):
        return [self.out]

    def execute(self, m, cfg):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        self.out.write_text("ok")
        return m


@pytest.fixture
def manifest():
    return Manifest(
        video_id="v1", created_at=datetime(2026, 8, 8, tzinfo=UTC),
        source="original", title="Hello",
    )


def test_runs_stages_in_order(tmp_path, manifest):
    order = []

    class Tracker(RecordingStage):
        def execute(self, m, cfg):
            order.append(self.name)
            return super().execute(m, cfg)

    for name in ("one", "two", "three"):
        register(Tracker(name, tmp_path / f"{name}.txt"))

    run_pipeline(manifest, Config(), ["one", "two", "three"], tmp_path / "manifest.json")

    assert order == ["one", "two", "three"]


def test_marks_each_stage_ok(tmp_path, manifest):
    register(RecordingStage("one", tmp_path / "one.txt"))

    result = run_pipeline(manifest, Config(), ["one"], tmp_path / "manifest.json")

    assert result.stages["one"].status == "ok"
    assert result.stages["one"].input_hash is not None
    assert result.stages["one"].finished_at is not None


def test_skips_satisfied_stages(tmp_path, manifest):
    stage = RecordingStage("one", tmp_path / "one.txt")
    register(stage)
    cfg = Config()
    path = tmp_path / "manifest.json"

    run_pipeline(manifest, cfg, ["one"], path)
    run_pipeline(manifest, cfg, ["one"], path)

    assert stage.calls == 1


def test_failure_records_error_and_raises(tmp_path, manifest):
    register(RecordingStage("one", tmp_path / "one.txt", fail=True))
    path = tmp_path / "manifest.json"

    with pytest.raises(StageFailure) as exc:
        run_pipeline(manifest, Config(), ["one"], path)

    assert exc.value.stage == "one"
    saved = Manifest.load(path)
    assert saved.stages["one"].status == "failed"
    assert "boom" in saved.stages["one"].error


def test_failure_halts_later_stages(tmp_path, manifest):
    register(RecordingStage("one", tmp_path / "one.txt", fail=True))
    later = RecordingStage("two", tmp_path / "two.txt")
    register(later)

    with pytest.raises(StageFailure):
        run_pipeline(manifest, Config(), ["one", "two"], tmp_path / "manifest.json")

    assert later.calls == 0


def test_start_from_forces_rerun_of_that_stage(tmp_path, manifest):
    one = RecordingStage("one", tmp_path / "one.txt")
    two = RecordingStage("two", tmp_path / "two.txt")
    register(one)
    register(two)
    cfg = Config()
    path = tmp_path / "manifest.json"

    run_pipeline(manifest, cfg, ["one", "two"], path)
    run_pipeline(manifest, cfg, ["one", "two"], path, start_from="two")

    assert one.calls == 1
    assert two.calls == 2


def test_manifest_persisted_after_each_stage(tmp_path, manifest):
    path = tmp_path / "manifest.json"

    class Checker(RecordingStage):
        def execute(self, m, cfg):
            if self.name == "two":
                assert Manifest.load(path).stages["one"].status == "ok"
            return super().execute(m, cfg)

    register(Checker("one", tmp_path / "one.txt"))
    register(Checker("two", tmp_path / "two.txt"))

    run_pipeline(manifest, Config(), ["one", "two"], path)
