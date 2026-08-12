from datetime import UTC, datetime
from pathlib import Path

import pytest

from nursery.config import Config
from nursery.manifest import Manifest, StageRecord
from nursery.stages.base import STAGE_ORDER, Stage, get_stage, register


class DummyStage(Stage):
    name = "dummy"

    def __init__(self, out: Path):
        self.out = out
        self.calls = 0

    def input_payload(self, m: Manifest, cfg: Config) -> object:
        return {"title": m.title}

    def outputs(self, m: Manifest, cfg: Config) -> list[Path]:
        return [self.out]

    def execute(self, m: Manifest, cfg: Config) -> Manifest:
        self.calls += 1
        self.out.write_text("done")
        return m


@pytest.fixture
def manifest() -> Manifest:
    return Manifest(
        video_id="v1", created_at=datetime(2026, 8, 8, tzinfo=UTC),
        source="original", title="Hello",
    )


def test_not_satisfied_when_never_run(tmp_path, manifest):
    stage = DummyStage(tmp_path / "a.txt")
    assert stage.is_satisfied(manifest, Config()) is False


def test_not_satisfied_when_output_missing(tmp_path, manifest):
    stage = DummyStage(tmp_path / "a.txt")
    manifest.stages["dummy"] = StageRecord(
        status="ok", input_hash=stage.current_hash(manifest, Config())
    )
    assert stage.is_satisfied(manifest, Config()) is False


def test_satisfied_when_ok_and_outputs_exist(tmp_path, manifest):
    stage = DummyStage(tmp_path / "a.txt")
    cfg = Config()
    stage.execute(manifest, cfg)
    manifest.stages["dummy"] = StageRecord(
        status="ok", input_hash=stage.current_hash(manifest, cfg)
    )
    assert stage.is_satisfied(manifest, cfg) is True


def test_stale_input_hash_invalidates(tmp_path, manifest):
    stage = DummyStage(tmp_path / "a.txt")
    cfg = Config()
    stage.execute(manifest, cfg)
    manifest.stages["dummy"] = StageRecord(
        status="ok", input_hash=stage.current_hash(manifest, cfg)
    )
    manifest.title = "Different"
    assert stage.is_satisfied(manifest, cfg) is False


def test_failed_status_is_not_satisfied(tmp_path, manifest):
    stage = DummyStage(tmp_path / "a.txt")
    cfg = Config()
    stage.execute(manifest, cfg)
    manifest.stages["dummy"] = StageRecord(
        status="failed", input_hash=stage.current_hash(manifest, cfg)
    )
    assert stage.is_satisfied(manifest, cfg) is False


def test_registry_round_trip(tmp_path):
    stage = DummyStage(tmp_path / "a.txt")
    register(stage)
    assert get_stage("dummy") is stage


def test_get_unknown_stage_raises():
    with pytest.raises(KeyError, match="nope"):
        get_stage("nope")


def test_stage_order_is_the_documented_sequence():
    assert STAGE_ORDER == [
        "seed", "script", "audio", "align",
        "cast", "images", "animate", "assemble", "metadata", "publish",
    ]
