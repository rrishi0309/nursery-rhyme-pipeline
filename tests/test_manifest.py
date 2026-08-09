import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nursery.manifest import SCHEMA_VERSION, Manifest, Scene, StageRecord


def make_manifest() -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        video_id="2026-08-08-twinkle-01",
        created_at=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
        source="classic",
        title="Twinkle Twinkle Little Star",
        scenes=[
            Scene(index=0, text="Twinkle twinkle little star",
                  visual_prompt="a star over a meadow", characters=["star"]),
        ],
    )


def test_round_trip_preserves_fields(tmp_path: Path):
    m = make_manifest()
    path = tmp_path / "manifest.json"
    m.save(path)

    loaded = Manifest.load(path)

    assert loaded.video_id == "2026-08-08-twinkle-01"
    assert loaded.title == "Twinkle Twinkle Little Star"
    assert loaded.scenes[0].text == "Twinkle twinkle little star"
    assert loaded.scenes[0].image_path is None
    assert loaded.audio is None


def test_save_is_atomic_leaving_no_temp_files(tmp_path: Path):
    m = make_manifest()
    path = tmp_path / "manifest.json"
    m.save(path)

    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]


def test_save_overwrites_existing_without_corruption(tmp_path: Path):
    path = tmp_path / "manifest.json"
    m = make_manifest()
    m.save(path)

    m.title = "Changed"
    m.save(path)

    assert Manifest.load(path).title == "Changed"
    json.loads(path.read_text())  # still valid JSON


def test_stage_records_default_to_empty():
    assert make_manifest().stages == {}


def test_stage_record_accepts_status_and_hash():
    m = make_manifest()
    m.stages["assemble"] = StageRecord(status="ok", input_hash="abc123")

    assert m.stages["assemble"].status == "ok"
    assert m.stages["assemble"].error is None


def test_rejects_unknown_schema_version(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 999, "video_id": "x",
                                "created_at": "2026-08-08T12:00:00Z",
                                "source": "classic", "title": "x", "scenes": []}))

    with pytest.raises(ValueError, match="schema_version"):
        Manifest.load(path)
