# Nursery Rhyme Pipeline — Foundation & Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the resumable stage pipeline and the deterministic video assembler, so that a complete 1080p MP4 with Ken Burns motion and karaoke captions can be produced end-to-end from fixture inputs, with no generative models involved.

**Architecture:** A per-video directory holds a `manifest.json` that is the single source of truth. Stages are independent modules implementing a common `Stage` protocol (`is_satisfied` / `run`); an orchestrator sequences them, skips satisfied stages via input hashing, and isolates failures. All model access sits behind provider interfaces, so fake providers let the whole pipeline run in CI in seconds.

**Tech Stack:** Python 3.12+, uv, pydantic v2, typer, PyYAML, pytest, ffmpeg (already installed at `/opt/homebrew/bin/ffmpeg`).

## Global Constraints

- Python 3.12 or newer. Dependency management with `uv`; never `pip install` into system Python.
- Zero paid APIs and zero network calls anywhere in this plan. Tests must pass with networking unavailable.
- No generative models in this plan. Every artifact comes from fixtures or fake providers.
- Stages never import a model library directly — only `nursery.providers.*` interfaces.
- Manifest writes are atomic: write to a temp file in the same directory, then `os.replace`.
- Target output: 1920x1080, 30 fps, H.264 `yuv420p`, AAC audio.
- `made_for_kids` is always `True` in any metadata structure; it is never configurable.
- All timestamps in the manifest are floats, in seconds, from the start of the audio.
- Every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, deps, pytest + ruff config |
| `nursery/config.py` | Typed config model, YAML + env loading |
| `nursery/manifest.py` | Pydantic manifest schema, atomic load/save |
| `nursery/hashing.py` | Stable input-hash computation for stage staleness |
| `nursery/stages/base.py` | `Stage` protocol, `StageResult`, registry |
| `nursery/orchestrator.py` | Stage sequencing, resume, failure isolation |
| `nursery/providers/base.py` | Provider protocols (image, song, tts, llm, align) |
| `nursery/providers/fake.py` | Deterministic fake providers for tests |
| `nursery/video/kenburns.py` | ffmpeg filter-graph construction |
| `nursery/video/captions.py` | ASS karaoke subtitle generation |
| `nursery/video/ffmpeg.py` | ffmpeg subprocess invocation and error surfacing |
| `nursery/stages/assemble.py` | The assemble stage, wiring the three above |
| `nursery/cli.py` | `nursery run \| stage \| list \| clean` |
| `tests/conftest.py` | Shared fixtures: tmp video dir, sample manifest |

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `nursery/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installed package named `nursery`, importable in tests. A working `uv run pytest`.

- [ ] **Step 1: Install uv**

`uv` is not currently installed. Run:

```bash
brew install uv
```

Verify:

```bash
uv --version
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "nursery"
version = "0.1.0"
description = "Local nursery rhyme video generation pipeline"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "typer>=0.12",
    "pyyaml>=6.0",
]

[project.scripts]
nursery = "nursery.cli:app"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nursery"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 3: Create package files**

`nursery/__init__.py`:

```python
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

- [ ] **Step 4: Write the failing test**

`tests/test_smoke.py`:

```python
def test_package_imports():
    import nursery

    assert nursery.__version__ == "0.1.0"
```

- [ ] **Step 5: Run it and confirm it passes**

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: PASS. (This task's test verifies packaging, so it passes once the
package exists — that is the deliverable.)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml nursery/__init__.py tests/__init__.py tests/test_smoke.py uv.lock
git commit -m "feat: scaffold nursery package with uv and pytest

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Manifest schema and atomic persistence

**Files:**
- Create: `nursery/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Scene(index: int, text: str, visual_prompt: str, characters: list[str], image_path: str | None, start_s: float | None, end_s: float | None)`
  - `StageRecord(status: Literal["pending","running","ok","failed"], input_hash: str | None, error: str | None, started_at: datetime | None, finished_at: datetime | None)`
  - `AudioSpec(mix_path: str, duration_s: float, provider: str, degraded: bool)`
  - `VideoSpec(final_path: str, thumbnail_path: str, duration_s: float)`
  - `Manifest` with `schema_version: int`, `video_id: str`, `created_at: datetime`, `source: Literal["classic","original"]`, `title: str`, `scenes: list[Scene]`, `audio: AudioSpec | None`, `video: VideoSpec | None`, `stages: dict[str, StageRecord]`
  - `Manifest.load(path: Path) -> Manifest`
  - `Manifest.save(self, path: Path) -> None` (atomic)
  - `SCHEMA_VERSION: int = 1`

- [ ] **Step 1: Write the failing test**

`tests/test_manifest.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.manifest'`.

- [ ] **Step 3: Write the implementation**

`nursery/manifest.py`:

```python
"""Per-video manifest: the single source of truth for a pipeline run."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

StageStatus = Literal["pending", "running", "ok", "failed"]


class Scene(BaseModel):
    index: int
    text: str
    visual_prompt: str
    characters: list[str] = Field(default_factory=list)
    image_path: str | None = None
    start_s: float | None = None
    end_s: float | None = None

    @property
    def duration_s(self) -> float:
        if self.start_s is None or self.end_s is None:
            raise ValueError(f"scene {self.index} has no timing yet")
        return self.end_s - self.start_s


class StageRecord(BaseModel):
    status: StageStatus = "pending"
    input_hash: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AudioSpec(BaseModel):
    mix_path: str
    duration_s: float
    provider: str
    degraded: bool = False


class VideoSpec(BaseModel):
    final_path: str
    thumbnail_path: str
    duration_s: float


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    video_id: str
    created_at: datetime
    source: Literal["classic", "original"]
    title: str
    scenes: list[Scene] = Field(default_factory=list)
    audio: AudioSpec | None = None
    video: VideoSpec | None = None
    stages: dict[str, StageRecord] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = path.read_text(encoding="utf-8")
        obj = cls.model_validate_json(data)
        if obj.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {obj.schema_version}, expected {SCHEMA_VERSION}"
            )
        return obj

    def save(self, path: Path) -> None:
        """Serialize atomically so an interrupted run never corrupts the manifest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def record(self, stage: str) -> StageRecord:
        return self.stages.setdefault(stage, StageRecord())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_manifest.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/manifest.py tests/test_manifest.py
git commit -m "feat: add manifest schema with atomic persistence

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Stable input hashing for stage staleness

**Files:**
- Create: `nursery/hashing.py`
- Test: `tests/test_hashing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `stable_hash(payload: object) -> str` returning a 16-char hex digest that is stable across processes and insensitive to dict ordering.

- [ ] **Step 1: Write the failing test**

`tests/test_hashing.py`:

```python
from nursery.hashing import stable_hash


def test_same_payload_gives_same_hash():
    assert stable_hash({"a": 1, "b": [2, 3]}) == stable_hash({"a": 1, "b": [2, 3]})


def test_key_order_does_not_change_hash():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_different_payload_gives_different_hash():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_list_order_does_change_hash():
    assert stable_hash([1, 2]) != stable_hash([2, 1])


def test_hash_is_short_hex():
    h = stable_hash({"a": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_hashing.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.hashing'`.

- [ ] **Step 3: Write the implementation**

`nursery/hashing.py`:

```python
"""Stable hashing used to detect when a stage's inputs have changed.

Python's built-in hash() is salted per process, so it cannot be persisted.
"""

from __future__ import annotations

import hashlib
import json


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_hashing.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/hashing.py tests/test_hashing.py
git commit -m "feat: add stable input hashing for stage staleness

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Config model

**Files:**
- Create: `nursery/config.py`
- Create: `config.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `VideoConfig(width: int = 1920, height: int = 1080, fps: int = 30, crossfade_s: float = 0.5, zoom_rate: float = 0.0008)`
  - `Config(out_dir: Path, assets_dir: Path, video: VideoConfig, review_gate: bool = True, providers: dict[str, str])`
  - `Config.load(path: Path | None = None) -> Config` — reads YAML if present, else defaults.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
from pathlib import Path

from nursery.config import Config


def test_defaults_when_no_file(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.video.width == 1920
    assert cfg.video.height == 1080
    assert cfg.video.fps == 30
    assert cfg.review_gate is True


def test_yaml_overrides_defaults(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("video:\n  fps: 24\nreview_gate: false\n")

    cfg = Config.load(path)

    assert cfg.video.fps == 24
    assert cfg.video.width == 1920  # untouched default
    assert cfg.review_gate is False


def test_paths_are_path_objects(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert isinstance(cfg.out_dir, Path)
    assert isinstance(cfg.assets_dir, Path)


def test_video_dir_is_derived_from_out_dir(tmp_path: Path):
    cfg = Config.load(tmp_path / "missing.yaml")

    assert cfg.video_dir("abc-01") == cfg.out_dir / "abc-01"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.config'`.

- [ ] **Step 3: Write the implementation**

`nursery/config.py`:

```python
"""Typed pipeline configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crossfade_s: float = 0.5
    zoom_rate: float = 0.0008
    max_zoom: float = 1.18


class Config(BaseModel):
    out_dir: Path = Path("out")
    assets_dir: Path = Path("assets")
    review_gate: bool = True
    video: VideoConfig = Field(default_factory=VideoConfig)
    providers: dict[str, str] = Field(
        default_factory=lambda: {
            "llm": "fake",
            "song": "fake",
            "tts": "fake",
            "image": "fake",
            "align": "fake",
        }
    )

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        if path is not None and path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return cls.model_validate(data)
        return cls()

    def video_dir(self, video_id: str) -> Path:
        return self.out_dir / video_id
```

`config.yaml` at the repo root:

```yaml
# Pipeline configuration. Providers default to "fake" until Plan 2 wires real
# models; see docs/superpowers/specs for the model selection table.
out_dir: out
assets_dir: assets
review_gate: true

video:
  width: 1920
  height: 1080
  fps: 30
  crossfade_s: 0.5
  zoom_rate: 0.0008
  max_zoom: 1.18

providers:
  llm: fake
  song: fake
  tts: fake
  image: fake
  align: fake
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/config.py config.yaml tests/test_config.py
git commit -m "feat: add typed pipeline configuration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Stage protocol and registry

**Files:**
- Create: `nursery/stages/__init__.py`
- Create: `nursery/stages/base.py`
- Test: `tests/test_stage_base.py`

**Interfaces:**
- Consumes: `Manifest`, `StageRecord` (Task 2); `Config` (Task 4); `stable_hash` (Task 3).
- Produces:
  - `class Stage(ABC)` with `name: str`, abstract `input_payload(m, cfg) -> object`, abstract `outputs(m, cfg) -> list[Path]`, abstract `execute(m, cfg) -> Manifest`, and concrete `is_satisfied(m, cfg) -> bool`.
  - `STAGE_ORDER: list[str]` — the canonical stage sequence.
  - `register(stage: Stage) -> None` and `get_stage(name: str) -> Stage`, backed by module-level `_REGISTRY: dict[str, Stage]`.

- [ ] **Step 1: Write the failing test**

`tests/test_stage_base.py`:

```python
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
        "cast", "images", "assemble", "metadata", "publish",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_stage_base.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.stages'`.

- [ ] **Step 3: Write the implementation**

`nursery/stages/__init__.py`: empty file.

`nursery/stages/base.py`:

```python
"""The stage contract every pipeline step implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from nursery.config import Config
from nursery.hashing import stable_hash
from nursery.manifest import Manifest

STAGE_ORDER = [
    "seed",
    "script",
    "audio",
    "align",
    "cast",
    "images",
    "assemble",
    "metadata",
    "publish",
]


class Stage(ABC):
    name: str

    @abstractmethod
    def input_payload(self, m: Manifest, cfg: Config) -> object:
        """The manifest fields this stage consumes, for staleness detection."""

    @abstractmethod
    def outputs(self, m: Manifest, cfg: Config) -> list[Path]:
        """Files this stage is responsible for producing."""

    @abstractmethod
    def execute(self, m: Manifest, cfg: Config) -> Manifest:
        """Do the work. Raise on failure; the orchestrator records it."""

    def current_hash(self, m: Manifest, cfg: Config) -> str:
        return stable_hash(self.input_payload(m, cfg))

    def is_satisfied(self, m: Manifest, cfg: Config) -> bool:
        record = m.stages.get(self.name)
        if record is None or record.status != "ok":
            return False
        if record.input_hash != self.current_hash(m, cfg):
            return False
        return all(p.exists() for p in self.outputs(m, cfg))


_REGISTRY: dict[str, Stage] = {}


def register(stage: Stage) -> None:
    _REGISTRY[stage.name] = stage


def get_stage(name: str) -> Stage:
    if name not in _REGISTRY:
        raise KeyError(f"no stage registered under {name!r}")
    return _REGISTRY[name]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_stage_base.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/stages tests/test_stage_base.py
git commit -m "feat: add stage protocol with staleness detection and registry

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Orchestrator

**Files:**
- Create: `nursery/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `Stage`, `get_stage`, `STAGE_ORDER` (Task 5); `Manifest`, `StageRecord` (Task 2); `Config` (Task 4).
- Produces:
  - `run_pipeline(m: Manifest, cfg: Config, stages: list[str], manifest_path: Path, start_from: str | None = None) -> Manifest`
  - `class StageFailure(Exception)` with attributes `stage: str` and `cause: BaseException`.

- [ ] **Step 1: Write the failing test**

`tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_orchestrator.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.orchestrator'`.

- [ ] **Step 3: Write the implementation**

`nursery/orchestrator.py`:

```python
"""Sequences stages, skips satisfied work, and isolates failures."""

from __future__ import annotations

import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path

from nursery.config import Config
from nursery.manifest import Manifest
from nursery.stages.base import get_stage

log = logging.getLogger(__name__)


class StageFailure(Exception):
    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"stage {stage!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


def run_pipeline(
    m: Manifest,
    cfg: Config,
    stages: list[str],
    manifest_path: Path,
    start_from: str | None = None,
) -> Manifest:
    forced = _forced_stages(stages, start_from)

    for name in stages:
        stage = get_stage(name)

        if name not in forced and stage.is_satisfied(m, cfg):
            log.info("skipping %s (satisfied)", name)
            continue

        record = m.record(name)
        record.status = "running"
        record.started_at = datetime.now(UTC)
        record.error = None
        m.save(manifest_path)

        try:
            m = stage.execute(m, cfg)
        except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
            record = m.record(name)
            record.status = "failed"
            record.error = "".join(traceback.format_exception_only(exc)).strip()
            record.finished_at = datetime.now(UTC)
            m.save(manifest_path)
            raise StageFailure(name, exc) from exc

        record = m.record(name)
        record.status = "ok"
        record.input_hash = stage.current_hash(m, cfg)
        record.finished_at = datetime.now(UTC)
        m.save(manifest_path)
        log.info("completed %s", name)

    return m


def _forced_stages(stages: list[str], start_from: str | None) -> set[str]:
    """Everything from start_from onward reruns even if satisfied."""
    if start_from is None:
        return set()
    if start_from not in stages:
        raise KeyError(f"{start_from!r} is not in the stage list")
    return set(stages[stages.index(start_from) :])
```

> Note on `input_hash`: it is computed *after* `execute`, because a stage may
> populate manifest fields that its own payload reads. Recomputing post-run is
> what makes the second invocation see a matching hash and skip.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_orchestrator.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add orchestrator with resume and failure isolation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Provider protocols and fakes

**Files:**
- Create: `nursery/providers/__init__.py`
- Create: `nursery/providers/base.py`
- Create: `nursery/providers/fake.py`
- Test: `tests/test_fake_providers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `ImageProvider.generate(prompt: str, refs: list[Path], seed: int, out: Path) -> Path`
  - `SongProvider.render(lyrics: str, style: str, out: Path) -> Path`
  - `TTSProvider.speak(text: str, out: Path) -> Path`
  - `LLMProvider.complete(prompt: str, schema: dict | None = None) -> str`
  - `WordTiming(word: str, start_s: float, end_s: float)`
  - `AlignProvider.align(audio: Path, text: str) -> list[WordTiming]`
  - `FakeImageProvider`, `FakeSongProvider`, `FakeTTSProvider`, `FakeLLMProvider`, `FakeAlignProvider`
  - `get_provider(kind: str, name: str) -> object`

- [ ] **Step 1: Write the failing test**

`tests/test_fake_providers.py`:

```python
import wave
from pathlib import Path

from nursery.providers.fake import (
    FakeAlignProvider,
    FakeImageProvider,
    FakeSongProvider,
)


def test_fake_image_writes_a_real_png(tmp_path: Path):
    out = tmp_path / "scene.png"
    FakeImageProvider().generate("a bunny", [], seed=7, out=out)

    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_fake_image_is_deterministic_for_a_seed(tmp_path: Path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    FakeImageProvider().generate("x", [], seed=3, out=a)
    FakeImageProvider().generate("x", [], seed=3, out=b)

    assert a.read_bytes() == b.read_bytes()


def test_fake_image_differs_across_seeds(tmp_path: Path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    FakeImageProvider().generate("x", [], seed=1, out=a)
    FakeImageProvider().generate("x", [], seed=2, out=b)

    assert a.read_bytes() != b.read_bytes()


def test_fake_song_writes_a_readable_wav(tmp_path: Path):
    out = tmp_path / "mix.wav"
    FakeSongProvider(duration_s=2.0).render("la la", "cheerful", out)

    with wave.open(str(out)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 44100
        assert abs(wf.getnframes() / 44100 - 2.0) < 0.01


def test_fake_align_spreads_words_across_duration():
    timings = FakeAlignProvider(duration_s=4.0).align(Path("ignored.wav"), "one two three four")

    assert [t.word for t in timings] == ["one", "two", "three", "four"]
    assert timings[0].start_s == 0.0
    assert abs(timings[-1].end_s - 4.0) < 1e-6


def test_fake_align_timings_are_contiguous_and_increasing():
    timings = FakeAlignProvider(duration_s=3.0).align(Path("x.wav"), "a b c")

    for earlier, later in zip(timings, timings[1:], strict=False):
        assert earlier.end_s == later.start_s
        assert later.start_s < later.end_s
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_fake_providers.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.providers'`.

- [ ] **Step 3: Write the implementation**

`nursery/providers/__init__.py`: empty file.

`nursery/providers/base.py`:

```python
"""Narrow interfaces between stages and models.

Stages depend only on these protocols, never on a model library. That is what
lets the pipeline run in CI with fakes and swap backends in one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_s: float
    end_s: float


@runtime_checkable
class ImageProvider(Protocol):
    def generate(self, prompt: str, refs: list[Path], seed: int, out: Path) -> Path: ...


@runtime_checkable
class SongProvider(Protocol):
    def render(self, lyrics: str, style: str, out: Path) -> Path: ...


@runtime_checkable
class TTSProvider(Protocol):
    def speak(self, text: str, out: Path) -> Path: ...


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, prompt: str, schema: dict | None = None) -> str: ...


@runtime_checkable
class AlignProvider(Protocol):
    def align(self, audio: Path, text: str) -> list[WordTiming]: ...
```

`nursery/providers/fake.py`:

```python
"""Deterministic fake providers.

These produce real, valid files - a decodable PNG, a readable WAV - so that
downstream stages including ffmpeg exercise their true code paths without a
model or a GPU.
"""

from __future__ import annotations

import math
import struct
import wave
import zlib
from pathlib import Path

from nursery.providers.base import WordTiming

_SAMPLE_RATE = 44100


def _png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


class FakeImageProvider:
    """Writes a solid-colour PNG whose colour is a pure function of the seed."""

    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height

    def generate(self, prompt: str, refs: list[Path], seed: int, out: Path) -> Path:
        rgb = ((seed * 53) % 256, (seed * 97) % 256, (seed * 151) % 256)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_png_bytes(self.width, self.height, rgb))
        return out


def _write_tone(out: Path, duration_s: float, freq: float) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = int(_SAMPLE_RATE * duration_s)
    with wave.open(str(out), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(
            b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * n / _SAMPLE_RATE)))
                for n in range(frames)
            )
        )
    return out


class FakeSongProvider:
    def __init__(self, duration_s: float = 8.0):
        self.duration_s = duration_s

    def render(self, lyrics: str, style: str, out: Path) -> Path:
        return _write_tone(out, self.duration_s, freq=440.0)


class FakeTTSProvider:
    def __init__(self, duration_s: float = 8.0):
        self.duration_s = duration_s

    def speak(self, text: str, out: Path) -> Path:
        return _write_tone(out, self.duration_s, freq=220.0)


class FakeLLMProvider:
    def __init__(self, response: str = "{}"):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str, schema: dict | None = None) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeAlignProvider:
    """Spreads words evenly across the duration - contiguous, increasing."""

    def __init__(self, duration_s: float = 8.0):
        self.duration_s = duration_s

    def align(self, audio: Path, text: str) -> list[WordTiming]:
        words = text.split()
        if not words:
            return []
        step = self.duration_s / len(words)
        return [
            WordTiming(word=w, start_s=i * step, end_s=(i + 1) * step)
            for i, w in enumerate(words)
        ]


_FAKES = {
    "image": FakeImageProvider,
    "song": FakeSongProvider,
    "tts": FakeTTSProvider,
    "llm": FakeLLMProvider,
    "align": FakeAlignProvider,
}


def get_provider(kind: str, name: str) -> object:
    if name != "fake":
        raise KeyError(f"provider {name!r} for {kind!r} is not available until Plan 2")
    return _FAKES[kind]()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_fake_providers.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/providers tests/test_fake_providers.py
git commit -m "feat: add provider protocols and deterministic fakes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Ken Burns filter-graph builder

**Files:**
- Create: `nursery/video/__init__.py`
- Create: `nursery/video/kenburns.py`
- Test: `tests/test_kenburns.py`

**Interfaces:**
- Consumes: `VideoConfig` (Task 4).
- Produces:
  - `PanDirection = Literal["in", "out", "left", "right"]`
  - `SceneClip(image: Path, duration_s: float, direction: PanDirection)`
  - `build_filter_graph(clips: list[SceneClip], cfg: VideoConfig, subtitles: Path | None) -> str`
  - `direction_for_index(i: int) -> PanDirection` — cycles so adjacent scenes never repeat a move.

**Why this comes before any generative stage:** it is pure and instant to test,
and it turns "what should the image stage output?" into an existing contract.

- [ ] **Step 1: Write the failing test**

`tests/test_kenburns.py`:

```python
from pathlib import Path

import pytest

from nursery.config import VideoConfig
from nursery.video.kenburns import SceneClip, build_filter_graph, direction_for_index


def clips(n: int) -> list[SceneClip]:
    return [
        SceneClip(image=Path(f"/tmp/s{i}.png"), duration_s=3.0, direction=direction_for_index(i))
        for i in range(n)
    ]


def test_directions_cycle_without_adjacent_repeats():
    seq = [direction_for_index(i) for i in range(8)]
    for a, b in zip(seq, seq[1:], strict=False):
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
        [SceneClip(image=Path("/tmp/a.png"), duration_s=2.0, direction="in")], cfg, None
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_kenburns.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.video'`.

- [ ] **Step 3: Write the implementation**

`nursery/video/__init__.py`: empty file.

`nursery/video/kenburns.py`:

```python
"""Builds the ffmpeg filter graph that turns stills into a moving video.

zoompan quantises its zoom factor per frame, which makes a 1080p source visibly
judder. Upscaling the source first pushes the quantisation below one output
pixel, which is the standard fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PanDirection = Literal["in", "out", "left", "right"]

_CYCLE: tuple[PanDirection, ...] = ("in", "left", "out", "right")
_UPSCALE = 4


@dataclass(frozen=True)
class SceneClip:
    image: Path
    duration_s: float
    direction: PanDirection


def direction_for_index(i: int) -> PanDirection:
    return _CYCLE[i % len(_CYCLE)]


def _zoom_expr(direction: PanDirection, cfg, frames: int) -> tuple[str, str, str]:
    """Return (zoom, x, y) expressions for the given move."""
    zmax = cfg.max_zoom
    rate = cfg.zoom_rate
    centre_x = "iw/2-(iw/zoom/2)"
    centre_y = "ih/2-(ih/zoom/2)"

    if direction == "in":
        return f"min(zoom+{rate},{zmax})", centre_x, centre_y
    if direction == "out":
        return f"max({zmax}-on*{rate},1.0)", centre_x, centre_y
    if direction == "left":
        return f"{zmax}", f"(iw-iw/zoom)*(1-on/{frames})", centre_y
    return f"{zmax}", f"(iw-iw/zoom)*(on/{frames})", centre_y


def build_filter_graph(clips: list[SceneClip], cfg, subtitles: Path | None) -> str:
    if not clips:
        raise ValueError("need at least one clip to build a filter graph")

    parts: list[str] = []

    for i, clip in enumerate(clips):
        frames = max(1, int(round(clip.duration_s * cfg.fps)))
        zoom, x, y = _zoom_expr(clip.direction, cfg, frames)
        parts.append(
            f"[{i}:v]"
            f"scale={cfg.width * _UPSCALE}:{cfg.height * _UPSCALE},"
            f"setsar=1,"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}"
            f":s={cfg.width}x{cfg.height}:fps={cfg.fps},"
            f"format=yuv420p"
            f"[v{i}]"
        )

    last = "v0"
    offset = clips[0].duration_s - cfg.crossfade_s
    for i in range(1, len(clips)):
        label = f"x{i}"
        parts.append(
            f"[{last}][v{i}]"
            f"xfade=transition=fade:duration={cfg.crossfade_s}:offset={offset:.3f}"
            f"[{label}]"
        )
        last = label
        offset += clips[i].duration_s - cfg.crossfade_s

    if subtitles is not None:
        parts.append(f"[{last}]subtitles='{subtitles}'[vout]")
    else:
        parts.append(f"[{last}]null[vout]")

    return ";".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_kenburns.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/video tests/test_kenburns.py
git commit -m "feat: add Ken Burns ffmpeg filter graph builder

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: ASS karaoke caption generation

**Files:**
- Create: `nursery/video/captions.py`
- Test: `tests/test_captions.py`

**Interfaces:**
- Consumes: `WordTiming` (Task 7).
- Produces:
  - `CaptionLine(words: list[WordTiming])` with `start_s` / `end_s` properties
  - `group_words_into_lines(timings: list[WordTiming], lyric_lines: list[str]) -> list[CaptionLine]`
  - `build_ass(lines: list[CaptionLine], width: int, height: int) -> str`
  - `write_ass(lines: list[CaptionLine], width: int, height: int, out: Path) -> Path`

**Why ASS:** the `\k` karaoke tag highlights words in time natively. Doing this
with `drawtext` would mean one filter per word and an unmanageable graph.

- [ ] **Step 1: Write the failing test**

`tests/test_captions.py`:

```python
import pytest

from nursery.providers.base import WordTiming
from nursery.video.captions import (
    CaptionLine,
    build_ass,
    group_words_into_lines,
    write_ass,
)


def timings(*pairs) -> list[WordTiming]:
    return [WordTiming(word=w, start_s=s, end_s=e) for w, s, e in pairs]


def test_groups_words_to_matching_lyric_lines():
    words = timings(("twinkle", 0.0, 0.5), ("little", 0.5, 1.0), ("star", 1.0, 1.5))

    lines = group_words_into_lines(words, ["twinkle little", "star"])

    assert [w.word for w in lines[0].words] == ["twinkle", "little"]
    assert [w.word for w in lines[1].words] == ["star"]


def test_line_span_comes_from_its_words():
    line = CaptionLine(words=timings(("a", 1.0, 1.5), ("b", 1.5, 2.25)))

    assert line.start_s == 1.0
    assert line.end_s == 2.25


def test_mismatched_word_count_raises():
    with pytest.raises(ValueError, match="word count"):
        group_words_into_lines(timings(("a", 0.0, 1.0)), ["a b c"])


def test_ass_has_required_sections():
    ass = build_ass([CaptionLine(words=timings(("hi", 0.0, 1.0)))], 1920, 1080)

    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass


def test_ass_declares_the_video_resolution():
    ass = build_ass([CaptionLine(words=timings(("hi", 0.0, 1.0)))], 1920, 1080)

    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass


def test_karaoke_durations_are_centiseconds():
    # 0.75s -> \k75
    ass = build_ass([CaptionLine(words=timings(("hi", 0.0, 0.75)))], 1920, 1080)

    assert r"{\k75}hi" in ass


def test_dialogue_timestamps_are_ass_formatted():
    ass = build_ass([CaptionLine(words=timings(("hi", 61.5, 62.0)))], 1920, 1080)

    assert "0:01:01.50" in ass


def test_each_line_becomes_one_dialogue_event():
    lines = [
        CaptionLine(words=timings(("a", 0.0, 1.0))),
        CaptionLine(words=timings(("b", 1.0, 2.0))),
    ]

    assert build_ass(lines, 1920, 1080).count("Dialogue:") == 2


def test_write_ass_creates_utf8_file(tmp_path):
    out = write_ass([CaptionLine(words=timings(("hi", 0.0, 1.0)))], 1920, 1080,
                    tmp_path / "c.ass")

    assert out.exists()
    assert "[Events]" in out.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_captions.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.video.captions'`.

- [ ] **Step 3: Write the implementation**

`nursery/video/captions.py`:

```python
"""Generates ASS subtitles with karaoke (\\k) timing from word timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nursery.providers.base import WordTiming

_STYLE = (
    "Style: Karaoke,Arial Rounded MT Bold,84,"
    "&H00FFFFFF,&H0000D7FF,&H00202020,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,6,2,2,120,120,90,1"
)


@dataclass
class CaptionLine:
    words: list[WordTiming]

    @property
    def start_s(self) -> float:
        return self.words[0].start_s

    @property
    def end_s(self) -> float:
        return self.words[-1].end_s

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def group_words_into_lines(
    timings: list[WordTiming], lyric_lines: list[str]
) -> list[CaptionLine]:
    expected = sum(len(line.split()) for line in lyric_lines)
    if expected != len(timings):
        raise ValueError(
            f"word count mismatch: lyrics have {expected}, alignment has {len(timings)}"
        )

    lines: list[CaptionLine] = []
    cursor = 0
    for line in lyric_lines:
        n = len(line.split())
        lines.append(CaptionLine(words=timings[cursor : cursor + n]))
        cursor += n
    return lines


def _ass_time(seconds: float) -> str:
    hours, rem = divmod(max(0.0, seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


def build_ass(lines: list[CaptionLine], width: int, height: int) -> str:
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        _STYLE,
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    events = []
    for line in lines:
        karaoke = "".join(
            f"{{\\k{int(round((w.end_s - w.start_s) * 100))}}}{w.word} " for w in line.words
        ).strip()
        events.append(
            f"Dialogue: 0,{_ass_time(line.start_s)},{_ass_time(line.end_s)},"
            f"Karaoke,,0,0,0,,{karaoke}"
        )

    return "\n".join(header + events) + "\n"


def write_ass(lines: list[CaptionLine], width: int, height: int, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_ass(lines, width, height), encoding="utf-8")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_captions.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/video/captions.py tests/test_captions.py
git commit -m "feat: add ASS karaoke caption generation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: ffmpeg invocation

**Files:**
- Create: `nursery/video/ffmpeg.py`
- Test: `tests/test_ffmpeg.py`

**Interfaces:**
- Consumes: `SceneClip` (Task 8), `VideoConfig` (Task 4).
- Produces:
  - `class FFmpegError(RuntimeError)` carrying `.stderr`
  - `build_command(clips, audio: Path, graph: str, out: Path, cfg) -> list[str]`
  - `run_ffmpeg(cmd: list[str]) -> None`
  - `extract_thumbnail(video: Path, at_s: float, out: Path) -> Path`

- [ ] **Step 1: Write the failing test**

`tests/test_ffmpeg.py`:

```python
from pathlib import Path

import pytest

from nursery.config import VideoConfig
from nursery.video.ffmpeg import FFmpegError, build_command, run_ffmpeg
from nursery.video.kenburns import SceneClip


def clips(n: int) -> list[SceneClip]:
    return [SceneClip(image=Path(f"/tmp/s{i}.png"), duration_s=2.0, direction="in")
            for i in range(n)]


def test_each_image_becomes_a_looped_input():
    cmd = build_command(clips(3), Path("/tmp/a.wav"), "graph", Path("/tmp/o.mp4"),
                        VideoConfig())

    assert cmd.count("-loop") == 3
    assert cmd.count("-i") == 4  # three images plus the audio


def test_audio_is_the_last_input():
    cmd = build_command(clips(2), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())
    idx = cmd.index("/tmp/a.wav")

    assert cmd[idx - 1] == "-i"
    assert "/tmp/s1.png" in cmd[: idx - 1]


def test_maps_graph_output_and_audio_stream():
    cmd = build_command(clips(2), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())

    assert "[vout]" in cmd
    assert "2:a" in cmd


def test_encodes_h264_yuv420p_and_aac():
    cmd = build_command(clips(1), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())

    assert "libx264" in cmd
    assert "yuv420p" in cmd
    assert "aac" in cmd


def test_uses_shortest_so_video_matches_audio():
    cmd = build_command(clips(1), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())

    assert "-shortest" in cmd


def test_overwrites_without_prompting():
    cmd = build_command(clips(1), Path("/tmp/a.wav"), "g", Path("/tmp/o.mp4"), VideoConfig())

    assert "-y" in cmd


def test_run_ffmpeg_raises_with_stderr_on_failure():
    with pytest.raises(FFmpegError) as exc:
        run_ffmpeg(["ffmpeg", "-i", "/nonexistent/nope.mp4", "-f", "null", "-"])

    assert exc.value.stderr
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_ffmpeg.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.video.ffmpeg'`.

- [ ] **Step 3: Write the implementation**

`nursery/video/ffmpeg.py`:

```python
"""Thin wrapper around the ffmpeg binary."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from nursery.video.kenburns import SceneClip

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    def __init__(self, message: str, stderr: str):
        super().__init__(message)
        self.stderr = stderr


def _binary() -> str:
    found = shutil.which("ffmpeg")
    if found is None:
        raise FFmpegError("ffmpeg not found on PATH", "")
    return found


def build_command(
    clips: list[SceneClip], audio: Path, graph: str, out: Path, cfg
) -> list[str]:
    cmd = [_binary(), "-y"]

    for clip in clips:
        cmd += ["-loop", "1", "-t", f"{clip.duration_s:.3f}", "-i", str(clip.image)]

    cmd += ["-i", str(audio)]
    cmd += [
        "-filter_complex", graph,
        "-map", "[vout]",
        "-map", f"{len(clips)}:a",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(cfg.fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(out),
    ]
    return cmd


def run_ffmpeg(cmd: list[str]) -> None:
    log.debug("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
        raise FFmpegError(f"ffmpeg exited {proc.returncode}", tail)


def extract_thumbnail(video: Path, at_s: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        _binary(), "-y", "-ss", f"{at_s:.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(out),
    ])
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ffmpeg.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add nursery/video/ffmpeg.py tests/test_ffmpeg.py
git commit -m "feat: add ffmpeg command builder and runner

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: The assemble stage, end to end

**Files:**
- Create: `nursery/stages/assemble.py`
- Create: `tests/conftest.py`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Consumes: everything from Tasks 2–10.
- Produces: `AssembleStage()` registered under `"assemble"`, setting
  `manifest.video = VideoSpec(...)`.

This task produces the plan's headline deliverable: a real, playable MP4.

- [ ] **Step 1: Write the shared fixture**

`tests/conftest.py`:

```python
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nursery.config import Config
from nursery.manifest import AudioSpec, Manifest, Scene
from nursery.providers.fake import FakeImageProvider, FakeSongProvider


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(out_dir=tmp_path / "out", assets_dir=tmp_path / "assets")


@pytest.fixture
def ready_manifest(cfg: Config) -> Manifest:
    """A manifest with real images, real audio, and timings already filled in."""
    video_dir = cfg.video_dir("v1")
    images = FakeImageProvider(width=1280, height=720)

    scenes = []
    for i, text in enumerate(["twinkle twinkle little star", "how i wonder what you are"]):
        path = video_dir / "images" / f"scene_{i:02d}.png"
        images.generate(text, [], seed=i, out=path)
        scenes.append(
            Scene(index=i, text=text, visual_prompt=text, characters=[],
                  image_path=str(path), start_s=i * 2.0, end_s=(i + 1) * 2.0)
        )

    audio_path = video_dir / "audio" / "mix.wav"
    FakeSongProvider(duration_s=4.0).render("la", "cheerful", audio_path)

    return Manifest(
        video_id="v1", created_at=datetime(2026, 8, 8, tzinfo=UTC),
        source="classic", title="Twinkle", scenes=scenes,
        audio=AudioSpec(mix_path=str(audio_path), duration_s=4.0,
                        provider="fake", degraded=False),
    )
```

- [ ] **Step 2: Write the failing test**

`tests/test_assemble.py`:

```python
import subprocess
from pathlib import Path

import pytest

from nursery.manifest import StageRecord
from nursery.stages.assemble import AssembleStage


def probe(path: Path, stream: str, entries: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", entries, "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_produces_a_playable_mp4(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    final = Path(result.video.final_path)
    assert final.exists()
    assert final.stat().st_size > 10_000


def test_video_has_expected_resolution(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    assert probe(Path(result.video.final_path), "v:0", "stream=width,height") == "1920,1080"


def test_video_has_an_audio_stream(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    assert probe(Path(result.video.final_path), "a:0", "stream=codec_name") == "aac"


def test_duration_roughly_matches_audio(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    duration = float(probe(Path(result.video.final_path), "v:0", "format=duration"))
    assert 3.0 < duration < 5.5


def test_thumbnail_is_written(ready_manifest, cfg):
    result = AssembleStage().execute(ready_manifest, cfg)

    assert Path(result.video.thumbnail_path).exists()


def test_scene_without_timing_raises(ready_manifest, cfg):
    ready_manifest.scenes[0].start_s = None

    with pytest.raises(ValueError, match="timing"):
        AssembleStage().execute(ready_manifest, cfg)


def test_scene_without_image_raises(ready_manifest, cfg):
    ready_manifest.scenes[1].image_path = None

    with pytest.raises(ValueError, match="image"):
        AssembleStage().execute(ready_manifest, cfg)


def test_is_satisfied_after_a_successful_run(ready_manifest, cfg):
    stage = AssembleStage()
    m = stage.execute(ready_manifest, cfg)
    m.stages["assemble"] = StageRecord(status="ok", input_hash=stage.current_hash(m, cfg))

    assert stage.is_satisfied(m, cfg) is True


def test_changing_a_scene_image_invalidates_satisfaction(ready_manifest, cfg):
    stage = AssembleStage()
    m = stage.execute(ready_manifest, cfg)
    m.stages["assemble"] = StageRecord(status="ok", input_hash=stage.current_hash(m, cfg))

    m.scenes[0].image_path = "/tmp/different.png"

    assert stage.is_satisfied(m, cfg) is False
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_assemble.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.stages.assemble'`.

- [ ] **Step 4: Write the implementation**

`nursery/stages/assemble.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_assemble.py -v
```

Expected: 9 passed. If `ffprobe` is missing, install it with `brew install ffmpeg`.

- [ ] **Step 6: Watch the output yourself**

The tests assert structure; look at one to confirm the motion and captions
actually read well:

```bash
uv run pytest tests/test_assemble.py::test_produces_a_playable_mp4 -v -s
```

Then generate a keepable sample and open it:

```bash
uv run python -c "
from datetime import UTC, datetime
from pathlib import Path
from nursery.config import Config
from nursery.manifest import AudioSpec, Manifest, Scene
from nursery.providers.fake import FakeImageProvider, FakeSongProvider
from nursery.stages.assemble import AssembleStage

cfg = Config(out_dir=Path('out'))
d = cfg.video_dir('sample')
scenes = []
for i, t in enumerate(['twinkle twinkle little star', 'how i wonder what you are']):
    p = d / 'images' / f'scene_{i:02d}.png'
    FakeImageProvider().generate(t, [], seed=i, out=p)
    scenes.append(Scene(index=i, text=t, visual_prompt=t, image_path=str(p),
                        start_s=i*3.0, end_s=(i+1)*3.0))
a = FakeSongProvider(duration_s=6.0).render('la', 'x', d / 'audio' / 'mix.wav')
m = Manifest(video_id='sample', created_at=datetime.now(UTC), source='classic',
             title='Sample', scenes=scenes,
             audio=AudioSpec(mix_path=str(a), duration_s=6.0, provider='fake'))
print(AssembleStage().execute(m, cfg).video.final_path)
"
open out/sample/video/final.mp4
```

Confirm: motion is smooth with no juddering, the crossfade is clean, and
captions highlight word by word.

- [ ] **Step 7: Commit**

```bash
git add nursery/stages/assemble.py tests/conftest.py tests/test_assemble.py
git commit -m "feat: add assemble stage producing final MP4 with captions

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: CLI

**Files:**
- Create: `nursery/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Config` (4), `Manifest` (2), `run_pipeline` (6), `STAGE_ORDER` (5).
- Produces: `app` (a `typer.Typer`) with commands `list`, `show`, `stage`, `clean`.

`run` is deliberately not implemented here — it needs the generative stages from
Plan 2. `stage` runs any single registered stage, which is enough to drive
`assemble` today.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
from typer.testing import CliRunner

from nursery.cli import app

runner = CliRunner()


def test_list_reports_nothing_when_out_dir_empty(tmp_path):
    result = runner.invoke(app, ["--out-dir", str(tmp_path), "list"])

    assert result.exit_code == 0
    assert "no videos" in result.stdout.lower()


def test_list_shows_video_ids(tmp_path, ready_manifest, cfg):
    ready_manifest.save(cfg.video_dir("v1") / "manifest.json")

    result = runner.invoke(app, ["--out-dir", str(cfg.out_dir), "list"])

    assert result.exit_code == 0
    assert "v1" in result.stdout


def test_show_prints_title_and_scene_count(ready_manifest, cfg):
    ready_manifest.save(cfg.video_dir("v1") / "manifest.json")

    result = runner.invoke(app, ["--out-dir", str(cfg.out_dir), "show", "v1"])

    assert result.exit_code == 0
    assert "Twinkle" in result.stdout
    assert "2" in result.stdout


def test_show_unknown_id_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["--out-dir", str(tmp_path), "show", "nope"])

    assert result.exit_code != 0


def test_stage_runs_assemble(ready_manifest, cfg):
    ready_manifest.save(cfg.video_dir("v1") / "manifest.json")

    result = runner.invoke(app, ["--out-dir", str(cfg.out_dir), "stage", "v1", "assemble"])

    assert result.exit_code == 0
    assert (cfg.video_dir("v1") / "video" / "final.mp4").exists()


def test_clean_removes_intermediates_but_keeps_final(ready_manifest, cfg):
    ready_manifest.save(cfg.video_dir("v1") / "manifest.json")
    runner.invoke(app, ["--out-dir", str(cfg.out_dir), "stage", "v1", "assemble"])

    result = runner.invoke(app, ["--out-dir", str(cfg.out_dir), "clean", "v1"])

    assert result.exit_code == 0
    assert (cfg.video_dir("v1") / "video" / "final.mp4").exists()
    assert (cfg.video_dir("v1") / "manifest.json").exists()
    assert not (cfg.video_dir("v1") / "images").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nursery.cli'`.

- [ ] **Step 3: Write the implementation**

`nursery/cli.py`:

```python
"""Command line entry point."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import typer

import nursery.stages.assemble  # noqa: F401 - import registers the stage
from nursery.config import Config
from nursery.manifest import Manifest
from nursery.orchestrator import StageFailure, run_pipeline

app = typer.Typer(help="Local nursery rhyme video generation pipeline.")
_state: dict[str, Config] = {}

_INTERMEDIATES = ("images", "audio", "align")


@app.callback()
def main(
    out_dir: Path = typer.Option(Path("out"), "--out-dir"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load(config)
    cfg.out_dir = out_dir
    _state["cfg"] = cfg


def _cfg() -> Config:
    return _state["cfg"]


def _load(video_id: str) -> tuple[Manifest, Path]:
    path = _cfg().video_dir(video_id) / "manifest.json"
    if not path.exists():
        typer.echo(f"no manifest for {video_id!r}", err=True)
        raise typer.Exit(1)
    return Manifest.load(path), path


@app.command("list")
def list_videos() -> None:
    """List every video in the output directory."""
    out = _cfg().out_dir
    manifests = sorted(out.glob("*/manifest.json")) if out.exists() else []
    if not manifests:
        typer.echo("no videos found")
        return
    for path in manifests:
        m = Manifest.load(path)
        done = sum(1 for r in m.stages.values() if r.status == "ok")
        flag = " [degraded audio]" if m.audio and m.audio.degraded else ""
        typer.echo(f"{m.video_id}  {m.title}  ({done} stages ok){flag}")


@app.command()
def show(video_id: str) -> None:
    """Print a manifest summary."""
    m, _ = _load(video_id)
    typer.echo(f"{m.video_id}: {m.title} [{m.source}]")
    typer.echo(f"scenes: {len(m.scenes)}")
    for name, record in m.stages.items():
        typer.echo(f"  {name}: {record.status}{' - ' + record.error if record.error else ''}")


@app.command()
def stage(video_id: str, name: str) -> None:
    """Run one stage for one video."""
    m, path = _load(video_id)
    try:
        run_pipeline(m, _cfg(), [name], path, start_from=name)
    except StageFailure as exc:
        typer.echo(f"{name} failed: {exc.cause}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{name} ok")


@app.command()
def clean(video_id: str) -> None:
    """Delete intermediates, keeping the manifest and the finished video."""
    base = _cfg().video_dir(video_id)
    for sub in _INTERMEDIATES:
        shutil.rmtree(base / sub, ignore_errors=True)
    typer.echo(f"cleaned {video_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run the whole suite**

```bash
uv run pytest -v && uv run ruff check .
```

Expected: all tests pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add nursery/cli.py tests/test_cli.py
git commit -m "feat: add CLI with list, show, stage, and clean commands

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Definition of done

- `uv run pytest` passes with no network access and no models installed.
- `uv run ruff check .` is clean.
- A real, playable 1920x1080 MP4 exists with Ken Burns motion, a clean
  crossfade, burned-in karaoke captions, and an AAC audio track.
- Re-running a satisfied stage skips it; changing an input re-runs it.
- A failing stage records its error in the manifest and halts the run.

## What Plan 2 adds

The `seed`, `script`, `audio`, `align`, `cast`, `images`, `metadata`, and
`publish` stages, plus real provider backends for the models selected in §10 of
the spec, the rhyme catalogs, the review gate, and batch mode.
