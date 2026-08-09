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
        except BaseException as exc:  # recorded then re-raised
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
