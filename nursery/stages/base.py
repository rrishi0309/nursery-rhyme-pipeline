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
