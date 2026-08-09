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
    out_dir: Path = typer.Option(Path("out"), "--out-dir"),  # noqa: B008
    config: Path = typer.Option(Path("config.yaml"), "--config"),  # noqa: B008
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
