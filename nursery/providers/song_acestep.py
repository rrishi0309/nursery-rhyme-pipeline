"""Song generation via ACE-Step 1.5 (MLX-native, Apple Silicon).

ACE-Step sings the full set of lyric lines into one continuous track; the
`align` stage (`nursery/providers/align_whisperx.py`) later recovers
per-line timings from that audio to drive scene boundaries.

`cli.py --help` exposes only `-c/--config`, `--configure`, `--backend`, and
`--log-level` - **there are no `--prompt`/`--lyrics`/`--output`/`--seed`
flags**. Per `docs/en/CLI.md` in the cloned repo (see
`docs/tool-help/acestep.txt`), generation is wizard-or-config-file only:
`cli.py --config <file.toml>`. This provider writes that TOML file rather
than building a flag-based command line.

`cli.py::main()` loads the file with:

    for key, value in config_from_file.items():
        setattr(args, key, value)

flat, with no section support and no key validation. A `[section]` header in
the TOML would set one attribute literally named "section" and leave every
real key - `caption`, `lyrics`, `seed`, everything - at cli.py's built-in
default. No error, no warning, just a song generated with the wrong
settings. A misspelled key is silently ignored the same way. So the TOML
this provider writes MUST be a flat top-level table, and `_dumps_flat`
enforces that twice: once by rejecting any dict-valued key before
serialising, and once by round-tripping its own output through `tomllib`
and confirming nothing came back as a dict.

Runs under ACE-Step's own venv (`python`/`repo_dir`), not the project's -
ACE-Step 1.5 requires Python <3.13 (project venv is 3.14).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)


class SongError(RuntimeError):
    pass


def _toml_scalar(value: bool | float | str) -> str:
    """Render one Python scalar as a TOML literal.

    `bool` is checked before `int` because `bool` is an `int` subclass in
    Python - `isinstance(True, int)` is `True`, so the int branch would
    otherwise render `True` as `1`.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # TOML basic-string escaping (\", \\, \n, \t, \uXXXX, ...) is a
        # superset of what json.dumps emits for a Python str (and Python's
        # json module never emits the \/ escape), so this always produces a
        # valid single-line TOML basic string - including for multi-line
        # lyrics, via the \n escape.
        return json.dumps(value)
    raise TypeError(f"unsupported TOML value type for {value!r}: {type(value)!r}")


def _dumps_flat(config: dict[str, bool | int | float | str]) -> str:
    """Serialise `config` as a flat TOML table with no `[section]` headers.

    See the module docstring for why a nested value here would be a silent,
    very expensive to diagnose bug. Verified two ways: reject any dict value
    up front, then round-trip the rendered text back through `tomllib` and
    confirm every top-level value parsed back as a non-dict.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            raise TypeError(
                f"ACE-Step config key {key!r} is a dict - this would render as a "
                "[section] header and silently reset every real key to its "
                "cli.py default instead of erroring. Flatten it before calling "
                "_dumps_flat."
            )

    text = "\n".join(f"{key} = {_toml_scalar(value)}" for key, value in config.items()) + "\n"

    parsed = tomllib.loads(text)
    if any(isinstance(v, dict) for v in parsed.values()):
        raise TypeError(
            "generated ACE-Step TOML did not round-trip as a flat table - refusing "
            "to write it. This should be unreachable given the check above; if it "
            "fires, _toml_scalar produced something tomllib parses as a table."
        )
    return text


class AceStepSongProvider:
    """Sings a full set of lyric lines into one continuous track."""

    name = "acestep"

    def __init__(
        self,
        python: str,
        repo_dir: str,
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        audio_format: str = "wav",
        duration: float = -1.0,
        backend: str = "mlx",
        variant: str | None = None,
    ):
        # ACE-Step 1.5 is not on PATH and is not runnable from the project's
        # own Python 3.14 venv (it requires <3.13) - both must be supplied,
        # there is no PATH-based default to fall back to.
        self.python = python
        self.repo_dir = repo_dir
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.audio_format = audio_format
        self.duration = duration
        self.backend = backend
        # Accepted so this constructor can be driven straight off
        # `SongConfig` (which carries a `variant` field, e.g.
        # "ACE-Step/acestep-v15-xl-turbo"), but deliberately unused here:
        # reading cli.py's source shows the DiT-variant selector is a
        # *different* TOML key (`config_path`) that expects a bare
        # SUBMODEL_REGISTRY name (e.g. "acestep-v15-xl-turbo"), not the
        # "org/name"-shaped HF repo id `variant` stores. Wiring the two
        # together needs a translation step this task's brief didn't
        # specify a source for, so it is left for whoever picks that up
        # rather than guessed at here - guessing wrong would be exactly the
        # kind of silent-wrong-config failure this file's docstring warns
        # about elsewhere.
        self.variant = variant

    def generate(self, lyrics: list[str], style: str, seed: int, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        save_dir = tempfile.mkdtemp(prefix="acestep-out-")

        config: dict[str, bool | int | float | str] = {
            "save_dir": save_dir,
            "audio_format": self.audio_format,
            "caption": style,
            "lyrics": "\n".join(lyrics),
            "duration": self.duration,
            "instrumental": False,
            "task_type": "text2music",
            "inference_steps": self.inference_steps,
            "seed": seed,
            "guidance_scale": self.guidance_scale,
            "backend": self.backend,
        }
        toml_text = _dumps_flat(config)

        with tempfile.NamedTemporaryFile(
            "w", suffix=".toml", prefix="acestep-config-", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(toml_text)
            config_path = fh.name

        try:
            cmd = [self.python, "cli.py", "--config", config_path]
            log.info("generating song -> %s (seed=%d, %d lines)", out.name, seed, len(lyrics))
            try:
                proc = subprocess.run(
                    cmd, cwd=self.repo_dir, capture_output=True, text=True, check=False
                )
            except FileNotFoundError as exc:
                raise SongError(
                    f"could not run {self.python!r} in {self.repo_dir!r}: {exc}"
                ) from exc

            if proc.returncode != 0:
                tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
                raise SongError(f"ACE-Step exited {proc.returncode}:\n{tail}")

            produced = sorted(
                Path(save_dir).glob(f"*.{self.audio_format}"),
                key=lambda p: p.stat().st_mtime,
            )
            if not produced:
                tail = "\n".join(proc.stdout.strip().splitlines()[-25:])
                raise SongError(
                    f"ACE-Step reported success but wrote no .{self.audio_format} file "
                    f"to {save_dir}:\n{tail}"
                )
            # batch_size defaults to 1, so this is normally exactly one file;
            # if more turn up, take the most recently written one.
            shutil.move(str(produced[-1]), str(out))
            return out
        finally:
            Path(config_path).unlink(missing_ok=True)
            shutil.rmtree(save_dir, ignore_errors=True)
