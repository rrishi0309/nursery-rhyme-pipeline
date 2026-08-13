"""Lyrics + cast + scene descriptions via Qwen (MLX-native, Apple Silicon).

Rendering shells out to the `mlx_lm.generate` console script, the same
shell-out pattern this codebase already uses for `mflux-generate`
(`nursery/providers/image_flux.py`) and `ffmpeg`, rather than an in-process
model API. `mlx_lm.generate` has no `--output` flag - generated text is
stdout-only - so the model is asked for strict JSON and the first balanced
`{...}` block in stdout is extracted and parsed.

Meter and rhyme are gated with `nursery.lyrics.validate.check`. A violation is
not fatal: the violation text is appended to the prompt and the model gets
another attempt, up to `max_retries` times, before `LyricsError` is raised.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess

from nursery.lyrics import validate

log = logging.getLogger(__name__)

# How many syllables off `syllables` a line may be before it counts as a
# meter violation. The heuristic in validate.py is approximate, so a single
# syllable of slack absorbs its common misses without accepting garbage.
SYLLABLE_TOLERANCE = 1

SYSTEM_PROMPT = (
    "You are a children's nursery rhyme lyricist and storyboard artist. "
    "You always respond with a single strict JSON object and nothing else - "
    "no markdown fences, no commentary before or after it."
)

PROMPT_TEMPLATE = """Write an original nursery rhyme inspired by: {slug_hint}

Respond with ONLY a single JSON object shaped exactly like this:
{{
  "title": "<short title>",
  "style": "<one-line music style/genre description, for an audio model>",
  "cast": [
    {{"name": "<character name>", "description": "<short visual description, for an illustrator>"}}
  ],
  "lines": [
    {{"text": "<one lyric line>", "visual": "<short scene description, for an illustrator>"}}
  ]
}}

Requirements for "lines":
- exactly {target_lines} entries
- each "text" should have close to {syllables} syllables
- the lines must follow rhyme scheme "{scheme}" (one letter per line, in \
order; lines sharing a letter must rhyme with each other)

Requirements for "cast":
- list every character that appears in "lines", each with a short, \
consistent visual description an illustrator could draw the same way twice
"""


class LyricsUnavailableError(RuntimeError):
    pass


class LyricsError(RuntimeError):
    """Raised when Qwen could not produce lyrics that pass validation."""


def _binary() -> str:
    found = shutil.which("mlx_lm.generate")
    if found is None:
        raise LyricsUnavailableError(
            "mlx_lm.generate not found on PATH. Install with `uv add mlx-lm`."
        )
    return found


def _extract_json_block(text: str) -> str:
    """Return the first balanced `{...}` block in `text`, brace-depth aware.

    Depth counting ignores braces inside quoted strings (respecting `\\"`
    escapes) so a description like `{"description": "a hat with a { on it"}`
    does not throw off the match.
    """
    start = text.find("{")
    if start == -1:
        raise LyricsError(f"no JSON object found in model output:\n{text[:500]!r}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise LyricsError(f"unbalanced JSON object in model output:\n{text[start:start + 500]!r}")


class LyricsProvider:
    """Writes title/style/cast/lines JSON with Qwen, gated by meter and rhyme."""

    name = "qwen-mlx-lm"

    def __init__(
        self,
        model_repo: str = "mlx-community/Qwen3.5-4B-MLX-4bit",
        max_tokens: int = 2048,
        temperature: float = 0.8,
        max_retries: int = 3,
    ):
        self.model_repo = model_repo
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    def _command(self, prompt: str, seed: int) -> list[str]:
        return [
            _binary(),
            "--model", self.model_repo,
            "--system-prompt", SYSTEM_PROMPT,
            "--prompt", prompt,
            "--max-tokens", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--seed", str(seed),
            # Print only the response, not mlx_lm's generation-stats banner -
            # keeps stdout close to just the JSON block.
            "--verbose", "False",
        ]

    def _run(self, prompt: str, seed: int) -> str:
        cmd = self._command(prompt, seed)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
            raise RuntimeError(f"mlx_lm.generate exited {proc.returncode}:\n{tail}")
        return proc.stdout

    def generate(
        self,
        slug_hint: str,
        target_lines: int,
        syllables: int,
        scheme: str,
        seed: int,
    ) -> dict:
        """Return a parsed `{title, style, cast, lines}` dict that passes `validate.check`.

        Retries up to `max_retries` times, appending the previous attempt's
        violations to the prompt each time. Raises `LyricsError` once retries
        are exhausted.
        """
        base_prompt = PROMPT_TEMPLATE.format(
            slug_hint=slug_hint, target_lines=target_lines, syllables=syllables, scheme=scheme
        )
        violations: list[str] = []

        for attempt in range(1, self.max_retries + 1):
            prompt = base_prompt
            if violations:
                prompt += "\n\nThe previous attempt was invalid, fix these problems:\n" + "\n".join(
                    f"- {v}" for v in violations
                )

            stdout = self._run(prompt, seed)

            try:
                data = json.loads(_extract_json_block(stdout))
            except (LyricsError, json.JSONDecodeError) as exc:
                violations = [f"could not parse a JSON object from the response: {exc}"]
                log.warning("lyrics attempt %d/%d: %s", attempt, self.max_retries, violations[0])
                continue

            lines = [entry.get("text", "") for entry in data.get("lines", [])]
            violations = validate.check(lines, syllables, SYLLABLE_TOLERANCE, scheme)
            if not violations:
                return data

            log.warning(
                "lyrics attempt %d/%d: %d violation(s)",
                attempt, self.max_retries, len(violations),
            )

        raise LyricsError(
            f"could not produce valid lyrics after {self.max_retries} attempt(s): "
            + "; ".join(violations)
        )
