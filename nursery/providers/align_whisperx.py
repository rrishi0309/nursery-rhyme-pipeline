"""Line-level timing recovery via whisperx-mlx word-level alignment.

ACE-Step sings every lyric line into one continuous track with no built-in
markers between lines. This provider recovers where each line falls in that
audio by running whisperx's ASR + forced alignment over it and matching the
recognised words back onto the supplied lyric lines, in order.

Per `docs/tool-help/whisperx-mlx.txt`, `--word_timestamps` is documented as
requiring `backend='lightning'`; under plain `mlx` the word-level spans may
come back empty. Sung vowels stretch across many frames and ASR both drops
and mishears words against a musical backing track, so exact-match alignment
is not viable even when word timestamps are present. Both of those are
exactly why an even-split fallback exists below and is required to never
raise: scene timing must always exist for the video stage to run, whether or
not the alignment itself produced anything usable.

Runs under whisperx-mlx's own venv (`bin`), not the project's - it is not
published on PyPI and requires Python <3.13 (project venv is 3.14).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import wave
from pathlib import Path

log = logging.getLogger(__name__)

# How many upcoming ASR words to scan when looking for the next expected
# lyric token. Bounded so that one line ASR badly mangles (e.g. an
# instrumental break misheard as words) can't consume the rest of the word
# list before later, easily-matched lines ever get a chance to look.
_LOOKAHEAD_WORDS = 20

_PUNCT_RE = re.compile(r"[^\w']+")


class AlignError(RuntimeError):
    pass


def _normalize(token: str) -> str:
    """Lowercase and strip everything but letters/digits/apostrophes.

    Applied to already whitespace-split tokens, so this never merges two
    words together - only punctuation attached to a single word is removed
    (commas, sung "oh-oh-oh" hyphens, trailing periods, ...).
    """
    return _PUNCT_RE.sub("", token.strip().lower())


def _line_tokens(line: str) -> list[str]:
    return [t for t in (_normalize(word) for word in line.split()) if t]


def _extract_words(payload: dict) -> list[dict]:
    """Flatten whisperx's JSON output into `[{"norm": ..., "start": ..., "end": ...}, ...]`.

    whisperx's `AlignedTranscriptionResult` carries a top-level
    `word_segments` list when alignment succeeded; fall back to each
    segment's own `words` list (same shape) if that key is absent. Either
    way, this comes back empty when `--word_timestamps` didn't apply (the
    plain-`mlx`-backend case the module docstring describes) - the caller
    treats an empty list as an alignment failure and falls back.
    """
    raw = payload.get("word_segments")
    if not raw:
        raw = []
        for seg in payload.get("segments", []):
            raw.extend(seg.get("words") or [])

    words = []
    for w in raw:
        text = w.get("word")
        start, end = w.get("start"), w.get("end")
        if not text or start is None or end is None:
            continue
        norm = _normalize(text)
        if not norm:
            continue
        words.append({"norm": norm, "start": float(start), "end": float(end)})
    return words


def _find_word(words: list[dict], start_idx: int, token: str) -> int | None:
    end = min(len(words), start_idx + _LOOKAHEAD_WORDS)
    for idx in range(start_idx, end):
        if words[idx]["norm"] == token:
            return idx
    return None


def _map_words_to_lines(
    words: list[dict], lines: list[str]
) -> list[tuple[float, float] | None]:
    """Walk the flat ASR word list forward once, consuming matches per line.

    For each line, each of its normalised tokens is searched for within a
    bounded lookahead of the shared cursor; a match advances the cursor past
    that word, a miss just moves on to the next token (tolerating dropped or
    mis-heard words without requiring an exact match). The cursor never
    rewinds, so a later line can never claim an earlier word - what comes
    back is either monotonic by construction or `None` for lines that
    matched nothing at all. Callers still need `_validate_spans`: an empty
    lyric line, or a run of lookahead misses, can still leave individual
    spans zero-length or (if a line's few matched words happen to be exactly
    where the previous line's cursor left off) too close together to trust.
    """
    cursor = 0
    spans: list[tuple[float, float] | None] = []
    for line in lines:
        starts: list[float] = []
        ends: list[float] = []
        for token in _line_tokens(line):
            idx = _find_word(words, cursor, token)
            if idx is None:
                continue
            starts.append(words[idx]["start"])
            ends.append(words[idx]["end"])
            cursor = idx + 1
        spans.append((min(starts), max(ends)) if starts else None)
    return spans


def _validate_spans(
    spans: list[tuple[float, float] | None],
) -> list[tuple[float, float]] | None:
    """All-or-nothing gate: one bad line invalidates the whole alignment.

    Required by the brief: a single garbled line shouldn't get an
    interpolated guess while its neighbours keep real timings - if any line
    failed to match, or any span is zero-length, or spans overlap/go
    backwards relative to the line before it, the caller discards the
    entire result and falls back to an even split for every line.
    """
    if any(s is None for s in spans):
        return None
    validated: list[tuple[float, float]] = []
    prev_end = 0.0
    for start, end in spans:  # type: ignore[misc]
        if end <= start:
            return None
        if start < prev_end:
            return None
        validated.append((start, end))
        prev_end = end
    return validated


def _even_split(duration_s: float, n_lines: int) -> list[tuple[float, float]]:
    """Spread `duration_s` evenly across `n_lines` contiguous spans.

    Never raises and never returns a zero-length or non-monotonic span:
    `duration_s <= 0` (unknown/unreadable audio) falls back to a fixed one
    second per line rather than collapsing every span to (0.0, 0.0).
    """
    if n_lines <= 0:
        return []
    if duration_s <= 0.0:
        duration_s = float(n_lines)
    step = duration_s / n_lines
    return [(round(i * step, 3), round((i + 1) * step, 3)) for i in range(n_lines)]


def _audio_duration_s(audio: Path) -> float:
    """Best-effort audio duration in seconds, used only to size the
    even-split fallback. Deliberately never raises: every song this
    pipeline generates is WAV (`SongConfig.audio_format`), so `wave` covers
    the real path; anything unreadable just falls through to
    `_even_split`'s fixed per-line duration.
    """
    try:
        with wave.open(str(audio), "rb") as wf:
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return wf.getnframes() / rate
    except (OSError, wave.Error, EOFError):
        return 0.0


class WhisperXAlignProvider:
    """Recovers one (start_s, end_s) span per lyric line from sung audio."""

    name = "whisperx"

    def __init__(
        self,
        bin: str,
        model: str = "small",
        backend: str = "lightning",
        max_drift_s: float = 0.75,
    ):
        # Not on PyPI and not runnable from the project's own venv - must be
        # supplied, there is no PATH-based default.
        self.bin = bin
        self.model = model
        self.backend = backend
        # Accepted so this constructor can be driven straight off
        # `AlignConfig`, but unused here: no drift-correction pass exists
        # yet, only the accept/reject-and-fall-back gate in
        # `_validate_spans`. Reserved for whoever adds one.
        self.max_drift_s = max_drift_s

    def align(self, audio: Path, lines: list[str], out_dir: Path) -> list[tuple[float, float]]:
        n = len(lines)
        if n == 0:
            return []

        fallback = _even_split(_audio_duration_s(audio), n)

        # Belt-and-suspenders: every expected failure mode below already
        # falls back explicitly, but "never raise" is a hard requirement
        # (scene timing must always exist), so an unanticipated failure
        # anywhere in this block - a missing binary, a malformed JSON file,
        # whatever - falls back too rather than propagating.
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                self.bin, str(audio),
                "--backend", self.backend,
                "--model", self.model,
                "-o", str(out_dir),
                "-f", "json",
                "--word_timestamps", "True",
            ]
            log.info("aligning %s against %d lines", audio.name, n)
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
                log.warning(
                    "whisperx exited %d, falling back to an even split:\n%s",
                    proc.returncode, tail,
                )
                return fallback

            json_path = out_dir / f"{audio.stem}.json"
            if not json_path.exists():
                log.warning(
                    "whisperx reported success but wrote no %s, falling back to an "
                    "even split", json_path,
                )
                return fallback

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            words = _extract_words(payload)
            if not words:
                log.warning(
                    "whisperx returned no word-level timestamps (--word_timestamps "
                    "needs backend='lightning'; got backend=%r), falling back to an "
                    "even split", self.backend,
                )
                return fallback

            spans = _map_words_to_lines(words, lines)
            validated = _validate_spans(spans)
            if validated is None:
                log.warning(
                    "word-to-line alignment produced an unmatched, zero-length, or "
                    "non-monotonic span, falling back to an even split"
                )
                return fallback
            return validated
        except Exception:
            log.exception("alignment raised unexpectedly, falling back to an even split")
            return fallback
