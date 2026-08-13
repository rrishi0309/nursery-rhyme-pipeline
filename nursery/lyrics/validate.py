"""Pure meter/rhyme heuristics for gating Qwen-generated lyrics.

No phoneme dictionary (e.g. CMUdict) is shipped, so syllable counting is a
vowel-group heuristic rather than a lookup - good enough to catch a line that
is wildly off-meter without adding a data dependency. Everything here is pure
and has no I/O, which is what makes it cheap to unit-test thoroughly; the
provider (`nursery/providers/lyrics_qwen.py`) is the only caller.
"""

from __future__ import annotations

import re

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
_WORD = re.compile(r"[A-Za-z']+")


def count_syllables(word: str) -> int:
    """Vowel-group heuristic: count runs of `[aeiouy]`, drop a silent trailing e.

    The trailing-e subtraction only applies when the word has more than one
    vowel group (so "the" stays 1, not 0), and the result always floors at 1
    (every word has at least one syllable).
    """
    word = word.lower()
    groups = _VOWEL_GROUPS.findall(word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def line_syllables(text: str) -> int:
    """Sum of `count_syllables` over the words in a line, punctuation ignored."""
    return sum(count_syllables(w) for w in _WORD.findall(text))


def rhyme_key(text: str) -> str:
    """The rhyme-relevant tail of a line's last word.

    Lowercases the last word, strips punctuation, and returns everything from
    its last vowel group onward: "star" -> "ar". A word with no vowel group at
    all (rare) returns itself unchanged; an empty line returns "".
    """
    words = _WORD.findall(text)
    if not words:
        return ""
    last = words[-1].lower()
    matches = list(_VOWEL_GROUPS.finditer(last))
    if not matches:
        return last
    return last[matches[-1].start() :]


def check(lines: list[str], target: int, tolerance: int, scheme: str) -> list[str]:
    """Human-readable meter/rhyme violations; empty when `lines` is valid.

    `scheme` is a string like "AABB" - one letter per line. Lines sharing a
    letter must share a `rhyme_key`; a line's syllable count must fall within
    `tolerance` of `target`.
    """
    violations: list[str] = []

    if len(lines) != len(scheme):
        violations.append(
            f"expected {len(scheme)} line(s) for rhyme scheme {scheme!r}, got {len(lines)}"
        )

    for i, line in enumerate(lines):
        n = line_syllables(line)
        if abs(n - target) > tolerance:
            violations.append(
                f"line {i + 1} has {n} syllables, expected {target}+/-{tolerance}: {line!r}"
            )

    group_keys: dict[str, str] = {}
    for i, letter in enumerate(scheme):
        if i >= len(lines):
            break
        key = rhyme_key(lines[i])
        first_key = group_keys.setdefault(letter, key)
        if key != first_key:
            violations.append(
                f"line {i + 1} breaks rhyme group {letter!r} "
                f"(rhyme key {key!r} != {first_key!r}): {lines[i]!r}"
            )

    return violations
