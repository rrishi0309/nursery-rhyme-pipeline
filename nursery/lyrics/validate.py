"""Pure meter/rhyme heuristics for gating Qwen-generated lyrics.

No phoneme dictionary (e.g. CMUdict) is shipped, so syllable counting is a
vowel-group heuristic rather than a lookup - good enough to catch a line that
is wildly off-meter without adding a data dependency. Everything here is pure
and has no I/O, which is what makes it cheap to unit-test thoroughly; the
provider (`nursery/providers/lyrics_qwen.py`) is the only caller.

NOTE ON DIVERGENCE FROM THE ORIGINAL TASK-3 BRIEF: the brief specified a
single rule - "subtract 1 for a silent trailing e when the word has >1 group,
floor at 1". Code review (fix round 1) found that rule alone misclassifies
two extremely common English patterns badly enough to break the archetypal
"Twinkle, twinkle, little star" line (7 real syllables, scored as 4). Two
exceptions were added on top of the brief's rule; see `count_syllables` and
`rhyme_key` below for what changed and why.
"""

from __future__ import annotations

import re

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
_WORD = re.compile(r"[A-Za-z']+")

# Exception 1 (count_syllables): a trailing "-Cle" (consonant + l + e - table,
# little, twinkle, candle) is NOT a silent e - the e forms its own syllable
# with the preceding consonant+l. Without this, "little" scores 1 instead of
# 2 and the canonical nursery-rhyme opening line undercounts by 3.
_CONSONANT_LE = re.compile(r"[^aeiouy]le$")

# Exception 2 (count_syllables): "-es"/"-ed" suffixes are usually silent
# (smiles, walked - do not add a syllable) but DO add one after a sibilant
# ("-es": boxes, wishes, watches) or after t/d ("-ed": wanted, needed). The
# plain vowel-group count already gets those two cases right (the suffix
# vowel is not isolated by a lone trailing consonant), so only the silent
# cases need an explicit subtraction.
_SIBILANT_ES = re.compile(r"(?:[sxz]|ch|sh)es$")
_TD_ED = re.compile(r"[td]ed$")


def count_syllables(word: str) -> int:
    """Vowel-group heuristic: count runs of `[aeiouy]`, drop a silent e.

    Three suffix patterns are treated as NOT adding their own syllable, each
    only when the word has more than one vowel group to begin with (so "the"
    stays 1, not 0) and the result always floors at 1:

    - a trailing "e" on its own (make, like, love) - EXCEPT a trailing
      "-Cle" (little, table, twinkle), where the e is not silent
    - a trailing "es" - EXCEPT after a sibilant (boxes, wishes, watches),
      where the e is not silent
    - a trailing "ed" - EXCEPT after t/d (wanted, needed), where the e is
      not silent
    """
    word = word.lower()
    groups = _VOWEL_GROUPS.findall(word)
    count = len(groups)

    if count > 1:
        if word.endswith("e") and not _CONSONANT_LE.search(word):
            count -= 1
        elif word.endswith("es") and not _SIBILANT_ES.search(word):
            count -= 1
        elif word.endswith("ed") and not _TD_ED.search(word):
            count -= 1

    return max(1, count)


def line_syllables(text: str) -> int:
    """Sum of `count_syllables` over the words in a line, punctuation ignored."""
    return sum(count_syllables(w) for w in _WORD.findall(text))


def rhyme_key(text: str) -> str:
    """The rhyme-relevant tail of a line's last word.

    Lowercases the last word, strips punctuation, and returns everything from
    its last vowel group onward: "star" -> "ar".

    A silent trailing e is its own vowel group by regex alone (separated from
    the stem vowel by a consonant), which - left unhandled - collapses every
    silent-e word down to the same generic "e" key: "love" and "smile" would
    both key as "e" and falsely count as rhyming, as would "little"/"fire"/
    "you've"/"are". When the last vowel group is exactly a trailing standalone
    "e" and there is more than one group, the tail starts from the group
    *before* it instead, so "love" -> "ove" and "smile" -> "ile" (correctly
    not rhyming).

    A word with no vowel group at all (rare) returns itself unchanged; an
    empty line returns "".
    """
    words = _WORD.findall(text)
    if not words:
        return ""
    last = words[-1].lower()
    matches = list(_VOWEL_GROUPS.finditer(last))
    if not matches:
        return last

    tail_start = matches[-1].start()
    if len(matches) > 1 and matches[-1].group() == "e" and matches[-1].end() == len(last):
        tail_start = matches[-2].start()
    return last[tail_start:]


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
