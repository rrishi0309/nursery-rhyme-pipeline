"""Stable hashing used to detect when a stage's inputs have changed.

Python's built-in hash() is salted per process, so it cannot be persisted.
"""

from __future__ import annotations

import hashlib
import json


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
