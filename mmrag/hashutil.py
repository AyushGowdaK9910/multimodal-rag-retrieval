from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def stable_id(*parts: object) -> str:
    raw = "||".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
