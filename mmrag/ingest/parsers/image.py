from __future__ import annotations

from pathlib import Path

from ...hashutil import stable_id
from .base import ParsedDocument, ParsedRegion


def parse_image(path: Path) -> ParsedDocument:
    raw = path.read_bytes()
    return ParsedDocument(
        stable_id(path.resolve()),
        path.name,
        [ParsedRegion("image", path.name, None, None, raw, [])],
    )
