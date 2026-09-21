from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedRegion:
    kind: str
    text: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    image_bytes: bytes | None = None
    section_path: list[str] | None = None


@dataclass
class ParsedDocument:
    document_id: str
    source_name: str
    regions: list[ParsedRegion] = field(default_factory=list)


Parser = Callable[[Path], ParsedDocument]


def parser_for(path: Path) -> Parser:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from .pdf import parse_pdf

        return parse_pdf
    if suffix in {".html", ".htm", ".md"}:
        from .web import parse_markup

        return parse_markup
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
        from .image import parse_image

        return parse_image
    raise ValueError(f"Unsupported file type: {suffix}")
