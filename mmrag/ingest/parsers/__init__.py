from __future__ import annotations

from .base import ParsedDocument, ParsedRegion, parser_for
from .image import parse_image
from .pdf import parse_pdf
from .web import parse_markup

__all__ = [
    "ParsedDocument",
    "ParsedRegion",
    "parse_image",
    "parse_markup",
    "parse_pdf",
    "parser_for",
]
