from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from ...hashutil import stable_id
from .base import ParsedDocument, ParsedRegion


def parse_markup(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".md":
        return _parse_markdown(path, raw)
    return _parse_html(path, raw)


def _parse_html(path: Path, raw: str) -> ParsedDocument:
    soup = BeautifulSoup(raw, "html.parser")
    regions: list[ParsedRegion] = []
    headings: list[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if node.name and node.name.startswith("h"):
            level = int(node.name[1:])
            headings = headings[: level - 1] + [text]
        kind = "table" if node.name == "table" else "text"
        regions.append(ParsedRegion(kind, text, None, None, None, headings.copy()))
    return ParsedDocument(stable_id(path.resolve()), path.name, regions)


def _parse_markdown(path: Path, raw: str) -> ParsedDocument:
    regions: list[ParsedRegion] = []
    headings: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            hashes = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[hashes:].strip()
            headings = headings[: max(hashes - 1, 0)] + [title]
            regions.append(ParsedRegion("text", title, None, None, None, headings.copy()))
        elif stripped.startswith("|") and stripped.endswith("|"):
            regions.append(
                ParsedRegion("table", stripped, None, None, None, headings.copy())
            )
        else:
            regions.append(
                ParsedRegion("text", stripped, None, None, None, headings.copy())
            )
    return ParsedDocument(stable_id(path.resolve()), path.name, regions)
