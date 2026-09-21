from __future__ import annotations

from pathlib import Path

import fitz

from ...hashutil import stable_id
from .base import ParsedDocument, ParsedRegion


def parse_pdf(path: Path) -> ParsedDocument:
    document_id = stable_id(path.resolve())
    regions: list[ParsedRegion] = []
    pdf = fitz.open(path)
    try:
        for page_index, page in enumerate(pdf):
            page_no = page_index + 1
            for block in page.get_text("blocks"):
                text = str(block[4]).strip()
                if text:
                    bbox = (float(block[0]), float(block[1]), float(block[2]), float(block[3]))
                    regions.append(
                        ParsedRegion("text", text, page_no, bbox, section_path=[])
                    )
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    image = pdf.extract_image(xref)
                except Exception:
                    continue
                raw = image.get("image")
                if not raw:
                    continue
                regions.append(
                    ParsedRegion(
                        "image",
                        f"Image on page {page_no}",
                        page_no,
                        None,
                        raw,
                        [],
                    )
                )
    finally:
        pdf.close()
    return ParsedDocument(document_id, path.name, regions)
