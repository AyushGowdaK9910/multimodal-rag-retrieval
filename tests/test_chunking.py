from mmrag.ingest.chunking import ChunkBuilder
from mmrag.ingest.parsers.base import ParsedDocument, ParsedRegion, parser_for
from mmrag.ingest.pipeline import IngestionPipeline


def test_neighbors_are_linked() -> None:
    document = ParsedDocument(
        "d",
        "x.md",
        [ParsedRegion("text", "one"), ParsedRegion("text", "two")],
    )
    chunks = ChunkBuilder().build(document)
    assert chunks[0].next_id == chunks[1].id
    assert chunks[1].prev_id == chunks[0].id


def test_long_text_is_split_with_parent() -> None:
    words = " ".join(f"w{i}" for i in range(200))
    chunks = ChunkBuilder(max_words=50).build(
        ParsedDocument("d", "x.md", [ParsedRegion("text", words)])
    )
    assert len(chunks) == 4
    assert chunks[0].parent_id == chunks[1].parent_id
    assert chunks[0].parent_id is not None


def test_parser_for_rejects_unknown_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "note.txt"
    path.write_text("hello")
    try:
        parser_for(path)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "Unsupported" in str(exc)


def test_ingest_image(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from PIL import Image

    path = tmp_path / "chart.png"
    Image.new("RGB", (8, 8), color=(20, 40, 80)).save(path)
    _, chunks = IngestionPipeline().ingest(path)
    assert chunks[0].modality.value == "image"
    assert chunks[0].image_bytes_b64


def test_ingest_pdf(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import fitz

    path = tmp_path / "page.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Revenue increased in Q3")
    pdf.save(path)
    pdf.close()
    _, chunks = IngestionPipeline().ingest(path)
    assert any("Revenue" in chunk.text for chunk in chunks)


def test_ingest_markdown_and_html(tmp_path) -> None:  # type: ignore[no-untyped-def]
    md = tmp_path / "doc.md"
    md.write_text("# Title\nRevenue increased in Q3.\n| a | b |\n")
    html = tmp_path / "doc.html"
    html.write_text(
        "<h1>Finance</h1><p>Revenue increased in Q3.</p><table><tr><td>a</td></tr></table>"
    )
    pipeline = IngestionPipeline()
    _, md_chunks = pipeline.ingest(md)
    _, html_chunks = pipeline.ingest(html)
    assert any(chunk.modality.value == "text" for chunk in md_chunks)
    assert any(chunk.modality.value == "table" for chunk in html_chunks)
    tree = pipeline.ingest_tree(tmp_path)
    assert len(tree) >= len(md_chunks) + len(html_chunks)
