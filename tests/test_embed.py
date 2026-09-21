from mmrag.embed.text import TextEmbedder, hashing_encode
from mmrag.embed.vision import VisualEmbedder


def test_hashing_encode_is_deterministic() -> None:
    left = hashing_encode(["Q3 revenue increased"])
    right = hashing_encode(["Q3 revenue increased"])
    other = hashing_encode(["unrelated weather"])
    assert left == right
    assert left != other


def test_text_embedder_uses_hashing_backend_by_default() -> None:
    embedder = TextEmbedder("BAAI/bge-m3")
    vectors = embedder.encode(["hello world", "hello world"])
    assert embedder.backend == "hashing"
    assert vectors[0] == vectors[1]
    assert len(vectors[0]) == embedder.dimension


def test_visual_embedder_hashing_backend() -> None:
    embedder = VisualEmbedder("ViT-B-32")
    empty, hashed = embedder.encode_images([None, "aGVsbG8="])
    assert empty == []
    assert hashed
    text_vectors = embedder.encode_text(["chart of revenue"])
    assert len(text_vectors[0]) == embedder.dimension
