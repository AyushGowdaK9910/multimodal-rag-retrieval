from mmrag.hashutil import stable_id, tokens


def test_stable_id_is_deterministic() -> None:
    assert stable_id("a", 1, "b") == stable_id("a", 1, "b")
    assert stable_id("a", 1) != stable_id("a", 2)


def test_tokens_lowercases_and_keeps_alnum() -> None:
    assert tokens("Hello, RAG-2!") == ["hello", "rag", "2"]
