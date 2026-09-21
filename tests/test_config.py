from mmrag.config import Settings


def test_settings_read_openai_key_without_prefix(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.openai_api_key == "sk-test"
    assert settings.storage == "memory"


def test_mmrag_prefix_overrides_storage(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MMRAG_STORAGE", "qdrant")
    settings = Settings(_env_file=None)
    assert settings.storage == "qdrant"
