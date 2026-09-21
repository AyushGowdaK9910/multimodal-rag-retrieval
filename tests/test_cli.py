from typer.testing import CliRunner

from mmrag.cli import app


def test_cli_eval_on_sample(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset = tmp_path / "q.jsonl"
    dataset.write_text(
        '{"question":"revenue","answer":"increased","relevant_chunk_ids":[]}\n',
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["eval", str(dataset)])
    assert result.exit_code == 0
    assert "retrieval" in result.stdout
