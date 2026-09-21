# MM-RAG Production

An independent, production-oriented multimodal RAG implementation for **text, tables, images, charts, and scanned PDFs**.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Runtime extras are layered by subsystem: `[ingest]`, `[index]`, `[ml]`, `[vision]`, `[providers]`, `[api]`.

Copy `.env.example` to `.env` for local configuration. Never commit secrets.
