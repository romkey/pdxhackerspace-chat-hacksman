# Chat Hacksman

Web-based chatbot with:

- Optional RAG from up to 3 Qdrant collections (or none, if unset).
- Configurable prompt/model/provider in the UI, persisted across runs.
- Per-collection Qdrant enable/disable checkboxes in the UI, persisted across runs.
- Topic buttons loaded from a JSON feed (default: `https://members.pdxhackerspace.org/rag.json`).
- If feed includes `events` (for example `https://events.pdxhackerspace.org/events.json`), events are persisted in SQLite.
- If feed includes `occurrences`, those are persisted too.
- Ollama and llama.cpp-compatible chat modes.
- History tracking for settings snapshot, prompt, question, and response.
- Docker, GitHub Actions CI, and image publishing.

## Quick Start (Local)

1. Create env file:

   ```bash
   cp .env.example .env
   ```

2. Install and run:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   python -m app.main
   ```

3. Open `http://localhost:8000`.

## Environment Variables

See `.env.example`. Key vars:

- `RAG_COLLECTIONS` as a comma-separated list, or `RAG_COLLECTION_1..3`.
- `RAG_TOPICS_URL` for topic feed buttons.
- `GET /api/events` returns stored events from the feed.
- `GET /api/occurrences` returns stored occurrences from the feed.
- `DEFAULT_PROVIDER` (`ollama` or `llama_cpp`).
- `DEFAULT_LLM_BASE_URL`, `DEFAULT_MODEL`, `DEFAULT_SYSTEM_PROMPT`.
- `EMBEDDING_TIMEOUT_SECONDS` for embedding calls used by RAG lookup.
- `LLM_TIMEOUT_SECONDS`, `LLM_RETRY_ATTEMPTS`, `LLM_RETRY_BACKOFF_SECONDS` to handle transient model server disconnects/timeouts.
- `LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, ...).

If Qdrant collections are not configured or embedding lookup fails, RAG is skipped gracefully.

## Provider Notes

- `ollama` mode uses `POST /api/chat`.
- `llama_cpp` mode uses OpenAI-compatible `POST /v1/chat/completions`.

The UI exposes shared controls (`temperature`, `top_p`, `max_tokens`, `repeat_penalty`) plus:

- Ollama-specific: `num_ctx`
- llama.cpp-specific: `seed`

## Docker

Build and run:

```bash
docker build -t chat-hacksman:local .
docker run --rm -p 8000:8000 --env-file .env chat-hacksman:local
```

Compose example:

```bash
cp docker-compose.example.yml docker-compose.yml
docker compose up --build
```

## Lint and Tests

```bash
ruff check .
pytest -q
```

## GitHub Actions

- `.github/workflows/ci.yml` runs lint + tests on PRs and pushes to `main`.
- `.github/workflows/docker-publish.yml` builds/pushes to GHCR with BuildKit cache (`type=gha`).
