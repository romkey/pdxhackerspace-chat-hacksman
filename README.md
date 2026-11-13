# Chat Hacksman

[![CI](https://github.com/romkey/pdxhackerspace-chat-hacksman/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-chat-hacksman/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/pdxhackerspace-chat-hacksman/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-chat-hacksman/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A web chatbot for PDX Hackerspace members. It answers with a local LLM
(Ollama or llama.cpp) and can ground its answers in up to five Qdrant
collections (wiki, events, Slack, Calibre books...), embedding each
question with any OpenAI-compatible embeddings API. Topic buttons, the
events feed, chat history and thumbs-up/down feedback are kept in SQLite.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/chat_hacksman/__main__.py` | The `chat-hacksman` command: reads settings, starts uvicorn |
| `src/chat_hacksman/config.py` | `Settings` from environment variables; embedding query prefixes |
| `src/chat_hacksman/server.py` | The FastAPI app (`create_app`) and its HTTP API |
| `src/chat_hacksman/rag.py` | Embeds questions and searches Qdrant, with per-collection filters |
| `src/chat_hacksman/rerank.py` | Optional reranker client (Jina/Cohere and TEI formats) |
| `src/chat_hacksman/llm.py` | Builds the prompt and calls Ollama or llama.cpp, with retries |
| `src/chat_hacksman/storage.py` | SQLite: settings, history, feedback, events, LLM server status |
| `src/chat_hacksman/topics.py` | Fetches and caches the topics/events feed |
| `src/chat_hacksman/models.py` | Pydantic request/response models |
| `src/chat_hacksman/static/` | The browser UI |
| `tests/` | pytest tests |
| `docs/configuration.md` | Every environment variable |

## Technical details

A question goes through these steps:

1. **Embed.** If RAG is on and collections are enabled, the question
   (with the model's query prefix, e.g. `search_query: ` for the default
   `nomic-ai/modernbert-embed-base`) is sent to
   `POST {EMBEDDING_URL}/embeddings`. Any OpenAI-compatible server works:
   OpenAI, Hugging Face TEI, Infinity, vLLM, llama.cpp's `llama-server`, or
   Ollama's `/v1` API. Use the same model and prefixes the collections were
   ingested with; [docs/configuration.md](docs/configuration.md#embeddings)
   lists the prefixes for common models.
2. **Search.** Each enabled Qdrant collection is searched, skipping any
   whose vector size doesn't match. Collections named `events*`, `slack*`
   and `calibre*` get filters picked from the question's wording (upcoming
   vs. past events, channel vs. thread summaries, catalog vs. passages).
   Hits below `RAG_MIN_SCORE` are dropped.
3. **Rerank** (optional, when `RERANK_URL` is set). More candidates are
   fetched per collection, and a cross-encoder reranker (default
   `BAAI/bge-reranker-v2-m3`, served by llama.cpp, vLLM, TEI, Infinity, or
   the Jina or Cohere APIs) scores each against the question; only the
   best go on, from whichever collections they come from. See
   [docs/configuration.md](docs/configuration.md#reranking). Schedule
   questions with no event hits fall back to upcoming events from the feed.
4. **Answer.** The chunks go into the system prompt and the question to
   the chat model: Ollama's `POST /api/chat`, or llama.cpp's
   OpenAI-compatible `POST /v1/chat/completions`.
5. **Record.** The question, answer, settings and timings are saved to
   history unless it's a temporary chat.

If Qdrant or the embedding server is unreachable, the chat still works
without RAG; if the reranker is, the vector search order is used.

HTTP API (all under optional basic auth except the health checks):

| Endpoint | What it does |
|----------|--------------|
| `GET /healthz`, `GET /health` | Health check (`/healthz` includes the version) |
| `GET /`, `/static/...` | The UI |
| `POST /api/chat` | Ask a question: `{"question", "use_rag", "temporary_chat"}` |
| `GET`/`PUT /api/settings` | Provider, model, system prompt, tweaks, enabled collections |
| `GET /api/rag/collections` | Configured and enabled collections |
| `GET /api/models`, `POST /api/models/pull` | List or pull Ollama models |
| `GET /api/llm-base-urls`, `GET /api/llm-base-urls/status` | LLM servers used, and whether they're up |
| `GET /api/topics`, `/api/events`, `/api/occurrences` | The topics feed and stored events |
| `GET`/`DELETE /api/history`, `DELETE /api/history/latest` | Chat history |
| `GET`/`POST /api/feedback` | Thumbs up/down with notes |
| `GET /api/meta` | Name, version, repo link |

The UI offers shared tweaks (`temperature`, `top_p`, `max_tokens`,
`repeat_penalty`), plus `num_ctx` for Ollama and `seed` for llama.cpp.

## Running it

Requires Docker. Configuration comes from environment variables, kept in
`.env` (see [docs/configuration.md](docs/configuration.md)):

```bash
cp .env.example .env   # then edit it
docker compose -f docker-compose.prod.yaml up -d
```

That runs the latest release from
`ghcr.io/romkey/pdxhackerspace-chat-hacksman`; set `APP_VERSION=X.Y.Z` to
pin one. The database is kept in the `chat-hacksman-data` volume. The app
answers on http://localhost:8000.

### With pip

Needs Python 3.15 or newer. Install into a virtual environment, from a
release's wheel:

```bash
python3 -m venv ~/.venvs/chat-hacksman
~/.venvs/chat-hacksman/bin/pip install https://github.com/romkey/pdxhackerspace-chat-hacksman/releases/download/vX.Y.Z/chat_hacksman-X.Y.Z-py3-none-any.whl
~/.venvs/chat-hacksman/bin/chat-hacksman --version
```

Set its environment variables yourself (see
[docs/configuration.md](docs/configuration.md)); it doesn't read `.env`.
It listens on 127.0.0.1 unless you set `APP_HOST`.

## Development

With Docker, nothing else to install:

```bash
docker compose up --build                                  # run from the working tree
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions
CI tests (`uv.lock`):

```bash
uv sync
uv run pytest
uv run mypy
uv run chat-hacksman
```

With pip (25.1 or newer, for `--group`). pip doesn't read `uv.lock`, so it
may pick newer dependency versions than CI tested:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . --group dev
pytest
mypy
chat-hacksman
```

Either way, `uvx pre-commit install` (or `pip install pre-commit &&
pre-commit install`) runs the lint checks on every commit.

## License

[MIT](LICENSE)
