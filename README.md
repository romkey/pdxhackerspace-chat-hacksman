# Chat Hacksman

[![CI](https://github.com/romkey/pdxhackerspace-chat-hacksman/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-chat-hacksman/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/pdxhackerspace-chat-hacksman/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-chat-hacksman/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The member chatbot for PDX Hackerspace. It answers members' questions
only from the shared RAG pipeline's member collections (`hs_member` and
`hs_catalog` in Qdrant), citing its sources, with any OpenAI-compatible
chat API (llama.cpp, vLLM...) and embeddings API. Topic buttons, the
events feed, the answer log and thumbs-up/down feedback are kept in SQLite.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/chat_hacksman/__main__.py` | The `chat-hacksman` command: reads settings, starts uvicorn |
| `src/chat_hacksman/config.py` | `Settings` from environment variables; embedding query prefixes |
| `src/chat_hacksman/server.py` | The FastAPI app (`create_app`) and its HTTP API |
| `src/chat_hacksman/rag.py` | Hybrid (dense + BM25) search of Qdrant, filters, authority pass |
| `src/chat_hacksman/rerank.py` | Optional reranker client (Jina/Cohere and TEI formats) |
| `src/chat_hacksman/llm.py` | Answering rules, prompt, query rewrite, streamed chat completions |
| `src/chat_hacksman/storage.py` | SQLite: settings, history, feedback, events, LLM server status |
| `src/chat_hacksman/topics.py` | Fetches and caches the topics/events feed |
| `src/chat_hacksman/models.py` | Pydantic request/response models |
| `src/chat_hacksman/static/` | The browser UI |
| `tests/` | pytest tests |
| `evals/`, `scripts/run-evals.py` | Golden set and red-team prompts, run against a live bot |
| `docs/configuration.md` | Every environment variable |

## Technical details

A question goes through these steps:

1. **Rewrite.** The browser keeps the conversation (it's forgotten on
   reload) and sends the last turns with each question. With earlier
   turns, the chat model rewrites the question into a standalone search
   query.
2. **Embed.** The query (with the model's query prefix, e.g.
   `search_query: ` for the default `nomic-ai/modernbert-embed-base`) is
   sent to `POST {EMBEDDING_URL}/embeddings`. Use the same model and
   prefixes the collections were ingested with;
   [docs/configuration.md](docs/configuration.md#embeddings) lists them.
3. **Search.** Each collection is searched by its dense vector and, by
   keyword, its sparse BM25 vector (inferred by Qdrant), fused with
   reciprocal rank fusion: `RAG_CANDIDATES` (50) hits each. Points whose
   `tier` isn't allowed, that are `superseded_by` another, or that are past
   `stale_after` are never returned.
4. **Rerank and pick** (reranking when `RERANK_URL` is set). A cross-encoder
   scores every candidate against the query; the best `RAG_TOP_N` (8) go
   on, preferring higher `authority` (policy and signs, then the wiki...)
   when relevance is close, and always including a relevant policy or
   sign. Schedule questions with no event hits get upcoming events from
   the live feed.
5. **Answer.** With no sources, the bot says it can't answer and gives
   next steps, without calling the model. Otherwise the fixed answering
   rules, the bot's prompt and the sources (marked as data) go to
   `POST {LLM_URL}/chat/completions`, and the answer streams back.
6. **Check and record.** The answer counts as answered if it cites a
   source. The question, answer, retrieved document IDs, topics and
   whether it was answered are saved (never who asked), unless it's a
   temporary chat. `GET /api/reports/unanswered` is the to-do list for the
   wiki.

If Qdrant or the embedding server is unreachable, nothing is found and the
bot says so; if the reranker is, the search order is used; if Qdrant can't
do BM25 inference, dense search alone is used.

HTTP API (everything but the health checks and login pages needs a login
when one is configured; see [Login](docs/configuration.md#login)):

| Endpoint | What it does |
|----------|--------------|
| `GET /healthz`, `GET /health` | Health check (`/healthz` includes the version) |
| `GET /`, `/static/...` | The UI |
| `GET`/`POST /login`, `POST /logout` | Login page (single sign-on button, local account) and logout |
| `GET /auth/oidc/login`, `/auth/oidc/callback` | OpenID Connect single sign-on |
| `GET /api/whoami` | Whether a login is required, and who is logged in |
| `POST /api/chat` | Ask a question: `{"question", "history", "temporary_chat", "stream"}`; with `stream`, newline-delimited JSON `delta` events then `done` |
| `GET`/`PUT /api/settings` | Model, system prompt, tweaks, enabled collections |
| `GET /api/rag/collections` | Configured and enabled collections |
| `GET /api/models` | Models the LLM server offers, and the `LLM_MODEL` default |
| `GET /api/llm/status` | Whether the LLM server is up, and since when |
| `GET /api/topics`, `/api/events`, `/api/occurrences` | The topics feed and stored events |
| `GET`/`DELETE /api/history`, `DELETE /api/history/latest` | Chat history |
| `GET`/`POST /api/feedback` | Thumbs up/down with notes |
| `GET /api/reports/unanswered?days=7` | Questions not answered from sources, grouped by topic |
| `GET /api/reports/negative-feedback?days=7` | Thumbs-down answers with their source documents |
| `GET /api/meta` | Name, version, repo link |

The UI offers the standard sampling options: `temperature`, `top_p`,
`max_tokens` and an optional `seed`.

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

### Evaluation

The tests fake every model. To check the real thing, run the golden set
and red-team prompts against a running bot after every change to prompts,
models or retrieval settings, and after every re-index:

```bash
uv run scripts/run-evals.py --url http://localhost:8000 --canary ADMIN-CANARY-1 --canary SLACK-CANARY-1
```

Add real member questions and their expected documents to
`evals/golden.json`; pass each planted canary string with `--canary`.

## License

[MIT](LICENSE)
