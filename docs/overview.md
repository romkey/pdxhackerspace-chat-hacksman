# Chat Hacksman overview

**What:** the PDX Hackerspace member chatbot: answers only from the shared
RAG pipeline's member collections in Qdrant (`hs_member`, `hs_catalog`), with citations.
It follows plans 00 (shared architecture) and 21 (chatbots) of the RAG pipeline.
**Runs as:** HTTP service (FastAPI + uvicorn) on port 8000; `chat-hacksman`.
**Talks to:** an OpenAI-compatible chat API (`LLM_URL`), an OpenAI-compatible embeddings
API, an optional reranker, Qdrant, the topics/events JSON feed, SQLite (local file), optionally Sentry.

## Files

- `src/chat_hacksman/__main__.py`: command line, logging, Sentry, uvicorn.
- `src/chat_hacksman/config.py`: `Settings` from environment variables; query prefixes.
- `src/chat_hacksman/server.py`: `create_app(settings)`; every HTTP endpoint; the answer
  pipeline (`prepare`: rewrite and retrieve; `finish`: check citations and log).
- `src/chat_hacksman/rag.py`: `RagService`: embed, hybrid dense + BM25 search with RRF,
  tier/stale/superseded filter, rerank; `select_chunks`: the authority pass.
- `src/chat_hacksman/rerank.py`: `Reranker`: rerank API client (Jina/Cohere or TEI format).
- `src/chat_hacksman/llm.py`: `ANSWERING_RULES`; `build_system_prompt` (sources as `<source>`
  data); `LlmService`: query rewrite, chat completions, streaming; `citations`.
- `src/chat_hacksman/storage.py`: `Storage`: SQLite schema and queries.
- `src/chat_hacksman/topics.py`: `TopicsService`: fetch and cache the feed.
- `src/chat_hacksman/models.py`: pydantic models for the API.
- `src/chat_hacksman/static/`: browser UI (plain HTML and JS).
- `evals/`, `scripts/run-evals.py`: golden set and red-team prompts against a live bot.
- `tests/`: pytest; `conftest.py` has `make_client` (app with a temp DB) and
  `mock_http` (fakes every httpx call).
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev), `.test`, `.lint`, `.prod`.

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Nothing reads the environment except `Settings.from_env()`, called in `main()`.
- Tests make no real network calls; fake httpx with `mock_http`.
- Tier is enforced by what the bot can read (its collections and Qdrant key) and the
  `tier` filter, never by the prompt alone. Chats are logged without identity.
- Not done yet: member-manager login (basic auth for now), realtime tools (plan 11),
  entity routing filters.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
