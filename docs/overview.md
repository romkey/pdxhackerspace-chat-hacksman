# Chat Hacksman overview

**What:** a web chatbot for hackerspace members, with optional RAG over Qdrant.
**Runs as:** HTTP service (FastAPI + uvicorn) on port 8000; `chat-hacksman`.
**Talks to:** Ollama or llama.cpp (chat), an OpenAI-compatible embeddings
API, Qdrant, the topics/events JSON feed, SQLite (local file), optionally Sentry.

## Files

- `src/chat_hacksman/__main__.py`: command line, logging, Sentry, uvicorn.
- `src/chat_hacksman/config.py`: `Settings` from environment variables; query prefixes.
- `src/chat_hacksman/server.py`: `create_app(settings)`; every HTTP endpoint.
- `src/chat_hacksman/rag.py`: `RagService`: embed the question, search and filter Qdrant.
- `src/chat_hacksman/llm.py`: `LlmService`: system prompt from chunks, Ollama/llama.cpp calls.
- `src/chat_hacksman/storage.py`: `Storage`: SQLite schema and queries.
- `src/chat_hacksman/topics.py`: `TopicsService`: fetch and cache the feed.
- `src/chat_hacksman/models.py`: pydantic models for the API.
- `src/chat_hacksman/static/`: browser UI (plain HTML and JS).
- `tests/`: pytest; `conftest.py` has `make_client` (app with a temp DB) and
  `mock_http` (fakes every httpx call).
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev), `.test`, `.lint`, `.prod`.

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Nothing reads the environment except `Settings.from_env()`, called in `main()`.
- Tests make no real network calls; fake httpx with `mock_http`.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
