# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases up to 0.5.7 are
described by their git tags and commit messages.

## [Unreleased]

### Changed

- **Breaking:** The chat LLM is any OpenAI-compatible chat completions API
  (`POST {LLM_URL}/chat/completions`): OpenAI, llama.cpp's `llama-server`,
  vLLM, LM Studio, LiteLLM... The Ollama-native API and the
  Ollama/llama.cpp provider choice are gone.
- **Breaking:** The server comes from the environment only: `LLM_URL`
  (default `http://localhost:8080/v1`), `LLM_API_KEY` and `LLM_MODEL`
  replace `DEFAULT_PROVIDER`, `DEFAULT_LLM_BASE_URL`, `DEFAULT_MODEL` and
  `OLLAMA_API_KEY`. The app won't start while an old variable is set,
  naming its replacement.
- The UI no longer asks for a provider or base URL. It shows the server
  and whether it's up, and picks a model from `GET {LLM_URL}/models`, with
  "Default" meaning `LLM_MODEL` (or the server's first model if unset).
- The sampling tweaks are the standard ones: `temperature`, `top_p`,
  `max_tokens` and an optional `seed`.
- The database moves to schema 2: saved settings lose the provider, base
  URL, model, `num_ctx` and `repeat_penalty`, so the model falls back to
  `LLM_MODEL`; the list of past base URLs is dropped.

### Removed

- `POST /api/models/pull`, `GET /api/llm-base-urls` and
  `GET /api/llm-base-urls/status` (replaced by `GET /api/llm/status`).

## [0.7.0] - 2026-11-13

### Added

- Optional reranking after the vector search: with `RERANK_URL` set, each
  collection is searched for `RERANK_CANDIDATES` chunks, a cross-encoder
  reranker (default `BAAI/bge-reranker-v2-m3`) scores them against the
  question, and the best `RERANK_TOP_N` go to the LLM. Speaks the
  Jina/Cohere rerank API (llama.cpp, vLLM, Infinity, Jina, Cohere) and
  Hugging Face TEI's (`RERANK_API=tei`). Falls back to vector order if the
  reranker fails. Chat responses' context chunks gain `rerank_score`.

### Fixed

- A Qdrant error other than a vector size mismatch no longer reuses the
  previous collection's hits (or fails) for that collection.
- The Release workflow no longer fails to push the image after a release
  PR is merged: it wrote the build cache from the closed PR's event, which
  isn't allowed.

## [0.6.0] - 2026-11-13

### Changed

- **Breaking:** query embeddings now come from an OpenAI-compatible
  embeddings API (`POST {EMBEDDING_URL}/embeddings`) instead of Ollama's
  `/api/embed` and `/api/embeddings`. `EMBEDDING_URL` is now a base URL
  including `/v1` (default `http://localhost:8080/v1`; for Ollama use
  `http://localhost:11434/v1`).
- **Breaking:** the default `EMBEDDING_MODEL` is now
  `nomic-ai/modernbert-embed-base` (768 dimensions), so collections ingested
  with another model need `EMBEDDING_MODEL` set to match.
- **Breaking:** `OLLAMA_API_KEY` is no longer sent to the embedding server;
  use `EMBEDDING_API_KEY`.
- **Breaking:** `APP_HOST` defaults to `127.0.0.1` when run natively (the
  Docker image still sets `0.0.0.0`).
- **Breaking:** the image is now `ghcr.io/romkey/pdxhackerspace-chat-hacksman`
  (was `.../pdxhackerspace-chat-hacksman/chat-hacksman`), and keeps its
  database in `/data` instead of `/app-data`. Run it with
  `docker-compose.prod.yaml`; copy an existing database into the new
  `chat-hacksman-data` volume.
- The app is now the `chat-hacksman` command (`python -m chat_hacksman`),
  in a `src/chat_hacksman` package, replacing `python -m app.main`.
- Settings are validated at startup: a bad value stops the app with a
  message naming the variable.
- Requires Python 3.15. Development and releases use uv, Docker compose
  files for dev/test/lint/prod, pre-commit, and the python-ai-rules CI,
  Lint and Release workflows.
- SQLite now uses WAL mode, foreign keys and a versioned schema.

### Added

- `EMBEDDING_QUERY_PREFIX`, defaulting to the query prefix of known models
  (ModernBERT and Nomic Embed, E5, mxbai, Qwen3 Embedding, OpenAI).
- `EMBEDDING_API_KEY`, sent as a bearer token to the embedding server.
- `GET /healthz` with the app's version, for health checks.
- `chat-hacksman --version`.
- MIT license, `docs/configuration.md`, `docs/overview.md`, SECURITY.md and
  AGENTS.md.
