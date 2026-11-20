# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases up to 0.5.7 are
described by their git tags and commit messages.

## [Unreleased]

## [0.11.0] - 2026-11-20

### Added

- Login page with OpenID Connect single sign-on (`OIDC_ISSUER`,
  `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, e.g. Authentik) and one
  local account (`LOCAL_LOGIN_USERNAME`, `LOCAL_LOGIN_PASSWORD`). The
  button reads `OIDC_LOGIN_LABEL` (default "Login via Authentik"); the local
  form is under a "Local login" disclosure. Sessions last `SESSION_DAYS`
  and survive restarts. The UI footer shows who is logged in, with a
  logout button.

### Removed

- HTTP basic auth. `BASIC_AUTH_USERNAME` and `BASIC_AUTH_PASSWORD` now stop
  the app with a message naming their replacements.

## [0.10.1] - 2026-11-19

### Changed

- Chat answers are rendered as markdown (headings, lists, emphasis, code,
  tables, quotes, links) instead of plain text. Citations like [2] link to
  their source below the answer. The renderer builds DOM nodes, so HTML in
  an answer shows as text, and links are limited to http and https.

## [0.10.0] - 2026-11-19

### Added

- `LLM_EXTRA_BODY`: a JSON object merged into every chat request, e.g.
  `{"chat_template_kwargs": {"enable_thinking": false}}` to turn a
  reasoning model's thinking off. `.env.example` sets that.

### Fixed

- An empty answer from the model is reported as an error saying why (for
  example, a reasoning model used all of `max_tokens` thinking), instead
  of showing nothing.
- A configured collection that doesn't exist in Qdrant is skipped quietly
  and looked for again every 5 minutes, instead of logging warnings on
  every question.

### Changed

- `.env.example` sets `RAG_CANDIDATES=20` and `RERANK_CONTEXT_LENGTH=512`,
  which rerank much faster; the built-in defaults are unchanged.

## [0.9.0] - 2026-11-18

### Changed

- **Breaking:** The bot is the member chatbot of the shared RAG pipeline.
  It searches `hs_member` and `hs_catalog` by default and reads the
  pipeline's point payload (`text`, `title`, `heading_path`, `url`,
  `source`, `tier`, `authority`, `stale_after`, `superseded_by`...). The
  special handling of `events*`, `slack*` and `calibre*` collections is
  gone.
- **Breaking:** Search is hybrid: each collection's dense vector
  (`RAG_DENSE_VECTOR`) and sparse BM25 vector (`RAG_SPARSE_VECTOR`, embedded
  by Qdrant) are queried and fused with reciprocal rank fusion. If the
  hybrid query fails, dense search alone is used.
- **Breaking:** `RAG_CANDIDATES` (50 per collection) and `RAG_TOP_N` (8)
  replace `RAG_TOP_K`, `RAG_TOP_K_EVENTS`, `RAG_TOP_K_SLACK`,
  `RAG_TOP_K_CALIBRE`, `RERANK_CANDIDATES` and `RERANK_TOP_N`;
  `RAG_COLLECTIONS` replaces `RAG_COLLECTION_1` ... `_5`. The app won't
  start while an old variable is set.
- Points with a `tier` outside `RAG_ALLOWED_TIERS` (default
  `public,member`; never `admin`), points `superseded_by` another and
  points past `stale_after` are never used.
- After reranking, an authority pass prefers policy, signs and the wiki
  over manuals and Slack when relevance is close
  (`RAG_AUTHORITY_WEIGHT`), and always includes a relevant policy or sign.
- The system prompt starts with fixed answering rules shared by both bots:
  answer only from the sources and cite them, treat sources as data, rules
  over manuals, training, verbatim configs, no individuals, incidents,
  credentials or admin procedures, safety first. `DEFAULT_SYSTEM_PROMPT` is
  now the member bot's own instructions; a saved copy of the old default is
  replaced by it.
- The browser keeps the conversation for the session and sends the last
  `CHAT_HISTORY_TURNS` with each question; the model rewrites it into a
  standalone search query.
- Answers stream (`"stream": true` on `POST /api/chat`, newline-delimited
  JSON) and the UI shows the cited sources as links.
- With no sources found, the bot answers `NO_ANSWER_MESSAGE` without asking
  the model. An answer counts as answered only if it cites a source.
- Chats record the retrieved document IDs, their topics and whether they
  were answered (schema 3); client IPs are no longer logged with chats or
  feedback.

### Added

- `GET /api/reports/unanswered` (questions not answered, grouped by topic)
  and `GET /api/reports/negative-feedback` (thumbs-down answers with their
  documents).
- `evals/` and `scripts/run-evals.py`: a golden set and red-team prompts to
  run against a live bot, with recall and canary leak checks.

### Removed

- The "Use RAG" switch and the `use_rag` request field: the bot always
  answers from its sources.

## [0.8.0] - 2026-11-13

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
