# Configuration

The app reads these environment variables at startup. Put them in `.env`
(copy `.env.example`); compose passes it to the container. An invalid
value stops the app with a message naming the variable. Empty values count
as unset.

## App

| Variable | Default | Meaning |
|----------|---------|---------|
| `APP_HOST` | `127.0.0.1` (`0.0.0.0` in the image) | Address to listen on |
| `APP_PORT` | `8000` | Port to listen on, 1-65535 |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |
| `CHAT_HACKSMAN_DB_PATH` | `data/chat_hacksman.db` (`/data/chat_hacksman.db` in the image) | SQLite database: settings, chat history, feedback, feed events |
| `REPO_URL` | `https://github.com/romkey/pdxhackerspace-chat-hacksman` | Link shown in the UI |
| `BASIC_AUTH_USERNAME` | unset | With `BASIC_AUTH_PASSWORD`, turns on HTTP basic auth for everything but `/health` and `/healthz` |
| `BASIC_AUTH_PASSWORD` | unset | Secret. See `BASIC_AUTH_USERNAME` |
| `SENTRY_DSN` | unset | Secret. Sentry DSN; set to report errors to Sentry |

## Topics feed

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAG_TOPICS_URL` | `https://members.pdxhackerspace.org/rag.json` | JSON feed of topic buttons; its `events` and `occurrences` are stored too |
| `TOPICS_CACHE_TTL_SECONDS` | `300` | How long to cache the feed |

## Qdrant (RAG)

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server |
| `QDRANT_API_KEY` | unset | Secret. Qdrant API key |
| `RAG_COLLECTIONS` | empty | Comma-separated collections to search. None means no RAG |
| `RAG_COLLECTION_1` ... `RAG_COLLECTION_5` | unset | More collections, one per variable, after `RAG_COLLECTIONS`; duplicates are dropped |
| `RAG_TOP_K` | `4` | Chunks per collection (at least 1) |
| `RAG_TOP_K_EVENTS` | `3` | Chunks from `events*` collections |
| `RAG_TOP_K_SLACK` | `3` | Chunks from `slack*` collections |
| `RAG_TOP_K_CALIBRE` | `2` | Chunks from `calibre*` collections |
| `RAG_MIN_SCORE` | `0.5` | Lowest similarity score to keep |

## Embeddings

Queries are embedded with an OpenAI-compatible embeddings API
(`POST {EMBEDDING_URL}/embeddings`, [API reference](https://platform.openai.com/docs/api-reference/embeddings/create)),
such as OpenAI, Hugging Face TEI, Infinity, vLLM, llama.cpp's
`llama-server` or Ollama's `/v1` API. It must be the model the collections
were ingested with. If the server is unreachable, RAG is skipped.

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_URL` | `http://localhost:8080/v1` | Base URL including `/v1`, e.g. `https://api.openai.com/v1` or `http://localhost:11434/v1` |
| `EMBEDDING_API_KEY` | unset | Secret. Sent as `Authorization: Bearer ...` |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Model name the server knows |
| `EMBEDDING_QUERY_PREFIX` | the model's own (below) | Put in front of every query. `none` for no prefix; `\n` is a newline; a space is added if it doesn't end in whitespace |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Tokens the model accepts; longer queries are cut (the prefix is kept whole) |
| `EMBEDDING_TIMEOUT_SECONDS` | `30` | Timeout for each embedding request |

With `EMBEDDING_QUERY_PREFIX` unset, these models get their prefix by name
(case ignored, with or without the `org/` part); others get none. The
ingester must embed documents with the matching document prefix.

| Model | Query prefix | Document prefix | Dimensions | Context |
|-------|--------------|-----------------|------------|---------|
| `nomic-ai/modernbert-embed-base` (default) | `search_query: ` | `search_document: ` | 768 (Matryoshka 256) | 8192 |
| `nomic-ai/nomic-embed-text-v1.5` | `search_query: ` | `search_document: ` | 768 | 8192 |
| `intfloat/e5-large-v2`, `intfloat/multilingual-e5-large` | `query: ` | `passage: ` | 1024 | 512 |
| `mixedbread-ai/mxbai-embed-large-v1` | `Represent this sentence for searching relevant passages: ` | none | 1024 | 512 |
| `Qwen/Qwen3-Embedding-0.6B` | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` | none | 1024 | 32768 |
| OpenAI `text-embedding-3-small` / `-large` | none | none | 1536 / 3072 | 8191 |

## Chat models

Defaults for the settings form; what's saved in the UI wins after that.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DEFAULT_PROVIDER` | `ollama` | `ollama` or `llama_cpp` |
| `DEFAULT_LLM_BASE_URL` | `http://localhost:11434` | Chat server URL |
| `DEFAULT_MODEL` | `llama3.2:latest` | Chat model |
| `DEFAULT_SYSTEM_PROMPT` | `You are Chat Hacksman, an expert assistant for hackerspace members.` | System prompt |
| `OLLAMA_API_KEY` | unset | Secret. Sent as `Authorization: Bearer ...` to Ollama (chat, model list, pull, availability checks) |
| `LLM_TIMEOUT_SECONDS` | `180` | Timeout for each chat request |
| `LLM_RETRY_ATTEMPTS` | `2` | Attempts per chat request when the connection fails or times out |
| `LLM_RETRY_BACKOFF_SECONDS` | `0.5` | Wait before retrying; the nth retry waits n times this |

## Docker (not read by the app)

| Variable | Meaning |
|----------|---------|
| `APP_VERSION` | Image tag `docker-compose.prod.yaml` runs; default `latest` |

## Publishing (not read by the app)

| Variable | Meaning |
|----------|---------|
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. Releases from GitHub don't need it. |
