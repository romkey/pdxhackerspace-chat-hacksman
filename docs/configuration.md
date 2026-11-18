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

## Qdrant (retrieval)

The collections and payload come from the shared RAG pipeline: each point
has `text`, `title`, `heading_path`, `url`, `source`, `tier`, `authority`
(5 policy/sign ... 1 Slack fact), optional `stale_after` and
`superseded_by`, and a dense and a sparse (BM25) vector. Each collection is
searched both ways and the results fused with reciprocal rank fusion. The
sparse query's text is embedded by Qdrant itself (server-side BM25
inference, `qdrant/bm25`); check that the running Qdrant supports it and
that the ingester's vector names match. If the hybrid query fails, dense
search alone is used.

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server |
| `QDRANT_API_KEY` | unset | Secret. Qdrant API key; use one that can read only the member bot's collections |
| `RAG_COLLECTIONS` | `hs_member,hs_catalog` | Comma-separated collections to search. Never `hs_public` alone for the member bot; never the member collections for the public bot |
| `RAG_CANDIDATES` | `50` | Hits per collection from the fused search, for the reranker to choose from |
| `RAG_TOP_N` | `8` | Chunks given to the model |
| `RAG_DENSE_VECTOR` | `dense` | Name of the dense vector; `none` for an unnamed vector |
| `RAG_SPARSE_VECTOR` | `sparse` | Name of the sparse (BM25) vector; `none` for dense-only search |
| `RAG_SPARSE_MODEL` | `qdrant/bm25` | Model Qdrant uses to embed the sparse query; must match the ingester's |
| `RAG_MIN_SCORE` | `0.5` | Lowest similarity to keep, in dense-only search only (fused scores are rank-based) |
| `RAG_ALLOWED_TIERS` | `public,member` | Points with any other `tier` are never used, whatever collection they are in. `admin` is refused |
| `RAG_AUTHORITY_WEIGHT` | `0.1` | How much authority lifts a chunk: relevance is scaled 0-1 over the candidates and a chunk gets up to this much more for authority 5. `0` is relevance alone. A policy or sign chunk with scaled relevance of at least 0.5 is always included |

## Conversation and answers

| Variable | Default | Meaning |
|----------|---------|---------|
| `CHAT_HISTORY_TURNS` | `6` | Earlier messages of the conversation used to rewrite the question into a search query and sent to the model. `0` treats every question alone |
| `NO_ANSWER_MESSAGE` | a pointer to the wiki, Slack and the open house | The answer when nothing relevant is found; the model isn't asked |

## Embeddings

Queries are embedded with an OpenAI-compatible embeddings API
(`POST {EMBEDDING_URL}/embeddings`, [API reference](https://platform.openai.com/docs/api-reference/embeddings/create)),
such as OpenAI, Hugging Face TEI, Infinity, vLLM or llama.cpp's
`llama-server`. It must be the model the collections were ingested with. If the server is unreachable, RAG is skipped.

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_URL` | `http://localhost:8080/v1` | Base URL including `/v1`, e.g. `https://api.openai.com/v1` |
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

## Reranking

An optional second pass after the search. The `RAG_CANDIDATES` hits from
every collection are sent with the query to a cross-encoder reranker,
which scores how well each chunk (with its title and heading) answers it;
the authority pass then picks the best `RAG_TOP_N` for the model, from
whichever collections they come from. Reranking is off unless `RERANK_URL`
is set, and if the reranker fails the search order is used.
`RERANK_MIN_SCORE` drops weak chunks; if none are left, the bot gives the
no-answer message.

| Variable | Default | Meaning |
|----------|---------|---------|
| `RERANK_URL` | unset (off) | Base URL; requests go to `{RERANK_URL}/rerank` |
| `RERANK_API` | `jina` | `jina`: Jina/Cohere format (`documents` in, `results[].relevance_score` out), spoken by Jina, Cohere, llama.cpp, vLLM and Infinity. `tei`: Hugging Face text-embeddings-inference format (`texts` in, `[{index, score}]` out) |
| `RERANK_API_KEY` | unset | Secret. Sent as `Authorization: Bearer ...` |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Model name (not sent with `tei`, which serves one model) |
| `RERANK_MIN_SCORE` | unset | Drop chunks scored below this. The scale depends on the server and model (TEI returns 0-1; llama.cpp can return raw scores, about -11 to 4 for `bge-reranker-v2-m3`), so check a few `rerank_score` values in chat responses first |
| `RERANK_CONTEXT_LENGTH` | `8192` | Tokens per question + chunk pair; the question gets at most a quarter, chunks are cut to fit the rest |
| `RERANK_TIMEOUT_SECONDS` | `30` | Timeout for each rerank request |

Where to get one (any of these serve the default model):

| Server | Start it | Settings |
|--------|----------|----------|
| llama.cpp | `llama-server -m <reranker>.gguf --reranking --port 8012` | `RERANK_URL=http://localhost:8012/v1` |
| vLLM | `vllm serve BAAI/bge-reranker-v2-m3` | `RERANK_URL=http://localhost:8000/v1` |
| [Hugging Face TEI](https://huggingface.github.io/text-embeddings-inference/) | `docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 --model-id BAAI/bge-reranker-v2-m3` (`cpu-arm64-1.9` on ARM) | `RERANK_URL=http://localhost:8080`, `RERANK_API=tei` |
| Infinity | its `/rerank` endpoint | `RERANK_URL=<base URL>` |
| Jina AI | hosted | `RERANK_URL=https://api.jina.ai/v1`, `RERANK_API_KEY` |
| Cohere | hosted | `RERANK_URL=https://api.cohere.com/v2`, `RERANK_API_KEY` |

Reranker models:

| Model | Languages | Context | License | Notes |
|-------|-----------|---------|---------|-------|
| `BAAI/bge-reranker-v2-m3` (default) | multilingual | 8192 | Apache-2.0 | |
| `Alibaba-NLP/gte-reranker-modernbert-base` | English | 8192 | Apache-2.0 | Smaller and faster |
| `BAAI/bge-reranker-base`, `-large` | English, Chinese | 512 | MIT | Set `RERANK_CONTEXT_LENGTH=512` |
| `mixedbread-ai/mxbai-rerank-base-v1` | English | 512 | Apache-2.0 | Set `RERANK_CONTEXT_LENGTH=512` |
| `jinaai/jina-reranker-v2-base-multilingual` | multilingual | 1024 | CC-BY-NC-4.0 | Non-commercial use only |

## Chat LLM

Answers come from an OpenAI-compatible chat completions API
(`POST {LLM_URL}/chat/completions`, [API reference](https://platform.openai.com/docs/api-reference/chat/create)),
such as OpenAI, llama.cpp's `llama-server`, vLLM, LM Studio or LiteLLM.
The server is set only here; the UI can pick a model from the ones it lists
at `GET {LLM_URL}/models`, or leave it on the default.

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_URL` | `http://localhost:8080/v1` | Base URL including `/v1`, e.g. `https://api.openai.com/v1` |
| `LLM_API_KEY` | unset | Secret. Sent as `Authorization: Bearer ...` (chat, model list, status check) |
| `LLM_MODEL` | unset | Model used unless one is picked in the UI. Unset: the first model the server lists, which suits `llama-server` (it serves one) |
| `DEFAULT_SYSTEM_PROMPT` | the member bot's instructions (entering and leaving, "where is", the library) | The bot's own prompt until one is saved in the UI. It always follows the fixed answering rules in `llm.ANSWERING_RULES`, which can't be changed |
| `LLM_TIMEOUT_SECONDS` | `180` | Timeout for each chat request |
| `LLM_RETRY_ATTEMPTS` | `2` | Attempts per chat request when the connection fails or times out |
| `LLM_RETRY_BACKOFF_SECONDS` | `0.5` | Wait before retrying; the nth retry waits n times this |

`DEFAULT_PROVIDER`, `DEFAULT_LLM_BASE_URL`, `DEFAULT_MODEL` and
`OLLAMA_API_KEY` were replaced by the variables above in 0.8.0;
`RAG_COLLECTION_1` ... `_5` (use `RAG_COLLECTIONS`), `RAG_TOP_K`,
`RAG_TOP_K_EVENTS`, `RAG_TOP_K_SLACK`, `RAG_TOP_K_CALIBRE` (use
`RAG_CANDIDATES` and `RAG_TOP_N`), `RERANK_CANDIDATES` and `RERANK_TOP_N`
in 0.9.0. The app won't start while any of them is set, so a stale `.env`
can't silently change what it does.

## Docker (not read by the app)

| Variable | Meaning |
|----------|---------|
| `APP_VERSION` | Image tag `docker-compose.prod.yaml` runs; default `latest` |

## Publishing (not read by the app)

| Variable | Meaning |
|----------|---------|
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. Releases from GitHub don't need it. |
