from datetime import datetime

# Any: payloads, metadata and metrics are free-form JSON objects.
from typing import Any, Literal

from pydantic import BaseModel, Field


class SamplingTweaks(BaseModel):
    """Sampling options sent with each chat request (OpenAI parameter names)."""

    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    seed: int | None = None


class AppSettings(BaseModel):
    # Empty means LLM_MODEL (or the server's first model if that's unset).
    model: str = ""
    system_prompt: str
    enabled_rag_collections: list[str] = Field(default_factory=list)
    tweaks: SamplingTweaks = Field(default_factory=SamplingTweaks)


class SettingsUpdate(BaseModel):
    model: str = Field(default="", max_length=200)
    system_prompt: str
    enabled_rag_collections: list[str] = Field(default_factory=list)
    tweaks: SamplingTweaks


class RagCollectionsResponse(BaseModel):
    available_collections: list[str]
    enabled_collections: list[str]


class ModelsResponse(BaseModel):
    # What an empty AppSettings.model resolves to (LLM_MODEL), if set.
    default_model: str
    models: list[str]
    error: str | None = None


class LlmStatusResponse(BaseModel):
    url: str
    available: bool
    last_changed_at: datetime
    last_checked_at: datetime


class MetaResponse(BaseModel):
    app_name: str
    version: str
    repo_url: str


class WhoAmIResponse(BaseModel):
    login_required: bool
    username: str | None = None


class EventRecord(BaseModel):
    event_id: int
    slug: str
    title: str
    description: str
    more_info_url: str
    visibility: str
    open_to: str
    recurrence_type: str
    start_time: str
    duration: int
    payload: dict[str, Any]


class OccurrenceRecord(BaseModel):
    occurrence_id: int
    slug: str
    occurs_at: str
    occurs_at_unix: int
    ends_at_unix: int
    duration: int
    is_cancelled: bool
    is_postponed: bool
    in_progress: bool
    postponed_until: str | None
    open_to: str
    event_id: int | None
    payload: dict[str, Any]


class ContextChunk(BaseModel):
    collection: str
    # Vector similarity from Qdrant.
    score: float
    text: str
    # Relevance from the reranker, when reranking ran (higher is better).
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatTurn(BaseModel):
    """One earlier message of this conversation, kept by the browser only."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=16000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    # Earlier turns, oldest first; only the last CHAT_HISTORY_TURNS are used.
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    temporary_chat: bool = False
    # Stream the answer as newline-delimited JSON events.
    stream: bool = False


class StageAverage(BaseModel):
    stage: str
    # How many chats ran this stage.
    count: int
    average_ms: float
    min_ms: float
    max_ms: float
    last_ms: float


class TimingStatsResponse(BaseModel):
    """Per-stage running tally, kept in the database, since `since` (the
    first chat timed, or the first after a reset)."""

    chats: int = 0
    errors: int = 0
    since: datetime | None = None
    stages: list[StageAverage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    # The standalone query searched for (the question, rewritten with the history).
    search_query: str = ""
    context: list[ContextChunk] = Field(default_factory=list)
    # Numbers (1-based, into context) of the sources the answer cites.
    citations: list[int] = Field(default_factory=list)
    # Whether the answer came from the sources: there were some and it cites them.
    answered: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    history_id: int | None = None
    # Milliseconds per stage of this chat, in run order, `total` last.
    timings: dict[str, float] = Field(default_factory=dict)
    # Running averages over every chat timed (see TimingStatsResponse).
    timing_stats: TimingStatsResponse | None = None


class HistoryRecord(BaseModel):
    id: int
    created_at: datetime
    prompt_id: int | None = None
    provider: str
    model: str
    system_prompt: str
    question: str
    answer: str
    rag_collections: list[str]
    rag_hits: int
    config_snapshot: dict[str, Any]
    doc_ids: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    # None for chats saved before this was recorded.
    answered: bool | None = None
    # Milliseconds per stage, `total` last; empty for chats saved before 0.12.
    timings: dict[str, float] = Field(default_factory=dict)


class FeedbackCreateRequest(BaseModel):
    rating: Literal["up", "down"]
    text: str = Field(default="", max_length=4000)
    history_id: int | None = None
    question: str = Field(default="", max_length=8000)
    answer: str = Field(default="", max_length=16000)
    provider: str = Field(default="", max_length=200)
    model: str = Field(default="", max_length=200)


class FeedbackRecord(BaseModel):
    id: int
    created_at: datetime
    rating: Literal["up", "down"]
    text: str
    history_id: int | None
    question: str
    answer: str
    provider: str
    model: str


class UnansweredQuestion(BaseModel):
    history_id: int
    created_at: datetime
    question: str


class UnansweredGroup(BaseModel):
    # A topic slug from the sources found, or "unclassified".
    topic: str
    questions: list[UnansweredQuestion]


class UnansweredReport(BaseModel):
    since: datetime
    total: int
    groups: list[UnansweredGroup]


class NegativeFeedbackItem(BaseModel):
    feedback_id: int
    created_at: datetime
    note: str
    question: str
    answer: str
    doc_ids: list[str]


class NegativeFeedbackReport(BaseModel):
    since: datetime
    items: list[NegativeFeedbackItem]
