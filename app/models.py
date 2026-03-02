from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ProviderName = Literal["ollama", "llama_cpp"]


class ProviderTweaks(BaseModel):
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    num_ctx: int = Field(default=4096, ge=256, le=32768)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=3.0)
    seed: int | None = None


class AppSettings(BaseModel):
    provider: ProviderName = "ollama"
    llm_base_url: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    system_prompt: str
    enabled_rag_collections: list[str] = Field(default_factory=list)
    tweaks: ProviderTweaks = Field(default_factory=ProviderTweaks)


class SettingsUpdate(BaseModel):
    provider: ProviderName
    llm_base_url: str
    model: str
    system_prompt: str
    enabled_rag_collections: list[str] = Field(default_factory=list)
    tweaks: ProviderTweaks


class RagCollectionsResponse(BaseModel):
    available_collections: list[str]
    enabled_collections: list[str]


class ModelsResponse(BaseModel):
    provider: ProviderName
    models: list[str]
    error: str | None = None


class ModelPullRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    provider: ProviderName | None = None
    base_url: str | None = None


class ModelPullResponse(BaseModel):
    provider: ProviderName
    pulled_model: str
    status: str


class MetaResponse(BaseModel):
    app_name: str
    version: str
    repo_url: str


class ContextChunk(BaseModel):
    collection: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    use_rag: bool = True
    temporary_chat: bool = False


class ChatResponse(BaseModel):
    answer: str
    context: list[ContextChunk] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class HistoryRecord(BaseModel):
    id: int
    created_at: datetime
    provider: str
    model: str
    system_prompt: str
    question: str
    answer: str
    rag_collections: list[str]
    rag_hits: int
    config_snapshot: dict[str, Any]
