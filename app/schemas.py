"""API istek/cevap şemaları (Pydantic modelleri)."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    index_size: int


class EmbeddingResponse(BaseModel):
    filename: str
    embedding_dim: int
    processing_time_ms: float
    vector: list[float] | None = None


class IndexAddResponse(BaseModel):
    image_id: str
    animal_id: str
    species: str | None
    index_size: int


class MatchQueryInfo(BaseModel):
    filename: str
    processing_time_ms: float


class MatchItem(BaseModel):
    rank: int
    image_id: str
    animal_id: str
    species: str | None
    similarity: float
    image_path: str


class MatchResponse(BaseModel):
    query: MatchQueryInfo
    matches: list[MatchItem]


class IndexReloadResponse(BaseModel):
    status: str
    index_size: int


class ErrorResponse(BaseModel):
    detail: str


class IndexMetadata(BaseModel):
    """/index isteğinde isteğe bağlı serbest-form metadata."""

    data: dict[str, Any] = Field(default_factory=dict)
