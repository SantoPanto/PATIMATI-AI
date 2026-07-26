# app/models.py
from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    embedding: list[float]          # 512 boyutlu CLIP vektörü
    labels: list[str]               # Eşleştirme skorunda kullanılan etiketler
    species: str                    # "cat" | "dog" | "unknown"
    species_confidence: float = 0.0
    breed: str | None = None        # Güven eşiğinin altındaysa None
    breed_confidence: float = 0.0
    pattern: str | None = None      # tabby | spotted | solid | bicolor
    colors: list[dict]              # Dominant renkler


class MatchCandidate(BaseModel):
    pet_id: str
    embedding: list[float]
    labels: list[str]
    species: str
    distance_km: float


class MatchRequest(BaseModel):
    embedding: list[float]
    labels: list[str]
    species: str
    candidates: list[MatchCandidate]


class MatchResult(BaseModel):
    pet_id: str
    score: float
    visual: float
    label: float
    location: float
    match: bool
