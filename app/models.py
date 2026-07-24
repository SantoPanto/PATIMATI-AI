# app/models.py
from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    embedding: list[float]   # 512 boyutlu CLIP vektörü
    labels: list[str]        # Vision API etiketleri
    species: str             # "cat" | "dog" | "unknown"
    colors: list[dict]       # Dominant renkler


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
