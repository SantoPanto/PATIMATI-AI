"""Sorgu görselini embedding'e çevirip indekste en benzer hayvanları arayan servis."""

from __future__ import annotations

from PIL import Image

from app.services.embedding_service import EmbeddingService
from app.services.index_service import SearchResult, VectorIndex


class MatchingService:
    """`EmbeddingService` ve `VectorIndex`'i birleştirerek görsel eşleştirme yapar."""

    def __init__(self, embedding_service: EmbeddingService, index: VectorIndex) -> None:
        self._embedding_service = embedding_service
        self._index = index

    def match(
        self,
        image: Image.Image,
        top_k: int = 5,
        exclude_image_id: str | None = None,
        species: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Sorgu görseli için indeksteki en benzer ilk `top_k` hayvanı döndürür."""
        query_embedding = self._embedding_service.embed_image(image)
        return self._index.search(
            query_embedding=query_embedding,
            top_k=top_k,
            exclude_image_id=exclude_image_id,
            species=species,
            min_similarity=min_similarity,
        )
