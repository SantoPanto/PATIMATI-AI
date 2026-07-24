"""Yerel, NumPy tabanlı vektör indeksi.

`VectorIndex`, arama arayüzünü somut depolama biçiminden ayırır; böylece
ileride (örn. pgvector) farklı bir backend'e geçilirse yalnızca bu dosyadaki
somut sınıf değişir, `matching_service` ve API kodu değişmeden kalır.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class IndexError_(RuntimeError):
    """İndeks işlemleri sırasında oluşan hatalar için (builtin IndexError ile çakışmasın diye)."""


@dataclass(frozen=True)
class IndexRecord:
    """İndekste saklanan tek bir görsel kaydı."""

    embedding: np.ndarray
    image_id: str
    image_path: str
    animal_id: str
    species: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Bir aramanın tek bir sonucu."""

    rank: int
    record: IndexRecord
    similarity: float


class VectorIndex(ABC):
    """Vektör indekslerinin uyması gereken soyut arayüz."""

    @abstractmethod
    def add(self, record: IndexRecord) -> None: ...

    @abstractmethod
    def add_batch(self, records: list[IndexRecord]) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        exclude_image_id: str | None = None,
        species: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __contains__(self, image_id: str) -> bool: ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalNumpyIndex(VectorIndex):
    """Küçük/orta ölçekli veri setleri için bellek içi, NumPy tabanlı cosine-similarity indeksi."""

    def __init__(self) -> None:
        self._records: list[IndexRecord] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, image_id: str) -> bool:
        return any(r.image_id == image_id for r in self._records)

    @property
    def records(self) -> list[IndexRecord]:
        return list(self._records)

    def add(self, record: IndexRecord) -> None:
        self.add_batch([record])

    def add_batch(self, records: list[IndexRecord]) -> None:
        if not records:
            return
        self._records.extend(records)
        self._matrix = None  # search matrisini tembel olarak yeniden kur

    def _matrix_view(self) -> np.ndarray:
        if self._matrix is None:
            if not self._records:
                raise IndexError_("İndeks boş; arama yapılamaz.")
            stacked = np.stack([r.embedding for r in self._records]).astype(np.float32)
            self._matrix = _normalize_rows(stacked)
        return self._matrix

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        exclude_image_id: str | None = None,
        species: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise IndexError_("top_k en az 1 olmalı.")
        if not self._records:
            return []

        matrix = self._matrix_view()
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise IndexError_("Sorgu embedding'i sıfır vektör olamaz.")
        query = query / query_norm

        similarities = (matrix @ query.T).ravel()

        candidates: list[tuple[float, IndexRecord]] = []
        for similarity, record in zip(similarities, self._records, strict=True):
            if exclude_image_id is not None and record.image_id == exclude_image_id:
                continue
            if species is not None and record.species != species:
                continue
            if min_similarity is not None and similarity < min_similarity:
                continue
            candidates.append((float(similarity), record))

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        top = candidates[:top_k]
        return [
            SearchResult(rank=i + 1, record=record, similarity=similarity)
            for i, (similarity, record) in enumerate(top)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            raise IndexError_("Boş bir indeks diske kaydedilemez.")
        embeddings = np.stack([r.embedding for r in self._records]).astype(np.float32)
        np.savez(
            path,
            embeddings=embeddings,
            image_id=np.array([r.image_id for r in self._records], dtype=object),
            image_path=np.array([r.image_path for r in self._records], dtype=object),
            animal_id=np.array([r.animal_id for r in self._records], dtype=object),
            species=np.array([r.species or "" for r in self._records], dtype=object),
            metadata=np.array(
                [json.dumps(r.metadata, ensure_ascii=False) for r in self._records], dtype=object
            ),
        )

    def load(self, path: Path) -> None:
        if not path.exists():
            raise IndexError_(f"İndeks dosyası bulunamadı: {path}")
        with np.load(path, allow_pickle=True) as data:
            embeddings = data["embeddings"]
            records = [
                IndexRecord(
                    embedding=embeddings[i],
                    image_id=str(data["image_id"][i]),
                    image_path=str(data["image_path"][i]),
                    animal_id=str(data["animal_id"][i]),
                    species=(str(data["species"][i]) or None),
                    metadata=json.loads(data["metadata"][i]) if data["metadata"][i] else {},
                )
                for i in range(len(embeddings))
            ]
        self._records = records
        self._matrix = None

    @classmethod
    def from_file(cls, path: Path) -> LocalNumpyIndex:
        index = cls()
        index.load(path)
        return index
