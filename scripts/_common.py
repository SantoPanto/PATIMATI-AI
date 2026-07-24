"""Script'ler arasında paylaşılan embedding çıkarma ve leave-one-out değerlendirme yardımcıları."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from app.services.embedding_service import EmbeddingError, EmbeddingService
from app.services.image_service import ImageLoadError, ScannedImage, load_rgb_image


def extract_with_fallback(
    embedding_service: EmbeddingService, images: list[Image.Image], batch_size: int
) -> np.ndarray:
    """MPS/bellek hatalarında batch_size'ı otomatik yarıya indirerek embedding çıkarır."""
    try:
        return embedding_service.embed_images(images, batch_size=batch_size)
    except EmbeddingError:
        if batch_size <= 1:
            raise
        new_batch_size = batch_size // 2
        print(f"  Uyarı: batch_size={batch_size} başarısız, {new_batch_size} ile tekrar deneniyor.")
        return extract_with_fallback(embedding_service, images, batch_size // 2)


def load_and_embed(
    scanned_images: list[ScannedImage], embedding_service: EmbeddingService, batch_size: int
) -> tuple[np.ndarray, list[ScannedImage], list[tuple[str, str]]]:
    """Taranmış görselleri yükleyip embedding çıkarır. Bozuk dosyaları atlayıp raporlar.

    Döndürür: (embeddings, yüklenen görsellerin metadatası, (yol, hata) atlanan görseller).
    """
    valid_images: list[Image.Image] = []
    valid_meta: list[ScannedImage] = []
    load_failures: list[tuple[str, str]] = []

    for scanned in tqdm(scanned_images, desc="Görseller yükleniyor", ncols=100):
        try:
            valid_images.append(load_rgb_image(scanned.path))
            valid_meta.append(scanned)
        except ImageLoadError as exc:
            load_failures.append((str(scanned.path), str(exc)))

    if not valid_images:
        return np.zeros((0, embedding_service.embedding_dim), dtype=np.float32), [], load_failures

    all_embeddings: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(valid_images), batch_size), desc="Embedding çıkarılıyor", ncols=100
    ):
        batch = valid_images[start : start + batch_size]
        all_embeddings.append(extract_with_fallback(embedding_service, batch, batch_size))

    return np.concatenate(all_embeddings, axis=0), valid_meta, load_failures


@dataclass
class QueryResult:
    """Bir leave-one-out sorgusunun sonucu (evaluate_baseline.py / evaluate_model.py için)."""

    query_path: str
    query_animal_id: str
    has_positive: bool
    top1_correct: bool = False
    topk_correct: bool = False
    reciprocal_rank: float = 0.0
    correct_similarity: float | None = None
    wrong_similarity: float | None = None
    latency_ms: float = 0.0
    top_candidates: list[tuple[str, str, float, bool]] | None = (
        None  # path, animal_id, sim, is_correct
    )


def evaluate_leave_one_out(
    embeddings: np.ndarray, metas: list[ScannedImage], top_k: int
) -> list[QueryResult]:
    """Her satırı sırayla sorgu kabul ederek (kendisi hariç) leave-one-out değerlendirme yapar."""
    animal_ids = np.array([m.animal_id for m in metas])
    results: list[QueryResult] = []

    for i in range(len(metas)):
        start = time.perf_counter()

        mask = np.ones(len(metas), dtype=bool)
        mask[i] = False
        candidate_indices = np.nonzero(mask)[0]

        if len(candidate_indices) == 0:
            results.append(
                QueryResult(
                    query_path=str(metas[i].path),
                    query_animal_id=metas[i].animal_id,
                    has_positive=False,
                )
            )
            continue

        similarities = embeddings[candidate_indices] @ embeddings[i]
        order = np.argsort(-similarities)
        ranked_indices = candidate_indices[order]
        ranked_similarities = similarities[order]

        query_animal_id = metas[i].animal_id
        is_correct = animal_ids[ranked_indices] == query_animal_id
        has_positive = bool(is_correct.any())
        latency_ms = (time.perf_counter() - start) * 1000

        top_candidates = [
            (str(metas[idx].path), str(metas[idx].animal_id), float(sim), bool(correct))
            for idx, sim, correct in zip(
                ranked_indices[:top_k], ranked_similarities[:top_k], is_correct[:top_k], strict=True
            )
        ]

        if not has_positive:
            results.append(
                QueryResult(
                    query_path=str(metas[i].path),
                    query_animal_id=query_animal_id,
                    has_positive=False,
                    latency_ms=latency_ms,
                    top_candidates=top_candidates,
                )
            )
            continue

        first_correct_rank = int(np.argmax(is_correct)) + 1  # 1-indexed
        correct_similarity = float(ranked_similarities[is_correct].max())
        wrong_similarity = (
            float(ranked_similarities[~is_correct].max()) if (~is_correct).any() else None
        )

        results.append(
            QueryResult(
                query_path=str(metas[i].path),
                query_animal_id=query_animal_id,
                has_positive=True,
                top1_correct=bool(is_correct[0]),
                topk_correct=bool(is_correct[:top_k].any()),
                reciprocal_rank=1.0 / first_correct_rank,
                correct_similarity=correct_similarity,
                wrong_similarity=wrong_similarity,
                latency_ms=latency_ms,
                top_candidates=top_candidates,
            )
        )

    return results


def summarize_results(results: list[QueryResult], top_k: int) -> dict[str, Any]:
    """`evaluate_leave_one_out` sonuçlarından toplu metrikler üretir."""
    processed = [r for r in results if r.has_positive]
    skipped = [r for r in results if not r.has_positive]

    if not processed:
        return {
            "processed_queries": 0,
            "skipped_queries": len(skipped),
            "top1_accuracy": None,
            f"top_{top_k}_accuracy": None,
            "mean_reciprocal_rank": None,
            "avg_correct_similarity": None,
            "avg_wrong_similarity": None,
            "avg_query_latency_ms": None,
        }

    correct_sims = [r.correct_similarity for r in processed if r.correct_similarity is not None]
    wrong_sims = [r.wrong_similarity for r in processed if r.wrong_similarity is not None]
    all_latencies = [r.latency_ms for r in results]

    return {
        "processed_queries": len(processed),
        "skipped_queries": len(skipped),
        "top1_accuracy": sum(r.top1_correct for r in processed) / len(processed),
        f"top_{top_k}_accuracy": sum(r.topk_correct for r in processed) / len(processed),
        "mean_reciprocal_rank": sum(r.reciprocal_rank for r in processed) / len(processed),
        "avg_correct_similarity": (sum(correct_sims) / len(correct_sims)) if correct_sims else None,
        "avg_wrong_similarity": (sum(wrong_sims) / len(wrong_sims)) if wrong_sims else None,
        "avg_query_latency_ms": (sum(all_latencies) / len(all_latencies))
        if all_latencies
        else None,
    }
