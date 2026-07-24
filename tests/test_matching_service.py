"""app/services/index_service.py ve app/services/matching_service.py için testler."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.image_service import load_rgb_image
from app.services.index_service import IndexRecord, LocalNumpyIndex
from app.services.matching_service import MatchingService


def make_record(
    image_id: str, animal_id: str, vector: np.ndarray, species: str | None = "cat"
) -> IndexRecord:
    return IndexRecord(
        embedding=vector.astype(np.float32),
        image_id=image_id,
        image_path=f"{animal_id}/{image_id}.jpg",
        animal_id=animal_id,
        species=species,
    )


def test_add_record_increases_index_size() -> None:
    index = LocalNumpyIndex()
    assert len(index) == 0
    index.add(make_record("img1", "kedi_001", np.array([1.0, 0.0, 0.0])))
    assert len(index) == 1
    assert "img1" in index


def test_search_returns_ranked_results_by_similarity() -> None:
    index = LocalNumpyIndex()
    base = np.array([1.0, 0.0, 0.0])
    index.add(make_record("img1", "kedi_001", base))
    index.add(make_record("img2", "kedi_002", np.array([0.0, 1.0, 0.0])))
    index.add(make_record("img3", "kedi_001", np.array([0.9, 0.1, 0.0])))

    results = index.search(base, top_k=2)

    assert len(results) == 2
    assert results[0].record.image_id == "img1"
    assert results[0].similarity >= results[1].similarity
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_search_excludes_same_image_id() -> None:
    index = LocalNumpyIndex()
    base = np.array([1.0, 0.0, 0.0])
    index.add(make_record("img1", "kedi_001", base))
    index.add(make_record("img2", "kedi_001", base))

    results = index.search(base, top_k=5, exclude_image_id="img1")

    assert all(r.record.image_id != "img1" for r in results)
    assert len(results) == 1


def test_search_on_empty_index_returns_empty_list() -> None:
    index = LocalNumpyIndex()
    results = index.search(np.array([1.0, 0.0, 0.0]), top_k=5)
    assert results == []


def test_species_filter_restricts_results() -> None:
    index = LocalNumpyIndex()
    base = np.array([1.0, 0.0, 0.0])
    index.add(make_record("img1", "kedi_001", base, species="cat"))
    index.add(make_record("img2", "kopek_001", base, species="dog"))

    results = index.search(base, top_k=5, species="dog")

    assert len(results) == 1
    assert results[0].record.species == "dog"


def test_min_similarity_threshold_filters_low_scores() -> None:
    index = LocalNumpyIndex()
    index.add(make_record("img1", "kedi_001", np.array([1.0, 0.0, 0.0])))
    index.add(make_record("img2", "kedi_002", np.array([-1.0, 0.0, 0.0])))

    results = index.search(np.array([1.0, 0.0, 0.0]), top_k=5, min_similarity=0.5)

    assert len(results) == 1
    assert results[0].record.image_id == "img1"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    index = LocalNumpyIndex()
    index.add(make_record("img1", "kedi_001", np.array([1.0, 0.0, 0.0])))
    path = tmp_path / "index.npz"

    index.save(path)
    reloaded = LocalNumpyIndex.from_file(path)

    assert len(reloaded) == 1
    assert reloaded.records[0].animal_id == "kedi_001"


def test_matching_service_finds_matching_animal(
    tiny_embedding_service, sample_image_path: Path
) -> None:
    image = load_rgb_image(sample_image_path)
    vector = tiny_embedding_service.embed_image(image)

    index = LocalNumpyIndex()
    index.add(make_record("img1", "kedi_001", vector))

    service = MatchingService(tiny_embedding_service, index)
    results = service.match(image, top_k=1)

    assert len(results) == 1
    assert results[0].record.animal_id == "kedi_001"
    assert results[0].similarity > 0.99
