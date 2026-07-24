# tests/test_attributes.py
# Gerçek fotoğraflarla zero-shot öznitelik testi.
# Gerektirir: python scripts/prepare_seed.py (tests/seed_data)
from pathlib import Path

import pytest

from app.attributes import attribute_analyzer

SEED = Path("tests/seed_data")
pytestmark = pytest.mark.skipif(
    not SEED.exists(), reason="seed_data yok — önce scripts/prepare_seed.py çalıştırın")


def _analyze(path):
    return attribute_analyzer.analyze(path.read_bytes())


def test_species_cat():
    # class_00 = Abyssinian (kedi) — 3 fotoğrafın en az 2'si doğru bilinmeli
    results = [_analyze(p)["species"] for p in sorted((SEED / "class_00").glob("*.jpg"))]
    assert results.count("cat") >= 2, f"beklenen cat, gelen: {results}"


def test_species_dog():
    # class_01 = American Bulldog (köpek)
    results = [_analyze(p)["species"] for p in sorted((SEED / "class_01").glob("*.jpg"))]
    assert results.count("dog") >= 2, f"beklenen dog, gelen: {results}"


def test_response_shape():
    d = _analyze(next((SEED / "class_00").glob("*.jpg")))
    assert d["labels"], "labels boş olmamalı"
    assert d["species"] in {"cat", "dog", "unknown"}
    assert d["colors"], "colors boş olmamalı"
    assert all(set(c) == {"r", "g", "b", "score"} for c in d["colors"])


def test_same_photo_same_labels():
    # Aynı fotoğraf iki kez analiz edilirse etiketler birebir aynı olmalı (Jaccard tutarlılığı)
    p = next((SEED / "class_05").glob("*.jpg"))
    assert _analyze(p)["labels"] == _analyze(p)["labels"]
