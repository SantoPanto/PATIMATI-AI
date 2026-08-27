# tests/test_attributes.py
# Gerçek fotoğraflarla zero-shot öznitelik testi.
# Gerektirir: python scripts/prepare_seed.py (tests/seed_data)
from pathlib import Path

import pytest

from app.attributes import BREEDS, attribute_analyzer
from app.embedder import embedder

BREED_SPECIES = {ad: tur for ad, tur in BREEDS}

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
    # Sözleşmenin beklediği cins alanları (docs/entegrasyon-sozlesmesi.md §4)
    assert d["breed"] is None or d["breed"] in BREED_SPECIES
    assert 0.0 <= d["breed_confidence"] <= 1.0
    assert 0.0 <= d["species_confidence"] <= 1.0


def test_breed_bilinen_irki_bulur():
    # class_25 = Pug — çok ayırt edici bir ırk, üç fotoğrafın en az ikisi bilinmeli
    sonuc = [_analyze(p)["breed"] for p in sorted((SEED / "class_25").glob("*.jpg"))]
    assert sonuc.count("Pug") >= 2, f"beklenen Pug, gelen: {sonuc}"


def test_breed_tur_daraltmasi_uygulaniyor():
    """Tür daraltması gerçekten aday listesini kısıtlamalı.

    Temiz fotoğraflarda daraltma sonucu değiştirmiyor (CLIP kediyle köpeği hiç
    karıştırmıyor), ama bozuk/bulanık görüntülerde güvence olarak duruyor.
    Bu test mekanizmanın çalıştığını kanıtlar: köpek fotoğrafı "cat" ile
    zorlandığında bir KEDİ ırkı dönmeli.
    """
    emb = embedder.embed_bytes((SEED / "class_25" / "0.jpg").read_bytes())  # Pug
    irk_kedi, _ = attribute_analyzer.predict_breed(emb, "cat")
    irk_kopek, _ = attribute_analyzer.predict_breed(emb, "dog")
    assert BREED_SPECIES[irk_kedi] == "cat", f"kedi bekleniyordu: {irk_kedi}"
    assert BREED_SPECIES[irk_kopek] == "dog", f"köpek bekleniyordu: {irk_kopek}"


def test_breed_esik_altinda_isim_dondurmez():
    """Güven eşiğinin altındaki tahmin isim olarak dönmemeli (yanlış cins göstermek yerine boş)."""
    d = _analyze(next((SEED / "class_25").glob("*.jpg")))
    if d["breed"] is not None:
        assert d["breed_confidence"] >= attribute_analyzer.BREED_MIN_PROB


def test_breed_top_esikten_bagimsiz_dolu():
    """breed eşik altında None olsa da breed_top EN İYİ tahminin adını taşır
    (ilan formunun düşük güvenli öneri dolumu, 2026-08-27 S5). Hayvansa
    breed_top boş kalmamalı; breed doluysa ikisi aynı isim olmalı."""
    d = _analyze(next((SEED / "class_25").glob("*.jpg")))
    if d["is_pet"]:
        assert d["breed_top"], "hayvan fotoğrafında breed_top boş dönmemeli"
        if d["breed"] is not None:
            assert d["breed"] == d["breed_top"]


def test_same_photo_same_labels():
    # Aynı fotoğraf iki kez analiz edilirse etiketler birebir aynı olmalı (Jaccard tutarlılığı)
    p = next((SEED / "class_05").glob("*.jpg"))
    assert _analyze(p)["labels"] == _analyze(p)["labels"]
