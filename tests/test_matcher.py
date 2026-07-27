# tests/test_matcher.py
import numpy as np
import pytest

from app.matcher import cosine_similarity, jaccard_score, compute_final_score
from app.surum import VEKTOR_BOYUTU


def make_embedding(seed=42, size=None):
    # Boyut sabit yazılmaz: kimlik modeli değişince (KIMLIK_MODEL ortam
    # değişkeni) vektör boyutu da değişiyor — CLIP 512, SigLIP2 768.
    rng = np.random.default_rng(seed)
    v = rng.random(size or VEKTOR_BOYUTU).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_identical_embeddings():
    emb = make_embedding()
    assert cosine_similarity(emb, emb) == pytest.approx(1.0, abs=1e-5)


def test_different_embeddings():
    a, b = make_embedding(1), make_embedding(2)
    sim = cosine_similarity(a, b)
    assert 0.0 <= sim <= 1.0


def test_jaccard_full_overlap():
    labels = ["cat", "orange", "fur"]
    assert jaccard_score(labels, labels) == 1.0


def test_jaccard_no_overlap():
    assert jaccard_score(["cat"], ["dog"]) == 0.0


def test_species_mismatch_returns_zero():
    emb = make_embedding()
    result = compute_final_score(emb, emb, ["cat"], ["dog"], 0.0, "cat", "dog")
    assert result["score"] == 0.0
    assert result["blocked_reason"] == "species_mismatch"
    # cevap şekli normal yolla AYNI olmalı — eksik "match" anahtarı gerçek bug'dı (demo'da bulundu)
    assert result["match"] is False


def test_perfect_match():
    emb = make_embedding()
    result = compute_final_score(emb, emb, ["cat"], ["cat"], 0.0, "cat", "cat")
    assert result["score"] >= 0.80
    assert result["match"] is True
