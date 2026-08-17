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


@pytest.mark.parametrize("a,b", [
    ("CAT", "cat"),      # Java'nın @Enumerated(EnumType.STRING) çıktısı
    ("CAT", "CAT"),
    ("  Cat ", "cat"),   # boşluk + karışık harf
    (None, "cat"),       # alan hiç gelmezse
])
def test_tur_karsilastirmasi_harf_duyarsiz(a, b):
    """Java enum'u BÜYÜK harf üretir, sözleşme küçük harf diyor.

    Düz metin karşılaştırmasında "CAT" != "cat" olduğu için tür kuralı HER adayı
    sıfırlardı — hata vermeden, sadece hiç eşleşme bulunmayarak. İki ayrı repoda
    derleyicinin yakalayamayacağı türden bir hata; testle kapatıldı.
    """
    emb = make_embedding()
    r = compute_final_score(emb, emb, ["cat"], ["cat"], 0.0, a, b)
    assert r.get("blocked_reason") is None, f"{a!r} vs {b!r} yanlışlıkla engellendi"
    assert r["score"] > 0.0


def test_gercek_tur_uyusmazligi_hala_engelleniyor():
    """Harf duyarsızlığı, asıl kuralı gevşetmemeli."""
    emb = make_embedding()
    r = compute_final_score(emb, emb, ["cat"], ["dog"], 0.0, "CAT", "DOG")
    assert r["blocked_reason"] == "species_mismatch"
    assert r["score"] == 0.0


# --------------------------------------------------------------------------
# Faz 2 — isteğe özel eşik (kaynak bazlı kalibrasyon, ör. Instagram)
# --------------------------------------------------------------------------

def test_threshold_verilmezse_modul_varsayilani_kullanilir():
    """threshold=None -> davranış MATCH_THRESHOLD ile birebir aynı kalmalı
    (native çağıranlar için sıfır davranış değişikliği)."""
    from app.matcher import MATCH_THRESHOLD
    emb = make_embedding()
    r_varsayilan = compute_final_score(emb, emb, ["cat"], ["cat"], 0.0, "cat", "cat")
    r_acik = compute_final_score(emb, emb, ["cat"], ["cat"], 0.0, "cat", "cat",
                                 threshold=MATCH_THRESHOLD)
    assert r_varsayilan["match"] == r_acik["match"]
    assert r_varsayilan["score"] == r_acik["score"]


def test_ozel_threshold_karari_degistirir():
    # Farklı iki vektör kullan: kimlikle eşleşme skoru 1.0 tavanına çarpar ve
    # "hemen üstünde bir eşik" diye bir şey kalmaz. Farklı vektörler orta
    # aralıkta, tavana yapışmayan bir skor verir.
    a, b = make_embedding(1), make_embedding(2)
    r = compute_final_score(a, b, ["cat"], ["cat"], 0.0, "cat", "cat")
    skor = r["score"]
    assert 0.0 < skor < 1.0, "test orta aralıkta bir skor varsayıyor"

    # Skorun hemen üstünde bir eşik: artık eşleşmemeli.
    assert compute_final_score(
        a, b, ["cat"], ["cat"], 0.0, "cat", "cat",
        threshold=skor + 0.01)["match"] is False

    # Skorun hemen altında bir eşik: eşleşmeli.
    assert compute_final_score(
        a, b, ["cat"], ["cat"], 0.0, "cat", "cat",
        threshold=skor - 0.01)["match"] is True
