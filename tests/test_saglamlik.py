# tests/test_saglamlik.py
"""Uç durum (edge case) testleri.

Buradaki her test, ÖLÇÜLEREK bulunmuş gerçek bir hatayı ya da riski koruyor.
Her testin başında hangi sorunu engellediği yazılıdır — biri bu davranışı
"gereksiz" diye sadeleştirmeye kalkarsa neyi kaybedeceğini görsün.
"""
import io

import numpy as np
import pytest
from PIL import Image

from app.embedder import ASGARI_KENAR, goruntu_ac
from app.hatalar import GecersizEmbedding, GecersizGoruntu
from app.matcher import (adaylari_eslestir, compute_final_score, cosine_similarity,
                         location_score)
from app.models import MatchCandidate
from app.surum import MODEL_SURUMU, VEKTOR_BOYUTU


def vektor(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(VEKTOR_BOYUTU).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def aday(ad_id, seed=0, **kwargs):
    varsayilan = {"ad_id": ad_id, "embedding": vektor(seed), "labels": ["cat"],
                  "species": "cat", "distance_km": 1.0, "model_version": MODEL_SURUMU}
    varsayilan.update(kwargs)
    return MatchCandidate(**varsayilan)


def jpg(img):
    b = io.BytesIO()
    img.save(b, format="JPEG")
    return b.getvalue()


# --------------------------------------------------------------------------
# Mesafe
# --------------------------------------------------------------------------

def test_negatif_mesafe_cokmez():
    """-5 km ZeroDivisionError ile servisi çökertiyordu."""
    assert location_score(-5.0) == 1.0
    assert location_score(-100.0) == 1.0


def test_negatif_mesafe_skoru_sismez():
    """-1 km, üst sınırı 1.0 olması gereken skoru 1.25 yapıyordu."""
    assert 0.0 <= location_score(-1.0) <= 1.0


def test_sonsuz_mesafe():
    assert location_score(float("inf")) == 0.0
    assert location_score(float("nan")) == 0.0


# --------------------------------------------------------------------------
# Embedding doğrulaması
# --------------------------------------------------------------------------

def test_nan_embedding_hata_verir():
    """NaN, skoru NaN yapıyordu; JSON NaN'ı desteklemediği için mesaj Java
    tarafında ayrıştırılamıyor ve sonuç tamamen kayboluyordu."""
    bozuk = [float("nan")] + [0.0] * (VEKTOR_BOYUTU - 1)
    with pytest.raises(GecersizEmbedding):
        cosine_similarity(vektor(), bozuk)


def test_sonsuz_embedding_hata_verir():
    bozuk = [float("inf")] + [0.0] * (VEKTOR_BOYUTU - 1)
    with pytest.raises(GecersizEmbedding):
        cosine_similarity(vektor(), bozuk)


def test_yanlis_uzunluk_hata_verir():
    """Yakalanmayan ValueError kuyruk tüketicisini çökertirdi."""
    with pytest.raises(GecersizEmbedding):
        cosine_similarity(vektor(), [1.0] * 10)


def test_bos_embedding_hata_verir():
    with pytest.raises(GecersizEmbedding):
        cosine_similarity(vektor(), [])


def test_skor_daima_araligin_icinde():
    a, b = vektor(1), vektor(2)
    for mesafe in (0.0, 5.0, 1000.0, -3.0):
        s = compute_final_score(a, b, ["cat"], ["cat"], mesafe, "cat", "cat")
        assert 0.0 <= s["score"] <= 1.0


# --------------------------------------------------------------------------
# Aday eleme
# --------------------------------------------------------------------------

def test_kendisiyle_eslesmez():
    """Java aday listesine ilanın kendisini koyarsa %100 eşleşme çıkıyordu."""
    m, atlanan = adaylari_eslestir(vektor(), ["cat"], "cat", [aday(7)], ad_id=7)
    assert m == []
    assert atlanan["kendisi"] == 1


def test_tekrar_eden_aday_bir_kez_sayilir():
    m, atlanan = adaylari_eslestir(vektor(), ["cat"], "cat", [aday(3), aday(3)])
    assert len(m) == 1
    assert atlanan["tekrar_eden"] == 1


def test_model_surumu_uyusmayan_aday_atlanir():
    """Farklı sürümle üretilmiş vektörler kıyaslanamaz; hata vermeden yanlış
    benzerlik üretirler. Atlanmalı ve raporlanmalı."""
    m, atlanan = adaylari_eslestir(
        vektor(), ["cat"], "cat", [aday(1, model_version="eski-surum/v0")])
    assert m == []
    assert atlanan["model_surumu_uyusmuyor"] == 1


def test_surumsuz_aday_atlanir():
    m, atlanan = adaylari_eslestir(vektor(), ["cat"], "cat", [aday(1, model_version=None)])
    assert atlanan["model_surumu_uyusmuyor"] == 1


def test_bozuk_aday_digerlerini_dusurmez():
    """Tek bozuk aday tüm isteği düşürmemeli, ama sessizce de yok sayılmamalı."""
    bozuk = aday(1, embedding=[float("nan")] * VEKTOR_BOYUTU)
    m, atlanan = adaylari_eslestir(vektor(), ["cat"], "cat", [bozuk, aday(2, seed=5)])
    assert len(m) == 1 and m[0]["ad_id"] == 2
    assert atlanan["gecersiz_embedding"] == 1


def test_aday_siniri_uygulanir():
    adaylar = [aday(i, seed=i % 7) for i in range(150)]
    m, atlanan = adaylari_eslestir(vektor(), ["cat"], "cat", adaylar)
    assert atlanan["aday_siniri_asildi"] == 50
    assert len(m) <= 20


def test_atlanan_toplami_tutarli():
    adaylar = [aday(1), aday(1), aday(2, model_version="v0")]
    _, atlanan = adaylari_eslestir(vektor(), ["cat"], "cat", adaylar)
    assert atlanan["toplam"] == sum(v for k, v in atlanan.items() if k != "toplam")


# --------------------------------------------------------------------------
# Görüntü açma
# --------------------------------------------------------------------------

def test_cok_kucuk_goruntu_reddedilir():
    """1x1 görüntü sessizce 512'lik bir çöp vektör üretiyordu."""
    with pytest.raises(GecersizGoruntu):
        goruntu_ac(jpg(Image.new("RGB", (8, 8), (10, 20, 30))))


def test_asgari_boyut_kabul_edilir():
    img = goruntu_ac(jpg(Image.new("RGB", (ASGARI_KENAR, ASGARI_KENAR), (10, 20, 30))))
    assert img.size == (ASGARI_KENAR, ASGARI_KENAR)


def test_bozuk_dosya_net_hata_verir():
    with pytest.raises(GecersizGoruntu):
        goruntu_ac(b"bu bir goruntu degil")


def test_seffaf_png_beyaza_yerlesir():
    """RGB'ye düz çevirmek şeffaf alanları siyaha çeviriyor, renk analizini bozuyordu."""
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))  # tamamen şeffaf kırmızı
    b = io.BytesIO()
    img.save(b, format="PNG")
    acilan = goruntu_ac(b.getvalue())
    assert acilan.mode == "RGB"
    assert acilan.getpixel((32, 32)) == (255, 255, 255)


def test_exif_dondurmesi_uygulanir():
    """Telefon fotoğrafları çoğu zaman 'yan' kaydedilir; düzeltilmezse vektör bozulur."""
    img = Image.new("RGB", (128, 64), (10, 20, 30))
    exif = img.getexif()
    exif[274] = 6  # Orientation: 90° döndür
    b = io.BytesIO()
    img.save(b, format="JPEG", exif=exif.tobytes())
    acilan = goruntu_ac(b.getvalue())
    assert acilan.size == (64, 128), "EXIF döndürmesi uygulanmadı"


# --------------------------------------------------------------------------
# "Hayvan mı" kapısı
# --------------------------------------------------------------------------

def test_hayvan_olmayan_goruntu_yakalanir():
    """Tür seçimi yalnızca kedi/köpek arasında yapılsaydı, hayvan olmayan
    fotoğraf da zorunlu olarak birine atanırdı."""
    from app.attributes import attribute_analyzer

    rng = np.random.default_rng(0)
    negatifler = {
        "duz gri": Image.new("RGB", (256, 256), (128, 128, 128)),
        "gurultu": Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)),
    }
    for ad, img in negatifler.items():
        d = attribute_analyzer.analyze(jpg(img))
        assert d["is_pet"] is False, f"{ad}: hayvan sanıldı"
        assert d["species"] == "unknown"
        assert d["breed"] is None, f"{ad}: hayvan değilken cins verildi"
