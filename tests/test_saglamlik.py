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
    varsayilan = {"ad_id": ad_id, "embeddings": [vektor(seed)], "labels": ["cat"],
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


def test_mesafe_bilinmiyorsa_sifir():
    """Eksik mesafe GEÇERSİZ mesafeyle aynı davranmalı.

    Eskiden `MatchCandidate.distance_km` varsayılanı 0.0 idi ve
    `location_score(0.0)` maksimum puandır (1.0). Yani alanı hiç göndermeyen
    çağıran, "mesafeyi bilmiyorum"un karşılığı olarak EN İYİ konum puanını
    alıyordu.
    """
    assert location_score(None) == 0.0


def test_aday_mesafesiz_kurulabilir_ve_none_olur():
    """Varsayılan artık 0.0 değil None — "0 km" ile "bilinmiyor" ayrı şeyler."""
    mesafesiz = MatchCandidate(ad_id=1, embeddings=[vektor()], labels=["cat"],
                               species="cat", model_version=MODEL_SURUMU)
    assert mesafesiz.distance_km is None


def test_eksik_mesafe_gecersiz_mesafeyle_ayni_skoru_verir():
    """Asıl iddia: "eksik" ile "geçersiz" AYNI yoldan geçmeli.

    Ayrı bir dal açılırsa (ör. None için başka bir puan) bu eşitlik bozulur.
    """
    a, b = vektor(1), vektor(2)
    mesafesiz = MatchCandidate(ad_id=1, embeddings=[b], labels=["cat"],
                               species="cat", model_version=MODEL_SURUMU)

    def skorla(mesafe):
        return compute_final_score(a, b, ["cat"], ["cat"], mesafe, "cat", "cat")

    eksik = skorla(mesafesiz.distance_km)
    assert eksik == skorla(float("nan"))
    assert eksik == skorla(float("inf"))
    assert eksik["location"] == 0.0


def test_mesafesiz_aday_gercek_eslestirme_yolundan_gecer():
    """Skorlama fonksiyonu değil, adayı ÜRÜNDEKİ yoldan geçirir.

    `adaylari_eslestir` mesafeyi `aday.distance_km` üzerinden okur; bu test o
    aktarımı da kapsar, yoksa yalnız `compute_final_score` sınanmış olurdu.
    """
    mesafesiz = MatchCandidate(ad_id=7, embeddings=[vektor(3)], labels=["cat"],
                               species="cat", model_version=MODEL_SURUMU)
    sonuclar, _ = adaylari_eslestir(vektor(1), ["cat"], "cat", [mesafesiz],
                                    model_version=MODEL_SURUMU)
    assert len(sonuclar) == 1
    assert sonuclar[0]["location"] == 0.0


def test_acik_sifir_km_hala_maksimum():
    """Karşıt kontrol: AÇIKÇA verilen 0.0 meşru "aynı noktada" demektir.

    Düzeltme yalnız bilginin HİÇ olmadığı durumu değiştirmeli; 0 km'yi de
    cezalandırırsa aynı sokakta bulunan hayvan puan kaybeder.
    """
    assert location_score(0.0) == 1.0
    a, b = vektor(1), vektor(2)
    assert compute_final_score(a, b, ["cat"], ["cat"], 0.0, "cat", "cat")["location"] == 1.0


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
    bozuk = aday(1, embeddings=[[float("nan")] * VEKTOR_BOYUTU])
    m, atlanan = adaylari_eslestir(vektor(), ["cat"], "cat", [bozuk, aday(2, seed=5)])
    assert len(m) == 1 and m[0]["ad_id"] == 2
    assert atlanan["gecersiz_embedding"] == 1


# --------------------------------------------------------------------------
# Çoklu fotoğraf
# --------------------------------------------------------------------------

def test_coklu_fotograf_en_iyi_cifti_secer():
    """Görsel skor, iki ilanın tüm fotoğraf çiftleri arasındaki EN İYİsi olmalı.

    Gerçek veriyle ölçüldü: tek fotoğrafla aynı hayvanın eşleşme oranı %24'te
    kalıyordu, ilan başına 3 fotoğrafla en iyi çifti almak eşiğin üstüne çıkardı.
    """
    hedef = vektor(1)
    # İkinci ilanda üç fotoğraf var ve sadece sonuncusu hedefle aynı
    aday_cok = aday(1, embeddings=[vektor(2), vektor(3), hedef])
    m, _ = adaylari_eslestir([hedef], ["cat"], "cat", [aday_cok])
    assert m[0]["visual"] == pytest.approx(1.0, abs=1e-4)
    assert m[0]["photo_b"] == 2, "en iyi çift 3. fotoğraf olmalıydı"


def test_coklu_fotograf_tekli_kadar_iyi_olmali():
    """Fotoğraf eklemek skoru asla düşürmemeli (en iyi çift alındığı için)."""
    a, b, c = vektor(1), vektor(2), vektor(3)
    tek, _ = adaylari_eslestir([a], ["cat"], "cat", [aday(1, embeddings=[b])])
    cok, _ = adaylari_eslestir([a], ["cat"], "cat", [aday(1, embeddings=[b, c])])
    assert cok[0]["visual"] >= tek[0]["visual"]


def test_bos_fotograf_listesi_reddedilir():
    with pytest.raises(Exception):
        MatchCandidate(ad_id=1, embeddings=[], model_version=MODEL_SURUMU)


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
