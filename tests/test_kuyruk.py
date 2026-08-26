# tests/test_kuyruk.py
"""Kuyruk köprüsünün testleri — RabbitMQ KURULU OLMADAN çalışır.

Nasıl: `app/kuyruk.py` iki katmana ayrıldı. İş mantığı (`istegi_isle`) saf bir
fonksiyon, AMQP'yi hiç bilmiyor. AMQP'ye dokunan kısım ise yalnızca kanal
nesnesi üzerinden konuşuyor, o da burada taklit ediliyor (`SahteKanal`).

Broker'a gerçekten bağlanan uçtan uca kanıt ayrı: `scripts/sahte_java.py`.
Bu ayrım bilerek yapıldı — testler her makinede, kurulum gerektirmeden
çalışabilmeli; yoksa ekipteki kimse koşturmaz.
"""
import json

import numpy as np
import pytest

from app import kuyruk, topoloji
from app.kuyruk import (DLQ, DLX, EXCHANGE, ISTEK_ANAHTARI, ISTEK_KUYRUGU,
                        SEMA_SURUMU, SONUC_ANAHTARI, SONUC_DLQ, SONUC_KUYRUGU,
                        istegi_isle, topolojiyi_kur)
from app.surum import MODEL_SURUMU, VEKTOR_BOYUTU


def vektor(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(VEKTOR_BOYUTU).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def istek(**degisiklikler):
    """Sözleşme §3'e uyan geçerli bir istek; testler istediğini değiştirir."""
    m = {
        "schema_version": SEMA_SURUMU,
        "request_id": "9f1c2b7e-3a44-4a1e-9d55-0c2f6b1a77de",
        "ad_id": 123,
        "ad_type": "LOST",
        "photo_urls": ["https://ornek.test/1.jpg"],
        "candidates": [],
    }
    m.update(degisiklikler)
    return m


def aday(ad_id=98, species="cat", model_version=None):
    return {"ad_id": ad_id, "embeddings": [vektor(1)], "labels": ["cat", "tabby"],
            "species": species, "distance_km": 1.0,
            "model_version": model_version or MODEL_SURUMU}


def external_aday(external_record_id=98, species="cat", model_version=None):
    """Faz 2: ad_id yerine external_record_id ile kimliklenen bir aday
    (Instagram kökenli external_pet_records kaydı)."""
    return {"external_record_id": external_record_id, "embeddings": [vektor(1)],
            "labels": ["cat", "tabby"], "species": species, "distance_km": 1.0,
            "model_version": model_version or MODEL_SURUMU}


@pytest.fixture
def sahte_analiz(monkeypatch):
    """Gerçek indirme + model çıkarımı yerine sabit bir analiz sonucu koyar.

    Testin konusu kuyruk mantığı; modeli burada çalıştırmak hem yavaşlatır hem
    de testi modelin doğruluğuna bağımlı yapar (ilgisiz bir sebeple kırmızıya
    döner).
    """
    def kur(species="cat", hata=None):
        def _sahte(urls):
            if hata:
                raise hata
            return {
                "embeddings": [vektor(1)], "species": species,
                "species_confidence": 0.95, "is_pet": True, "breed": "Tekir",
                "breed_confidence": 0.81, "pattern": "tabby",
                "colors": [{"name": "brown", "score": 0.4}],
                "labels": ["cat", "tabby"], "model_version": MODEL_SURUMU,
                "photo_count": len(urls), "failed_photos": [],
            }
        monkeypatch.setattr(kuyruk, "urlleri_analiz_et", _sahte)
    return kur


# --------------------------------------------------------------------------
# Hata yolları — hepsi sözleşme §4'teki kodlara karşılık gelmeli
# --------------------------------------------------------------------------

def test_tanimayan_sema_surumu_reddedilir():
    """Bilmediğimiz sürümdeki bir mesajı ayrıştırmaya ÇALIŞMAMALIYIZ.

    Alanların anlamı değişmiş olabilir; okuyup işlemek hata vermeden yanlış
    sonuç üretmenin en kolay yoludur.
    """
    sonuc = istegi_isle(istek(schema_version=99))
    assert sonuc["status"] == "error"
    assert sonuc["error"]["code"] == "UNSUPPORTED_SCHEMA"


def test_eksik_alan_gonderen_tarafin_hatasi_olarak_bildirilir():
    """`ad_id` yoksa kod INVALID_REQUEST olmalı, INTERNAL değil.

    INTERNAL, hatayı ayıklayan kişiyi AI servisinin içinde arattırır; oysa
    sorun gönderendedir.
    """
    m = istek()
    del m["ad_id"]
    sonuc = istegi_isle(m)
    assert sonuc["error"]["code"] == "INVALID_REQUEST"
    assert "ad_id" in sonuc["error"]["message"]


def test_sema_disi_mesajda_bile_request_id_geri_doner():
    """Java'nın hangi isteğin başarısız olduğunu bilmesi gerekir.

    request_id dönmezse o ilan sonsuza kadar PENDING'de kalır ve kimse
    sebebini bulamaz.
    """
    m = istek()
    del m["ad_id"]
    sonuc = istegi_isle(m)
    assert sonuc["request_id"] == m["request_id"]


def test_fotograf_yoksa_no_photos():
    """Boş `photo_urls` Pydantic hatası değil, sözleşmedeki NO_PHOTOS olmalı."""
    sonuc = istegi_isle(istek(photo_urls=[]))
    assert sonuc["error"]["code"] == "NO_PHOTOS"


def test_model_patlarsa_model_error(sahte_analiz):
    """Model hatası INTERNAL'e karışmamalı — ayrı kodu var, görünür kalsın."""
    sahte_analiz(hata=RuntimeError("tensör boyutu uyuşmadı"))
    sonuc = istegi_isle(istek())
    assert sonuc["error"]["code"] == "MODEL_ERROR"


def test_sorgu_embeddingi_bozuksa_model_error(monkeypatch):
    """GecersizEmbedding INTERNAL'e düşmemeli.

    Burada patlayan embedding Java'dan gelmiyor, urlleri_analiz_et'in ürettiği
    vektör — yani sorun çağıranda değil bizim model çıktımızda. Önceden
    ValueError'dan türeyen bu hata genel except'e düşüp INTERNAL dönüyordu;
    Java "bizde bir şey patladı" diye AI servisinin içinde arıyordu.
    """
    def _sahte(urls):
        return {
            "embeddings": [[1.0] * 10],  # yanlış boyut
            "species": "cat", "species_confidence": 0.95, "is_pet": True,
            "breed": "Tekir", "breed_confidence": 0.81, "pattern": "tabby",
            "colors": [{"name": "brown", "score": 0.4}],
            "labels": ["cat", "tabby"], "model_version": MODEL_SURUMU,
            "photo_count": len(urls), "failed_photos": [],
        }
    monkeypatch.setattr(kuyruk, "urlleri_analiz_et", _sahte)

    sonuc = istegi_isle(istek(candidates=[aday()]))
    assert sonuc["status"] == "error"
    assert sonuc["error"]["code"] == "MODEL_ERROR"


def test_istegi_isle_asla_firlatmaz():
    """Fırlatırsa mesaj onaylanmaz, sonsuza kadar yeniden teslim edilir.

    Bu fonksiyonun sözleşmesi: ne olursa olsun bir sonuç mesajı DÖNDÜR.
    """
    for bozuk in ({}, {"schema_version": 1}, istek(candidates="liste değil"),
                  istek(ad_id="sayı değil")):
        sonuc = istegi_isle(bozuk)
        assert sonuc["status"] == "error"


# --------------------------------------------------------------------------
# Başarılı yol
# --------------------------------------------------------------------------

def test_basarili_sonuc_sozlesmedeki_alanlari_tasir(sahte_analiz):
    sahte_analiz()
    sonuc = istegi_isle(istek(candidates=[aday()]))

    assert sonuc["status"] == "ok"
    assert sonuc["schema_version"] == SEMA_SURUMU
    assert sonuc["ad_id"] == 123
    assert sonuc["model_version"] == MODEL_SURUMU
    assert set(sonuc["analysis"]) == {
        "embeddings", "species", "species_confidence", "is_pet", "breed",
        "breed_confidence", "pattern", "colors", "labels"}
    assert len(sonuc["matches"]) == 1
    assert sonuc["matches"][0]["ad_id"] == 98
    assert "toplam" in sonuc["skipped_candidates"]


def test_sonuc_json_e_cevrilebilir(sahte_analiz):
    """JSON'a çevrilemeyen bir değer (numpy float, NaN) kuyruğu kırar.

    Daha önce aynı sınıf hata yaşandı: NaN JSON standardında yok, mesaj Java
    tarafında ayrıştırılamıyor ve sonuç sessizce kayboluyordu.
    """
    sahte_analiz()
    sonuc = istegi_isle(istek(candidates=[aday()]))
    yeniden = json.loads(json.dumps(sonuc, ensure_ascii=False))
    assert yeniden["ad_id"] == 123


def test_processed_at_utc_ve_z_ile_biter(sahte_analiz):
    sahte_analiz()
    sonuc = istegi_isle(istek())
    assert sonuc["processed_at"].endswith("Z")
    assert "T" in sonuc["processed_at"]


def test_surumu_tutmayan_aday_atlanir_ve_raporlanir(sahte_analiz):
    """Farklı sürümle üretilmiş vektör kıyaslanamaz; sessizce atlanmamalı."""
    sahte_analiz()
    sonuc = istegi_isle(istek(candidates=[aday(model_version="eski-model/v0")]))
    assert sonuc["matches"] == []
    assert sonuc["skipped_candidates"]["model_surumu_uyusmuyor"] == 1


# --------------------------------------------------------------------------
# Sözleşme §7 kural 2 — kullanıcı beyanı AI tahmininden önceliklidir
# --------------------------------------------------------------------------

def test_kullanici_beyani_ai_tahminini_ezer(sahte_analiz):
    """AI "dog" dese bile kullanıcı "cat" dediyse kedi adayları elenmemeli.

    Tersi olsaydı: AI'ın tek bir yanlış tür tahmini, doğru eşleşmeyi tamamen
    yok ederdi — kullanıcı hayvanını bulamazdı.
    """
    sahte_analiz(species="dog")
    sonuc = istegi_isle(istek(declared_species="cat", candidates=[aday(species="cat")]))
    assert sonuc["matches"][0]["score"] > 0.0


def test_beyan_yoksa_ai_tahmini_kullanilir(sahte_analiz):
    """Beyan yoksa AI "dog" dedi diye kedi adayı sıfırlanmalı (tür filtresi)."""
    sahte_analiz(species="dog")
    sonuc = istegi_isle(istek(candidates=[aday(species="cat")]))
    assert sonuc["matches"][0]["score"] == 0.0


# --------------------------------------------------------------------------
# Faz 2 — metin analizi tetikleme kuralı (caption VEYA triggering_comment)
# --------------------------------------------------------------------------

def test_caption_ve_yorum_bossa_nlp_attributes_bos_kalir(sahte_analiz):
    """Regresyon koruması: native ilan akışı hiçbir zaman caption/comment
    göndermez — nlp_attributes/extracted_features eskisi gibi tam boş kalmalı."""
    sahte_analiz()
    sonuc = istegi_isle(istek())
    assert sonuc["nlp_attributes"] == {}
    assert sonuc["extracted_features"] == []


def test_yalnizca_caption_varsa_metin_analizi_calisir(sahte_analiz, monkeypatch):
    from app.models import TextAnalysisResult

    cagrilar = []

    class _SahteAnalyzer:
        def analyze(self, caption, triggering_comment, photo_bytes=None):
            cagrilar.append((caption, triggering_comment))
            return TextAnalysisResult(category="ADOPTION", category_confidence=0.7)

    monkeypatch.setattr(kuyruk, "get_text_analyzer", lambda: _SahteAnalyzer())
    sahte_analiz()
    sonuc = istegi_isle(istek(caption="Sahiplendirilecek yavru kedi"))

    assert cagrilar == [("Sahiplendirilecek yavru kedi", None)]
    assert sonuc["nlp_attributes"]["category"] == "ADOPTION"


def test_yorum_varsa_caption_bos_olsa_da_nlp_calisir(sahte_analiz, monkeypatch):
    """BUG DÜZELTMESİ: eski davranış yalnızca `caption` doluysa metin analizini
    çalıştırıyordu. Ama caption boş/alakasız olup triggering_comment'in tek
    başına anlamlı olduğu durumlar gerçek — "@patimati bu kediyi Görükle'de
    buldum" gibi bir yorum caption olmadan da FOUND + konum çıkarımı
    yapılabilmeli. Bu test, yalnızca yorum doluyken analyzer'ın GERÇEKTEN
    çağrıldığını doğrular (önceki davranışta hiç çağrılmazdı)."""
    from app.models import TextAnalysisResult

    cagrilar = []

    class _SahteAnalyzer:
        def analyze(self, caption, triggering_comment, photo_bytes=None):
            cagrilar.append((caption, triggering_comment))
            return TextAnalysisResult(
                category="FOUND", category_confidence=0.85,
                location_text="Bursa, Görükle")

    monkeypatch.setattr(kuyruk, "get_text_analyzer", lambda: _SahteAnalyzer())
    sahte_analiz()
    sonuc = istegi_isle(istek(
        caption=None,
        triggering_comment="@patimati bu kediyi Görükle'de buldum, sahibi ulaşsın"))

    assert len(cagrilar) == 1, "caption boşken ve yorum doluyken analyzer çağrılmadı"
    assert cagrilar[0] == (None, "@patimati bu kediyi Görükle'de buldum, sahibi ulaşsın")
    assert sonuc["nlp_attributes"]["category"] == "FOUND"
    assert sonuc["nlp_attributes"]["location_text"] == "Bursa, Görükle"
    assert sonuc["nlp_attributes"] != {}


def test_caption_ve_yorum_bossa_ama_gorsel_varsa_external_kayitta_nlp_calisir(
        monkeypatch):
    """BUG DÜZELTMESİ (2026-08-19, kullanıcı raporu): kayıp/bulundu bilgisi
    bazı Instagram gönderilerinde caption/yorumda değil, doğrudan fotoğrafın
    (afiş/poster) içindeki yazıda geçiyor. caption VE triggering_comment
    ikisi de boşken önceki davranış metin analizini hiç çalıştırmıyordu —
    kategori hep UNCERTAIN'a düşüyor, bu da aşağıda Java'nın aday havuzunu
    tamamen boş bırakmasına yol açıyordu. external_record_id dolu (Instagram
    kökenli) ve en az bir fotoğraf işlenebildiyse artık analyzer'a görsel de
    geçirilerek çağrılmalı."""
    from app.models import TextAnalysisResult

    def _sahte_urlleri_analiz_et(urls):
        return {
            "embeddings": [vektor(1)], "species": "cat",
            "species_confidence": 0.4, "is_pet": True, "breed": None,
            "breed_confidence": 0.0, "pattern": None, "colors": [],
            "labels": [], "model_version": MODEL_SURUMU,
            "photo_count": len(urls), "failed_photos": [],
            "photo_bytes": [b"sahte-jpeg-baytlari"],
        }

    monkeypatch.setattr(kuyruk, "urlleri_analiz_et", _sahte_urlleri_analiz_et)

    cagrilar = []

    class _SahteAnalyzer:
        def analyze(self, caption, triggering_comment, photo_bytes=None):
            cagrilar.append((caption, triggering_comment, photo_bytes))
            return TextAnalysisResult(category="LOST", category_confidence=0.8)

    monkeypatch.setattr(kuyruk, "get_text_analyzer", lambda: _SahteAnalyzer())

    sonuc = istegi_isle(istek(
        ad_id=None, external_record_id=777,
        candidates=[external_aday(external_record_id=555)],
        caption=None, triggering_comment=None))

    assert len(cagrilar) == 1, "görsel varken ve external kayıtken analyzer çağrılmadı"
    assert cagrilar[0] == (None, None, [b"sahte-jpeg-baytlari"])
    assert sonuc["nlp_attributes"]["category"] == "LOST"


def test_caption_doluysa_gorsel_analyzera_gecirilmez(monkeypatch):
    """arkadaş incelemesi (PR #30): caption zaten anlamlı içerik taşıyorsa
    görsel gönderilmesi GEREKSİZ maliyet/gecikme -- görsel yalnızca caption
    VE triggering_comment ikisi de boş/alakasız olduğunda (yukarıdaki
    DÜZELTME 2) gönderilmeli. photo_bytes ve external_record_id dolu olsa
    bile caption doluysa analyzer'a photo_bytes=None geçirilmeli."""
    from app.models import TextAnalysisResult

    def _sahte_urlleri_analiz_et(urls):
        return {
            "embeddings": [vektor(1)], "species": "cat",
            "species_confidence": 0.4, "is_pet": True, "breed": None,
            "breed_confidence": 0.0, "pattern": None, "colors": [],
            "labels": [], "model_version": MODEL_SURUMU,
            "photo_count": len(urls), "failed_photos": [],
            "photo_bytes": [b"sahte-jpeg-baytlari"],
        }

    monkeypatch.setattr(kuyruk, "urlleri_analiz_et", _sahte_urlleri_analiz_et)

    cagrilar = []

    class _SahteAnalyzer:
        def analyze(self, caption, triggering_comment, photo_bytes=None):
            cagrilar.append((caption, triggering_comment, photo_bytes))
            return TextAnalysisResult(category="FOUND", category_confidence=0.8)

    monkeypatch.setattr(kuyruk, "get_text_analyzer", lambda: _SahteAnalyzer())

    sonuc = istegi_isle(istek(
        ad_id=None, external_record_id=777,
        candidates=[external_aday(external_record_id=555)],
        caption="bu kediyi Görükle'de buldum", triggering_comment=None))

    assert len(cagrilar) == 1
    assert cagrilar[0] == ("bu kediyi Görükle'de buldum", None, None), (
        "caption doluyken photo_bytes analyzer'a gecirilmemeli")
    assert sonuc["nlp_attributes"]["category"] == "FOUND"


def test_gorsel_var_ama_native_ilansa_nlp_calismaz(monkeypatch):
    """Aynı fotoğraf-tetikleyici native (ad_id'li) bir istekte ASLA
    devreye girmemeli -- native akışta caption/comment zaten hiç
    gönderilmiyor ve kullanıcı zaten ad_type'ı kendi seçiyor, metin analizi
    burada anlamsız bir ek maliyet/gecikme olurdu."""
    def _sahte_urlleri_analiz_et(urls):
        return {
            "embeddings": [vektor(1)], "species": "cat",
            "species_confidence": 0.9, "is_pet": True, "breed": None,
            "breed_confidence": 0.0, "pattern": None, "colors": [],
            "labels": [], "model_version": MODEL_SURUMU,
            "photo_count": len(urls), "failed_photos": [],
            "photo_bytes": [b"sahte-jpeg-baytlari"],
        }

    monkeypatch.setattr(kuyruk, "urlleri_analiz_et", _sahte_urlleri_analiz_et)

    def _cagrilmamasi_gereken():
        raise AssertionError("analyzer hiç çağrılmamalıydı (native ilan)")

    monkeypatch.setattr(kuyruk, "get_text_analyzer", _cagrilmamasi_gereken)

    sonuc = istegi_isle(istek(caption=None, triggering_comment=None))
    assert sonuc["nlp_attributes"] == {}


def test_bos_dizeli_caption_ve_yorum_da_bos_sayilir(sahte_analiz, monkeypatch):
    """Sadece None değil, yalnızca boşluk içeren dizeler de "boş" sayılmalı —
    aksi halde Java'nın gönderebileceği "" gibi bir değer yanlışlıkla
    analyzer'ı tetikler."""
    cagrildi = []

    def _cagrilmamasi_gereken():
        cagrildi.append(True)
        raise AssertionError("analyzer hiç çağrılmamalıydı")

    monkeypatch.setattr(kuyruk, "get_text_analyzer", _cagrilmamasi_gereken)
    sahte_analiz()
    sonuc = istegi_isle(istek(caption="   ", triggering_comment=""))
    assert sonuc["nlp_attributes"] == {}
    assert cagrildi == []


# --------------------------------------------------------------------------
# Faz 2 — external_record_id kimlikli adaylar
# --------------------------------------------------------------------------

def test_external_record_id_ile_gelen_aday_dogru_eslenir(sahte_analiz):
    """ad_id yerine external_record_id taşıyan bir aday (Instagram kökenli
    external_pet_records kaydı) doğru skorlanmalı ve sonuçta kendi
    external_record_id'siyle geri dönmeli, ad_id ile karıştırılmamalı."""
    sahte_analiz()
    sonuc = istegi_isle(istek(candidates=[external_aday(external_record_id=555)]))

    assert len(sonuc["matches"]) == 1
    eslesme = sonuc["matches"][0]
    assert eslesme["external_record_id"] == 555
    assert eslesme["ad_id"] is None
    assert sonuc["skipped_candidates"]["toplam"] == 0


def test_ad_id_ve_external_record_id_ikisi_de_bos_olamaz():
    """Bir aday ne native bir ilan ne de bir external kayıt olarak
    kimliklenemiyorsa mesaj INVALID_REQUEST olarak reddedilmeli — sessizce
    "kimliksiz" bir adayla devam edip kendisi/tekrar eleme mantığını
    bozmasın (bkz. matcher.py:_aday_kimligi)."""
    bozuk_aday = aday()
    del bozuk_aday["ad_id"]  # ne ad_id ne external_record_id kaldı
    sonuc = istegi_isle(istek(candidates=[bozuk_aday]))
    assert sonuc["status"] == "error"
    assert sonuc["error"]["code"] == "INVALID_REQUEST"


def test_ad_id_ve_external_record_id_ayni_anda_dolu_olamaz():
    cift_kimlikli = aday()
    cift_kimlikli["external_record_id"] = 999
    sonuc = istegi_isle(istek(candidates=[cift_kimlikli]))
    assert sonuc["status"] == "error"
    assert sonuc["error"]["code"] == "INVALID_REQUEST"


def test_match_threshold_gonderilirse_kullanilir(sahte_analiz):
    """match_threshold verilmezse modülün MATCH_THRESHOLD'u kullanılır;
    verilirse o isteğe özel eşik geçerli olmalı — native davranış değişmeden."""
    sahte_analiz()
    # Çok yüksek bir eşik ver: normalde eşleşecek aday artık match=False olmalı.
    sonuc = istegi_isle(istek(candidates=[aday()], match_threshold=0.999))
    assert sonuc["matches"][0]["match"] is False

    sonuc_dusuk_esik = istegi_isle(istek(candidates=[aday()], match_threshold=0.0))
    assert sonuc_dusuk_esik["matches"][0]["match"] is True


# --------------------------------------------------------------------------
# AMQP katmanı — kanal taklidiyle
# --------------------------------------------------------------------------

class SahteKanal:
    """pika kanalının kaydeden taklidi. Sıra önemli olduğu için tek listede."""

    def __init__(self):
        self.olaylar = []

    def exchange_declare(self, exchange, **kw):
        self.olaylar.append(("exchange", exchange, kw))

    def queue_declare(self, queue, **kw):
        self.olaylar.append(("kuyruk", queue, kw))

    def queue_bind(self, queue, exchange, routing_key=None, **kw):
        self.olaylar.append(("bag", queue, exchange, routing_key))

    def basic_qos(self, **kw):
        self.olaylar.append(("qos", kw))

    def basic_publish(self, exchange, routing_key, body, properties=None):
        self.olaylar.append(("yayin", exchange, routing_key, body, properties))

    def basic_ack(self, delivery_tag):
        self.olaylar.append(("onay", delivery_tag))

    def basic_nack(self, delivery_tag, requeue=True):
        self.olaylar.append(("ret", delivery_tag, requeue))

    def tipler(self):
        return [o[0] for o in self.olaylar]


class SahteMethod:
    delivery_tag = 7


def test_topoloji_sozlesmedeki_nesneleri_ilan_eder():
    kanal = SahteKanal()
    topolojiyi_kur(kanal)

    exchangeler = {o[1] for o in kanal.olaylar if o[0] == "exchange"}
    kuyruklar = {o[1] for o in kanal.olaylar if o[0] == "kuyruk"}
    assert exchangeler == {EXCHANGE, DLX}
    assert kuyruklar == {ISTEK_KUYRUGU, SONUC_KUYRUGU, DLQ, SONUC_DLQ}


def test_dlq_orijinal_yonlendirme_anahtariyla_baglanir():
    """RabbitMQ ölü mektuba düşerken orijinal routing key'i KORUR.

    DLQ, DLX'e kuyruk adıyla bağlanırsa mesajlar hata vermeden yok olur —
    bulunması çok zor bir sessiz kayıp. Bu test o tuzağı kapatıyor.
    """
    kanal = SahteKanal()
    topolojiyi_kur(kanal)

    dlq_baglari = [o for o in kanal.olaylar if o[0] == "bag" and o[1] == DLQ]
    assert dlq_baglari == [("bag", DLQ, DLX, ISTEK_ANAHTARI)]


def test_istek_kuyrugu_dlx_e_baglidir():
    """x-dead-letter-exchange yoksa bozuk mesaj DLQ'ya değil, hiçliğe gider."""
    kanal = SahteKanal()
    topolojiyi_kur(kanal)

    ilan = next(o for o in kanal.olaylar
                if o[0] == "kuyruk" and o[1] == ISTEK_KUYRUGU)
    assert ilan[2]["arguments"]["x-dead-letter-exchange"] == DLX
    assert ilan[2]["durable"] is True


def test_sonuc_dlq_orijinal_yonlendirme_anahtariyla_baglanir():
    """İstek tarafındaki aynı tuzak, sonuç tarafı için de geçerli: SONUC_DLQ
    kuyruk adıyla değil `analysis.result` anahtarıyla bağlanmalı."""
    kanal = SahteKanal()
    topolojiyi_kur(kanal)

    dlq_baglari = [o for o in kanal.olaylar if o[0] == "bag" and o[1] == SONUC_DLQ]
    assert dlq_baglari == [("bag", SONUC_DLQ, DLX, SONUC_ANAHTARI)]


def test_sonuc_kuyrugu_varsayilan_olarak_dlx_e_baglanmaz(monkeypatch):
    """`SONUC_KUYRUGU_DLQ_ENABLED` KAPALI kalmalı: bu kuyruk Java tarafında
    çoktan (bu argüman olmadan) canlıda ilan edilmiş olabilir -- argümanı
    tek taraflı eklemek PRECONDITION_FAILED'a yol açar (bkz. app/topoloji.py).
    Java'da eşdeğer değişiklik + kuyruğun yeniden kurulmasıyla BİRLİKTE
    açılana kadar varsayılan kapalı kalır."""
    monkeypatch.setattr(topoloji, "SONUC_KUYRUGU_DLQ_ENABLED", False)
    kanal = SahteKanal()
    topolojiyi_kur(kanal)

    ilan = next(o for o in kanal.olaylar
                if o[0] == "kuyruk" and o[1] == SONUC_KUYRUGU)
    assert ilan[2]["arguments"] is None
    assert ilan[2]["durable"] is True


def test_sonuc_kuyrugu_dlx_e_baglidir_flag_acikken(monkeypatch):
    """Java'daki onResult bir istisna fırlatırsa (ya da mesaj hiç
    ayrıştırılamazsa) mesaj SONUC_DLQ'ya düşsün diye -- x-dead-letter-exchange
    yoksa aynı kuyruğa sonsuz döngüyle geri döner (zehirli mesaj). Bu, yalnızca
    `SONUC_KUYRUGU_DLQ_ENABLED` açıldığında (Java tarafı hazır olduğunda)
    devreye girer."""
    monkeypatch.setattr(topoloji, "SONUC_KUYRUGU_DLQ_ENABLED", True)
    kanal = SahteKanal()
    topolojiyi_kur(kanal)

    ilan = next(o for o in kanal.olaylar
                if o[0] == "kuyruk" and o[1] == SONUC_KUYRUGU)
    assert ilan[2]["arguments"]["x-dead-letter-exchange"] == DLX
    assert ilan[2]["durable"] is True


def test_bozuk_json_dlq_ya_gider_ve_cevap_yayinlanmaz():
    """Ayrıştırılamayan mesajın request_id'si okunamaz; cevap verilemez (§8)."""
    kanal = SahteKanal()
    kuyruk._mesaj_geldi(kanal, SahteMethod(), None, b"{bu json degil")

    assert kanal.tipler() == ["ret"]
    assert kanal.olaylar[0] == ("ret", 7, False)   # requeue=False: geri gelmesin


def test_json_dizisi_de_dlq_ya_gider():
    """Geçerli JSON ama nesne değil — alanları okunamaz."""
    kanal = SahteKanal()
    kuyruk._mesaj_geldi(kanal, SahteMethod(), None, b"[1, 2, 3]")
    assert kanal.tipler() == ["ret"]


def test_once_yayinlanir_sonra_onaylanir(sahte_analiz):
    """Sıra tersine dönerse: onay ile yayın arasında çökersek mesaj kuyruktan
    silinmiş ama sonuç hiç yazılmamış olur. İlan sessizce PENDING'de kalır.
    """
    sahte_analiz()
    kanal = SahteKanal()
    govde = json.dumps(istek()).encode("utf-8")
    kuyruk._mesaj_geldi(kanal, SahteMethod(), None, govde)

    assert kanal.tipler() == ["yayin", "onay"]


def test_hatali_istek_de_cevaplanir_ve_onaylanir():
    """Hata sonucu da yayınlanmalı ve mesaj onaylanmalı.

    Onaylanmazsa bozuk mesaj sonsuza kadar yeniden teslim edilir ve servis
    başka iş yapamaz (§8: "kuyruğa geri koymaz, cevaplar").
    """
    kanal = SahteKanal()
    govde = json.dumps(istek(schema_version=99)).encode("utf-8")
    kuyruk._mesaj_geldi(kanal, SahteMethod(), None, govde)

    assert kanal.tipler() == ["yayin", "onay"]
    govde_yayinlanan = kanal.olaylar[0][3]
    assert json.loads(govde_yayinlanan)["status"] == "error"


def test_sonuc_dogru_anahtarla_ve_kalici_yayinlanir(sahte_analiz):
    """Kalıcı olmayan mesaj, broker yeniden başlayınca kaybolur."""
    sahte_analiz()
    kanal = SahteKanal()
    kuyruk._mesaj_geldi(kanal, SahteMethod(), None,
                        json.dumps(istek()).encode("utf-8"))

    _, exchange, anahtar, govde, ozellikler = kanal.olaylar[0]
    assert exchange == EXCHANGE
    assert anahtar == SONUC_ANAHTARI
    assert ozellikler.delivery_mode == 2
    assert ozellikler.content_type == "application/json"
    assert json.loads(govde)["status"] == "ok"


def test_turkce_karakterler_bozulmadan_gider(sahte_analiz):
    """ensure_ascii=False + utf-8: "Tekir" gibi değerler kaçış dizisine dönmesin.

    Java tarafı UTF-8 okuyor; ASCII kaçışı da çalışırdı ama günlükleri
    okunmaz hâle getirirdi.
    """
    sahte_analiz()
    kanal = SahteKanal()
    kuyruk._mesaj_geldi(kanal, SahteMethod(), None,
                        json.dumps(istek()).encode("utf-8"))
    govde = kanal.olaylar[0][3]
    assert json.loads(govde.decode("utf-8"))["analysis"]["breed"] == "Tekir"


# ---------------------------------------------------------------------------
# Kapanış sinyalleri
#
# NEDEN TEST EDİLİYOR: konteynerde `docker stop` SIGTERM gönderiyor ve
# SERVIS_ROLU=kuyruk rolünde bu süreç PID 1 oluyor. Linux, PID 1'e gelen ve
# İŞLEYİCİSİ OLMAYAN sinyalin varsayılan eylemini uygulamaz — sinyali yutar.
# Yani işleyici olmadan konteyner `docker stop` ile hiç durmaz, 10 sn sonra
# SIGKILL yer ve dinle()'deki düzgün kapanış hiç çalışmaz.
#
# Uçtan uca kanıtı CI'daki "kuyruk rolü + PID 1'de düzgün kapanış" adımı
# veriyor (çıkış kodu 0 mı 137 mi diye bakıyor). Buradaki testler o adımın
# yerine geçmez; işleyicinin MANTIĞINI saniyeler içinde sabitler.
# ---------------------------------------------------------------------------

@pytest.fixture
def sinyal_duzeni_korunsun():
    """Sinyal düzeni SÜREÇ GENELİNDE geçerli — test onu kalıcı bozmasın."""
    import signal
    onceki = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, isleyici in onceki.items():
        signal.signal(s, isleyici)


def test_sigterm_keyboardinterrupt_a_cevriliyor(sinyal_duzeni_korunsun):
    """dinle() zaten KeyboardInterrupt'ı yakalayıp düzgün kapanıyor (Ctrl+C
    yolu). İkinci bir kapanış yolu yazmak yerine sinyali oraya BAĞLIYORUZ."""
    import signal
    kuyruk._kapanis_sinyallerini_yakala()
    for numara in (signal.SIGTERM, signal.SIGINT):
        isleyici = signal.getsignal(numara)
        assert callable(isleyici), f"{numara} için işleyici kurulmamış"
        with pytest.raises(KeyboardInterrupt):
            isleyici(numara, None)


def test_ikinci_sinyal_varsayilana_dusuyor(sinyal_duzeni_korunsun):
    """İşleyicinin ilk satırı SIG_DFL kurmalı.

    Yoksa ikinci SIGTERM, dinle()'nin kapanış bloğundaki
    `try: ... except Exception: pass` içinde patlar — KeyboardInterrupt bir
    Exception DEĞİLDİR, oradan kaçar ve süreç düzgün kapanmanın tam ortasında
    iz dökerek ölür.
    """
    import signal
    kuyruk._kapanis_sinyallerini_yakala()
    isleyici = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        isleyici(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, (
        "ikinci sinyal hâlâ işleyiciye düşüyor; kapanış ortasında iz dökme riski")

# --------------------------------------------------------------------------
# Eşik: `match` bayrağının anlamı zamanla değişiyor (B6)
# --------------------------------------------------------------------------

def test_sonuc_o_anki_esigi_tasir(sahte_analiz):
    """`match: true` = "score >= eşik", ama eşik SABİT DEĞİL: 0.70 → 0.80 (PR #17).

    Java kaydı "bu eşleşme üretilirken eşik neydi" bilgisini saklıyor
    (`ad_match.threshold_at_time`, NOT NULL). Değer buradan gitmezse Java ya
    kaydı yazamaz ya da ikinci bir kaynaktan tahmin eder; iki kaynak sessizce
    kayar ve kayıt geçmişe dair yanlış konuşur.
    """
    sahte_analiz()
    sonuc = istegi_isle(istek(candidates=[aday()]))
    assert sonuc["match_threshold"] == pytest.approx(kuyruk.MATCH_THRESHOLD)


def test_esik_sabit_yazilmamis_calisan_degeri_izler(sahte_analiz, monkeypatch):
    """Sayı sabit yazılsaydı üstteki test YİNE GEÇERDİ — eşik bugün zaten o değer.

    Bu yüzden çalışan eşiği değiştirip cevabın onu izlediğini ölçüyoruz.
    Bekçi bu mutasyonla sınandı: `"match_threshold": 0.80` yazınca düşüyor.
    """
    monkeypatch.setattr(kuyruk, "MATCH_THRESHOLD", 0.42)
    sahte_analiz()
    sonuc = istegi_isle(istek(candidates=[aday()]))
    assert sonuc["match_threshold"] == pytest.approx(0.42), (
        "cevaptaki eşik çalışan değeri izlemiyor — sabit yazılmış olabilir")


def test_esik_json_e_cevrilebilir_sayidir(sahte_analiz):
    """Java bunu `DOUBLE PRECISION NOT NULL` sütuna yazıyor: sayı olmalı, None değil."""
    sahte_analiz()
    sonuc = istegi_isle(istek(candidates=[aday()]))
    yeniden = json.loads(json.dumps(sonuc, ensure_ascii=False))
    assert isinstance(yeniden["match_threshold"], float)
    assert yeniden["match_threshold"] is not None
