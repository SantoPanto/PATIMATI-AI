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

from app import kuyruk
from app.kuyruk import (DLQ, DLX, EXCHANGE, ISTEK_ANAHTARI, ISTEK_KUYRUGU,
                        SEMA_SURUMU, SONUC_ANAHTARI, SONUC_KUYRUGU,
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
    assert kuyruklar == {ISTEK_KUYRUGU, SONUC_KUYRUGU, DLQ}


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
