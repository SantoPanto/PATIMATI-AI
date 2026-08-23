# tests/test_main.py
"""`app/main.py`'deki HTTP uçlarının testleri.

Buraya kadar `test_kuyruk.py` AMQP tarafını, `test_matcher.py`/`test_saglamlik.py`
skorlama mantığını kapsıyordu ama Java'nın senkron deneme/demo için kullandığı
asıl HTTP yüzeyinin (`/analyze`, `/compare`, `/match`, `/analyze_url`, `/health`)
hiç testi yoktu.

Ağır model çıkarımı `test_kuyruk.py`deki `sahte_analiz` ile aynı gerekçeyle
taklit edilir: burada sınanan uç noktaların HTTP sözleşmesi (durum kodu, cevap
şekli, hata kodu) — modelin doğruluğu değil. Modelin kendisi `test_attributes.py`de
gerçek seed veriyle ayrıca sınanıyor.
"""
import io
import json

import numpy as np
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.hatalar import FotografIndirilemedi
from app.surum import MODEL_SURUMU, VEKTOR_BOYUTU

#: Testlerin kullandığı paylaşılan sır. A1 bölümündeki kilit testleri de bunu
#: kullanır; tek yerde durması "hangi anahtarla koştuk" sorusunu tek cevaplı yapar.
ANAHTAR = "cok-gizli-anahtar"

#: VARSAYILAN İSTEMCİ KİMLİKLİDİR. Kilit artık anahtar verilmediğinde de
#: reddediyor (bkz. `anahtari_dogrula`), yani başlıksız bir istemciyle bu
#: dosyadaki HTTP sözleşmesi testlerinin hepsi 401 alırdı ve ölçtükleri şeyi
#: (cevap şekli, hata kodları) hiç ölçemezlerdi.
client = TestClient(main.app, headers={"X-Api-Key": ANAHTAR})

#: Kilidin KENDİSİNİ ölçen testler için. Varsayılan istemci başlığı otomatik
#: eklediği için "anahtarsız istek" onunla kurulamaz.
anahtarsiz_client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _anahtar_yapilandirildi(monkeypatch):
    """Her teste anahtarı yapılandırılmış bir servis verir.

    Ortam değişkeni her istekte okunuyor (`api_anahtari`), dolayısıyla bunu
    fikstürde sabitlemek şart: geliştiricinin `.env` dosyası `load_dotenv`
    ile yüklenirse test hangi anahtarla koştuğunu bilemezdi.

    Kilidi ölçen testler bunu `monkeypatch` ile kendileri geri alıyor.
    """
    monkeypatch.setenv("AI_API_KEY", ANAHTAR)


def vektor(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(VEKTOR_BOYUTU).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def sahte_oznitelik(**degisiklikler):
    d = {"labels": ["cat", "tabby"], "species": "cat", "species_confidence": 0.95,
         "is_pet": True, "breed": "Tekir", "breed_confidence": 0.81,
         "pattern": "tabby", "colors": [{"r": 100, "g": 80, "b": 60, "score": 0.4}],
         "is_designed_graphic": False, "graphic_confidence": 0.02}
    d.update(degisiklikler)
    return d


def _resim_baytlari(renk=(120, 80, 40), boyut=(64, 64)) -> bytes:
    tampon = io.BytesIO()
    Image.new("RGB", boyut, renk).save(tampon, format="JPEG")
    return tampon.getvalue()


@pytest.fixture
def sahte_model(monkeypatch):
    """Gerçek CLIP çıkarımı yerine sabit embedding + öznitelik koyar.

    Testin konusu main.py'nin HTTP davranışı; modeli burada çalıştırmak hem
    yavaşlatır hem de testi modelin doğruluğuna bağımlı kılar (bkz.
    test_kuyruk.py'deki `sahte_analiz` ile aynı gerekçe).
    """
    def kur(embedding=None, oznitelik=None):
        monkeypatch.setattr(main.kimlik_gomucu, "embed_bytes",
                            lambda b: embedding or vektor(1))
        monkeypatch.setattr(main.attribute_analyzer, "analyze",
                            lambda b, embedding=None: oznitelik or sahte_oznitelik())
    return kur


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == MODEL_SURUMU
    assert body["vektor_boyutu"] == VEKTOR_BOYUTU


# --------------------------------------------------------------------------
# /analyze
# --------------------------------------------------------------------------

def test_analyze_basarili_yanit_sekli(sahte_model):
    sahte_model()
    r = client.post("/analyze",
                    files={"file": ("kedi.jpg", _resim_baytlari(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["embedding"]) == VEKTOR_BOYUTU
    assert body["species"] == "cat"
    assert body["model_version"] == MODEL_SURUMU


def test_analyze_goruntu_disi_dosya_reddedilir():
    r = client.post("/analyze", files={"file": ("not.txt", b"merhaba", "text/plain")})
    assert r.status_code == 400


def test_analyze_cok_buyuk_dosya_reddedilir():
    buyuk = b"\x00" * (10 * 1024 * 1024 + 1)
    r = client.post("/analyze", files={"file": ("buyuk.jpg", buyuk, "image/jpeg")})
    assert r.status_code == 413


def test_analyze_bozuk_goruntu_400_doner():
    """Model hiç çalıştırılmadan `goruntu_ac` içinde patlar (bkz. app/embedder.py)."""
    r = client.post("/analyze",
                    files={"file": ("bozuk.jpg", b"bu bir goruntu degil", "image/jpeg")})
    assert r.status_code == 400


# --------------------------------------------------------------------------
# /compare
# --------------------------------------------------------------------------

def test_compare_iki_farkli_turu_sifirlar(monkeypatch):
    cagri = {"n": 0}

    def _oznitelik(b, embedding=None):
        cagri["n"] += 1
        return sahte_oznitelik(species="cat" if cagri["n"] == 1 else "dog")

    monkeypatch.setattr(main.kimlik_gomucu, "embed_bytes", lambda b: vektor(1))
    monkeypatch.setattr(main.attribute_analyzer, "analyze", _oznitelik)

    r = client.post("/compare", files={
        "file1": ("a.jpg", _resim_baytlari(), "image/jpeg"),
        "file2": ("b.jpg", _resim_baytlari(), "image/jpeg"),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["sonuc"]["score"] == 0.0
    assert body["sonuc"]["match"] is False


def test_compare_ayni_tur_skor_uretir(sahte_model):
    sahte_model()
    r = client.post("/compare", files={
        "file1": ("a.jpg", _resim_baytlari(), "image/jpeg"),
        "file2": ("b.jpg", _resim_baytlari(), "image/jpeg"),
    })
    assert r.status_code == 200
    assert 0.0 <= r.json()["sonuc"]["score"] <= 1.0


# --------------------------------------------------------------------------
# /match
# --------------------------------------------------------------------------

def test_match_kendisi_elenir():
    v = vektor(0)
    r = client.post("/match", json={
        "embeddings": [v], "labels": ["cat"], "species": "cat", "ad_id": 5,
        "candidates": [{"ad_id": 5, "embeddings": [v], "labels": ["cat"],
                        "species": "cat", "distance_km": 0.0,
                        "model_version": MODEL_SURUMU}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["matches"] == []
    assert body["skipped_candidates"]["kendisi"] == 1


def test_match_ad_id_ve_external_record_id_ikisi_birden_reddedilir():
    """models.py: MatchRequest._en_fazla_bir_kimlik. Düzeltme öncesi ikisi de
    dolu gelebiliyordu ve matcher.py._aday_kimligi ad_id'yi sessizce
    önceliklendiriyordu (external_record_id sorgusu yanlış kimlikle
    eşleştiriliyordu) -- artık 422 ile açıkça reddedilir."""
    r = client.post("/match", json={
        "embeddings": [vektor(0)], "labels": ["cat"], "species": "cat",
        "ad_id": 5, "external_record_id": 7,
        "candidates": [],
    })
    assert r.status_code == 422


def test_match_ikisi_de_bos_hala_gecerli():
    """MatchCandidate/KuyrukIstegi'nin aksine MatchRequest'te "hiçbiri"
    hâlâ geçerli bir durum -- self-exclusion o zaman devre dışı kalır (bkz.
    models.py._en_fazla_bir_kimlik docstring'i). Bu davranışı bilerek
    KORUYORUZ; test_match_bos_aday_listesiyle_bos_sonuc_doner ile aynı
    girdi şeklini kapsıyor, burada ayrıca 'kırılmadı' diye adı geçiyor."""
    r = client.post("/match", json={
        "embeddings": [vektor(0)], "labels": ["cat"], "species": "cat",
        "candidates": [],
    })
    assert r.status_code == 200


def test_match_bozuk_sorgu_embeddingi_400_doner():
    """Sorgunun kendi embedding'i bozuksa 'aday elendi' gibi yutulmamalı, açık
    400 dönmeli (bkz. app/matcher.py: adaylari_eslestir)."""
    r = client.post("/match", json={
        "embeddings": [[0.1, 0.2, 0.3]], "labels": [], "species": "cat",
        "candidates": [{"ad_id": 1, "embeddings": [vektor(1)], "labels": [],
                        "species": "cat", "distance_km": 1.0,
                        "model_version": MODEL_SURUMU}],
    })
    assert r.status_code == 400


def test_match_bos_aday_listesiyle_bos_sonuc_doner():
    r = client.post("/match", json={
        "embeddings": [vektor(0)], "labels": ["cat"], "species": "cat",
        "candidates": [],
    })
    assert r.status_code == 200
    assert r.json()["matches"] == []


# --------------------------------------------------------------------------
# /analyze_url
# --------------------------------------------------------------------------

def test_analyze_url_indirilemezse_502(monkeypatch):
    def _patlar(urls):
        raise FotografIndirilemedi("adres cozulemedi")

    monkeypatch.setattr(main, "urlleri_analiz_et", _patlar)
    r = client.post("/analyze_url", json={"photo_urls": ["https://ornek.test/1.jpg"]})
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "PHOTO_DOWNLOAD_FAILED"


def test_analyze_url_basarili_analiz_sonucu_doner(monkeypatch):
    beklenen = {
        "embeddings": [vektor(1)], "species": "cat", "species_confidence": 0.9,
        "is_pet": True, "breed": "Tekir", "breed_confidence": 0.8,
        "pattern": "tabby", "colors": [], "labels": ["cat"],
        "model_version": MODEL_SURUMU, "photo_count": 1, "failed_photos": [],
    }
    monkeypatch.setattr(main, "urlleri_analiz_et", lambda urls: beklenen)
    r = client.post("/analyze_url", json={"photo_urls": ["https://ornek.test/1.jpg"]})
    assert r.status_code == 200
    assert r.json() == beklenen


def test_analyze_url_bos_liste_pydantic_tarafindan_reddedilir():
    r = client.post("/analyze_url", json={"photo_urls": []})
    assert r.status_code == 422


def test_analyze_url_photo_bytes_yanita_sizmaz(monkeypatch):
    """arkadaş incelemesi (PR #30): urlleri_analiz_et artık dönüşe
    `photo_bytes` (ham bayt listesi, kuyruk.py'nin metin analizine görsel
    geçirmek için kullandığı bir ara değer) ekliyor. Bu alan JSON'a
    çevrilemez -- eski stub'lu test (yukarıdaki) bunu hiç sınamıyordu ve
    üretimde bu uç fotoğraflı HER çağrıda 500 veriyordu."""
    beklenen = {
        "embeddings": [vektor(1)], "species": "cat", "species_confidence": 0.9,
        "is_pet": True, "breed": "Tekir", "breed_confidence": 0.8,
        "pattern": "tabby", "colors": [], "labels": ["cat"],
        "model_version": MODEL_SURUMU, "photo_count": 1, "failed_photos": [],
        "photo_bytes": [b"sahte-jpeg-baytlari"],
    }
    monkeypatch.setattr(main, "urlleri_analiz_et", lambda urls: dict(beklenen))
    r = client.post("/analyze_url", json={"photo_urls": ["https://ornek.test/1.jpg"]})
    assert r.status_code == 200
    assert "photo_bytes" not in r.json()


# --------------------------------------------------------------------------
# OpenAPI şeması
# --------------------------------------------------------------------------

def test_openapi_semasi_kati_json_olarak_uretilebilir():
    """Hiçbir alan varsayılanı JSON'a çevrilemeyen bir değer olmamalı.

    Ölçülerek bulundu: `MatchCandidate.distance_km` varsayılanını
    `float("nan")` yapmak cazip görünüyordu — eksik mesafe böylece
    `location_score`'un mevcut "sonlu değil" dalından geçer, yeni dal
    gerekmezdi. Ama pydantic varsayılanı OpenAPI şemasına koyuyor
    (`{'default': nan, ...}`) ve Starlette'in JSONResponse'u
    `json.dumps(..., allow_nan=False)` kullanıyor ⇒ `/openapi.json` 500 verir
    ve `/docs` tamamen ölür. Tasarım bu ölçüm yüzünden `None`'a çevrildi.

    Bu test yalnız MatchCandidate'i değil TÜM şemayı kapsar; aynı tuzağa
    başka bir modelde düşülmesini de engeller.
    """
    r = client.get("/openapi.json")
    assert r.status_code == 200

    # TestClient gövdeyi zaten çözdü; asıl kontrol KATI JSON'a geri çevrilebilmesi.
    # (json.dumps varsayılanı NaN/Infinity'yi sessizce yazar, allow_nan=False yazmaz.)
    json.dumps(r.json(), allow_nan=False)


# --------------------------------------------------------------------------
# A1 — uçların kimliği (paylaşılan anahtar)
# --------------------------------------------------------------------------

#: Kimlik isteyen uçlar. `/health` bilerek dışarıda: canlılık yoklaması
#: kimlik isteseydi, doğru çalışan servis "ölü" görünürdü.
#: (`ANAHTAR` sabiti dosyanın başında — istemciler de onu kullanıyor.)
KORUNAN_UCLAR = ["/analyze", "/analyze_url", "/compare", "/match"]


@pytest.mark.parametrize("yol", KORUNAN_UCLAR)
def test_anahtar_yapilandirilmamissa_uc_401_verir(monkeypatch, yol):
    """Varsayılan REDDET: yapılandırma eksikse kapı KAPANIR, açılmaz.

    Bu testin eskisi tam tersini söylüyordu ("401 vermemeli") ve haklıydı:
    servis o zaman tarayıcıdan da çağrılıyordu, anahtarı zorunlu kılmak ilan
    oluşturma ekranını kırardı. O engel kalktı — ön yüz AI'ı doğrudan
    çağırmıyor, backend `X-Api-Key` gönderiyor. Artık korunması gereken şey
    ters yön: unutulmuş bir ortam değişkeni yüzünden servisin herkese açık
    hâle gelmesi.
    """
    monkeypatch.delenv("AI_API_KEY", raising=False)

    r = anahtarsiz_client.post(yol)

    assert r.status_code == 401, (
        f"{yol} anahtar YAPILANDIRILMADIĞI hâlde {r.status_code} verdi — "
        f"eksik yapılandırma kapıyı AÇIYOR. Canlıda bu, kimliksiz AI demek.")


@pytest.mark.parametrize("yol", KORUNAN_UCLAR)
def test_anahtar_varken_anahtarsiz_istek_reddedilir(yol):
    r = anahtarsiz_client.post(yol)

    assert r.status_code == 401, (
        f"{yol} anahtar zorunluyken anahtarsız isteği kabul etti "
        f"(durum {r.status_code}). Uç korumasız kalmış olabilir.")


@pytest.mark.parametrize("yol", KORUNAN_UCLAR)
def test_yanlis_anahtar_reddedilir(yol):
    """Yalnız son harfin BÜYÜKLÜĞÜ farklı: ön ek ya da harf duyarsız eşleşme yok."""
    r = anahtarsiz_client.post(yol, headers={"X-Api-Key": "cok-gizli-anahtaR"})

    assert r.status_code == 401, f"{yol} yanlış anahtarı kabul etti."


def test_dogru_anahtar_gecer(sahte_model):
    """Ön koşul: kilit yalnızca reddetmiyor, DOĞRU anahtarla geçiriyor da.

    Bu olmadan yukarıdaki üç test "her şeyi reddet" diyen bozuk bir kilitle de
    yeşil yanardı — ki kilit artık gerçekten "varsayılan reddet" olduğu için
    bu ön koşul eskisinden daha kritik.
    """
    sahte_model()

    r = anahtarsiz_client.post("/analyze",
                    headers={"X-Api-Key": ANAHTAR},
                    files={"file": ("kedi.jpg", _resim_baytlari(), "image/jpeg")})

    assert r.status_code == 200, f"Doğru anahtarla istek geçmedi: {r.text}"
    assert len(r.json()["embedding"]) == VEKTOR_BOYUTU


def test_health_anahtarsiz_da_acik_kalir():
    r = anahtarsiz_client.get("/health")

    assert r.status_code == 200, (
        "/health kimlik istemeye başladı — canlılık yoklaması artık çalışan "
        "servisi ölü gösterir.")
    assert r.json()["api_anahtari_yapilandirildi"] is True


def test_health_anahtar_verilmediyse_gercegi_soyler(monkeypatch):
    """Anahtarsız servis AYAKTA ama İŞE YARAMAZ; /health bunu söylemeli.

    Aksi hâlde 200 dönen bir sağlık yoklaması, korumalı uçların hepsi 401
    verirken "her şey yolunda" der ve yanlış yapılandırılmış dağıtım
    çalışan bir dağıtımdan ayırt edilemez.
    """
    monkeypatch.delenv("AI_API_KEY", raising=False)

    assert anahtarsiz_client.get("/health").json()["api_anahtari_yapilandirildi"] is False


def test_butun_POST_uclari_korunuyor():
    """Uç listesi ELLE değil, uygulamanın kendi yönlendirme tablosundan türetilir.

    Elle yazılmış bir liste yalnız bugünü ölçer: yarın eklenen bir uç listeye
    yazılmadığı sürece korumasız kalır ve hiçbir test bunu söylemez.
    """
    korunan, korumasiz = [], []
    for route in main.app.routes:
        if not isinstance(route, APIRoute) or "POST" not in route.methods:
            continue
        cagrilar = [alt.call for alt in route.dependant.dependencies]
        (korunan if main.anahtari_dogrula in cagrilar else korumasiz).append(route.path)

    # Ön koşul: türetme çalışmadıysa aşağıdaki iddia HİÇBİR ŞEY kanıtlamaz —
    # boş liste her zaman "korumasız uç yok" der.
    assert korunan, "Düzenek bozuk: hiç POST ucu bulunamadı, iddia anlamsız."

    assert korumasiz == [], (
        f"Kimlik doğrulaması olmayan POST ucu var: {korumasiz}. "
        f"Yeni uç eklendiyse dependencies=[Depends(anahtari_dogrula)] unutulmuş.")
