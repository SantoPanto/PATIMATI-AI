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

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.hatalar import FotografIndirilemedi
from app.surum import MODEL_SURUMU, VEKTOR_BOYUTU

client = TestClient(main.app)


def vektor(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(VEKTOR_BOYUTU).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def sahte_oznitelik(**degisiklikler):
    d = {"labels": ["cat", "tabby"], "species": "cat", "species_confidence": 0.95,
         "is_pet": True, "breed": "Tekir", "breed_confidence": 0.81,
         "pattern": "tabby", "colors": [{"r": 100, "g": 80, "b": 60, "score": 0.4}]}
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
