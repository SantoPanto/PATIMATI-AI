# tests/test_metin_analiz.py
"""TextAnalyzer soyutlamasının testleri — gerçek ağ çağrısı YAPMAZ.

`app/surum.py`'nin KIMLIK_MODEL seçim deseniyle aynı fikir: get_text_analyzer()
bir ortam değişkenine göre uygulama seçer. Burada sağlayıcı çağrısının
kendisi monkeypatch ile taklit ediliyor — testin konusu "sağlayıcı hatası
UNCERTAIN'a düşüyor mu" gibi davranış, gerçek bir LLM'in doğruluğu değil.
"""
import httpx
import pytest

from app import metin_analiz
from app.metin_analiz import (HttpTextAnalyzer, NullTextAnalyzer,
                              get_text_analyzer)


@pytest.fixture(autouse=True)
def _singleton_sifirla():
    """Her testten önce/sonra get_text_analyzer()'ın önbelleğini temizler —
    testler env değişkenlerini değiştirdiğinde bir öncekinin kalıntı
    örneğiyle karışmasınlar."""
    metin_analiz._sifirla_singleton_testler_icin()
    yield
    metin_analiz._sifirla_singleton_testler_icin()


def test_saglayici_bos_ise_null_analyzer_donuyor(monkeypatch):
    monkeypatch.delenv("TEXT_AI_PROVIDER", raising=False)
    analyzer = get_text_analyzer()
    assert isinstance(analyzer, NullTextAnalyzer)


def test_null_analyzer_her_zaman_uncertain_doner():
    sonuc = NullTextAnalyzer().analyze("herhangi bir metin", None)
    assert sonuc.category == "UNCERTAIN"
    assert sonuc.needs_review is True
    assert sonuc.species is None  # uydurma yok


def test_saglayici_ayarli_ama_endpoint_eksikse_null_analyzera_dusuluyor(monkeypatch):
    monkeypatch.setenv("TEXT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.delenv("TEXT_AI_ENDPOINT", raising=False)
    monkeypatch.delenv("TEXT_AI_API_KEY", raising=False)
    analyzer = get_text_analyzer()
    assert isinstance(analyzer, NullTextAnalyzer)


def test_saglayici_basarili_cevabi_dogru_esler(monkeypatch):
    monkeypatch.setenv("TEXT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("TEXT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("TEXT_AI_API_KEY", "sahte-anahtar")
    monkeypatch.setenv("TEXT_AI_MODEL", "sahte-model")

    def _sahte_yaniti_ayristir(self, veri):
        from app.models import TextAnalysisResult
        return TextAnalysisResult(
            category="FOUND", category_confidence=0.9, species="cat",
            location_text="Bursa, Görükle", needs_review=False)

    class _SahteCevap:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return _SahteCevap()

    monkeypatch.setattr(HttpTextAnalyzer, "_yaniti_ayristir", _sahte_yaniti_ayristir)
    monkeypatch.setattr(metin_analiz.httpx, "Client", _SahteIstemci)

    analyzer = get_text_analyzer()
    assert isinstance(analyzer, HttpTextAnalyzer)
    sonuc = analyzer.analyze("Bulduk bu kediyi, sahibi arıyoruz", None)
    assert sonuc.category == "FOUND"
    assert sonuc.location_text == "Bursa, Görükle"


def test_saglayici_zaman_asimina_ugrarsa_uncertaine_duser(monkeypatch):
    monkeypatch.setenv("TEXT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("TEXT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("TEXT_AI_API_KEY", "sahte-anahtar")

    class _PatlayanIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            raise TimeoutError("zaman aşımı")

    monkeypatch.setattr(metin_analiz.httpx, "Client", _PatlayanIstemci)

    analyzer = get_text_analyzer()
    sonuc = analyzer.analyze("herhangi bir caption", "herhangi bir yorum")
    assert sonuc.category == "UNCERTAIN"
    assert sonuc.needs_review is True


def test_gorsel_verilince_govde_multimodal_icerik_tasir(monkeypatch):
    """BUG DÜZELTMESİ (2026-08-19): afiş/poster fotoğraflarındaki yazı
    caption/yorumda hiç yoksa eskiden hiç okunmuyordu. photo_bytes
    verildiğinde istek gövdesi metin + base64 image_url parçalarından oluşan
    bir liste olmalı (Gemini'nin OpenAI-uyumlu multimodal zarfı)."""
    monkeypatch.setenv("TEXT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("TEXT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("TEXT_AI_API_KEY", "sahte-anahtar")

    analyzer = get_text_analyzer()
    assert isinstance(analyzer, HttpTextAnalyzer)
    govde = analyzer._istek_govdesi(None, None, photo_bytes=[b"\xff\xd8\xff"])

    icerik = govde["messages"][0]["content"]
    assert isinstance(icerik, list)
    assert icerik[0]["type"] == "text"
    assert icerik[1]["type"] == "image_url"
    assert icerik[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_gorsel_yoksa_govde_duz_metin_kalir(monkeypatch):
    """Geriye dönük uyumluluk: photo_bytes verilmezse (eski davranış) gövde
    hâlâ düz bir metin dizesi olmalı, önceki sağlayıcı zarfını bozmamalı."""
    monkeypatch.setenv("TEXT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("TEXT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("TEXT_AI_API_KEY", "sahte-anahtar")

    analyzer = get_text_analyzer()
    govde = analyzer._istek_govdesi("caption metni", None)
    assert isinstance(govde["messages"][0]["content"], str)


def test_gecici_503_hatasi_yeniden_denenip_basariyla_sonuclanir(monkeypatch):
    """BUG DÜZELTMESİ (2026-08-19, kullanıcı raporu): önceki sürüm tek bir
    geçici 503'te bile pes edip UNCERTAIN'a düşüyordu -- kayıt bir daha asla
    yeniden denenmediği için (ai_status=DONE terminal) bu kalıcı bir kayıp
    oluyordu. Canlı bir örnekte doğrulandı: afişin üzerinde açıkça yazan
    kategori bilgisi sırf o anki 503 yüzünden hiç okunamadı. Şimdi ilk
    deneme 503 dönse bile ikinci deneme başarılıysa gerçek sonuç kullanılır."""
    from app.models import TextAnalysisResult

    denemeler = []

    class _SahteCevap:
        def __init__(self, basarili):
            self._basarili = basarili

        def raise_for_status(self):
            if not self._basarili:
                istek = httpx.Request("POST", "https://ornek.test/v1/analyze")
                yanit = httpx.Response(503, request=istek)
                raise httpx.HTTPStatusError("503", request=istek, response=yanit)

        def json(self):
            return {}

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            denemeler.append(1)
            return _SahteCevap(basarili=len(denemeler) >= 2)

    monkeypatch.setattr(metin_analiz.httpx, "Client", _SahteIstemci)
    monkeypatch.setattr(
        HttpTextAnalyzer, "_yaniti_ayristir",
        lambda self, veri: TextAnalysisResult(category="ADOPTION", category_confidence=0.9),
    )

    uykular = []
    analyzer = HttpTextAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar", sleep=uykular.append,
    )
    sonuc = analyzer.analyze("caption", None)

    assert len(denemeler) == 2, "ikinci denemede başarılı olmalıydı"
    assert len(uykular) == 1, "yalnızca iki deneme arasında bir kez beklenmeliydi"
    assert sonuc.category == "ADOPTION"


def test_gecici_hata_azami_deneme_sayisini_asarsa_uncertaine_duser(monkeypatch):
    """Geçici hata HER denemede tekrar ederse (sağlayıcı gerçekten çökmüş),
    sonunda yine UNCERTAIN'a düşülür -- retry sonsuz döngü değildir."""
    denemeler = []

    class _SahteCevap:
        def raise_for_status(self):
            istek = httpx.Request("POST", "https://ornek.test/v1/analyze")
            yanit = httpx.Response(503, request=istek)
            raise httpx.HTTPStatusError("503", request=istek, response=yanit)

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            denemeler.append(1)
            return _SahteCevap()

    monkeypatch.setattr(metin_analiz.httpx, "Client", _SahteIstemci)

    uykular = []
    analyzer = HttpTextAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar", sleep=uykular.append,
    )
    sonuc = analyzer.analyze("caption", None)

    assert len(denemeler) == metin_analiz._MAX_ATTEMPTS
    assert len(uykular) == metin_analiz._MAX_ATTEMPTS - 1
    assert sonuc.category == "UNCERTAIN"
    assert sonuc.needs_review is True


def test_kalici_400_hatasi_hic_yeniden_denenmez(monkeypatch):
    """400 gibi kalıcı bir istemci hatası (isteğin kendisiyle ilgili) hiç
    yeniden denenmemeli -- yeniden denemek aynı sonucu tekrar üretir,
    yalnızca gecikme ekler."""
    denemeler = []

    class _SahteCevap:
        def raise_for_status(self):
            istek = httpx.Request("POST", "https://ornek.test/v1/analyze")
            yanit = httpx.Response(400, request=istek)
            raise httpx.HTTPStatusError("400", request=istek, response=yanit)

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            denemeler.append(1)
            return _SahteCevap()

    monkeypatch.setattr(metin_analiz.httpx, "Client", _SahteIstemci)

    uykular = []
    analyzer = HttpTextAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar", sleep=uykular.append,
    )
    sonuc = analyzer.analyze("caption", None)

    assert len(denemeler) == 1, "kalıcı hata yeniden denenmemeliydi"
    assert uykular == []
    assert sonuc.category == "UNCERTAIN"


def test_gecersiz_kategori_uncertaine_normalize_edilir(monkeypatch):
    monkeypatch.setenv("TEXT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("TEXT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("TEXT_AI_API_KEY", "sahte-anahtar")

    class _SahteCevap:
        def raise_for_status(self):
            pass

        def json(self):
            return {"category": "BILINMEYEN_KATEGORI"}

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return _SahteCevap()

    monkeypatch.setattr(metin_analiz.httpx, "Client", _SahteIstemci)

    analyzer = get_text_analyzer()
    sonuc = analyzer.analyze("caption", None)
    assert sonuc.category == "UNCERTAIN"
