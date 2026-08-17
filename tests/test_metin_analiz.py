# tests/test_metin_analiz.py
"""TextAnalyzer soyutlamasının testleri — gerçek ağ çağrısı YAPMAZ.

`app/surum.py`'nin KIMLIK_MODEL seçim deseniyle aynı fikir: get_text_analyzer()
bir ortam değişkenine göre uygulama seçer. Burada sağlayıcı çağrısının
kendisi monkeypatch ile taklit ediliyor — testin konusu "sağlayıcı hatası
UNCERTAIN'a düşüyor mu" gibi davranış, gerçek bir LLM'in doğruluğu değil.
"""
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
