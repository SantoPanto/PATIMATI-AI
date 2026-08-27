# tests/test_instagram_caption.py
"""InstagramCaptionGenerator soyutlamasının ve instagram_caption_olustur()
entegrasyon kapısının testleri — gerçek ağ çağrısı YAPMAZ.

Yapı `tests/test_pet_raporu.py`'nin birebir aynısı (bkz. app/instagram_caption.py
docstring'i): sağlayıcı çağrısı monkeypatch ile taklit edilir, testin konusu
"sağlayıcı hatası None'a düşüyor mu" gibi davranış, gerçek bir LLM'in
doğruluğu değil.
"""
import httpx
import pytest

from app import instagram_caption
from app.instagram_caption import (HttpInstagramCaptionGenerator,
                                   NullInstagramCaptionGenerator,
                                   get_instagram_caption_generator,
                                   instagram_caption_olustur)


@pytest.fixture(autouse=True)
def _singleton_sifirla():
    instagram_caption._sifirla_singleton_testler_icin()
    yield
    instagram_caption._sifirla_singleton_testler_icin()


# --------------------------------------------------------------------------
# Sağlayıcı seçimi
# --------------------------------------------------------------------------

def test_saglayici_bos_ise_null_generator_donuyor(monkeypatch):
    monkeypatch.delenv("INSTAGRAM_CAPTION_AI_PROVIDER", raising=False)
    generator = get_instagram_caption_generator()
    assert isinstance(generator, NullInstagramCaptionGenerator)


def test_null_generator_her_zaman_none_doner():
    sonuc = NullInstagramCaptionGenerator().generate(
        "Kayıp", "Kedi", None, None, None, None, None)
    assert sonuc is None


def test_saglayici_ayarli_ama_endpoint_eksikse_null_generatora_dusuluyor(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.delenv("INSTAGRAM_CAPTION_AI_ENDPOINT", raising=False)
    monkeypatch.delenv("INSTAGRAM_CAPTION_AI_API_KEY", raising=False)
    generator = get_instagram_caption_generator()
    assert isinstance(generator, NullInstagramCaptionGenerator)


def test_pet_raporu_env_degiskenleri_bu_ozelligi_etkilemez(monkeypatch):
    """PET_REPORT_AI_* ile INSTAGRAM_CAPTION_AI_* bilerek AYRI -- biri
    açıkken diğeri yapılandırılmamışsa sessizce onun ayarını miras almamalı."""
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "gemini")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/pet")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "pet-anahtari")
    monkeypatch.delenv("INSTAGRAM_CAPTION_AI_PROVIDER", raising=False)
    generator = get_instagram_caption_generator()
    assert isinstance(generator, NullInstagramCaptionGenerator)


# --------------------------------------------------------------------------
# instagram_caption_olustur() -- entegrasyon kapısı
# --------------------------------------------------------------------------

def test_ilan_turu_turkceye_cevrilir(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_ENDPOINT", "https://ornek.test/v1/caption")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_API_KEY", "sahte-anahtar")

    yakalanan = {}

    def _sahte_generate(self, ilan_turu, tur, irk, renkler, il_ilce, aciklama, ayirt_edici):
        yakalanan["ilan_turu"] = ilan_turu
        return "üretilen metin"

    monkeypatch.setattr(HttpInstagramCaptionGenerator, "generate", _sahte_generate)

    instagram_caption_olustur("LOST", "Kedi")
    assert yakalanan["ilan_turu"] == "Kayıp"

    instagram_caption_olustur("FOUND", "Kedi")
    assert yakalanan["ilan_turu"] == "Bulundu"

    instagram_caption_olustur("ADOPTION", "Kedi")
    assert yakalanan["ilan_turu"] == "Sahiplendirme"


# --------------------------------------------------------------------------
# HttpInstagramCaptionGenerator -- istek gövdesi
# --------------------------------------------------------------------------

def test_istek_govdesi_gorsel_icermez():
    """pet_raporu'nun aksine burada görsel YOK -- içerik düz metin olmalı."""
    generator = HttpInstagramCaptionGenerator(
        endpoint="https://ornek.test/v1/caption", model="sahte-model",
        api_key="sahte-anahtar")
    govde = generator._istek_govdesi("Kayıp", "Kedi", None, None, None, None, None)
    icerik = govde["messages"][0]["content"]
    assert isinstance(icerik, str)
    assert govde["response_format"] == {"type": "json_object"}


def test_placeholderlar_prompt_metnine_islenir():
    generator = HttpInstagramCaptionGenerator(
        endpoint="https://ornek.test/v1/caption", model="sahte-model",
        api_key="sahte-anahtar")
    govde = generator._istek_govdesi(
        "Kayıp", "Kedi", "Van Kedisi", "Beyaz", "İstanbul/Kadıköy",
        "Sokakta kayboldu", ["Sol kulakta çentik"])
    metin = govde["messages"][0]["content"]
    assert "ilan_turu: Kayıp" in metin
    assert "irk: Van Kedisi" in metin
    assert "il_ilce: İstanbul/Kadıköy" in metin
    assert "Sol kulakta çentik" in metin
    assert "{{TUR}}" not in metin and "{{IRK}}" not in metin


def test_irk_yoksa_belirlenemediye_duser():
    generator = HttpInstagramCaptionGenerator(
        endpoint="https://ornek.test/v1/caption", model="sahte-model",
        api_key="sahte-anahtar")
    govde = generator._istek_govdesi("Kayıp", "Kedi", None, None, None, None, None)
    metin = govde["messages"][0]["content"]
    assert "irk: BELIRLENEMEDI" in metin


# --------------------------------------------------------------------------
# Yanıt ayrıştırma / hata dayanıklılığı
# --------------------------------------------------------------------------

def _sahte_istemci_kur(monkeypatch, cevap_json):
    class _SahteCevap:
        def raise_for_status(self):
            pass

        def json(self):
            return cevap_json

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return _SahteCevap()

    monkeypatch.setattr(instagram_caption.httpx, "Client", _SahteIstemci)


def test_basarili_cevap_caption_doner(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_ENDPOINT", "https://ornek.test/v1/caption")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_API_KEY", "sahte-anahtar")
    _sahte_istemci_kur(monkeypatch, {"choices": [{"message": {"content": {"caption": "Kayıp kedimiz var!"}}}]})

    generator = get_instagram_caption_generator()
    assert isinstance(generator, HttpInstagramCaptionGenerator)
    sonuc = generator.generate("Kayıp", "Kedi", None, None, None, None, None)
    assert sonuc == "Kayıp kedimiz var!"


def test_bos_caption_none_a_duser(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_ENDPOINT", "https://ornek.test/v1/caption")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_API_KEY", "sahte-anahtar")
    _sahte_istemci_kur(monkeypatch, {"choices": [{"message": {"content": {"caption": "   "}}}]})

    generator = get_instagram_caption_generator()
    sonuc = generator.generate("Kayıp", "Kedi", None, None, None, None, None)
    assert sonuc is None


def test_kalici_hata_none_a_duser_crash_etmez(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_ENDPOINT", "https://ornek.test/v1/caption")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_API_KEY", "sahte-anahtar")

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            request = httpx.Request("POST", "https://ornek.test/v1/caption")
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError("yetkisiz", request=request, response=response)

    monkeypatch.setattr(instagram_caption.httpx, "Client", _SahteIstemci)

    generator = get_instagram_caption_generator()
    sonuc = generator.generate("Kayıp", "Kedi", None, None, None, None, None)
    assert sonuc is None


def test_gecici_hata_tekrar_denenip_basarili_olabilir(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_ENDPOINT", "https://ornek.test/v1/caption")
    monkeypatch.setenv("INSTAGRAM_CAPTION_AI_API_KEY", "sahte-anahtar")

    deneme_sayaci = {"n": 0}

    class _SahteCevapBasarili:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": {"caption": "ikinci denemede basarili"}}}]}

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            deneme_sayaci["n"] += 1
            if deneme_sayaci["n"] == 1:
                request = httpx.Request("POST", "https://ornek.test/v1/caption")
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError("gecici", request=request, response=response)
            return _SahteCevapBasarili()

    monkeypatch.setattr(instagram_caption.httpx, "Client", _SahteIstemci)

    generator = HttpInstagramCaptionGenerator(
        endpoint="https://ornek.test/v1/caption", model="sahte-model",
        api_key="sahte-anahtar", sleep=lambda _s: None)
    sonuc = generator.generate("Kayıp", "Kedi", None, None, None, None, None)
    assert sonuc == "ikinci denemede basarili"
    assert deneme_sayaci["n"] == 2
