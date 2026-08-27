# tests/test_pet_raporu.py
"""PetReportAnalyzer soyutlamasının ve pet_raporu_olustur() entegrasyon
kapısının testleri — gerçek ağ çağrısı YAPMAZ.

Yapı `tests/test_metin_analiz.py`'nin birebir aynısı (bkz. app/pet_raporu.py
docstring'i): sağlayıcı çağrısı monkeypatch ile taklit edilir, testin konusu
"sağlayıcı hatası güvenli varsayılana düşüyor mu" gibi davranış, gerçek bir
LLM'in doğruluğu değil.
"""
import httpx
import pytest

from app import pet_raporu
from app.pet_raporu import (HttpPetReportAnalyzer, NullPetReportAnalyzer,
                            get_pet_report_analyzer, pet_raporu_olustur)


@pytest.fixture(autouse=True)
def _singleton_sifirla():
    pet_raporu._sifirla_singleton_testler_icin()
    yield
    pet_raporu._sifirla_singleton_testler_icin()


# --------------------------------------------------------------------------
# Sağlayıcı seçimi
# --------------------------------------------------------------------------

def test_saglayici_bos_ise_null_analyzer_donuyor(monkeypatch):
    monkeypatch.delenv("PET_REPORT_AI_PROVIDER", raising=False)
    analyzer = get_pet_report_analyzer()
    assert isinstance(analyzer, NullPetReportAnalyzer)


def test_null_analyzer_her_zaman_gecersiz_ve_servis_kullanilamiyor_doner():
    sonuc = NullPetReportAnalyzer().analyze(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI", "BELIRLENEMEDI")
    assert sonuc.gecerli is False
    assert sonuc.hata_nedeni == "SERVIS_KULLANILAMIYOR"
    assert sonuc.karakter_profili is None  # uydurma yok


def test_saglayici_ayarli_ama_endpoint_eksikse_null_analyzera_dusuluyor(monkeypatch):
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.delenv("PET_REPORT_AI_ENDPOINT", raising=False)
    monkeypatch.delenv("PET_REPORT_AI_API_KEY", raising=False)
    analyzer = get_pet_report_analyzer()
    assert isinstance(analyzer, NullPetReportAnalyzer)


def test_metin_analiz_env_degiskenleri_bu_ozelligi_etkilemez(monkeypatch):
    """TEXT_AI_* ile PET_REPORT_AI_* bilerek AYRI -- biri açıkken diğeri
    yapılandırılmamışsa sessizce onun ayarını miras almamalı."""
    monkeypatch.setenv("TEXT_AI_PROVIDER", "gemini")
    monkeypatch.setenv("TEXT_AI_ENDPOINT", "https://ornek.test/v1/text")
    monkeypatch.setenv("TEXT_AI_API_KEY", "metin-anahtari")
    monkeypatch.delenv("PET_REPORT_AI_PROVIDER", raising=False)
    analyzer = get_pet_report_analyzer()
    assert isinstance(analyzer, NullPetReportAnalyzer)


# --------------------------------------------------------------------------
# pet_raporu_olustur() -- entegrasyon kapısı (asıl yeni davranış burada)
# --------------------------------------------------------------------------

def test_tur_belirsizse_llm_hic_cagrilmaz(monkeypatch):
    """species 'cat'/'dog' değilse (attributes.py 'unknown' döndürür)
    HttpPetReportAnalyzer'a hiç ulaşılmamalı -- gereksiz LLM çağrısı yok."""
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "sahte-anahtar")

    cagrildi = []
    monkeypatch.setattr(
        HttpPetReportAnalyzer, "analyze",
        lambda self, *a, **kw: cagrildi.append(1))

    sonuc = pet_raporu_olustur(b"\xff\xd8\xff", species="unknown", breed=None,
                               pattern=None, is_pet=True)
    assert sonuc.gecerli is False
    assert sonuc.hata_nedeni == "KEDI_KOPEK_DEGIL"
    assert cagrildi == [], "tür belirsizken LLM çağrılmamalıydı"


def test_is_pet_false_ise_llm_hic_cagrilmaz(monkeypatch):
    """species 'cat' dönse bile is_pet=False ise (CLIP 'hayvan var mı'
    kapısı reddetmiş) yine LLM'e gidilmemeli."""
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "sahte-anahtar")

    cagrildi = []
    monkeypatch.setattr(
        HttpPetReportAnalyzer, "analyze",
        lambda self, *a, **kw: cagrildi.append(1))

    sonuc = pet_raporu_olustur(b"\xff\xd8\xff", species="cat", breed=None,
                               pattern=None, is_pet=False)
    assert sonuc.gecerli is False
    assert sonuc.hata_nedeni == "KEDI_KOPEK_DEGIL"
    assert cagrildi == []


def test_placeholderlar_dogru_cevriliyor(monkeypatch):
    """species 'cat'->'Kedi', breed None->'BELIRLENEMEDI',
    pattern 'unknown'->'BELIRLENEMEDI' -- LLM'e ham İngilizce/None sızmamalı."""
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "sahte-anahtar")

    yakalanan = {}

    def _sahte_analyze(self, image_bytes, tur, irk, desen, kullanici_notu=None):
        yakalanan.update(tur=tur, irk=irk, desen=desen)
        from app.models import PetReportResult
        return PetReportResult(gecerli=True)

    monkeypatch.setattr(HttpPetReportAnalyzer, "analyze", _sahte_analyze)

    pet_raporu_olustur(b"\xff\xd8\xff", species="cat", breed=None,
                       pattern="unknown", is_pet=True)

    assert yakalanan == {"tur": "Kedi", "irk": "BELIRLENEMEDI", "desen": "BELIRLENEMEDI"}


def test_irk_ve_desen_belirliyse_oldugu_gibi_gecer(monkeypatch):
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "sahte-anahtar")

    yakalanan = {}

    def _sahte_analyze(self, image_bytes, tur, irk, desen, kullanici_notu=None):
        yakalanan.update(tur=tur, irk=irk, desen=desen)
        from app.models import PetReportResult
        return PetReportResult(gecerli=True)

    monkeypatch.setattr(HttpPetReportAnalyzer, "analyze", _sahte_analyze)

    pet_raporu_olustur(b"\xff\xd8\xff", species="dog", breed="Pug",
                       pattern="solid", is_pet=True)

    assert yakalanan == {"tur": "Köpek", "irk": "Pug", "desen": "solid"}


def test_breed_confidence_hicbir_yere_sizmiyor():
    """Kullanıcı kararı: breed_confidence (kalibre değil) prompt'a hiç
    taşınmamalı. pet_raporu_olustur'un imzasında bu parametre YOK -- varlığı
    kendisi bu kuralı garanti eder, ama açıkça da doğrulayalım."""
    import inspect
    parametreler = inspect.signature(pet_raporu_olustur).parameters
    assert "breed_confidence" not in parametreler


# --------------------------------------------------------------------------
# HttpPetReportAnalyzer -- istek gövdesi
# --------------------------------------------------------------------------

def test_istek_govdesi_daima_multimodal():
    """Metin analizinden farklı olarak burada görsel HER ZAMAN zorunlu --
    içerik hep [text, image_url] listesi olmalı."""
    analyzer = HttpPetReportAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar")
    govde = analyzer._istek_govdesi(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI",
                                    "BELIRLENEMEDI", None)
    icerik = govde["messages"][0]["content"]
    assert isinstance(icerik, list)
    assert icerik[0]["type"] == "text"
    assert icerik[1]["type"] == "image_url"
    assert icerik[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert govde["response_format"] == {"type": "json_object"}


def test_placeholderlar_prompt_metnine_islenir():
    analyzer = HttpPetReportAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar")
    govde = analyzer._istek_govdesi(b"\xff\xd8\xff", "Kedi", "Van Kedisi",
                                    "solid", None)
    metin = govde["messages"][0]["content"][0]["text"]
    assert "tur: Kedi" in metin
    assert "irk: Van Kedisi" in metin
    assert "desen: solid" in metin
    assert "{{TUR}}" not in metin and "{{IRK}}" not in metin and "{{DESEN}}" not in metin


def test_kullanici_notu_verilince_govdeye_eklenir():
    analyzer = HttpPetReportAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar")
    govde = analyzer._istek_govdesi(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI",
                                    "BELIRLENEMEDI", "3 aylık, sokakta bulundu")
    metin = govde["messages"][0]["content"][0]["text"]
    assert "3 aylık, sokakta bulundu" in metin


def test_kullanici_notu_yoksa_govdeye_eklenmez():
    analyzer = HttpPetReportAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar")
    govde = analyzer._istek_govdesi(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI",
                                    "BELIRLENEMEDI", None)
    metin = govde["messages"][0]["content"][0]["text"]
    assert "Kullanıcı notu" not in metin


# --------------------------------------------------------------------------
# Yanıt ayrıştırma
# --------------------------------------------------------------------------

def test_basarili_cevap_dogru_esler(monkeypatch):
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "sahte-anahtar")

    veri = {
        "gecerli": True,
        "hata_nedeni": None,
        "irka_ozel_icerik": True,
        "renk_tarifi": {"deger": "Kahverengi tekir", "guven": 90},
        "goz_rengi": {"deger": "Kehribar", "guven": 88},
        "tahmini_yas": {"aralik": "2-4 yaş", "yasam_evresi": "Yetişkin", "guven": 60},
        "cinsiyet": {"tahmin": "Erkek", "guven": 45},
        "tahmini_boyut": {"deger": "Orta boy", "guven": 35},
        "ayirt_edici_isaretler": ["Sol pati beyaz"],
        "genel_durum_gozlemi": "Bakımlı görünüyor.",
        "karakter_profili": "Meraklı ve oyuncu.",
        "sasirtici_bilgiler": ["a", "b", "c"],
        "dikkat_edilmesi_gerekenler": ["x", "y", "z"],
        "bakim_ipuclari": {"beslenme": "...", "tuy_bakimi": "...", "aktivite": "..."},
        "ek_hayvanlar": None,
        "goruntu_kalite_notu": None,
    }

    class _SahteCevap:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": veri}}]}

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return _SahteCevap()

    monkeypatch.setattr(pet_raporu.httpx, "Client", _SahteIstemci)

    analyzer = get_pet_report_analyzer()
    assert isinstance(analyzer, HttpPetReportAnalyzer)
    sonuc = analyzer.analyze(b"\xff\xd8\xff", "Kedi", "Tekir", "solid")

    assert sonuc.gecerli is True
    assert sonuc.renk_tarifi.deger == "Kahverengi tekir"
    assert sonuc.renk_tarifi.guven == 90
    assert sonuc.tahmini_yas.yasam_evresi == "Yetişkin"
    assert sonuc.cinsiyet.tahmin == "Erkek"
    assert sonuc.ayirt_edici_isaretler == ["Sol pati beyaz"]
    assert sonuc.bakim_ipuclari.beslenme == "..."


def test_gecersiz_alan_bicimi_none_a_dusurur_crash_etmez(monkeypatch):
    """renk_tarifi beklenmedik bir şekilde (örn. düz string) gelirse
    Pydantic'i patlatıp kalıcı hataya düşürmek yerine None'a düşürülmeli."""
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "sahte-anahtar")

    class _SahteCevap:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content":
                {"gecerli": True, "renk_tarifi": "sadece bir string"}}}]}

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return _SahteCevap()

    monkeypatch.setattr(pet_raporu.httpx, "Client", _SahteIstemci)

    analyzer = get_pet_report_analyzer()
    sonuc = analyzer.analyze(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI", "BELIRLENEMEDI")
    assert sonuc.gecerli is True
    assert sonuc.renk_tarifi is None


def test_beklenmeyen_hata_nedeni_crash_etmez_sadece_loglanir(monkeypatch, caplog):
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "sahte-anahtar")

    class _SahteCevap:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content":
                {"gecerli": False, "hata_nedeni": "SERBEST_METIN_UYDURMA"}}}]}

    class _SahteIstemci:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return _SahteCevap()

    monkeypatch.setattr(pet_raporu.httpx, "Client", _SahteIstemci)

    analyzer = get_pet_report_analyzer()
    sonuc = analyzer.analyze(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI", "BELIRLENEMEDI")
    assert sonuc.gecerli is False
    assert sonuc.hata_nedeni == "SERBEST_METIN_UYDURMA"  # olduğu gibi bırakılır


# --------------------------------------------------------------------------
# Hata/retry davranışı (bkz. test_metin_analiz.py'nin aynı testleri)
# --------------------------------------------------------------------------

def test_gecici_503_hatasi_yeniden_denenip_basariyla_sonuclanir(monkeypatch):
    from app.models import PetReportResult

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

    monkeypatch.setattr(pet_raporu.httpx, "Client", _SahteIstemci)
    monkeypatch.setattr(
        HttpPetReportAnalyzer, "_yaniti_ayristir",
        lambda self, veri: PetReportResult(gecerli=True, karakter_profili="ok"),
    )

    uykular = []
    analyzer = HttpPetReportAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar", sleep=uykular.append,
    )
    sonuc = analyzer.analyze(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI", "BELIRLENEMEDI")

    assert len(denemeler) == 2
    assert len(uykular) == 1
    assert sonuc.gecerli is True
    assert sonuc.karakter_profili == "ok"


def test_gecici_hata_azami_deneme_sayisini_asarsa_guvenli_varsayilana_duser(monkeypatch):
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

    monkeypatch.setattr(pet_raporu.httpx, "Client", _SahteIstemci)

    uykular = []
    analyzer = HttpPetReportAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar", sleep=uykular.append,
    )
    sonuc = analyzer.analyze(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI", "BELIRLENEMEDI")

    assert len(denemeler) == pet_raporu._MAX_ATTEMPTS
    assert len(uykular) == pet_raporu._MAX_ATTEMPTS - 1
    assert sonuc.gecerli is False
    assert sonuc.hata_nedeni == "SERVIS_KULLANILAMIYOR"


def test_kalici_400_hatasi_hic_yeniden_denenmez(monkeypatch):
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

    monkeypatch.setattr(pet_raporu.httpx, "Client", _SahteIstemci)

    uykular = []
    analyzer = HttpPetReportAnalyzer(
        endpoint="https://ornek.test/v1/analyze", model="sahte-model",
        api_key="sahte-anahtar", sleep=uykular.append,
    )
    sonuc = analyzer.analyze(b"\xff\xd8\xff", "Kedi", "BELIRLENEMEDI", "BELIRLENEMEDI")

    assert len(denemeler) == 1
    assert uykular == []
    assert sonuc.gecerli is False
    assert sonuc.hata_nedeni == "SERVIS_KULLANILAMIYOR"


# --------------------------------------------------------------------------
# Kimlik alanları (tur/irk/desen) -- cevapta taşınıyor mu (2026-08-27 eki)
# --------------------------------------------------------------------------

def test_kimlik_alanlari_cevapta_tasiniyor(monkeypatch):
    """FE başlığı ("Sen bir Köpeksin -- Pug!") sınıflandırıcı kimliğini
    /analyze_pet cevabından okur; LLM raporuna eklenmiş çevrilmiş girdiler
    kaybolursa bu test düşer."""
    monkeypatch.setenv("PET_REPORT_AI_PROVIDER", "ornek-saglayici")
    monkeypatch.setenv("PET_REPORT_AI_ENDPOINT", "https://ornek.test/v1/analyze")
    monkeypatch.setenv("PET_REPORT_AI_API_KEY", "sahte-anahtar")

    def _sahte_analyze(self, image_bytes, tur, irk, desen, kullanici_notu=None):
        from app.models import PetReportResult
        return PetReportResult(gecerli=True)

    monkeypatch.setattr(HttpPetReportAnalyzer, "analyze", _sahte_analyze)

    sonuc = pet_raporu_olustur(b"\xff\xd8\xff", species="dog", breed="Pug",
                               pattern=None, is_pet=True)

    assert (sonuc.tur, sonuc.irk, sonuc.desen) == ("Köpek", "Pug", "BELIRLENEMEDI")


def test_saglayici_dusse_de_kimlik_alanlari_dolu(monkeypatch):
    """Sağlayıcı başarısız olsa bile (gecerli=False) sınıflandırıcı kimliği
    elimizde -- FE en azından tür/ırk başlığını gösterebilmeli."""
    monkeypatch.delenv("PET_REPORT_AI_PROVIDER", raising=False)  # NullAnalyzer

    sonuc = pet_raporu_olustur(b"\xff\xd8\xff", species="cat", breed=None,
                               pattern="tabby", is_pet=True)

    assert sonuc.gecerli is False
    assert (sonuc.tur, sonuc.irk, sonuc.desen) == ("Kedi", "BELIRLENEMEDI", "tabby")


def test_erken_donuste_kimlik_alanlari_bos(monkeypatch):
    """KEDI_KOPEK_DEGIL erken dönüşünde tür güvenilir değil -- alanlar None
    kalmalı, yanlış bir "Kedi" başlığı kurulmamalı."""
    monkeypatch.delenv("PET_REPORT_AI_PROVIDER", raising=False)

    sonuc = pet_raporu_olustur(b"\xff\xd8\xff", species="bird", breed=None,
                               pattern=None, is_pet=True)

    assert sonuc.gecerli is False
    assert sonuc.tur is None and sonuc.irk is None and sonuc.desen is None
