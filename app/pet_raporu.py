# app/pet_raporu.py
"""Fotoğraf + sınıflandırıcı çıktısından zengin, anlatı ağırlıklı bir "pet
raporu" üretir (karakter profili, bakım ipuçları, şaşırtıcı bilgiler...).

Bu modülün yapısı `app/metin_analiz.py`'nin BİREBİR AYNISI (bilerek -- iki
modül de "sağlayıcıdan bağımsız arayüz + Null/Http uygulama + env ile seçilen
singleton" desenini kendi başına taşır, birbirinden import ETMEZ; tıpkı
`app/surum.py`'nin kimlik modeli seçim deseninin metin_analiz.py'de de
tekrarlanması gibi, burada da o iki modülün deseni tekrarlanıyor). Amaç aynı:
yarın sağlayıcı değişirse yalnızca burada yeni bir `PetReportAnalyzer` alt
sınıfı eklenir.

ÖNEMLİ SINIR: bu modül LLM'e ASLA ham sınıflandırıcı çıktısı (İngilizce
"cat"/"dog", None, "unknown") göndermez. `pet_raporu_olustur()` -- tüm
placeholder doldurma ve "LLM'i hiç çağırma" kapısının GEÇTİĞİ TEK YER --
`app/attributes.py::AttributeAnalyzer.analyze()`'in ham çıktısını (species,
breed, pattern, is_pet) alıp Türkçeye çevirir/BELIRLENEMEDI'ye indirger.
`PetReportAnalyzer.analyze()`'i DOĞRUDAN çağıran hiçbir kod bu çeviriyi
atlayamaz -- imzası zaten yalnızca önceden çevrilmiş string'leri kabul eder.

Sağlayıcı çağrısı BAŞARISIZ olursa (zaman aşımı, ağ hatası, bozuk yanıt) bu
modül hiçbir zaman dışarı istisna fırlatmaz -- metin_analiz.py'deki aynı
ilke: tek bir LLM hatası, zaten hesaplanmış embedding/öznitelik sonucunu
kaybettirmemeli. Başarısızlık `gecerli=False` + dahili bir `hata_nedeni`
ile ele alınır.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

import httpx

from .models import PetReportResult
from .pet_raporu_prompt import PET_RAPORU_PROMPT

# bkz. metin_analiz.py:_RETRYABLE_HTTP_STATUS -- aynı sınıflandırma, ayrı
# kopya (bu modül kendi başına taşınabilir olsun diye, bkz. modül docstring'i).
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = int(os.getenv("PET_REPORT_AI_MAX_ATTEMPTS", "3"))
_RETRY_BACKOFF_BASE_SECONDS = float(os.getenv("PET_REPORT_AI_RETRY_BACKOFF_SECONDS", "1.0"))

# Metin sınıflandırmasından (kısa bir JSON) çok daha uzun, anlatı ağırlıklı
# bir yanıt isteniyor (karakter profili, 3 şaşırtıcı bilgi, bakım ipuçları...)
# -- metin_analiz.py'nin 15sn varsayılanı burada dar olabilir.
_VARSAYILAN_ZAMAN_ASIMI = float(os.getenv("PET_REPORT_AI_TIMEOUT", "25"))

logger = logging.getLogger(__name__)

# Prompt'un kendi "ALAN KURALLARI" bölümünde sabitlediği 6 değer (bkz.
# pet_raporu_prompt.py). LLM bunun dışında bir şey yazarsa (serbest metin)
# ayıklanır -- metin_analiz.py'nin geçersiz category'yi UNCERTAIN'a
# indirgemesiyle aynı ilke, ama burada "hepsini kapsayan" tek bir değer
# tanımlı değil (prompt öyle bir yedek tanımlamıyor), o yüzden yalnızca
# uyarı loglanıp ham değer olduğu gibi bırakılır.
_GECERLI_HATA_NEDENLERI = frozenset({
    "HAYVAN_YOK", "KEDI_KOPEK_DEGIL", "INSAN_FOTOGRAFI",
    "GORUNTU_COK_BULANIK", "HAYVAN_COK_UZAK", "SINIFLANDIRICI_CELISKISI",
})

# LLM'e hiç gitmeyen, yalnızca bu modülün kendi ürettiği dahili sebep --
# sağlayıcı yapılandırılmamış (NullPetReportAnalyzer) ya da çağrı sonunda
# hâlâ başarısız (HttpPetReportAnalyzer'ın retry'ları tükendi).
_SERVIS_KULLANILAMIYOR = "SERVIS_KULLANILAMIYOR"

_BELIRLENEMEDI = "BELIRLENEMEDI"

_TUR_TR = {"cat": "Kedi", "dog": "Köpek"}


class _RetryableProviderError(Exception):
    """İç kullanım: HttpPetReportAnalyzer._tek_deneme()'nin, hatanın YENİDEN
    DENENEBİLİR olduğunu analyze()'daki retry döngüsüne işaretlemesi için.
    Asla bu modülün dışına sızmaz (bkz. analyze())."""


class PetReportAnalyzer(ABC):
    """Sağlayıcıdan bağımsız arayüz. Tek sorumluluk: fotoğraf + ÖNCEDEN
    ÇEVRİLMİŞ sınıflandırıcı bağlamı -> zengin, anlatı ağırlıklı rapor.

    `tur`, `irk`, `desen` burada ÇAĞRILDIĞINDA zaten Türkçe/BELIRLENEMEDI
    olmalıdır -- ham İngilizce/None geçirmek bu arayüzün sözleşmesini bozar
    (bkz. modül docstring'i, `pet_raporu_olustur()`).
    """

    @abstractmethod
    def analyze(self, image_bytes: bytes, tur: str, irk: str, desen: str,
               kullanici_notu: str | None = None) -> PetReportResult:
        raise NotImplementedError


class NullPetReportAnalyzer(PetReportAnalyzer):
    """`PET_REPORT_AI_PROVIDER` boşsa kullanılan güvenli varsayılan.

    Ağ çağrısı yapmaz, API anahtarı gerektirmez -- testler ve yerel
    geliştirme hiçbir dış bağımlılık olmadan çalışabilsin diye (bkz.
    metin_analiz.py::NullTextAnalyzer, aynı gerekçe).
    """

    def analyze(self, image_bytes: bytes, tur: str, irk: str, desen: str,
               kullanici_notu: str | None = None) -> PetReportResult:
        return PetReportResult(gecerli=False, hata_nedeni=_SERVIS_KULLANILAMIYOR)


class HttpPetReportAnalyzer(PetReportAnalyzer):
    """Düz HTTP üzerinden yapılandırılmış-çıktı isteyen bir LLM sağlayıcısını
    çağırır (bkz. metin_analiz.py::HttpTextAnalyzer -- aynı yaklaşım, yeni
    SDK bağımlılığı EKLENMEDİ, `httpx` zaten üretim bağımlılığı)."""

    def __init__(self, endpoint: str, model: str, api_key: str,
                timeout: float = _VARSAYILAN_ZAMAN_ASIMI, sleep=time.sleep):
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        # Testler gerçek zaman geçirmeden retry-backoff yolunu kanıtlayabilsin
        # diye enjekte edilebilir (bkz. metin_analiz.py'nin aynı deseni).
        self._sleep = sleep

    def analyze(self, image_bytes: bytes, tur: str, irk: str, desen: str,
               kullanici_notu: str | None = None) -> PetReportResult:
        # DÜZELTME İLKESİ metin_analiz.py::HttpTextAnalyzer.analyze() ile
        # AYNI (bkz. oradaki 2026-08-19 kullanıcı raporu notu): geçici
        # (429/5xx, ağ/zaman aşımı) hatalar en fazla _MAX_ATTEMPTS kez kısa
        # bir backoff ile yeniden denenir; kalıcı hatalar (400/401/403,
        # bozuk yanıt) İLK denemede güvenli varsayılana düşer.
        son_hata: Exception | None = None
        for deneme in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._tek_deneme(image_bytes, tur, irk, desen, kullanici_notu)
            except _RetryableProviderError as e:
                son_hata = e.__cause__ or e
                if deneme < _MAX_ATTEMPTS:
                    bekleme = _RETRY_BACKOFF_BASE_SECONDS * (2 ** (deneme - 1))
                    logger.warning(
                        "Pet raporu sağlayıcı hatası (geçici, deneme %d/%d, %.1fsn sonra tekrar): %s",
                        deneme, _MAX_ATTEMPTS, bekleme, son_hata,
                    )
                    self._sleep(bekleme)
                    continue
            except Exception as e:
                son_hata = e
                break

        # Sağlayıcı ne sebeple olursa olsun başarısız olsun -- bu modül asla
        # fırlatmaz (bkz. modül docstring'i).
        logger.warning("Pet raporu sağlayıcı hatası, güvenli varsayılana düşülüyor: %s", son_hata)
        return PetReportResult(gecerli=False, hata_nedeni=_SERVIS_KULLANILAMIYOR)

    def _tek_deneme(self, image_bytes: bytes, tur: str, irk: str, desen: str,
                    kullanici_notu: str | None) -> PetReportResult:
        """Tek bir sağlayıcı çağrısı. Geçici (yeniden denenebilir) bir hata
        _RetryableProviderError'a sarılıp fırlatılır; kalıcı bir hata olduğu
        gibi fırlatılır -- analyze()'daki retry döngüsü ikisini ayırt eder."""
        gövde = self._istek_govdesi(image_bytes, tur, irk, desen, kullanici_notu)
        try:
            with httpx.Client(timeout=self._timeout) as istemci:
                cevap = istemci.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json"},
                    json=gövde,
                )
                cevap.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in _RETRYABLE_HTTP_STATUS:
                raise _RetryableProviderError() from e
            raise
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise _RetryableProviderError() from e
        return self._yaniti_ayristir(cevap.json())

    def _istek_govdesi(self, image_bytes: bytes, tur: str, irk: str, desen: str,
                       kullanici_notu: str | None) -> dict:
        prompt = (PET_RAPORU_PROMPT
                  .replace("{{TUR}}", tur)
                  .replace("{{IRK}}", irk)
                  .replace("{{DESEN}}", desen))

        # Prompt metninin kendisinde (pet_raporu_prompt.py) kullanıcı notu
        # için bir placeholder YOK -- yalnızca "# GİRDİ" bölümünde opsiyonel
        # bir girdi olarak SÖZÜ ediliyor. Prompt sabitini değiştirmemek için
        # (bkz. modül docstring'i: "Prompt'u koda gömme, ayrı bir sabitte
        # tut") notu ayrı bir ek olarak, sistem talimatından SONRA ekliyoruz.
        if kullanici_notu and kullanici_notu.strip():
            prompt += (
                "\n\nKullanıcı notu (bağlam olarak dikkate al; fotoğrafla "
                f"çelişiyorsa fotoğrafa güven): {kullanici_notu.strip()}"
            )

        import base64
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        return {"model": self._model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}

    def _yaniti_ayristir(self, veri: dict) -> PetReportResult:
        # bkz. metin_analiz.py::HttpTextAnalyzer._yaniti_ayristir -- aynı dar
        # nokta deseni, standart chat/completions zarfı (OpenAI, Gemini'nin
        # OpenAI-uyumlu katmanı): {"choices": [{"message": {"content": "<json>"}}]}.
        icerik = None
        if isinstance(veri, dict):
            secenekler = veri.get("choices")
            if isinstance(secenekler, list) and secenekler:
                ilk = secenekler[0]
                mesaj = ilk.get("message") if isinstance(ilk, dict) else None
                icerik = mesaj.get("content") if isinstance(mesaj, dict) else None
        if isinstance(icerik, str):
            import json
            icerik = json.loads(icerik)
        alanlar = icerik if isinstance(icerik, dict) else veri

        gecerli = bool(alanlar.get("gecerli", False))

        hata_nedeni = alanlar.get("hata_nedeni")
        if hata_nedeni is not None and hata_nedeni not in _GECERLI_HATA_NEDENLERI:
            logger.warning("Pet raporu beklenmeyen hata_nedeni döndürdü: %r", hata_nedeni)

        def _deger_guven(anahtar: str) -> dict | None:
            d = alanlar.get(anahtar)
            if not isinstance(d, dict) or "deger" not in d:
                return None
            return {"deger": d.get("deger"), "guven": float(d.get("guven") or 0.0)}

        tahmini_yas = alanlar.get("tahmini_yas")
        if isinstance(tahmini_yas, dict) and "aralik" in tahmini_yas:
            tahmini_yas = {
                "aralik": tahmini_yas.get("aralik"),
                "yasam_evresi": tahmini_yas.get("yasam_evresi") or "Belirsiz",
                "guven": float(tahmini_yas.get("guven") or 0.0),
            }
        else:
            tahmini_yas = None

        cinsiyet = alanlar.get("cinsiyet")
        if isinstance(cinsiyet, dict) and "tahmin" in cinsiyet:
            cinsiyet = {"tahmin": cinsiyet.get("tahmin"),
                       "guven": float(cinsiyet.get("guven") or 0.0)}
        else:
            cinsiyet = None

        bakim = alanlar.get("bakim_ipuclari")
        if not isinstance(bakim, dict):
            bakim = None

        return PetReportResult(
            gecerli=gecerli,
            hata_nedeni=hata_nedeni,
            irka_ozel_icerik=alanlar.get("irka_ozel_icerik"),
            renk_tarifi=_deger_guven("renk_tarifi"),
            goz_rengi=_deger_guven("goz_rengi"),
            tahmini_yas=tahmini_yas,
            cinsiyet=cinsiyet,
            tahmini_boyut=_deger_guven("tahmini_boyut"),
            ayirt_edici_isaretler=list(alanlar.get("ayirt_edici_isaretler") or []),
            genel_durum_gozlemi=alanlar.get("genel_durum_gozlemi"),
            karakter_profili=alanlar.get("karakter_profili"),
            sasirtici_bilgiler=list(alanlar.get("sasirtici_bilgiler") or []),
            dikkat_edilmesi_gerekenler=list(alanlar.get("dikkat_edilmesi_gerekenler") or []),
            bakim_ipuclari=bakim,
            ek_hayvanlar=alanlar.get("ek_hayvanlar"),
            goruntu_kalite_notu=alanlar.get("goruntu_kalite_notu"),
        )


_analyzer_ornegi: PetReportAnalyzer | None = None


def get_pet_report_analyzer() -> PetReportAnalyzer:
    """`app/metin_analiz.py::get_text_analyzer()` deseninin birebir aynısı --
    bilerek AYRI env değişkenleri kullanır (`PET_REPORT_AI_*`, `TEXT_AI_*`
    DEĞİL): iki özellik birbirinden bağımsız açılıp kapanabilsin, biri
    yapılandırılmadan diğeri sessizce onun ayarını miras almasın.
    """
    global _analyzer_ornegi
    if _analyzer_ornegi is not None:
        return _analyzer_ornegi

    saglayici = os.getenv("PET_REPORT_AI_PROVIDER", "").strip()
    if not saglayici:
        _analyzer_ornegi = NullPetReportAnalyzer()
        return _analyzer_ornegi

    endpoint = os.getenv("PET_REPORT_AI_ENDPOINT", "")
    model = os.getenv("PET_REPORT_AI_MODEL", "")
    api_key = os.getenv("PET_REPORT_AI_API_KEY", "")
    if not endpoint or not api_key:
        logger.warning(
            "PET_REPORT_AI_PROVIDER=%r ayarlı ama PET_REPORT_AI_ENDPOINT/"
            "PET_REPORT_AI_API_KEY eksik — NullPetReportAnalyzer'a düşülüyor.",
            saglayici)
        _analyzer_ornegi = NullPetReportAnalyzer()
        return _analyzer_ornegi

    _analyzer_ornegi = HttpPetReportAnalyzer(endpoint=endpoint, model=model, api_key=api_key)
    return _analyzer_ornegi


def _sifirla_singleton_testler_icin() -> None:
    """Yalnızca testler için (bkz. metin_analiz.py'nin aynı adlı fonksiyonu)."""
    global _analyzer_ornegi
    _analyzer_ornegi = None


def pet_raporu_olustur(image_bytes: bytes, *, species: str, breed: str | None,
                       pattern: str | None, is_pet: bool,
                       kullanici_notu: str | None = None) -> PetReportResult:
    """ENTEGRASYON NOKTASI -- `app/attributes.py::AttributeAnalyzer.analyze()`
    çıktısını (ham, İngilizce, None/"unknown" içerebilir) alır, LLM'e güvenli
    hale getirip iletir. `PetReportAnalyzer.analyze()`'i BAŞKA HİÇBİR YERDEN
    doğrudan çağırma -- bu çeviri/kapı mantığını atlarsın.

    Kurallar (kullanıcı kararı, bkz. proje geçmişi):
      - species "cat"/"dog" DEĞİLSE ya da is_pet False İSE: LLM'i HİÇ ÇAĞIRMA
        (gereksiz maliyet + prompt zaten "tur"un doğru olduğunu varsayıyor).
        gecerli=False, hata_nedeni="KEDI_KOPEK_DEGIL" ile erken dön.
      - breed None ise (attributes.py'de zaten BREED_MIN_PROB=0.70 eşiğinin
        altında kalmış demektir, bkz. attributes.py:313) -> "BELIRLENEMEDI".
        breed_confidence'ı BURAYA DA, prompt'a da HİÇ taşıma: kalibre değil,
        ham top-1 skoru düşük güvenilirlikte bir sayıyı kesinmiş gibi
        gösterip yanıltır (kullanıcı kararı).
      - pattern None/"unknown" ise -> "BELIRLENEMEDI".
    """
    if species not in ("cat", "dog") or not is_pet:
        return PetReportResult(gecerli=False, hata_nedeni="KEDI_KOPEK_DEGIL")

    tur = _TUR_TR[species]
    irk = breed if breed else _BELIRLENEMEDI
    desen = pattern if (pattern and pattern != "unknown") else _BELIRLENEMEDI

    return get_pet_report_analyzer().analyze(
        image_bytes, tur=tur, irk=irk, desen=desen, kullanici_notu=kullanici_notu)
