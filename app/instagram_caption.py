# app/instagram_caption.py
"""Bir ilanın (kayıp/bulundu/sahiplendirme) bilgilerinden Instagram gönderi
metni (caption) üretir.

Bu modülün yapısı `app/pet_raporu.py`'nin BİREBİR AYNISI (bilerek --
"sağlayıcıdan bağımsız arayüz + Null/Http uygulama + env ile seçilen
singleton" deseni burada da tekrarlanıyor, birbirinden import ETMEZ). Amaç
aynı: yarın sağlayıcı değişirse yalnızca burada yeni bir
`InstagramCaptionGenerator` alt sınıfı eklenir.

`pet_raporu.py`'den TEK farkı: girdi zaten temiz/Türkçe alanlardır (Java
tarafı, Ad entity'sinden doldurur) -- burada pet_raporu.py::pet_raporu_olustur()
gibi bir "ham sınıflandırıcı çıktısını çevir" kapısı YOK, çünkü çevrilecek
ham bir girdi yok. Ayrıca görsele HİÇ ihtiyaç duymaz (yalnızca metin girdisi).

Sağlayıcı çağrısı BAŞARISIZ olursa (zaman aşımı, ağ hatası, bozuk yanıt) bu
modül hiçbir zaman dışarı istisna fırlatmaz -- pet_raporu.py'deki AYNI ilke:
tek bir LLM hatası, ilan oluşturma akışını hiçbir zaman etkilememeli. Java
tarafı `caption=None` gördüğünde kendi şablon caption'ına düşer.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

import httpx

from .instagram_caption_prompt import INSTAGRAM_CAPTION_PROMPT

# bkz. pet_raporu.py::_RETRYABLE_HTTP_STATUS -- aynı sınıflandırma, ayrı kopya
# (bu modül kendi başına taşınabilir olsun diye, bkz. modül docstring'i).
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = int(os.getenv("INSTAGRAM_CAPTION_AI_MAX_ATTEMPTS", "3"))
_RETRY_BACKOFF_BASE_SECONDS = float(os.getenv("INSTAGRAM_CAPTION_AI_RETRY_BACKOFF_SECONDS", "1.0"))
_VARSAYILAN_ZAMAN_ASIMI = float(os.getenv("INSTAGRAM_CAPTION_AI_TIMEOUT", "15"))

logger = logging.getLogger(__name__)

_BELIRLENEMEDI = "BELIRLENEMEDI"

_ILAN_TURU_TR = {"LOST": "Kayıp", "FOUND": "Bulundu", "ADOPTION": "Sahiplendirme"}


class _RetryableProviderError(Exception):
    """İç kullanım: HttpInstagramCaptionGenerator._tek_deneme()'nin, hatanın
    YENİDEN DENENEBİLİR olduğunu generate()'daki retry döngüsüne
    işaretlemesi için. Asla bu modülün dışına sızmaz (bkz. generate())."""


class InstagramCaptionGenerator(ABC):
    """Sağlayıcıdan bağımsız arayüz. Tek sorumluluk: ilan alanları -> tek bir
    Türkçe Instagram caption metni."""

    @abstractmethod
    def generate(self, ilan_turu: str, tur: str, irk: str | None,
                renkler: str | None, il_ilce: str | None,
                aciklama: str | None, ayirt_edici: list[str] | None) -> str | None:
        raise NotImplementedError


class NullInstagramCaptionGenerator(InstagramCaptionGenerator):
    """`INSTAGRAM_CAPTION_AI_PROVIDER` boşsa kullanılan güvenli varsayılan.

    Ağ çağrısı yapmaz, API anahtarı gerektirmez -- testler ve yerel
    geliştirme hiçbir dış bağımlılık olmadan çalışabilsin diye (bkz.
    pet_raporu.py::NullPetReportAnalyzer, aynı gerekçe).
    """

    def generate(self, ilan_turu: str, tur: str, irk: str | None,
                renkler: str | None, il_ilce: str | None,
                aciklama: str | None, ayirt_edici: list[str] | None) -> str | None:
        return None


class HttpInstagramCaptionGenerator(InstagramCaptionGenerator):
    """Düz HTTP üzerinden yapılandırılmış-çıktı isteyen bir LLM sağlayıcısını
    çağırır (bkz. pet_raporu.py::HttpPetReportAnalyzer -- aynı yaklaşım)."""

    def __init__(self, endpoint: str, model: str, api_key: str,
                timeout: float = _VARSAYILAN_ZAMAN_ASIMI, sleep=time.sleep):
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        # Testler gerçek zaman geçirmeden retry-backoff yolunu kanıtlayabilsin
        # diye enjekte edilebilir (bkz. pet_raporu.py'nin aynı deseni).
        self._sleep = sleep

    def generate(self, ilan_turu: str, tur: str, irk: str | None,
                renkler: str | None, il_ilce: str | None,
                aciklama: str | None, ayirt_edici: list[str] | None) -> str | None:
        son_hata: Exception | None = None
        for deneme in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._tek_deneme(ilan_turu, tur, irk, renkler, il_ilce,
                                        aciklama, ayirt_edici)
            except _RetryableProviderError as e:
                son_hata = e.__cause__ or e
                if deneme < _MAX_ATTEMPTS:
                    bekleme = _RETRY_BACKOFF_BASE_SECONDS * (2 ** (deneme - 1))
                    logger.warning(
                        "Instagram caption sağlayıcı hatası (geçici, deneme %d/%d, %.1fsn sonra tekrar): %s",
                        deneme, _MAX_ATTEMPTS, bekleme, son_hata,
                    )
                    self._sleep(bekleme)
                    continue
            except Exception as e:
                son_hata = e
                break

        # Sağlayıcı ne sebeple olursa olsun başarısız olsun -- bu modül asla
        # fırlatmaz (bkz. modül docstring'i).
        logger.warning("Instagram caption sağlayıcı hatası, None'a düşülüyor: %s", son_hata)
        return None

    def _tek_deneme(self, ilan_turu: str, tur: str, irk: str | None,
                    renkler: str | None, il_ilce: str | None,
                    aciklama: str | None, ayirt_edici: list[str] | None) -> str | None:
        gövde = self._istek_govdesi(ilan_turu, tur, irk, renkler, il_ilce,
                                    aciklama, ayirt_edici)
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

    def _istek_govdesi(self, ilan_turu: str, tur: str, irk: str | None,
                       renkler: str | None, il_ilce: str | None,
                       aciklama: str | None, ayirt_edici: list[str] | None) -> dict:
        prompt = (INSTAGRAM_CAPTION_PROMPT
                  .replace("{{ILAN_TURU}}", ilan_turu)
                  .replace("{{TUR}}", tur)
                  .replace("{{IRK}}", irk or _BELIRLENEMEDI)
                  .replace("{{RENKLER}}", renkler or _BELIRLENEMEDI)
                  .replace("{{IL_ILCE}}", il_ilce or "")
                  .replace("{{ACIKLAMA}}", aciklama or "(yok)")
                  .replace("{{AYIRT_EDICI}}", ", ".join(ayirt_edici) if ayirt_edici else "(yok)"))

        return {"model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}}

    def _yaniti_ayristir(self, veri: dict) -> str | None:
        # bkz. pet_raporu.py::HttpPetReportAnalyzer._yaniti_ayristir -- aynı
        # dar nokta deseni, standart chat/completions zarfı.
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

        caption = alanlar.get("caption")
        return caption if isinstance(caption, str) and caption.strip() else None


_generator_ornegi: InstagramCaptionGenerator | None = None


def get_instagram_caption_generator() -> InstagramCaptionGenerator:
    """`app/pet_raporu.py::get_pet_report_analyzer()` deseninin birebir
    aynısı -- bilerek AYRI env değişkenleri kullanır (`INSTAGRAM_CAPTION_AI_*`,
    `PET_REPORT_AI_*`/`TEXT_AI_*` DEĞİL): her özellik birbirinden bağımsız
    açılıp kapanabilsin, biri yapılandırılmadan diğeri sessizce onun ayarını
    miras almasın.
    """
    global _generator_ornegi
    if _generator_ornegi is not None:
        return _generator_ornegi

    saglayici = os.getenv("INSTAGRAM_CAPTION_AI_PROVIDER", "").strip()
    if not saglayici:
        _generator_ornegi = NullInstagramCaptionGenerator()
        return _generator_ornegi

    endpoint = os.getenv("INSTAGRAM_CAPTION_AI_ENDPOINT", "")
    model = os.getenv("INSTAGRAM_CAPTION_AI_MODEL", "")
    api_key = os.getenv("INSTAGRAM_CAPTION_AI_API_KEY", "")
    if not endpoint or not api_key:
        logger.warning(
            "INSTAGRAM_CAPTION_AI_PROVIDER=%r ayarlı ama INSTAGRAM_CAPTION_AI_ENDPOINT/"
            "INSTAGRAM_CAPTION_AI_API_KEY eksik — NullInstagramCaptionGenerator'a düşülüyor.",
            saglayici)
        _generator_ornegi = NullInstagramCaptionGenerator()
        return _generator_ornegi

    _generator_ornegi = HttpInstagramCaptionGenerator(endpoint=endpoint, model=model, api_key=api_key)
    return _generator_ornegi


def _sifirla_singleton_testler_icin() -> None:
    """Yalnızca testler için (bkz. pet_raporu.py'nin aynı adlı fonksiyonu)."""
    global _generator_ornegi
    _generator_ornegi = None


def instagram_caption_olustur(ilan_turu: str, tur: str, irk: str | None = None,
                              renkler: str | None = None, il_ilce: str | None = None,
                              aciklama: str | None = None,
                              ayirt_edici: list[str] | None = None) -> str | None:
    """ENTEGRASYON NOKTASI -- Java tarafının Ad entity'sinden doldurduğu
    alanları alır, sağlayıcıya iletir. `ilan_turu` "LOST"/"FOUND"/"ADOPTION"
    (Ad.AdType'ın adı) olmalı, Türkçeye burada çevrilir.
    """
    ilan_turu_tr = _ILAN_TURU_TR.get(ilan_turu, ilan_turu)
    return get_instagram_caption_generator().generate(
        ilan_turu_tr, tur, irk, renkler, il_ilce, aciklama, ayirt_edici)
