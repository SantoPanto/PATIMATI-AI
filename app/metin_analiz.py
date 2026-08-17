# app/metin_analiz.py
"""Caption/yorum metninden LOST/FOUND/ADOPTION/IRRELEVANT/UNCERTAIN çıkarımı.

Faz 2 (Instagram entegrasyonu) revised blueprint'inin "TextAnalyzer" soyutlaması.
`app/surum.py`'nin `KIMLIK_MODEL` ortam değişkeniyle kimlik modelini seçtiği
yöntemin BİREBİR aynısı burada da kullanılıyor: gerçek sağlayıcı `TEXT_AI_PROVIDER`
ortam değişkeniyle seçilir, geri kalan kod hangi sağlayıcının çalıştığını bilmez.
Amaç: yarın sağlayıcı değişirse (yerel model, başka bir API...) yalnızca burada
yeni bir `TextAnalyzer` alt sınıfı eklenir, `kuyruk.py` tek satır bile değişmez.

Türkçe caption/yorum örnekleri için LOST/FOUND/ADOPTION/IRRELEVANT/UNCERTAIN
sınıflandırması ve yapılandırılmış alan çıkarımı (tür, cins, renk, konum...)
temel iştir. Kaynak metinde olmayan bilgi UYDURULMAZ — çıkarılamayan alanlar
None kalır; emin olunamayan durumda kategori zorla LOST/FOUND'a itilmez,
UNCERTAIN döner (bkz. `KuyrukIstegi`/`kuyruk.py` çağrı noktası).

Sağlayıcı çağrısı BAŞARISIZ olursa (zaman aşımı, ağ hatası, bozuk yanıt) bu
modül hiçbir zaman dışarı istisna fırlatmaz — `istegi_isle` saf bir fonksiyon
ve tek bir NLP hatası yüzünden tüm analiz sonucunu kaybetmemeli. Başarısızlık
UNCERTAIN'a düşer, tıpkı "emin değilim" gibi ele alınır.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import httpx

from .models import TextAnalysisResult

logger = logging.getLogger(__name__)

_GECERLI_KATEGORILER = {"LOST", "FOUND", "ADOPTION", "IRRELEVANT", "UNCERTAIN"}


class TextAnalyzer(ABC):
    """Sağlayıcıdan bağımsız arayüz. Tek sorumluluk: metin -> yapılandırılmış çıktı."""

    @abstractmethod
    def analyze(self, caption: str | None, triggering_comment: str | None) -> TextAnalysisResult:
        raise NotImplementedError


class NullTextAnalyzer(TextAnalyzer):
    """`TEXT_AI_PROVIDER` boşsa kullanılan güvenli varsayılan.

    Ağ çağrısı yapmaz, API anahtarı gerektirmez — testler ve yerel geliştirme
    hiçbir dış bağımlılık olmadan çalışabilsin diye. Her zaman UNCERTAIN
    döner: metni hiç okumadan LOST/FOUND gibi kesin bir kategori üretmek,
    "hiç analiz edilmedi" ile "analiz edildi ama emin değilim" durumlarını
    aynı sonuca düşürüp yanlış güven verirdi.
    """

    def analyze(self, caption: str | None, triggering_comment: str | None) -> TextAnalysisResult:
        return TextAnalysisResult(category="UNCERTAIN", category_confidence=0.0,
                                  needs_review=True)


class HttpTextAnalyzer(TextAnalyzer):
    """Düz HTTP üzerinden yapılandırılmış-çıktı isteyen bir LLM sağlayıcısını çağırır.

    Yeni bir SDK bağımlılığı EKLENMEDİ: `httpx` zaten `app/indirici.py`
    tarafından kullanılan bir üretim bağımlılığı. Sağlayıcı REST API'si
    olduğu sürece bu sınıf herhangi bir yapılandırılmış-çıktı destekleyen
    LLM servisine (yerel ya da bulut) konfigürasyon değiştirilerek işaret
    edilebilir; kod değişmez.
    """

    def __init__(self, endpoint: str, model: str, api_key: str, timeout: float = 15.0):
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def analyze(self, caption: str | None, triggering_comment: str | None) -> TextAnalysisResult:
        try:
            gövde = self._istek_govdesi(caption, triggering_comment)
            with httpx.Client(timeout=self._timeout) as istemci:
                cevap = istemci.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json"},
                    json=gövde,
                )
                cevap.raise_for_status()
                return self._yaniti_ayristir(cevap.json())
        except Exception as e:
            # Sağlayıcı ne sebeple olursa olsun başarısız olsun (zaman aşımı,
            # 5xx, bozuk JSON, beklenmeyen şema) — bu modül asla fırlatmaz.
            # İstegi_isle'nin "hiçbir zaman istisna fırlatmaz" sözleşmesiyle
            # aynı ilke: bir NLP hatası tüm görsel analizi de düşürmemeli.
            logger.warning("Metin analizi sağlayıcı hatası, UNCERTAIN'a düşülüyor: %s", e)
            return TextAnalysisResult(category="UNCERTAIN", category_confidence=0.0,
                                      needs_review=True)

    def _istek_govdesi(self, caption: str | None, triggering_comment: str | None) -> dict:
        yonerge = (
            "Aşağıdaki Instagram gönderi açıklamasını ve/veya yorumunu "
            "PatiMati kayıp/bulunmuş/sahiplendirme platformu için sınıflandır. "
            "Yalnızca metinde AÇIKÇA belirtilen bilgiyi çıkar, tahmin/uydurma yapma. "
            "Emin değilsen category alanına UNCERTAIN yaz.\n\n"
            f"category seçenekleri: {sorted(_GECERLI_KATEGORILER)}\n"
            f"caption: {caption or '(yok)'}\n"
            f"triggering_comment: {triggering_comment or '(yok)'}\n\n"
            "Yanıtı TextAnalysisResult şemasına uyan JSON olarak ver: "
            "category, category_confidence, species, breed, colors, gender, "
            "age_text, pet_name, location_text, location_confidence, event_date, "
            "event_date_estimated, distinguishing_features, pet_count, needs_review."
        )
        return {"model": self._model, "input": yonerge,
                "response_format": {"type": "json_object"}}

    def _yaniti_ayristir(self, veri: dict) -> TextAnalysisResult:
        # Sağlayıcılar arası zarf biçimi değişir; burada tek bir dar nokta var
        # ki sağlayıcı değişince tek yer güncellensin.
        icerik = veri.get("output") if isinstance(veri, dict) else None
        if isinstance(icerik, str):
            import json
            icerik = json.loads(icerik)
        alanlar = icerik if isinstance(icerik, dict) else veri

        kategori = str(alanlar.get("category") or "UNCERTAIN").upper()
        if kategori not in _GECERLI_KATEGORILER:
            kategori = "UNCERTAIN"

        return TextAnalysisResult(
            category=kategori,
            category_confidence=float(alanlar.get("category_confidence") or 0.0),
            species=alanlar.get("species"),
            breed=alanlar.get("breed"),
            colors=list(alanlar.get("colors") or []),
            gender=alanlar.get("gender"),
            age_text=alanlar.get("age_text"),
            pet_name=alanlar.get("pet_name"),
            location_text=alanlar.get("location_text"),
            location_confidence=alanlar.get("location_confidence"),
            event_date=alanlar.get("event_date"),
            event_date_estimated=bool(alanlar.get("event_date_estimated") or False),
            distinguishing_features=alanlar.get("distinguishing_features"),
            pet_count=alanlar.get("pet_count"),
            needs_review=bool(alanlar.get("needs_review") or False),
        )


_analyzer_ornegi: TextAnalyzer | None = None


def get_text_analyzer() -> TextAnalyzer:
    """`app/surum.py::KIMLIK_MODEL` seçim desenini birebir izler.

    Modül seviyesinde tek seferlik başlatma (singleton), tekrar tekrar env
    okumasın ve HttpTextAnalyzer her çağrıda yeniden kurulmasın diye.
    """
    global _analyzer_ornegi
    if _analyzer_ornegi is not None:
        return _analyzer_ornegi

    saglayici = os.getenv("TEXT_AI_PROVIDER", "").strip()
    if not saglayici:
        _analyzer_ornegi = NullTextAnalyzer()
        return _analyzer_ornegi

    endpoint = os.getenv("TEXT_AI_ENDPOINT", "")
    model = os.getenv("TEXT_AI_MODEL", "")
    api_key = os.getenv("TEXT_AI_API_KEY", "")
    if not endpoint or not api_key:
        logger.warning(
            "TEXT_AI_PROVIDER=%r ayarlı ama TEXT_AI_ENDPOINT/TEXT_AI_API_KEY eksik "
            "— NullTextAnalyzer'a düşülüyor.", saglayici)
        _analyzer_ornegi = NullTextAnalyzer()
        return _analyzer_ornegi

    _analyzer_ornegi = HttpTextAnalyzer(endpoint=endpoint, model=model, api_key=api_key)
    return _analyzer_ornegi


def _sifirla_singleton_testler_icin() -> None:
    """Yalnızca testler için: env değişkenleri monkeypatch edildiğinde
    `get_text_analyzer()`'ın önbelleklenmiş sonucu yeniden hesaplaması gerekir."""
    global _analyzer_ornegi
    _analyzer_ornegi = None
