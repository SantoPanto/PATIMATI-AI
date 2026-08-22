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

import base64
import logging
import os
import time
from abc import ABC, abstractmethod

import httpx

from .models import TextAnalysisResult

# Sağlayıcının GEÇİCİ olduğu bilinen hataları -- yeniden denemek gerçekten
# kurtarabilir, aynı isteği tekrar tekrar aynı biçimde başarısız etmez.
# 429/5xx dışındaki HTTP hataları (400/401/403 gibi) VE bozuk yanıt/JSON
# hataları buna dahil DEĞİL: bunlar isteğin kendisiyle ilgili kalıcı
# sorunlardır, yeniden denemek aynı sonucu tekrar üretir, sadece gecikmeyi
# artırır.
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = int(os.getenv("TEXT_AI_MAX_ATTEMPTS", "3"))
_RETRY_BACKOFF_BASE_SECONDS = float(os.getenv("TEXT_AI_RETRY_BACKOFF_SECONDS", "1.0"))

# Gemini'ye (ve OpenAI-uyumlu zarfı destekleyen her sağlayıcıya) tek istekte
# gönderilecek azami görsel sayısı. app/indirici.py zaten bir ilanın en fazla
# kaç fotoğrafını indirdiğini AZAMI_FOTOGRAF ile sınırlıyor (varsayılan 5) --
# burada ayrıca sınırlamak yerine aynı üst sınırı miras alıyoruz; kuyruk.py
# zaten en fazla o kadar bayt geçirir.
_MAX_IMAGES_PER_REQUEST = 5

logger = logging.getLogger(__name__)

_GECERLI_KATEGORILER = {"LOST", "FOUND", "ADOPTION", "IRRELEVANT", "UNCERTAIN"}


class _RetryableProviderError(Exception):
    """İç kullanım: HttpTextAnalyzer._tek_deneme()'nin, hatanın YENİDEN
    DENENEBİLİR olduğunu analyze()'daki retry döngüsüne işaretlemesi için.
    Asla bu modülün dışına sızmaz (bkz. analyze())."""


class TextAnalyzer(ABC):
    """Sağlayıcıdan bağımsız arayüz. Tek sorumluluk: metin (+ opsiyonel görsel)
    -> yapılandırılmış çıktı.

    `photo_bytes` opsiyoneldir ve geriye dönük uyumludur: eski çağıranlar
    (yalnızca 2 pozisyonel argümanla) hâlâ çalışır. Eklenme sebebi (2026-08-19
    kullanıcı raporu): kayıp/bulundu bilgisi bazı Instagram gönderilerinde
    caption/yorumda DEĞİL, doğrudan fotoğrafın (afiş/poster) İÇİNDEKİ yazıda
    geçiyor -- salt metin analizi bunu asla göremez, kategori hep UNCERTAIN'a
    düşüyordu."""

    @abstractmethod
    def analyze(self, caption: str | None, triggering_comment: str | None,
               photo_bytes: list[bytes] | None = None) -> TextAnalysisResult:
        raise NotImplementedError


class NullTextAnalyzer(TextAnalyzer):
    """`TEXT_AI_PROVIDER` boşsa kullanılan güvenli varsayılan.

    Ağ çağrısı yapmaz, API anahtarı gerektirmez — testler ve yerel geliştirme
    hiçbir dış bağımlılık olmadan çalışabilsin diye. Her zaman UNCERTAIN
    döner: metni hiç okumadan LOST/FOUND gibi kesin bir kategori üretmek,
    "hiç analiz edilmedi" ile "analiz edildi ama emin değilim" durumlarını
    aynı sonuca düşürüp yanlış güven verirdi.
    """

    def analyze(self, caption: str | None, triggering_comment: str | None,
               photo_bytes: list[bytes] | None = None) -> TextAnalysisResult:
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

    def __init__(self, endpoint: str, model: str, api_key: str, timeout: float = 15.0,
                sleep=time.sleep):
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        # Testler gerçek zaman geçirmeden retry-backoff yolunu kanıtlayabilsin
        # diye enjekte edilebilir (bkz. app/detector.py'nin aynı deseni,
        # scan()'in kendi test edilebilirliği için page parametresi alması gibi).
        self._sleep = sleep

    def analyze(self, caption: str | None, triggering_comment: str | None,
               photo_bytes: list[bytes] | None = None) -> TextAnalysisResult:
        # DÜZELTME (2026-08-19, kullanıcı raporu): önceki sürüm TEK denemeden
        # sonra pes ediyordu -- sağlayıcının (Gemini) tek seferlik, geçici bir
        # 503'ü bile kaydı SONSUZA KADAR UNCERTAIN'a kilitliyordu: bu kayıt
        # bir daha asla yeniden denenmez (ai_status=DONE terminaldir, hiçbir
        # süpürücü onu tekrar kuyruğa almaz). Canlı bir örnekte doğrulandı:
        # afişin üzerinde açıkça "Ömürlük Yuva Arıyoruz" yazan bir görsel,
        # sırf o anki 503 yüzünden hiç okunmadan UNCERTAIN'a düştü. Şimdi
        # geçici (429/5xx, ağ/zaman aşımı) hatalar en fazla _MAX_ATTEMPTS kez
        # kısa bir backoff ile yeniden denenir; kalıcı hatalar (400/401/403,
        # bozuk yanıt) hâlâ İLK denemede UNCERTAIN'a düşer -- yeniden denemek
        # aynı isteği aynı şekilde tekrar başarısız eder, yalnızca gecikme
        # ekler.
        son_hata: Exception | None = None
        for deneme in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._tek_deneme(caption, triggering_comment, photo_bytes)
            except _RetryableProviderError as e:
                son_hata = e.__cause__ or e
                if deneme < _MAX_ATTEMPTS:
                    bekleme = _RETRY_BACKOFF_BASE_SECONDS * (2 ** (deneme - 1))
                    logger.warning(
                        "Metin analizi sağlayıcı hatası (geçici, deneme %d/%d, %.1fsn sonra tekrar): %s",
                        deneme, _MAX_ATTEMPTS, bekleme, son_hata,
                    )
                    self._sleep(bekleme)
                    continue
            except Exception as e:
                son_hata = e
                break

        # Sağlayıcı ne sebeple olursa olsun başarısız olsun (kalıcı hata, ya
        # da geçici hata _MAX_ATTEMPTS kez tekrarlandıktan sonra hâlâ devam
        # ediyorsa) — bu modül asla fırlatmaz. İstegi_isle'nin "hiçbir zaman
        # istisna fırlatmaz" sözleşmesiyle aynı ilke: bir NLP hatası tüm
        # görsel analizi de düşürmemeli.
        logger.warning("Metin analizi sağlayıcı hatası, UNCERTAIN'a düşülüyor: %s", son_hata)
        return TextAnalysisResult(category="UNCERTAIN", category_confidence=0.0,
                                  needs_review=True)

    def _tek_deneme(self, caption: str | None, triggering_comment: str | None,
                    photo_bytes: list[bytes] | None) -> TextAnalysisResult:
        """Tek bir sağlayıcı çağrısı. Geçici (yeniden denenebilir) bir hata
        _RetryableProviderError'a sarılıp fırlatılır; kalıcı bir hata olduğu
        gibi fırlatılır -- analyze()'daki retry döngüsü ikisini ayırt eder."""
        gövde = self._istek_govdesi(caption, triggering_comment, photo_bytes)
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
            # Zaman aşımı / bağlantı reddi / DNS vb. -- isteğin kendisiyle
            # değil, sağlayıcıya o an ulaşılamamasıyla ilgili, yeniden
            # denenebilir.
            raise _RetryableProviderError() from e
        return self._yaniti_ayristir(cevap.json())

    def _istek_govdesi(self, caption: str | None, triggering_comment: str | None,
                       photo_bytes: list[bytes] | None = None) -> dict:
        gorsel_notu = (
            "\n\nEkli fotoğraf(lar)a da bak: bilgi çoğu zaman caption/yorumda "
            "değil, fotoğrafın kendisinde (bir afiş/poster/el ilanı üzerindeki "
            "yazıda) yer alır. Fotoğrafın üzerinde görünen HER metni (\"KAYIP "
            "KEDİ\", telefon numarası, tarih, semt/mahalle adı vb.) oku ve "
            "ilgili alanlara aynı şekilde yansıt. Yine de yalnızca AÇIKÇA "
            "görünen/yazan bilgiyi kullan, görselin geri kalanından tahmin "
            "yürütme." if photo_bytes else ""
        )
        yonerge = (
            "Aşağıdaki Instagram gönderi açıklamasını ve/veya yorumunu "
            "PatiMati kayıp/bulunmuş/sahiplendirme platformu için sınıflandır. "
            "Yalnızca metinde AÇIKÇA belirtilen bilgiyi çıkar, tahmin/uydurma yapma. "
            "Emin değilsen category alanına UNCERTAIN yaz."
            f"{gorsel_notu}\n\n"
            f"category seçenekleri: {sorted(_GECERLI_KATEGORILER)}\n"
            f"caption: {caption or '(yok)'}\n"
            f"triggering_comment: {triggering_comment or '(yok)'}\n\n"
            "Yanıtı TextAnalysisResult şemasına uyan JSON olarak ver: "
            "category, category_confidence, species, breed, colors, gender, "
            "age_text, pet_name, location_text, location_confidence, event_date, "
            "event_date_estimated, distinguishing_features, pet_count, needs_review."
        )
        if not photo_bytes:
            content: str | list[dict] = yonerge
        else:
            content = [{"type": "text", "text": yonerge}]
            for ham in photo_bytes[:_MAX_IMAGES_PER_REQUEST]:
                b64 = base64.b64encode(ham).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        return {"model": self._model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}

    def _yaniti_ayristir(self, veri: dict) -> TextAnalysisResult:
        # Sağlayıcılar arası zarf biçimi değişir; burada tek bir dar nokta var
        # ki sağlayıcı değişince tek yer güncellensin. Standart chat/completions
        # zarfı (OpenAI, Gemini'nin OpenAI-uyumlu katmanı, çoğu yerel sunucu):
        # {"choices": [{"message": {"content": "<json string>"}}]}.
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
