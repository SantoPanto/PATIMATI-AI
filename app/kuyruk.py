# app/kuyruk.py
"""RabbitMQ köprüsü — Java backend ile asenkron bağlantı.

Akış (sözleşme §1): Java ilanı kaydeder, adayları PostGIS ile süzer ve
`ai.analysis.request` kuyruğuna bir mesaj bırakır. Biz o mesajı alır,
fotoğrafları indirir, analiz eder, adaylarla skorlar ve sonucu
`ai.analysis.result` kuyruğuna yazarız. Kullanıcı bu sırada beklemez.

Dosyanın iki katmanı var ve ayrım kasıtlı:

  `istegi_isle()`   SAF fonksiyon — dict alır, dict döner. AMQP bilmez,
                    bağlantı açmaz. Bu yüzden broker kurmadan test edilebilir
                    ve testlerin çoğu buraya bakar.
  `dinle()`         Kabuk — bağlanır, topolojiyi ilan eder, mesaj başına
                    `istegi_isle` çağırır, sonucu yayınlar, onaylar.

Neden `pika` (senkron istemci) ve neden ayrı süreç: model çıkarımı CPU'yu
saniyelerce meşgul eden SENKRON bir iştir. Asenkron bir istemci burada hiçbir
şey kazandırmaz, olay döngüsünü bloklar. Kuyruk tüketicisini FastAPI'nin
içine gömmek de HTTP isteklerini aç bırakırdı — bu yüzden ayrı süreç:

    venv\\Scripts\\python.exe -m app.kuyruk
"""
import json
import logging
import signal
import time
from datetime import datetime, timezone

import pika
import pika.exceptions
from pydantic import ValidationError

# Topoloji ve ayarlar ayrı bir modülde: kuyruğa mesaj yazan araçlar modeli
# yüklemek zorunda kalmasın (bkz. app/topoloji.py). Buradan yeniden dışa
# aktarılıyorlar ki `from app.kuyruk import EXCHANGE` gibi kullanımlar
# çalışmaya devam etsin — çağıranın hangi dosyada durduğunu bilmesi gerekmez.
from .topoloji import (AMQP_URL, DLQ, DLX, EXCHANGE, ISTEK_ANAHTARI,  # noqa: F401
                       ISTEK_KUYRUGU, KALP_ATISI, PREFETCH, SEMA_SURUMU,
                       SONUC_ANAHTARI, SONUC_KUYRUGU, topolojiyi_kur)

# DİKKAT: `.topoloji` yukarıda load_dotenv() çağırıyor ve bu satırın ÜSTÜNDE
# durması şart — `.surum` modülü ortam değişkenlerini içe aktarma anında
# okuyor. Sıra bozulursa .env hiç okunmamış gibi davranır ve servis, yazdığın
# ayarları sessizce yok sayıp varsayılanlarla çalışır.
from .analiz import urlleri_analiz_et
from .hatalar import (AIHatasi, DesteklenmeyenSema, GecersizEmbedding,
                      GecersizIstek, ModelHatasi)
from .matcher import adaylari_eslestir
from .models import KuyrukIstegi
from .surum import MODEL_SURUMU

logger = logging.getLogger(__name__)


def _simdi() -> str:
    """Sözleşmedeki `processed_at` biçimi: UTC, saniye çözünürlüğü, sonda Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def _hata_sonucu(request_id, ad_id, kod: str, mesaj: str) -> dict:
    """Sözleşme §4 'Hatalı' biçimi. Java bunu görünce ai_status = FAILED yazar."""
    return {
        "schema_version": SEMA_SURUMU,
        "request_id": request_id,
        "ad_id": ad_id,
        "status": "error",
        "error": {"code": kod, "message": mesaj},
        "processed_at": _simdi(),
    }


def istegi_isle(mesaj: dict) -> dict:
    """Bir istek mesajını sonuç mesajına çevirir. SAF: ağ/AMQP yok.

    Hiçbir zaman istisna fırlatmaz — her hata sözleşmedeki `status: "error"`
    biçimine dönüşür. Sebebi §8'de: işleyemediğimiz mesajı kuyruğa geri
    koymuyoruz, cevaplıyoruz. Aksi hâlde bozuk bir mesaj sonsuza kadar
    yeniden teslim edilip servisi meşgul ederdi.
    """
    # request_id ve ad_id'yi doğrulamadan ÖNCE okuyoruz: mesaj şemaya uymasa
    # bile Java'nın hangi ilana ait olduğunu bilmesi gerekiyor, yoksa o ilan
    # sonsuza kadar PENDING'de kalır.
    request_id = mesaj.get("request_id")
    ad_id = mesaj.get("ad_id")

    try:
        # Şema sürümü önce: tanımadığımız bir sürümü alan alan ayrıştırmaya
        # çalışmak, yanlış anlamda okunan bir alanla sessizce yanlış sonuç
        # üretme riski taşır.
        gelen_surum = mesaj.get("schema_version")
        if gelen_surum != SEMA_SURUMU:
            raise DesteklenmeyenSema(
                f"schema_version={gelen_surum!r} tanınmıyor "
                f"(bu servis: {SEMA_SURUMU})")

        try:
            istek = KuyrukIstegi.model_validate(mesaj)
        except ValidationError as e:
            # Pydantic'in tam metni çok uzun; ilk üç sorunu özetliyoruz ki
            # günlük okunabilir kalsın ama sorun da anlaşılsın.
            ozet = "; ".join(
                f"{'.'.join(str(p) for p in h['loc'])}: {h['msg']}"
                for h in e.errors()[:3])
            raise GecersizIstek(f"mesaj sözleşmeye uymuyor -> {ozet}") from e

        request_id, ad_id = istek.request_id, istek.ad_id

        try:
            analiz = urlleri_analiz_et(istek.photo_urls)
        except AIHatasi:
            # NO_PHOTOS / PHOTO_DOWNLOAD_FAILED / INVALID_IMAGE — kendi
            # kodlarıyla yukarı çıksınlar.
            raise
        except Exception as e:
            # İndirme başarılıysa ve buraya kadar geldiyse sorun modeldedir.
            # Bunu INTERNAL'e katmak, model hatalarını görünmez yapardı.
            raise ModelHatasi(f"analiz sırasında hata: {e}") from e

        # Sözleşme §7 kural 2: kullanıcı beyanı AI tahmininden ÖNCELİKLİ.
        # Beyan yoksa AI'ın tahmini kullanılır; AI "unknown" derse tür filtresi
        # kendiliğinden devre dışı kalır (compute_final_score "unknown"u
        # eşleşmeyi engellemeyen değer sayıyor) — yanlış eleme yapmamak için.
        tur = istek.declared_species or analiz["species"]

        try:
            matches, atlanan = adaylari_eslestir(
                embeddings=analiz["embeddings"],
                labels=analiz["labels"],
                species=tur,
                candidates=istek.candidates,
                ad_id=istek.ad_id,
            )
        except GecersizEmbedding as e:
            # Burada patlayan embedding Java'dan gelmiyor, bir satır yukarıdaki
            # urlleri_analiz_et'in ürettiği vektör — yani sorun çağıranda değil
            # bizim model çıktımızda. INTERNAL'e düşseydi Java "bizde bir şey
            # patladı" diye AI servisinin içinde arardı; MODEL_ERROR doğru yere
            # işaret ediyor (bkz. app/matcher.py'deki adaylari_eslestir).
            raise ModelHatasi(f"sorgu embedding'i geçersiz: {e}") from e

        # `analysis` bloğu sözleşmede sabit bir alan kümesi. urlleri_analiz_et
        # bunlara ek olarak photo_count/failed_photos/model_version döndürüyor;
        # ilk ikisi sözleşmede yok, model_version ise üst seviyede duruyor.
        # Fazlasını göndermek kırıcı değil (§9) ama sözleşmeyi belge olarak
        # doğru tutmak için burada açıkça seçiyoruz.
        return {
            "schema_version": SEMA_SURUMU,
            "request_id": istek.request_id,
            "ad_id": istek.ad_id,
            "status": "ok",
            "model_version": MODEL_SURUMU,
            "analysis": {
                "embeddings": analiz["embeddings"],
                "species": analiz["species"],
                "species_confidence": analiz["species_confidence"],
                "is_pet": analiz["is_pet"],
                "breed": analiz["breed"],
                "breed_confidence": analiz["breed_confidence"],
                "pattern": analiz["pattern"],
                "colors": analiz["colors"],
                "labels": analiz["labels"],
            },
            # TODO (NLP Entegrasyonu): PatiMatiTextExtractor modülü ana projeye dâhil
            # edildiğinde, kullanıcının ilan açıklaması analiz edilecek
            # ve dönen nitelikler (niyet, tasma durumu vb.) aşağıdaki alanlara bağlanacaktır.
            "nlp_attributes": {},
            "extracted_features": [],
            "matches": matches,
            "skipped_candidates": atlanan,
            # Sözleşmede yok ama eklemek kırıcı değil (§9): indirilemeyen
            # fotoğraflar sessizce kaybolmasın. Kullanıcı 3 fotoğraf yükleyip
            # 1'iyle analiz edildiyse bunun bir izi olmalı.
            "failed_photos": analiz["failed_photos"],
            "processed_at": _simdi(),
        }

    except AIHatasi as e:
        logger.warning("İstek hatayla sonuçlandı (ad_id=%s, kod=%s): %s",
                       ad_id, e.KOD, e)
        return _hata_sonucu(request_id, ad_id, e.KOD, str(e))
    except Exception as e:
        logger.exception("Beklenmeyen hata (ad_id=%s)", ad_id)
        return _hata_sonucu(request_id, ad_id, "INTERNAL", str(e))


def sonucu_yayinla(kanal, sonuc: dict) -> None:
    """Sonucu `patimati.ai` exchange'ine `analysis.result` anahtarıyla yazar."""
    kanal.basic_publish(
        exchange=EXCHANGE,
        routing_key=SONUC_ANAHTARI,
        body=json.dumps(sonuc, ensure_ascii=False).encode("utf-8"),
        properties=pika.BasicProperties(
            content_type="application/json",
            # delivery_mode=2 → kalıcı. Kuyruk durable olsa bile mesaj kalıcı
            # değilse broker yeniden başladığında SONUÇ KAYBOLUR: ilan sonsuza
            # kadar PENDING'de kalır. İkisi birlikte gerekir.
            delivery_mode=2,
            correlation_id=str(sonuc.get("request_id") or ""),
        ),
    )


def _mesaj_geldi(kanal, method, ozellikler, govde: bytes) -> None:
    """Tek bir mesajın tüm yaşam döngüsü."""
    try:
        mesaj = json.loads(govde.decode("utf-8"))
        if not isinstance(mesaj, dict):
            raise ValueError(f"JSON nesnesi bekleniyordu, {type(mesaj).__name__} geldi")
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        # Ayrıştırılamayan mesaja CEVAP VEREMEYİZ: request_id'sini bile
        # okuyamıyoruz. Sözleşme §8 bu durumu DLQ'ya yönlendiriyor.
        # requeue=False olmazsa mesaj sonsuza kadar geri gelir.
        logger.error("Ayrıştırılamayan mesaj DLQ'ya gönderildi: %s", e)
        kanal.basic_nack(method.delivery_tag, requeue=False)
        return

    baslangic = time.monotonic()
    sonuc = istegi_isle(mesaj)

    # SIRA ÖNEMLİ: önce yayınla, sonra onayla. Tersi yapılırsa, onayla yayın
    # arasında süreç çökerse mesaj kuyruktan silinmiş ama sonuç hiç
    # yazılmamış olur — ilan sessizce PENDING'de kalır ve kimse fark etmez.
    # Bu sırayla, çökme hâlinde mesaj yeniden teslim edilir; yeniden işlemek
    # zararsızdır çünkü AI tarafı saftır (§8).
    sonucu_yayinla(kanal, sonuc)
    kanal.basic_ack(method.delivery_tag)

    logger.info("ad_id=%s durum=%s eşleşme=%d süre=%.1fsn",
                sonuc.get("ad_id"), sonuc.get("status"),
                len(sonuc.get("matches") or []), time.monotonic() - baslangic)


def dinle(amqp_url: str | None = None) -> None:
    """Kuyruğu dinler. Bağlantı koparsa artan bekleme ile yeniden dener."""
    url = amqp_url or AMQP_URL
    bekleme = 1.0

    while True:
        baglanti = kanal = None
        try:
            parametreler = pika.URLParameters(url)
            parametreler.heartbeat = KALP_ATISI
            # Broker bellek/disk sınırına dayanınca yayıncıyı bloklar. Sınırsız
            # beklemek yerine hata verip yeniden bağlanmayı tercih ediyoruz;
            # sessizce asılı kalan bir tüketici, çökmüş olandan daha kötüdür.
            parametreler.blocked_connection_timeout = 300

            baglanti = pika.BlockingConnection(parametreler)
            kanal = baglanti.channel()
            topolojiyi_kur(kanal)
            kanal.basic_qos(prefetch_count=PREFETCH)
            kanal.basic_consume(queue=ISTEK_KUYRUGU, on_message_callback=_mesaj_geldi)

            bekleme = 1.0  # başarılı bağlantı: bekleme sıfırlanır
            logger.info("Kuyruk dinleniyor: %s (model=%s, prefetch=%d)",
                        ISTEK_KUYRUGU, MODEL_SURUMU, PREFETCH)
            kanal.start_consuming()

        except KeyboardInterrupt:
            logger.info("Kapatılıyor...")
            try:
                if kanal is not None:
                    kanal.stop_consuming()
                if baglanti is not None and baglanti.is_open:
                    baglanti.close()
            except Exception:
                pass
            return

        except pika.exceptions.AMQPError as e:
            # Broker kapalı ya da bağlantı düştü. Çıkmak yerine bekleyip
            # yeniden deniyoruz: RabbitMQ'nun yeniden başlaması AI servisinin
            # elle başlatılmasını gerektirmemeli.
            logger.warning("Bağlantı sorunu (%s). %.0f sn sonra yeniden denenecek.",
                           e, bekleme)
            time.sleep(bekleme)
            bekleme = min(bekleme * 2, 30.0)


def _kapanis_sinyallerini_yakala() -> None:
    """SIGTERM/SIGINT'i KeyboardInterrupt'a çevirir — MEVCUT kapanış yolunu kullanır.

    NEDEN: `docker stop` ve Railway SIGTERM gönderiyor. Python'un SIGTERM için
    varsayılan düzeni SIG_DFL ve bu, sürecin nerede durduğuna göre İKİ FARKLI
    biçimde yanlış:

      - Konteynerde SERVIS_ROLU=kuyruk ile: entrypoint `exec` ettiği için bu
        süreç PID 1'dir. Linux, PID 1'e gelen ve İŞLEYİCİSİ OLMAYAN sinyalin
        varsayılan eylemini UYGULAMAZ — sinyali sessizce yutar. Yani süreç
        `docker stop` ile HİÇ durmaz; 10 sn beklenir ve SIGKILL gelir,
        aşağıdaki düzgün kapanış hiç çalışmaz.
      - SERVIS_ROLU=hepsi ile (PID 1 değil): varsayılan eylem uygulanır ve
        süreç anında ölür; dinle() içindeki stop_consuming + connection.close
        yine çalışmaz.

    NEDEN YENİ KAPANIŞ MANTIĞI YAZMIYORUZ: dinle() zaten KeyboardInterrupt'ı
    yakalayıp düzgün kapanıyor (Ctrl+C yolu). Sinyali oraya BAĞLAMAK, ikinci bir
    kapanış yolu yazmaktan iyidir — iki yol zamanla birbirinden kayar.

    NEDEN __main__ İÇİNDE, dinle() İÇİNDE DEĞİL:
      - signal.signal() yalnızca ANA yorumlayıcının ANA iş parçacığında çalışır,
        başka yerde ValueError fırlatır. dinle()'nin içine koymak onu ileride
        bir iş parçacığında çalıştırmayı imkânsız kılardı.
      - Sinyal düzeni SÜRECE aittir, bir kuyruk bağlantısına değil.
        `from app import kuyruk` yapan testler süreç genelinde sinyal düzenini
        değiştirmemeli. logging.basicConfig() de aynı sebeple burada.

    Analiz sırasında sinyal gelirse: mesaj ONAYLANMAZ, RabbitMQ yeniden teslim
    eder. Sözleşme §8 bunu zaten kapsıyor ("en az bir kez"; Java tarafı ad_id
    üzerinden idempotent). Doğru takas: `docker stop` mühleti 10 sn, en kötü
    analiz ~20 sn — beklemek SIGKILL'i garanti ederdi.
    """
    def _isle(numara, _cerceve):
        # BU SATIR ÖNEMLİ: aynı sinyal ikinci kez gelirse varsayılan davranış
        # (anında ölüm) devreye girsin. Yoksa ikinci KeyboardInterrupt,
        # dinle()'deki `try: ... except Exception: pass` bloğunun İÇİNDE
        # patlar — KeyboardInterrupt bir Exception DEĞİLDİR, oradan kaçar ve
        # süreç düzgün kapanmanın tam ortasında iz dökerek ölür.
        signal.signal(numara, signal.SIG_DFL)
        logger.info("Sinyal %s alındı, kapatılıyor...", numara)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _isle)
    signal.signal(signal.SIGINT, _isle)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _kapanis_sinyallerini_yakala()
    dinle()
