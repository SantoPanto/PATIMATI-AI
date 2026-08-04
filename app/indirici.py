# app/indirici.py
"""Fotoğrafı adresinden indirir — kuyruk mesajı dosya değil URL taşıyor.

Rastgele adres indiren bir servis, korunmazsa saldırganın eline geçen bir
araca dönüşür (SSRF): iç ağdaki adreslere, bulut sağlayıcının kimlik bilgisi
döndüren uçlarına (169.254.169.254 gibi) ya da localhost'taki başka servislere
istek attırılabilir. Adres kendi backend'imizden gelse bile korumak gerekir;
tek bir hatalı kayıt yeter.

Korumalar:
  - şema kısıtı (varsayılan yalnızca https)
  - alan adı beyaz listesi (PHOTO_ALLOWED_HOSTS) — **boşsa hiçbir şey inmez**
  - özel/yerel IP engeli (10.x, 192.168.x, 127.x, 169.254.x, ::1 ...)
  - yönlendirme takip edilmez (her sıçrama yeniden doğrulanamayacağı için)
  - indirme sırasında boyut sınırı (Content-Length'e güvenilmez)
  - zaman aşımı
"""
import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse

import httpx

from .hatalar import FotografIndirilemedi, FotografYok

logger = logging.getLogger(__name__)

AZAMI_BAYT = int(os.getenv("PHOTO_MAX_BYTES", str(10 * 1024 * 1024)))   # 10 MB
ZAMAN_ASIMI = float(os.getenv("PHOTO_TIMEOUT", "10"))                   # saniye
AZAMI_FOTOGRAF = int(os.getenv("PHOTO_MAX_COUNT", "5"))

# TEK GÜVEN MEKANİZMASI: bu listede AÇIKÇA yazan adres güvenilir sayılır.
#   üretim : PHOTO_ALLOWED_HOSTS=patimati-media.s3.eu-central-1.amazonaws.com
#   lokal  : PHOTO_ALLOWED_HOSTS=127.0.0.1,localhost,minio
#
# Neden tek mekanizma: önce "özel ağa izin ver" diye ayrı bir bayrak vardı ve
# açıldığında SSRF korumasının TAMAMINI kapatıyordu — bulut kimlik ucu dâhil.
# Lokal geliştirme için açıp üretimde kapatmayı unutmak, servisi tamamen
# savunmasız bırakırdı. Artık lokal adres de listeye YAZILARAK izinli oluyor,
# yani izin her zaman açık ve dar kapsamlı.
IZINLI_HOSTLAR = [h.strip().lower() for h in
                  os.getenv("PHOTO_ALLOWED_HOSTS", "").split(",") if h.strip()]


def _acikca_izinli(host: str) -> bool:
    """Host beyaz listede açıkça yer alıyor mu? (Liste boşsa hiçbir şey açık değil.)

    Nokta ile başlayan girdi alt alan adlarını kapsar: ".amazonaws.com"
    """
    host = host.lower()
    for izinli in IZINLI_HOSTLAR:
        if host == izinli or (izinli.startswith(".") and host.endswith(izinli)):
            return True
    return False


def _ip_denetle(host: str, acik_izinli: bool) -> None:
    """Adresin çözüldüğü IP'leri denetler."""
    try:
        kayitlar = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise FotografIndirilemedi(f"Adres çözülemedi: {host}") from e

    for kayit in kayitlar:
        ip = ipaddress.ip_address(kayit[4][0])

        # Bağlantı-yerel aralık (169.254.x, fe80::) bulut sağlayıcıların kimlik
        # bilgisi döndüren uçlarını barındırır. Bizim için hiçbir meşru kullanımı
        # yok — beyaz listeye yazılsa bile İZİN VERİLMEZ.
        if ip.is_link_local:
            raise FotografIndirilemedi(
                f"Bağlantı-yerel adres her koşulda reddedilir: {host} -> {ip}")

        if (ip.is_private or ip.is_loopback or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            if acik_izinli:
                logger.info("Yerel adrese izin verildi (beyaz listede): %s -> %s",
                            host, ip)
                continue
            raise FotografIndirilemedi(
                f"Yerel/özel ağ adresi reddedildi: {host} -> {ip}")


def dogrula(url: str) -> None:
    """İndirmeden ÖNCE adresi denetler; uygun değilse hata fırlatır."""
    p = urlparse(url)
    if not p.hostname:
        raise FotografIndirilemedi("Adreste alan adı yok")

    acik_izinli = _acikca_izinli(p.hostname)

    # https her zaman; http yalnızca açıkça izin verilmiş adresler için
    # (lokal MinIO gibi). Böylece şifresiz trafik de dar kapsamda kalıyor.
    if p.scheme == "http":
        if not acik_izinli:
            raise FotografIndirilemedi(
                f"http yalnızca beyaz listedeki adresler için: {p.hostname}")
    elif p.scheme != "https":
        raise FotografIndirilemedi(
            f"Desteklenmeyen şema: {p.scheme or '(yok)'} (izinli: https)")

    # Liste boşsa HİÇBİR ŞEY indirilmez (kapalıya düşer).
    #
    # Eskiden boş liste "dış adresleri serbest bırak" anlamına geliyordu ve bu
    # bilinçli bir tercihti (lokalde rastgele bir fotoğraf adresiyle deneme
    # yapmak kolay olsun diye). Değiştirildi çünkü sessizce yanlış tarafa
    # düşüyordu: değişkeni yazmayı unutan bir dağıtım, korumanın AÇIK olduğunu
    # sanarak kapalı çalışıyordu. Bir güvenlik denetiminin varsayılanı,
    # unutulduğunda güvenli olan taraf olmalı.
    #
    # Geliştirmeye maliyeti düşük: yerel deneme için zaten `127.0.0.1`
    # yazılması gerekiyordu (bkz. scripts/sahte_java.py).
    if not IZINLI_HOSTLAR:
        raise FotografIndirilemedi(
            "PHOTO_ALLOWED_HOSTS boş — hiçbir adres indirilemez. "
            ".env dosyasına indirilmesine izin verilen alan adlarını yazın "
            "(yerel deneme için 127.0.0.1, üretimde S3 alan adınız).")

    if not acik_izinli:
        raise FotografIndirilemedi(f"Alan adı beyaz listede değil: {p.hostname}")

    _ip_denetle(p.hostname, acik_izinli)


def indir(url: str) -> bytes:
    """Fotoğrafı indirir. Sorun olursa FotografIndirilemedi fırlatır."""
    dogrula(url)
    try:
        # follow_redirects=False: yönlendirilen adresi yeniden doğrulayamayız,
        # bu yüzden hiç takip etmiyoruz. Presigned S3 adresleri yönlendirmez.
        with httpx.stream("GET", url, timeout=ZAMAN_ASIMI,
                          follow_redirects=False) as cevap:
            if cevap.is_redirect:
                raise FotografIndirilemedi(
                    f"Yönlendirme takip edilmiyor (HTTP {cevap.status_code})")
            cevap.raise_for_status()

            tur = cevap.headers.get("content-type", "")
            # Sunucu tür bildirmezse engellemiyoruz; asıl doğrulamayı görüntüyü
            # açarken PIL yapıyor. Ama yanlış tür bildirilmişse erken duruyoruz.
            if tur and not tur.split(";")[0].strip().startswith("image/"):
                raise FotografIndirilemedi(f"Görüntü değil: content-type={tur}")

            parcalar, toplam = [], 0
            for parca in cevap.iter_bytes():
                toplam += len(parca)
                # Content-Length'e güvenmiyoruz; yalan olabilir ya da hiç gelmez
                if toplam > AZAMI_BAYT:
                    raise FotografIndirilemedi(
                        f"Dosya çok büyük ({AZAMI_BAYT} bayt sınırı aşıldı)")
                parcalar.append(parca)
    except FotografIndirilemedi:
        raise
    except httpx.TimeoutException as e:
        raise FotografIndirilemedi(f"Zaman aşımı ({ZAMAN_ASIMI} sn): {url}") from e
    except httpx.HTTPStatusError as e:
        raise FotografIndirilemedi(
            f"HTTP {e.response.status_code}: {url}") from e
    except httpx.HTTPError as e:
        raise FotografIndirilemedi(f"İndirilemedi: {e}") from e

    if not parcalar:
        raise FotografIndirilemedi(f"Boş cevap: {url}")
    return b"".join(parcalar)


def hepsini_indir(urls: list[str]) -> tuple[list[bytes], list[dict]]:
    """Adreslerin hepsini indirir.

    Tek bir adresin bozuk olması diğerlerini düşürmez — ilanda 3 fotoğraf varsa
    biri silinmiş olsa bile kalan ikisiyle eşleştirme yapılabilir. Başarısızlar
    ayrıca döndürülür ki sonuç mesajında raporlanabilsin.

    Dönüş: (indirilen_baytlar, basarisizlar)
    """
    if not urls:
        raise FotografYok("photo_urls boş")
    if len(urls) > AZAMI_FOTOGRAF:
        logger.warning("%d fotoğraf geldi, ilk %d tanesi işlenecek",
                       len(urls), AZAMI_FOTOGRAF)
        urls = urls[:AZAMI_FOTOGRAF]

    baytlar, basarisizlar = [], []
    for url in urls:
        try:
            baytlar.append(indir(url))
        except FotografIndirilemedi as e:
            logger.warning("Fotoğraf indirilemedi (%s): %s", url, e)
            basarisizlar.append({"url": url, "error": str(e)})

    if not baytlar:
        raise FotografIndirilemedi(
            "Hiçbir fotoğraf indirilemedi: "
            + "; ".join(b["error"] for b in basarisizlar))
    return baytlar, basarisizlar
