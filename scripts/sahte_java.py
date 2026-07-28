# scripts/sahte_java.py
"""Java backend'in yerine geçen sahte üretici — uçtan uca kanıt.

Java tarafı henüz yazılmadı. Ama entegrasyonun çalıştığını GÖRMEDEN "bitti"
demek, bu projede iki kez yanlış çıkmış bir alışkanlık (bkz. docs/olcum-raporu.md:
sentetik testler gerçek veride tutmadı). Bu betik Java'nın yapacağı işi birebir
yapar:

  1. `patimati.ai` exchange'ine `analysis.request` anahtarıyla bir istek yayınlar
  2. `ai.analysis.result` kuyruğunu dinler
  3. Gelen sonucu sözleşmeye göre denetler ve ekrana basar

Yani gerçekten kanıtladığı şey: mesaj biçimi, topoloji ve akış tutuyor mu.

Kullanım (RabbitMQ ve `python -m app.kuyruk` çalışırken):

    venv\\Scripts\\python.exe scripts/sahte_java.py --url https://.../kedi.jpg

Fotoğraf adresi vermezsen yerel bir dosyayı geçici bir HTTP sunucusundan
servis eder — böylece internet olmadan da denenebilir. O durumda
PHOTO_ALLOWED_HOSTS=127.0.0.1 gerekir, betik bunu hatırlatır.
"""
import argparse
import json
import sys
import threading
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pika

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# `app.kuyruk` DEĞİL `app.topoloji`: kuyruk modülünü içe aktarmak modeli de
# yükler (~25 sn). Bu betiğin modele ihtiyacı yok, sadece JSON yayınlıyor.
from app.topoloji import (AMQP_URL, EXCHANGE, ISTEK_ANAHTARI, SEMA_SURUMU,
                          SONUC_KUYRUGU, topolojiyi_kur)


def yerel_sunucu_baslat(dosya: Path) -> tuple[str, ThreadingHTTPServer]:
    """Verilen dosyayı 127.0.0.1 üzerinden servis eder, adresini döner.

    Neden: kuyruk akışı dosya değil ADRES taşıyor. İnternete çıkmadan gerçek
    akışı denemek için fotoğrafı kendi makinemizde yayınlıyoruz.
    """
    islemci = partial(SimpleHTTPRequestHandler, directory=str(dosya.parent))
    sunucu = ThreadingHTTPServer(("127.0.0.1", 0), islemci)
    threading.Thread(target=sunucu.serve_forever, daemon=True).start()
    port = sunucu.server_address[1]
    return f"http://127.0.0.1:{port}/{dosya.name}", sunucu


def istek_uret(photo_urls: list[str], ad_id: int) -> dict:
    """Sözleşme §3 biçiminde bir istek mesajı."""
    return {
        "schema_version": SEMA_SURUMU,
        "request_id": str(uuid.uuid4()),
        "ad_id": ad_id,
        "ad_type": "LOST",
        "declared_species": None,
        "photo_urls": photo_urls,
        # Aday listesini Java PostGIS ile süzüp gönderiyor. Burada boş
        # bırakıyoruz: amaç eşleştirme kalitesini değil, KÖPRÜNÜN çalıştığını
        # göstermek. Aday üretmek için gerçek vektör gerekir, o da modeli
        # ikinci kez çalıştırmak demek.
        "candidates": [],
    }


def sonucu_denetle(sonuc: dict, beklenen_request_id: str) -> list[str]:
    """Sonucu sözleşmeye göre denetler; sorunları liste olarak döner."""
    sorunlar = []
    if sonuc.get("schema_version") != SEMA_SURUMU:
        sorunlar.append(f"schema_version {sonuc.get('schema_version')}")
    if sonuc.get("request_id") != beklenen_request_id:
        sorunlar.append("request_id geri dönmedi (izleme kopar)")
    if sonuc.get("status") not in ("ok", "error"):
        sorunlar.append(f"status={sonuc.get('status')!r}")
    if "processed_at" not in sonuc:
        sorunlar.append("processed_at yok")

    if sonuc.get("status") == "ok":
        analiz = sonuc.get("analysis") or {}
        for alan in ("embeddings", "species", "is_pet", "labels"):
            if alan not in analiz:
                sorunlar.append(f"analysis.{alan} yok")
        if not sonuc.get("model_version"):
            sorunlar.append("model_version yok — hangi modelle üretildiği kaybolur")
        vektorler = analiz.get("embeddings") or []
        if vektorler and not all(isinstance(x, float) for x in vektorler[0][:5]):
            sorunlar.append("embedding float listesi değil")
    else:
        if not (sonuc.get("error") or {}).get("code"):
            sorunlar.append("error.code yok")
    return sorunlar


def main():
    ayristirici = argparse.ArgumentParser()
    ayristirici.add_argument("--url", action="append", default=[],
                             help="Fotoğraf adresi (birden çok kez verilebilir)")
    ayristirici.add_argument("--dosya", type=Path,
                             help="Yerel fotoğraf — geçici sunucudan servis edilir")
    ayristirici.add_argument("--ad-id", type=int, default=123)
    ayristirici.add_argument("--bekle", type=float, default=120.0,
                             help="Sonuç için azami bekleme (saniye)")
    ayristirici.add_argument("--amqp", default=AMQP_URL)
    a = ayristirici.parse_args()

    sunucu = None
    urls = list(a.url)
    if a.dosya:
        if not a.dosya.exists():
            print(f"HATA: dosya yok: {a.dosya}")
            return 1
        adres, sunucu = yerel_sunucu_baslat(a.dosya.resolve())
        urls.append(adres)
        print(f"Yerel sunucu: {adres}")
        print("  (AI servisinin .env dosyasında PHOTO_ALLOWED_HOSTS=127.0.0.1 "
              "olmalı, yoksa SSRF koruması bu adresi reddeder.)")
    if not urls:
        print("HATA: --url ya da --dosya vermelisin.")
        return 1

    baglanti = pika.BlockingConnection(pika.URLParameters(a.amqp))
    kanal = baglanti.channel()
    topolojiyi_kur(kanal)

    # Eski sonuçlar kalmışsa bu koşunun sonucuyla karışmasın.
    kanal.queue_purge(SONUC_KUYRUGU)

    mesaj = istek_uret(urls, a.ad_id)
    kanal.basic_publish(
        exchange=EXCHANGE, routing_key=ISTEK_ANAHTARI,
        body=json.dumps(mesaj, ensure_ascii=False).encode("utf-8"),
        properties=pika.BasicProperties(content_type="application/json",
                                        delivery_mode=2))
    print(f"\n→ İstek yayınlandı  request_id={mesaj['request_id']}  ad_id={a.ad_id}")
    print(f"  fotoğraf: {', '.join(urls)}")
    print(f"\n← {SONUC_KUYRUGU} dinleniyor (en fazla {a.bekle:.0f} sn)...")

    baslangic = time.monotonic()
    govde = None
    while time.monotonic() - baslangic < a.bekle:
        method, ozellikler, govde = kanal.basic_get(SONUC_KUYRUGU, auto_ack=True)
        if method:
            break
        # İlk çağrıda model yükleniyor olabilir; sıkı döngüye girmeyelim.
        baglanti.sleep(0.5)
        govde = None

    if sunucu:
        sunucu.shutdown()

    if not govde:
        print(f"\n✗ {a.bekle:.0f} saniyede sonuç gelmedi.")
        print("  Kontrol: `python -m app.kuyruk` çalışıyor mu? Günlüğünde hata var mı?")
        baglanti.close()
        return 1

    gecen = time.monotonic() - baslangic
    sonuc = json.loads(govde.decode("utf-8"))
    sorunlar = sonucu_denetle(sonuc, mesaj["request_id"])

    print(f"\n← Sonuç geldi ({gecen:.1f} sn, {len(govde)/1024:.1f} KB)\n")
    if sonuc["status"] == "ok":
        analiz = sonuc["analysis"]
        print(f"  durum        : ok")
        print(f"  model        : {sonuc['model_version']}")
        print(f"  tür          : {analiz['species']} "
              f"({analiz['species_confidence']:.2f})")
        print(f"  hayvan mı    : {analiz['is_pet']}")
        print(f"  cins         : {analiz['breed']} ({analiz['breed_confidence']:.2f})")
        print(f"  etiketler    : {', '.join(analiz['labels'])}")
        print(f"  vektör       : {len(analiz['embeddings'])} fotoğraf x "
              f"{len(analiz['embeddings'][0])} boyut")
        print(f"  eşleşme      : {len(sonuc['matches'])}")
        print(f"  atlanan aday : {sonuc['skipped_candidates']['toplam']}")
    else:
        print(f"  durum : error")
        print(f"  kod   : {sonuc['error']['code']}")
        print(f"  mesaj : {sonuc['error']['message']}")

    print()
    if sorunlar:
        print("✗ SÖZLEŞME DENETİMİ BAŞARISIZ:")
        for s in sorunlar:
            print(f"    - {s}")
    else:
        print("✓ Sözleşme denetimi geçti — mesaj biçimi Java'nın beklediği gibi.")

    baglanti.close()
    return 1 if sorunlar else 0


if __name__ == "__main__":
    sys.exit(main())
