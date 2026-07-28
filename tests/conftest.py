# tests/conftest.py
"""Test oturumunun kimlik modelini AÇIKÇA sabitler.

Sorun neydi: `app/surum.py` kimlik modelini `KIMLIK_MODEL` ortam değişkeninden
İÇE AKTARMA ANINDA okuyor. `.env` dosyasını ise `load_dotenv()` yüklüyor ve o
da yalnızca bazı modüllerde çağrılıyor. Yani testlerin hangi modelle koştuğu
"hangi test dosyası önce içe aktarıldı" sorusunun cevabına bağlıydı:

  - `test_attributes.py` önce gelirse  -> .env okunmamış -> clip / 512
  - `app.topoloji` önce gelirse        -> .env okunmuş   -> siglip2 / 768

Bu sessiz bir tuzak. Testler her iki durumda da GEÇER (vektör boyutunu
`VEKTOR_BOYUTU`den alıyorlar, kendi içinde tutarlılar), ama hangi
yapılandırmanın sınandığı belirsiz kalır. 2026-07-28'de gerçekten böyle oldu:
servis siglip2/768 ile çalışırken testler clip/512 ile geçiyordu.

Çözüm: varsayılanı burada, her şeyden önce sabitliyoruz.

  - `setdefault` kullanılıyor, düz atama değil: kabuktan açıkça verilen değer
    kazanmalı ki kazanan modeli de sınayabilelim:
        $env:KIMLIK_MODEL="siglip2-animal"; pytest tests/
  - Varsayılan `clip`, çünkü SigLIP2 ağırlıkları 1,4 GB ve depoda değil.
    Testler ağırlık indirmeden, her ekip arkadaşının makinesinde çalışmalı.
    Testin ne sınadığı makineden makineye değişmemeli.
"""
import os

os.environ.setdefault("KIMLIK_MODEL", "clip")


def pytest_report_header(config):
    """Hangi yapılandırmanın sınandığını çıktının başına yazar.

    Görünmeyen varsayılan, yanlış varsayılandan daha tehlikeli: kimse
    "hangi modelle geçti bu testler?" diye sormaz.
    """
    from app.surum import MODEL_SURUMU, SECILEN_KIMLIK, VEKTOR_BOYUTU
    return (f"kimlik modeli: {SECILEN_KIMLIK} "
            f"(vektör {VEKTOR_BOYUTU}, sürüm {MODEL_SURUMU})")
