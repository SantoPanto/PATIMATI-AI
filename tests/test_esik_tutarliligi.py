"""Bildirim eşiği tek yerden okunsun — belgeler koddan kaymasın.

NEDEN VAR: 2026-08-13'te `app/matcher.py`'deki varsayılan 0.70'ten 0.65'e
indirildi, ama `.env.example` 0.70 demeye devam etti. Sonuç: altı ayrı dosya
iki farklı değer söylüyordu ve bakan kişi hangisinin doğru olduğunu
bilemiyordu. Hiçbir test eşiğe bakmadığı için CI olsaydı bile yakalayamazdı.

Bu bekçi iki şeyi zorluyor:
  1. `.env.example`'daki MATCH_THRESHOLD, koddaki varsayılanla AYNI olacak.
     (`.env.example` kurulum yapan kişinin kopyaladığı dosya; koddan farklıysa
     kurulumun eşiği sessizce başka olur.)
  2. README eşik sayısını TEKRARLAMAYACAK. Tekrarlanan her sayı kayabilecek
     bir sayıdır; README `MATCH_THRESHOLD`'a işaret etmekle yetinir.

Kapsam dışı bırakılanlar ve sebebi:
  - `docs/olcum-raporu.md` ve `degerlendirme/`: bunlar TARİHSEL ölçüm kayıtları.
    "O ölçüm 0.70 eşiğiyle yapıldı" cümlesi eşik değişince yanlış olmaz, tam
    tersine doğru kalması gerekir.
  - `scripts/` içindeki eşik TARAMALARI (`for esik in (0.65, 0.70, ...)`):
    bunlar bir aralığı gezmek için var, yürürlükteki eşiği iddia etmiyorlar.
"""
import re
from pathlib import Path

import pytest

from app.matcher import MATCH_THRESHOLD

KOK = Path(__file__).resolve().parent.parent

# `os.getenv("MATCH_THRESHOLD", "0.70")` içindeki VARSAYILANI yakalar.
# Modülden okunan MATCH_THRESHOLD'a bakmıyoruz: ortam değişkeni ayarlıysa
# (CI, yerel .env) o değer gelir ve test kendi ölçmek istediği şeyi kaçırır.
KOD_VARSAYILANI = re.compile(
    r'MATCH_THRESHOLD\s*=\s*float\(\s*os\.getenv\(\s*["\']MATCH_THRESHOLD["\']\s*,\s*["\']([\d.]+)["\']'
)
ENV_SATIRI = re.compile(r'^\s*MATCH_THRESHOLD\s*=\s*([\d.]+)\s*$', re.MULTILINE)


def kod_varsayilani() -> float:
    kaynak = (KOK / "app" / "matcher.py").read_text(encoding="utf-8")
    eslesme = KOD_VARSAYILANI.search(kaynak)
    assert eslesme, (
        "app/matcher.py içinde MATCH_THRESHOLD varsayılanı bulunamadı. "
        "Satırın biçimi değiştiyse bu bekçinin deseni de güncellenmeli — "
        "sessizce geçmesindense burada patlaması doğru."
    )
    return float(eslesme.group(1))


def test_env_example_kodla_ayni_esigi_soyluyor():
    """.env.example'daki değer, app/matcher.py'deki varsayılanla aynı olmalı."""
    ornek = (KOK / ".env.example").read_text(encoding="utf-8")
    eslesme = ENV_SATIRI.search(ornek)
    assert eslesme, ".env.example içinde MATCH_THRESHOLD satırı yok."

    env_degeri = float(eslesme.group(1))
    kod = kod_varsayilani()
    assert env_degeri == pytest.approx(kod), (
        f".env.example MATCH_THRESHOLD={env_degeri} diyor ama app/matcher.py "
        f"varsayılanı {kod}. İkisi aynı olmalı: .env.example kurulum yapan "
        f"kişinin kopyaladığı dosya, koddan farklıysa kurulumun eşiği sessizce "
        f"başka olur. Değeri bilerek değiştirdiysen İKİSİNİ BİRDEN değiştir."
    )


def test_readme_esik_sayisini_tekrarlamiyor():
    """README sayıyı tekrar etmesin; `MATCH_THRESHOLD`'a işaret etsin."""
    readme = (KOK / "README.md").read_text(encoding="utf-8")

    # Yalnız eşikten söz eden satırdaki çıplak ondalık sayıyı arıyoruz;
    # README'deki başka sayılar (%55/%30/%15 ağırlıkları, port numaraları) masum.
    #
    # ⚠️ "eşik" diye aramak YETMEZ: Türkçede k→ğ yumuşaması var ve README'de
    # kelime "eşiği" diye geçiyor — "eşik" alt dizgisi o kelimede HİÇ yok.
    # Bu bekçinin ilk hâli tam bu yüzden bozuktu ve mutasyon denemesinde
    # yakalandı (bozdum, yeşil kaldı). `eşi[kğ]` ikisini de kapsar.
    tetikleyici = re.compile(r"eşi[kğ]|MATCH_THRESHOLD", re.IGNORECASE)
    suclu = [
        satir.strip()
        for satir in readme.splitlines()
        if tetikleyici.search(satir) and re.search(r"\b0[.,]\d{2}\b", satir)
    ]
    assert not suclu, (
        "README eşik sayısını tekrarlıyor: "
        + " | ".join(suclu)
        + " — tekrarlanan sayı kayabilir. Sayıyı yazma, `MATCH_THRESHOLD`'a ve "
        "app/matcher.py'ye işaret et."
    )


def test_calisan_esik_kod_varsayilani_ile_uyusuyor_ya_da_bilerek_ezilmis(monkeypatch):
    """Modülden okunan değer ya varsayılan ya da ortam değişkeninden gelmeli.

    Bu, yukarıdaki ikisinin sessizce anlamsızlaşmasına karşı: eğer bir gün
    MATCH_THRESHOLD ortam değişkeninden okunmayı bırakırsa (ör. sabit yazılırsa)
    `.env.example` ile kod eşit görünmeye devam eder ama .env'deki değer artık
    hiçbir şey yapmaz. Kurulumun ayar dosyası sessizce etkisiz kalır.
    """
    import importlib

    import app.matcher as matcher

    monkeypatch.setenv("MATCH_THRESHOLD", "0.42")
    yeniden = importlib.reload(matcher)
    try:
        assert yeniden.MATCH_THRESHOLD == pytest.approx(0.42), (
            "MATCH_THRESHOLD ortam değişkeninden okunmuyor. .env ile ayarlamak "
            "işe yaramaz hâle gelmiş demektir — sözleşme ve .env.example bunun "
            "ayarlanabildiğini söylüyor."
        )
    finally:
        monkeypatch.delenv("MATCH_THRESHOLD", raising=False)
        importlib.reload(matcher)

    # Diğer testler ve modülü zaten içe aktarmış olanlar bozulmasın.
    assert matcher.MATCH_THRESHOLD == pytest.approx(MATCH_THRESHOLD)
