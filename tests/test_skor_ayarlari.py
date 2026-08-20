"""Skor ağırlıkları ve konum eğrisi ortamdan ayarlanabilir — ama SESSİZCE bozulamaz.

NEDEN VAR: 19.08 ölçümü, aynı hayvanın birebir aynı fotoğrafının 3 km uzaktayken
eşiği geçemediğini gösterdi (cebirsel azami mesafe 432 m, aday arama yarıçapı
25 km). Düzeltmenin yolu bir sayıyı gevşetmek, ama hangi sayının doğru olduğu
YANLIŞ POZİTİF bedeli ölçülmeden bilinemez. Bu yüzden hiçbir sayı değiştirilmedi,
yalnız ayarlanabilir hâle getirildi.

Ayarlanabilirlik iki yeni risk açar ve ikisi de burada kapatılıyor:

  1. VARSAYILAN KAYMASI — "ayarlanabilir yaptık" derken bugünkü davranış
     değişirse, kimse fark etmeden eşleşme davranışı değişmiş olur. Bu yüzden
     varsayılanlarla hesaplanan sayılar ESKİ değerlere karşı sınanıyor.

  2. ANLAMSIZ AYAR — ağırlıklar toplamı 1 değilse skor artık [0,1] aralığında
     olmaz ve eşik (0.80) başka bir şeyi ölçmeye başlar. Yanlış ayar dağıtımda
     değil, ilk saniyede görünmeli.

Ayrıca düğmelerin GERÇEKTEN bağlı olduğu sınanıyor: yalnız "varsayılan değişmedi"
sınanırsa, hiçbir şeye bağlanmamış süs bir ortam değişkeni de testi geçer.

Bu dosya model ağırlığı ya da tohum verisi İSTEMEZ — `tests/test_attributes.py`
CI'da baştan sona atlanıyor (tests/seed_data depoda yok) ve oraya konan bir bekçi
hiç koşmaz.
"""
import importlib

import pytest

import app.matcher as matcher

# Raporun canlı ölçtüğü çift (TUR8 BULGU-4 tablosu).
OLCULEN_GORSEL = 0.9417
OLCULEN_ETIKET = 0.4800


@pytest.fixture
def yeniden_yukle(monkeypatch):
    """Ortam değişkenleriyle modülü yeniden yükler, sonunda eski hâle döner."""
    def yukle(**ortam):
        for anahtar, deger in ortam.items():
            monkeypatch.setenv(anahtar, str(deger))
        return importlib.reload(matcher)

    yield yukle
    # Diğer testler bozulmuş bir modülle karşılaşmasın.
    monkeypatch.undo()
    importlib.reload(matcher)


def test_varsayilanlar_eski_davranisin_birebir_aynisi():
    assert (matcher.GORSEL_AGIRLIK, matcher.ETIKET_AGIRLIK,
            matcher.KONUM_AGIRLIK) == (0.55, 0.30, 0.15)
    assert matcher.KONUM_YARI_MESAFE_KM == 5.0


@pytest.mark.parametrize("km, beklenen", [
    (0, 1.00), (5, 0.50), (20, 0.20), (50, 0.0909),
])
def test_konum_egrisi_varsayilanla_degismedi(km, beklenen):
    # Bu dört sayı `location_score` docstring'inde de yazılı; oradan kayarsa
    # belge ile kod iki ayrı şey söylemeye başlar.
    assert matcher.location_score(km) == pytest.approx(beklenen, abs=0.001)


def test_raporun_olctugu_cift_ayni_skoru_veriyor():
    # Uçtan uca ölçümde 3 km'de 0.7558 çıkmıştı. Ağırlıklar ortama taşınırken
    # aritmetiğin kaymadığının kanıtı.
    skor = (matcher.GORSEL_AGIRLIK * OLCULEN_GORSEL
            + matcher.ETIKET_AGIRLIK * OLCULEN_ETIKET
            + matcher.KONUM_AGIRLIK * matcher.location_score(3.0))
    assert skor == pytest.approx(0.7558, abs=0.001)


def test_agirliklar_toplami_bir_degilse_modul_acilmiyor(yeniden_yukle):
    with pytest.raises(ValueError, match="toplamı 1.0 olmalı"):
        yeniden_yukle(GORSEL_AGIRLIK="0.90")


def test_yari_mesafe_pozitif_degilse_modul_acilmiyor(yeniden_yukle):
    with pytest.raises(ValueError, match="pozitif olmalı"):
        yeniden_yukle(KONUM_YARI_MESAFE_KM="0")


def test_yari_mesafe_dugmesi_gercekten_bagli(yeniden_yukle):
    # Süs değişken sınaması: yarı mesafe 10 km olursa 10 km'de skor 0.50
    # OLMALI. Değişken bağlı değilse burası 0.3333 kalır ve test kırmızı yanar.
    m = yeniden_yukle(KONUM_YARI_MESAFE_KM="10")
    assert m.location_score(10) == pytest.approx(0.50, abs=0.001)
    assert m.location_score(0) == pytest.approx(1.00, abs=0.001)


def test_agirlik_dugmeleri_gercekten_bagli(yeniden_yukle):
    m = yeniden_yukle(GORSEL_AGIRLIK="0.70", ETIKET_AGIRLIK="0.15",
                      KONUM_AGIRLIK="0.15")
    assert (m.GORSEL_AGIRLIK, m.ETIKET_AGIRLIK, m.KONUM_AGIRLIK) == (0.70, 0.15, 0.15)

    # Aynı çift, etiket ağırlığı düşürülünce eşiği 3 km'de artık geçmeli.
    # (Ölçülen kusurun tam olarak neye bağlı olduğunu da gösteriyor: etiket
    # kanalı 0.48'de takılı olduğu için toplam eşiğin altında kalıyordu.)
    eski = (0.55 * OLCULEN_GORSEL + 0.30 * OLCULEN_ETIKET
            + 0.15 * m.location_score(3.0))
    yeni = (m.GORSEL_AGIRLIK * OLCULEN_GORSEL
            + m.ETIKET_AGIRLIK * OLCULEN_ETIKET
            + m.KONUM_AGIRLIK * m.location_score(3.0))
    assert yeni > eski


# ---------------------------------------------------------------------------
# .env.example ile kod aynı sayıyı söylüyor mu
#
# `test_esik_tutarliligi.py` bu kuralı MATCH_THRESHOLD için kuruyor; sebebi
# 13.08'de kodun 0.65'e inip .env.example'ın 0.70 demeye devam etmesiydi. Yeni
# eklenen dört düğme aynı riski taşıyor: .env.example kurulum yapan kişinin
# KOPYALADIĞI dosya, koddan farklıysa kurulumun ağırlıkları sessizce başka olur.
# ---------------------------------------------------------------------------
import re  # noqa: E402
from pathlib import Path  # noqa: E402

KOK = Path(__file__).resolve().parent.parent

DUGMELER = {
    "GORSEL_AGIRLIK": 0.55,
    "ETIKET_AGIRLIK": 0.30,
    "KONUM_AGIRLIK": 0.15,
    "KONUM_YARI_MESAFE_KM": 5.0,
}


def _kod_varsayilani(ad: str) -> float:
    kaynak = (KOK / "app" / "matcher.py").read_text(encoding="utf-8")
    desen = re.compile(
        r'%s\s*=\s*float\(\s*os\.getenv\(\s*["\']%s["\']\s*,\s*["\']([\d.]+)["\']'
        % (re.escape(ad), re.escape(ad)))
    eslesme = desen.search(kaynak)
    assert eslesme, (
        f"app/matcher.py içinde {ad} varsayılanı bulunamadı. Satırın biçimi "
        f"değiştiyse bu bekçinin deseni de güncellenmeli — 0 eşleşme şüphedir, "
        f"sessizce geçmesindense burada patlaması doğru."
    )
    return float(eslesme.group(1))


@pytest.mark.parametrize("ad, beklenen", sorted(DUGMELER.items()))
def test_kod_varsayilani_beklenen_deger(ad, beklenen):
    # Pozitif kontrol: desen GERÇEKTEN okuyor mu? Yalnız .env karşılaştırması
    # yapılsaydı, iki taraf da birlikte kayınca test yeşil kalırdı.
    assert _kod_varsayilani(ad) == pytest.approx(beklenen)


@pytest.mark.parametrize("ad", sorted(DUGMELER))
def test_env_example_kodla_ayni_degeri_soyluyor(ad):
    ornek = (KOK / ".env.example").read_text(encoding="utf-8")
    eslesme = re.search(r"^\s*%s\s*=\s*([\d.]+)\s*$" % re.escape(ad),
                        ornek, re.MULTILINE)
    assert eslesme, f".env.example içinde {ad} satırı yok."
    assert float(eslesme.group(1)) == pytest.approx(_kod_varsayilani(ad)), (
        f".env.example {ad}={eslesme.group(1)} diyor ama app/matcher.py "
        f"varsayılanı {_kod_varsayilani(ad)}. Bilerek değiştirdiysen İKİSİNİ "
        f"BİRDEN değiştir."
    )
