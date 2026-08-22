# tests/test_kirpma.py
"""app/kirpma.py: hayvana-kırpma yeteneği.

Gerçek dedektörü (torchvision, ~74 MB, ilk kullanımda iner) İNDİRMEDEN
sahte bir `HayvanDedektoru` enjekte ederek kırpma/pay/fallback mantığını
sınar -- aynı test felsefesi: app/matcher.py'nin gerçek bir broker/model
olmadan sınanması gibi. Gerçek dedektörle uçtan uca bir doğrulama, seed_data
gibi opsiyonel/isteğe bağlı bırakılmıştır (bkz. test_gercek_dedektor_*).

CROP_TO_ANIMAL varsayılan KAPALI olduğu için bu testler canlı davranışı
DEĞİL, yalnızca yetenek doğru çalışıyor mu onu doğrular.
"""
import io
import os

import pytest
from PIL import Image

from app.kirpma import KENAR_PAYI_ORANI, _bool, hayvana_kirp


def jpg(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="JPEG")
    return b.getvalue()


class _SahteDedektor:
    """En_iyi_kutu() sabit bir değer döner -- gerçek model hiç yüklenmez."""

    def __init__(self, kutu):
        self._kutu = kutu

    def en_iyi_kutu(self, img):
        return self._kutu


# --------------------------------------------------------------------------
# _bool: env değişkeni ayrıştırma (app/config.py'deki collector deseniyle aynı)
# --------------------------------------------------------------------------


def test_bool_ayristirma_varsayilan_deger_kullanilir(monkeypatch):
    monkeypatch.delenv("OLMAYAN_DEGISKEN", raising=False)
    assert _bool("OLMAYAN_DEGISKEN", False) is False
    assert _bool("OLMAYAN_DEGISKEN", True) is True


@pytest.mark.parametrize("deger", ["1", "true", "True", "yes", "on"])
def test_bool_ayristirma_dogru_degerler(monkeypatch, deger):
    monkeypatch.setenv("TEST_BOOL_DEGISKENI", deger)
    assert _bool("TEST_BOOL_DEGISKENI", False) is True


@pytest.mark.parametrize("deger", ["0", "false", "no", "off", ""])
def test_bool_ayristirma_yanlis_degerler(monkeypatch, deger):
    monkeypatch.setenv("TEST_BOOL_DEGISKENI", deger)
    assert _bool("TEST_BOOL_DEGISKENI", True) is False


# --------------------------------------------------------------------------
# hayvana_kirp: kutuya göre kırpma + pay + fallback
# --------------------------------------------------------------------------


def test_hayvana_kirp_tespit_edilen_kutuya_kirpar():
    img = Image.new("RGB", (200, 200), (10, 20, 30))
    # Kenar payı olmadan test edebilmek için payı geçici olarak sıfırla.
    dedektor = _SahteDedektor((50.0, 50.0, 150.0, 150.0))

    import app.kirpma as kirpma_modulu
    eski_pay = kirpma_modulu.KENAR_PAYI_ORANI
    kirpma_modulu.KENAR_PAYI_ORANI = 0.0
    try:
        sonuc = hayvana_kirp(img, dedektor=dedektor)
    finally:
        kirpma_modulu.KENAR_PAYI_ORANI = eski_pay

    assert sonuc.size == (100, 100)


def test_hayvana_kirp_kutu_etrafina_pay_ekler():
    img = Image.new("RGB", (200, 200), (10, 20, 30))
    dedektor = _SahteDedektor((50.0, 50.0, 150.0, 150.0))  # 100x100 kutu

    sonuc = hayvana_kirp(img, dedektor=dedektor)

    # %10 pay: her yöne 10px eklenmeli -> 120x120
    beklenen_kenar = 100 * (1 + 2 * KENAR_PAYI_ORANI)
    assert abs(sonuc.size[0] - beklenen_kenar) <= 1
    assert abs(sonuc.size[1] - beklenen_kenar) <= 1


def test_hayvana_kirp_payi_goruntu_sinirinda_kirpar():
    """Kutu görüntünün kenarına yakınsa, pay görüntü dışına taşmamalı."""
    img = Image.new("RGB", (200, 200), (10, 20, 30))
    dedektor = _SahteDedektor((0.0, 0.0, 100.0, 100.0))  # sol üst köşede

    sonuc = hayvana_kirp(img, dedektor=dedektor)

    # Pay sol/üstte 0'ın altına inemez, sağ/altta en fazla 100+10=110'a çıkar.
    assert sonuc.size[0] <= 111
    assert sonuc.size[1] <= 111


def test_hayvana_kirp_kutu_bulunamazsa_orijinali_doner():
    """Ekibin ölçümünde 7 fotoğrafın 2'sinde (%29) dedektör hayvanı hiç
    bulamamıştı (docs/olcum-raporu.md §4) -- bu durumda görsel OLDUĞU GİBİ
    dönmeli, sessizce boş/anlamsız bir kırpma üretmemeli."""
    img = Image.new("RGB", (200, 200), (10, 20, 30))
    dedektor = _SahteDedektor(None)

    sonuc = hayvana_kirp(img, dedektor=dedektor)

    assert sonuc.size == img.size


def test_hayvana_kirp_cok_kucuk_kirpma_orijinale_geri_doner():
    """Dedektörün minik (muhtemelen yanlış) bir kutu bulduğu durumda, çok
    küçük bir kırpma üretmektense tam görüntüye geri dönülmeli -- çok küçük
    bir kırpma hiç kırpmamaktan daha kötü bir sonuçtur."""
    img = Image.new("RGB", (200, 200), (10, 20, 30))
    dedektor = _SahteDedektor((0.0, 0.0, 5.0, 5.0))  # ASGARI_KENAR'ın (32) altı

    sonuc = hayvana_kirp(img, dedektor=dedektor)

    assert sonuc.size == img.size


def test_crop_to_animal_varsayilan_kapali():
    """Bu bayrağın YANLIŞLIKLA açık gelmesi, hiçbir açık env değişkeni
    olmadan mevcut tüm ilanların embedding'lerini sessizce değiştirir --
    bkz. app/kirpma.py'nin docstring'i. Varsayılan kapalı olduğunu doğrudan
    kanıtlamak, bu regresyonun fark edilmeden geçmesini engeller."""
    assert os.getenv("CROP_TO_ANIMAL") is None, (
        "test ortamında CROP_TO_ANIMAL ayarlanmamış olmalı ki bu test "
        "gerçek varsayılanı sınasın"
    )
    from app.kirpma import CROP_TO_ANIMAL
    assert CROP_TO_ANIMAL is False


# --------------------------------------------------------------------------
# Gerçek dedektörle uçtan uca (opsiyonel -- ağırlık indirmeyi gerektirir)
# --------------------------------------------------------------------------

_GERCEK_DEDEKTOR_ISTENDI = os.getenv("RUN_KIRPMA_ENTEGRASYON_TESTI") == "1"
pytestmark_gercek = pytest.mark.skipif(
    not _GERCEK_DEDEKTOR_ISTENDI,
    reason="RUN_KIRPMA_ENTEGRASYON_TESTI=1 verilmedi -- gerçek dedektör "
           "(~74 MB, ilk kullanımda iner) indirilmeden atlanır, aynı "
           "seed_data/CI ağırlık indirmeme kuralı (bkz. tests/conftest.py)",
)


@pytestmark_gercek
def test_gercek_dedektor_duz_renkte_hayvan_bulamaz():
    from app.kirpma import _dedektoru_al

    img = Image.new("RGB", (256, 256), (128, 128, 128))
    kutu = _dedektoru_al().en_iyi_kutu(img)
    assert kutu is None
