# tests/test_tur_kapisi.py
"""Tür atama kapısı: hayvan kanıtı zayıfken `species` atanmamalı.

NEDEN AYRI DOSYA: `test_attributes.py` gerçek fotoğraflarla ölçüyor ama
`tests/seed_data` depoda yok ve CI `prepare_seed.py` çalıştırmıyor ⇒ o dosya
CI'da BAŞTAN SONA ATLANIYOR. Oraya konan bir bekçi hiç koşmaz.

BU DOSYA NEYİ ÖLÇER: kararın kendisini — "hayvan kanıtı şu kadarken tür
atanır mı". Tür kapısının girdisi (`_tur_kapisi_feats` ile hesaplanan
benzerlikler) bilerek elle kuruluyor, böylece test modelin o günkü
davranışına değil, KURALA bağlı kalıyor ve ağırlık indirmeden koşuyor.

BU DOSYA NEYİ ÖLÇMEZ: modelin gerçek fotoğraflarda ürettiği skorları. Eşiğin
değeri (0.50) gerçek veriyle seçildi — 26 etiketli hayvansız fotoğraf +
81 gerçek hayvan fotoğrafı; yeniden ölçmek için `scripts/hayvansiz_olcum.py`.
Fikstür model katmanını atlıyor, dolayısıyla "yangın tüpüne artık dog demiyor"
iddiasını BU test kanıtlamaz, o iddia betikle ölçülür.
"""
import math

import pytest
import torch

from app.attributes import attribute_analyzer


def _kapiyi_kur(monkeypatch, p_kedi, p_kopek, p_celdirici):
    """`predict_species`'e verilecek girdiyi hedef olasılıkları verecek şekilde kurar.

    `predict_species` içeride `softmax(img_feat @ feats.T * 100)` hesaplıyor.
    `feats` birim matris yapılırsa benzerlikler doğrudan `img_feat` olur;
    `log(p)/100` yazınca softmax tam olarak `p`'yi geri verir. Böylece
    "hayvan toplamı 0.30 iken ne oluyor" sorusu doğrudan sorulabiliyor.
    """
    hedefler = [p_kedi, p_kopek, p_celdirici]
    assert abs(sum(hedefler) - 1.0) < 1e-9, "olasılıklar 1'e toplanmalı"

    monkeypatch.setattr(attribute_analyzer, "_species_keys", ["cat", "dog"])
    monkeypatch.setattr(attribute_analyzer, "_tur_kapisi_feats", torch.eye(3))
    return torch.tensor([[math.log(p) / 100 for p in hedefler]], dtype=torch.float32)


def test_zayif_hayvan_kanitinda_tur_atanmiyor(monkeypatch):
    """Yangın tüpü vakası: kapı "hayvan var" diyor ama kanıt zayıf.

    Ölçümde bu, hayvansız 26 fotoğrafın 3'ünde gerçekleşiyordu: hayvan
    toplamı 0.115–0.322 arasındayken kedi/köpek yarışması İKİ seçenek
    üzerinden yapıldığı için güven %82'ye çıkıyor ve "dog" atanıyordu.
    Tür ataması eşleştirmede ELEME ölçütü (matcher: species_mismatch), yani
    yanlış tür kayıp hayvanı listeden düşürüyor.
    """
    girdi = _kapiyi_kur(monkeypatch, 0.25, 0.05, 0.70)   # hayvan toplamı 0.30

    tur, guven, hayvan_mi = attribute_analyzer.predict_species(girdi)

    assert tur == "unknown", (
        f"hayvan kanıtı 0.30 iken tür atandı ({tur}) — "
        f"SPECIES_GATE_MIN={attribute_analyzer.SPECIES_GATE_MIN} çalışmıyor.")
    assert guven == pytest.approx(0.833, abs=0.01), (
        "güvenin anlamı değişmemeli: hâlâ 'hayvansa hangisi' sorusunun cevabı.")
    assert hayvan_mi is True, (
        "is_pet DEĞİŞMEMELİ. Sözleşme §4 gereği eleme ölçütü değil ve bilerek "
        "gevşek; tür kapısını eklerken onu da sıkmak gerçek ilanları elerdi.")


def test_guclu_hayvan_kanitinda_tur_atanmaya_devam_ediyor(monkeypatch):
    """Ön koşul: kapı yalnızca reddetmiyor, geçiriyor da.

    Bu olmadan yukarıdaki test "her şeye unknown de" diyen bozuk bir kapıyla
    da yeşil yanardı. Gerçek hayvan fotoğraflarında ölçülen hayvan toplamı
    0.76–0.98 aralığında.
    """
    girdi = _kapiyi_kur(monkeypatch, 0.85, 0.05, 0.10)   # hayvan toplamı 0.90

    tur, guven, hayvan_mi = attribute_analyzer.predict_species(girdi)

    assert tur == "cat", f"güçlü kanıtta tür atanmadı: {tur}"
    assert hayvan_mi is True


def test_esigin_tam_ustunde_tur_atanir(monkeypatch):
    """Sınır bilerek dışlayıcı değil: eşiğe EŞİT olan kanıt yeter.

    Kural `< SPECIES_GATE_MIN` diye yazıldı; `<=` olsaydı eşiğe oturan bir
    fotoğraf sessizce türsüz kalırdı ve eşik belgede yazandan farklı davranırdı.
    """
    esik = attribute_analyzer.SPECIES_GATE_MIN
    girdi = _kapiyi_kur(monkeypatch, esik - 0.05, 0.05, 1.0 - esik)

    tur, _, _ = attribute_analyzer.predict_species(girdi)

    assert tur == "cat", f"hayvan toplamı tam eşikte ({esik}) tür atanmadı: {tur}"


def test_tur_ayrimi_belirsizken_hala_unknown(monkeypatch):
    """Eski kural yerinde duruyor: kanıt güçlü ama kedi/köpek ayrımı net değil.

    Yeni kapı eklenirken bu dalın kaybolmadığını gösterir — iki koşul
    birbirinin yerine geçmiyor, ikisi de gerekiyor.
    """
    girdi = _kapiyi_kur(monkeypatch, 0.48, 0.47, 0.05)   # toplam 0.95, güven 0.505

    tur, guven, hayvan_mi = attribute_analyzer.predict_species(girdi)

    assert tur == "unknown", "kedi/köpek ayrımı belirsizken tür atanmamalı"
    assert guven < attribute_analyzer.SPECIES_MIN_PROB
    assert hayvan_mi is True


# --------------------------------------------------------------------------
# İlan seviyesi: birden çok fotoğraflı ilanda BİRİNCİL kare seçimi
# --------------------------------------------------------------------------

def _kare(species, guven, is_pet=True, etiket="x"):
    """`_birincil_sec`'in okuduğu alanlarla asgari bir öznitelik sözlüğü."""
    return {"species": species, "species_confidence": guven, "is_pet": is_pet,
            "labels": [etiket], "breed": None, "breed_confidence": 0.0,
            "pattern": None, "colors": []}


def test_birincil_kare_turu_atanmis_fotografi_secer():
    """Hayvansız kare, tür güveni daha yüksek diye ilanı ele geçirmemeli.

    ÖLÇÜLDÜ (gerçek fotoğraflar): gerçek kedi karesi güven 0.8177, hayvansız
    kare güven 0.8293 ⇒ eski kural hayvansız kareyi birincil seçiyordu ve
    ilanın türü `cat` yerine `unknown` oluyordu; etiketler/desen de o kareden
    geliyordu. `species_confidence` "hayvan var mı"yı ölçmüyor.
    """
    from app.analiz import _birincil_sec

    hayvansiz = _kare("unknown", 0.8293, etiket="hayvansiz")
    gercek_kedi = _kare("cat", 0.8177, etiket="kedi")

    secilen = _birincil_sec([hayvansiz, gercek_kedi])

    assert secilen["labels"] == ["kedi"], (
        "türü atanmamış kare birincil seçildi — ilan türünü ve etiketlerini "
        "hayvansız fotoğraftan alır.")


def test_birincil_kare_iki_tur_atanmissa_guveni_yuksek_olani_secer():
    """Eski kural ikinci basamak olarak DURUYOR.

    Bu olmadan yukarıdaki test, "listedeki ilk türü atanmış kareyi al" diyen
    daha kaba bir kuralla da yeşil yanardı.
    """
    from app.analiz import _birincil_sec

    bulanik = _kare("cat", 0.83, etiket="bulanik")
    net = _kare("cat", 0.99, etiket="net")

    assert _birincil_sec([bulanik, net])["labels"] == ["net"]


def test_birincil_kare_hicbiri_turlu_degilse_eski_davranis():
    """Hiçbir karenin türü atanamadıysa seçim yine güvene düşer — ilan yine de
    bir kareden etiket almalı, boş kalmamalı."""
    from app.analiz import _birincil_sec

    dusuk = _kare("unknown", 0.55, etiket="dusuk")
    yuksek = _kare("unknown", 0.79, etiket="yuksek")

    assert _birincil_sec([dusuk, yuksek])["labels"] == ["yuksek"]
