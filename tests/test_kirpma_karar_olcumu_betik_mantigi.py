# tests/test_kirpma_karar_olcumu_betik_mantigi.py
"""scripts/kirpma_karar_olcumu.py'nin MEKANİĞİNİ sınar -- gerçek bir kırpma
kararı üretmez (gerçek CatIndividualImages verisi bu ortamda yok, bkz. o
betiğin kendi docstring'i). Burada sentetik, düz renkli görüntülerle
kurulan sahte bir <birey>/<foto> klasörü kullanılıyor -- amaç betiğin uçtan
uca ÇALIŞTIĞINI (klasör okuma, gruplama, AUC hesaplama, kırpma entegrasyonu)
doğrulamak, "bu renklerden hangisi hangisiyle eşleşir" gibi bir doğruluk
iddiası TAŞIMIYOR (düz renkli sentetik görüntülerde CLIP'in ayrımı anlamsız
olurdu -- tıpkı is_pet kapısının sentetik testinin gerçek negatiflerde
yanıltıcı çıkması gibi, bkz. tests/test_saglamlik.py)."""
import io

import numpy as np
import pytest
from PIL import Image

import scripts.kirpma_karar_olcumu as olcum


def _jpg(renk) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (128, 128), renk).save(b, format="JPEG")
    return b.getvalue()


@pytest.fixture
def sahte_veri_koku(tmp_path):
    """3 birey × 2 fotoğraf, model_yarisi.py'nin beklediği <kök>/<birey>/
    <foto> düzeninde."""
    renkler = {
        "kedi01": [(200, 50, 50), (190, 60, 55)],
        "kedi02": [(50, 200, 50), (55, 190, 60)],
        "kedi03": [(50, 50, 200), (60, 55, 190)],
    }
    for birey, fotolar in renkler.items():
        dizin = tmp_path / birey
        dizin.mkdir()
        for i, renk in enumerate(fotolar):
            (dizin / f"{i}.jpg").write_bytes(_jpg(renk))
    return tmp_path


def test_tabloyu_kur_klasor_yapisini_dogru_okur(sahte_veri_koku):
    satirlar = olcum._tabloyu_kur(sahte_veri_koku)
    assert len(satirlar) == 6
    bireyler = {b for b, _ in satirlar}
    assert bireyler == {"kedi01", "kedi02", "kedi03"}


def test_bireylere_grupla_dogru_sayar(sahte_veri_koku):
    satirlar = olcum._tabloyu_kur(sahte_veri_koku)
    gruplar = olcum._bireylere_grupla(satirlar)
    assert {b: len(f) for b, f in gruplar.items()} == {
        "kedi01": 2, "kedi02": 2, "kedi03": 2,
    }


def test_auc_kusursuz_ayrimda_bir_doner():
    """Tüm 'aynı' skorları tüm 'farklı' skorlarından yüksekse AUC 1.0 olmalı
    -- teshis_arkaplan.py'deki AYNI fonksiyonla aynı formül."""
    assert olcum.auc(np.array([0.9, 0.85]), np.array([0.5, 0.4])) == 1.0


def test_auc_yazi_turaysa_yarim_doner():
    assert olcum.auc(np.array([0.5]), np.array([0.5])) == 0.5


def test_auc_veri_yoksa_nan_doner():
    assert np.isnan(olcum.auc([], [0.5]))
    assert np.isnan(olcum.auc([0.5], []))


def test_farkli_indeksleri_ornekle_kendisiyle_eslesmez():
    """a == b (bir bireyin kendisiyle 'farklı birey' sayılması) örneklemeye
    hiç girmemeli -- rng.choice(..., replace=False) bunu garanti eder."""
    bireyler = {"a": [0, 1], "b": [0, 1], "c": [0, 1]}
    rng = np.random.default_rng(0)
    ornekler = olcum._farkli_indeksleri_ornekle(bireyler, rng, hedef_sayi=50)
    assert len(ornekler) == 50
    assert all(a != b for a, b, _, _ in ornekler)


def test_farkli_indeksleri_ornekle_gecerli_foto_indeksleri_uretir():
    """Bireylerin foto sayıları farklıysa (bazı fotoğraflar ASGARI_KENAR
    altında elenip kısa kalmış olabilir), üretilen indeks o bireyin GERÇEK
    foto sayısını aşmamalı."""
    bireyler = {"a": [0], "b": [0, 1, 2], "c": [0, 1]}
    rng = np.random.default_rng(0)
    ornekler = olcum._farkli_indeksleri_ornekle(bireyler, rng, hedef_sayi=200)
    for a, b, i, j in ornekler:
        assert 0 <= i < len(bireyler[a])
        assert 0 <= j < len(bireyler[b])


def test_farkli_cift_sayisi_birey_sayisindan_bagimsiz_kalir(sahte_veri_koku):
    """BUG DÜZELTMESİ: gerçek MPDD verisiyle (191 birey) ilk denemede farklı-
    birey karşılaştırma sayısı ~290.000'e çıkıp ölçüm 20+ dakikada bitmedi --
    C(n,2) karesel büyüme. MAKS_FARKLI_CIFT ile örnekleme, süreyi birey
    sayısından bağımsız kılmalı. Burada üç bireyle çalışan küçük bir kümede,
    MAKS_FARKLI_CIFT'in olası tüm çiftlerden (C(3,2)=3) küçük olduğu durumu
    değil, tam tersini (istenen sayı mevcut çift sayısını AŞARSA taşma
    olmaması) doğruluyoruz."""
    orijinal = olcum.MAKS_FARKLI_CIFT
    olcum.MAKS_FARKLI_CIFT = 10  # 3 bireyle mümkün olan C(3,2)=3'ten fazla
    try:
        sonuc = olcum.olcumu_calistir(sahte_veri_koku, birey_sayisi=3, foto_sayisi_birey=2)
    finally:
        olcum.MAKS_FARKLI_CIFT = orijinal
    assert 0.0 <= sonuc["tam_auc"] <= 1.0


def test_olcumu_calistir_uctan_uca_calisir_ve_beklenen_seklen_doner(sahte_veri_koku):
    """Gerçek CLIP + gerçek kırpma dedektörü çalıştırır (her ikisi de zaten
    diğer testlerde -- test_saglamlik.py/test_kirpma.py -- gerekli) ama
    SENTETİK renk verisiyle: doğruluk hakkında hiçbir iddiada bulunmaz,
    yalnızca betiğin çökmeden, beklenen şekilde bir sonuç ürettiğini
    doğrular."""
    sonuc = olcum.olcumu_calistir(sahte_veri_koku, birey_sayisi=3, foto_sayisi_birey=2)

    assert sonuc["birey_sayisi"] == 3
    assert 0.0 <= sonuc["tam_auc"] <= 1.0
    assert 0.0 <= sonuc["hayvan_auc"] <= 1.0


def test_olcumu_calistir_veri_yoksa_acikca_hata_verir(tmp_path):
    bos = tmp_path / "bos"
    bos.mkdir()
    with pytest.raises(SystemExit, match="bulunamadı"):
        olcum.olcumu_calistir(bos, birey_sayisi=5, foto_sayisi_birey=2)
