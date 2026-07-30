import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.model_yarisi import gomucu_kur, veri_yukle


VERI_ADI = "CatIndividualImages"
MODEL_ANAHTARI = "google-siglip2"
TOHUM = 42
BIREY_SAYISI = 100
FOTO_SAYISI = 3


def hesapla_hibrit_skor(ham_skor, gercek_eslesme_mi):
    """Ham görsel skoru, ürün skorundaki etiket ve konum katkılarıyla simüle eder."""
    ham_skor = np.asarray(ham_skor, dtype=np.float32)

    # Gerçek eşleşmede aynı kediye baktığımızı varsayıyoruz:
    # etiketler tam örtüşür, konum da aynı ilan/sahip bağlamında tam puan alır.
    if gercek_eslesme_mi:
        etiket_skoru = 1.0
        konum_skoru = 1.0
    else:
        # İbrahim'in zorlu "aynı şehirde iki benzer tekir kedi" senaryosu:
        # etiket katmanı ayırt edemiyor, konum da yüksek ama tam değil.
        etiket_skoru = 1.0
        konum_skoru = 0.80

    return (0.55 * ham_skor) + (0.30 * etiket_skoru) + (0.15 * konum_skoru)


def kimlikleri_bol(kimlikler, tohum):
    """Kimlikleri kayıtlı/yabancı iki açık-küme parçasına deterministik böler."""
    essiz_kimlikler = np.unique(kimlikler)
    if len(essiz_kimlikler) < 2:
        raise ValueError(
            f"Open-set testi için en az 2 farklı kimlik gerekir "
            f"(bulunan: {len(essiz_kimlikler)})."
        )

    # np.random.default_rng global rastgelelik durumunu değiştirmez; aynı tohumla
    # herkes aynı kayıtlı/yabancı ayrımını görür.
    rng = np.random.default_rng(tohum)
    karisik_kimlikler = rng.permutation(essiz_kimlikler)

    # Tek sayıda kimlik varsa fazla olan taraf yabancı kümede kalır; böylece
    # açık-küme yanlış alarm testi olabildiğince zor kalır.
    kayitli_sayisi = len(karisik_kimlikler) // 2
    kayitli_kimlikler = karisik_kimlikler[:kayitli_sayisi]
    yabanci_kimlikler = karisik_kimlikler[kayitli_sayisi:]
    return kayitli_kimlikler, yabanci_kimlikler


def galeri_ve_sorgulari_sec(kimlikler, kayitli_kimlikler, yabanci_kimlikler):
    """Galeri, pozitif sorgu ve negatif sorgu indekslerini üretir."""
    galeri_idx = []
    pozitif_sorgu_idx = []

    # Her kayıtlı kimlik için ilk fotoğraf galeriye gider; aynı kimliğin kalan
    # fotoğrafları TPR ölçen gerçek eşleşme sorguları olur.
    for kimlik in kayitli_kimlikler:
        aday_idx = np.flatnonzero(kimlikler == kimlik)
        if len(aday_idx) < 2:
            # veri_yukle normalde bunu engeller; yine de script başka veriyle
            # çağrılırsa sessizce hatalı TPR üretmeyelim.
            continue
        galeri_idx.append(aday_idx[0])
        pozitif_sorgu_idx.extend(aday_idx[1:])

    # Yabancı kimliklerin tüm fotoğrafları FPR ölçen negatif sorgudur.
    negatif_sorgu_idx = np.flatnonzero(np.isin(kimlikler, yabanci_kimlikler))

    if not galeri_idx:
        raise ValueError("Galeri oluşturulamadı: kayıtlı kimliklerde yeterli fotoğraf yok.")
    if not pozitif_sorgu_idx:
        raise ValueError("Pozitif sorgu oluşturulamadı: galeri dışı kayıtlı fotoğraf yok.")
    if len(negatif_sorgu_idx) == 0:
        raise ValueError("Negatif sorgu oluşturulamadı: yabancı kimlik fotoğrafı yok.")

    return (
        np.asarray(galeri_idx, dtype=np.int64),
        np.asarray(pozitif_sorgu_idx, dtype=np.int64),
        negatif_sorgu_idx.astype(np.int64),
    )


def skor_dagilimlarini_hesapla(vektorler, kimlikler, galeri_idx, pozitif_idx, negatif_idx):
    """Pozitif ve negatif ham görsel skorları vektörel olarak hesaplar."""
    galeri_vektorleri = vektorler[galeri_idx]
    galeri_kimlikleri = kimlikler[galeri_idx]

    pozitif_vektorleri = vektorler[pozitif_idx]
    pozitif_kimlikleri = kimlikler[pozitif_idx]
    negatif_vektorleri = vektorler[negatif_idx]

    # Pozitif sorguda yalnızca galerideki doğru kimlik skorlanır.
    # Bir kimlik -> tek galeri satırı olduğu için eşleşme maskesi tek True içerir.
    dogru_galeri_maskesi = pozitif_kimlikleri[:, None] == galeri_kimlikleri[None, :]
    pozitif_benzerlikler = pozitif_vektorleri @ galeri_vektorleri.T
    pozitif_ham_skorlar = pozitif_benzerlikler[dogru_galeri_maskesi]

    # Negatif sorguda sistemin düşebileceği en büyük tuzağı ölçüyoruz:
    # yabancı fotoğrafın tüm galeriyle benzerliklerinden maksimum olanı.
    negatif_benzerlikler = negatif_vektorleri @ galeri_vektorleri.T
    negatif_ham_skorlar = np.max(negatif_benzerlikler, axis=1)

    return pozitif_ham_skorlar, negatif_ham_skorlar


def tablo_bas(baslik, pozitif_skorlar, negatif_skorlar):
    """Eşik taraması için FPR ve TPR tablosunu basar."""
    esikler = np.round(np.arange(0.50, 0.951, 0.05), 2)

    # Karşılaştırmaları tek seferde matris olarak yapıyoruz. Satır: eşik,
    # sütun: sorgu. Ortalama, ilgili oranın doğrudan vektörel hesabıdır.
    negatif_gecer = negatif_skorlar[None, :] >= esikler[:, None]
    pozitif_gecer = pozitif_skorlar[None, :] >= esikler[:, None]

    yanlis_alarm_sayilari = np.sum(negatif_gecer, axis=1)
    gercek_yakalama_sayilari = np.sum(pozitif_gecer, axis=1)
    fpr_oranlari = yanlis_alarm_sayilari / len(negatif_skorlar)
    tpr_oranlari = gercek_yakalama_sayilari / len(pozitif_skorlar)

    print(f"\n--- {baslik} ---")
    print("| Eşik | Yanlış Alarm (FPR) | Gerçek Yakalama (TPR) |")
    print("|---:|---:|---:|")
    for esik, fp, tp, fpr, tpr in zip(
        esikler,
        yanlis_alarm_sayilari,
        gercek_yakalama_sayilari,
        fpr_oranlari,
        tpr_oranlari,
    ):
        print(
            f"| {esik:.2f} | "
            f"{fp}/{len(negatif_skorlar)} (%{fpr * 100:.1f}) | "
            f"{tp}/{len(pozitif_skorlar)} (%{tpr * 100:.1f}) |"
        )


def main():
    print("--- AÇIK KÜME (OPEN-SET) TPR ve FPR ÖLÇÜM TESTİ ---")

    # 1) Veri kümesini yükle: her kimlikten en fazla FOTO_SAYISI fotoğraf alınır.
    # En az iki fotoğraflı kimlikler seçildiği için bir fotoğrafı galeriye,
    # kalanları pozitif sorguya ayırabiliyoruz.
    df = veri_yukle(
        VERI_ADI,
        ROOT / "data" / VERI_ADI,
        birey_sayisi=BIREY_SAYISI,
        foto_sayisi=FOTO_SAYISI,
        tohum=TOHUM,
    )

    kimlikler = np.asarray(df["identity"].to_numpy())
    yollar = df["tam_yol"].tolist()

    # 2) Kimlikleri açık-küme mantığıyla ikiye böl:
    # kayıtlı kimlikler galeride yer alır, yabancı kimlikler galeride hiç yoktur.
    kayitli_kimlikler, yabanci_kimlikler = kimlikleri_bol(kimlikler, TOHUM)
    galeri_idx, pozitif_idx, negatif_idx = galeri_ve_sorgulari_sec(
        kimlikler, kayitli_kimlikler, yabanci_kimlikler
    )

    print(f"\nVeri kümesi: {VERI_ADI}")
    print(f"Model: {MODEL_ANAHTARI}")
    print(f"Kayıtlı kimlik: {len(kayitli_kimlikler)}")
    print(f"Yabancı kimlik: {len(yabanci_kimlikler)}")
    print(f"Galeri fotoğrafı: {len(galeri_idx)}")
    print(f"Pozitif sorgu fotoğrafı (TPR): {len(pozitif_idx)}")
    print(f"Negatif sorgu fotoğrafı (FPR): {len(negatif_idx)}")

    # 3) Tüm fotoğrafları bir kez vektöre çevir; sonra skor hesapları NumPy ile
    # matris çarpımı üzerinden ilerler.
    print(f"\nModel yükleniyor: {MODEL_ANAHTARI}...")
    gomucu = gomucu_kur(MODEL_ANAHTARI)

    print("Fotoğraflar vektöre çevriliyor...")
    vektorler = gomucu.kodla(yollar)

    pozitif_ham_skorlar, negatif_ham_skorlar = skor_dagilimlarini_hesapla(
        vektorler, kimlikler, galeri_idx, pozitif_idx, negatif_idx
    )

    # 4) Hibrit skor simülasyonu: pozitiflerde gerçek eşleşme varsayımı,
    # negatiflerde "aynı şehirde benzer tekir" gibi zor yanlış alarm senaryosu.
    pozitif_hibrit_skorlar = hesapla_hibrit_skor(
        pozitif_ham_skorlar, gercek_eslesme_mi=True
    )
    negatif_hibrit_skorlar = hesapla_hibrit_skor(
        negatif_ham_skorlar, gercek_eslesme_mi=False
    )

    # 5) İki ayrı eşik taraması: biri yalnızca ham görsel skor, diğeri hibrit skor.
    tablo_bas("TABLO 1: SADECE HAM GÖRSEL SKOR", pozitif_ham_skorlar, negatif_ham_skorlar)
    tablo_bas("TABLO 2: HİBRİT SKOR", pozitif_hibrit_skorlar, negatif_hibrit_skorlar)


if __name__ == "__main__":
    main()
