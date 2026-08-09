import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VERI_ADI = "CatIndividualImages"
MODEL_ANAHTARI = "google-siglip2"
TOHUM = 42
BIREY_SAYISI = 100
FOTO_SAYISI = 3

# compute_final_score embedding boyutunu app.surum uzerinden dogrular.
# Bu test google-siglip2 vektorleri urettigi icin matcher'i ayni kimlik
# modeliyle yukluyoruz.
os.environ["KIMLIK_MODEL"] = MODEL_ANAHTARI

from app.matcher import compute_final_score  # noqa: E402
from scripts.model_yarisi import gomucu_kur, veri_yukle  # noqa: E402


# Skorlayiciya verilen etiket/konum girdileri kor sabittir; ayni kedi olup
# olmadigina gore degismez. Etiketler 1/6 Jaccard = 0.1667 uretir.
KOR_ETIKETLER_A = ["cat"]
KOR_ETIKETLER_B = ["cat", "tabby", "short_hair", "green_eyes", "adult", "outdoor"]
KOR_MESAFE_KM = 5.0
KOR_TUR = "cat"


def kimlikleri_bol(kimlikler, tohum):
    """Kimlikleri kayitli/yabanci iki acik-kume parcasina deterministik boler."""
    essiz_kimlikler = np.unique(kimlikler)
    if len(essiz_kimlikler) < 2:
        raise ValueError(
            f"Open-set testi icin en az 2 farkli kimlik gerekir "
            f"(bulunan: {len(essiz_kimlikler)})."
        )

    # np.random.default_rng global rastgelelik durumunu degistirmez; ayni tohumla
    # herkes ayni kayitli/yabanci ayrimini gorur.
    rng = np.random.default_rng(tohum)
    karisik_kimlikler = rng.permutation(essiz_kimlikler)

    # Tek sayida kimlik varsa fazla olan taraf yabanci kumede kalir; boylece
    # acik-kume yanlis alarm testi olabildigince zor kalir.
    kayitli_sayisi = len(karisik_kimlikler) // 2
    kayitli_kimlikler = karisik_kimlikler[:kayitli_sayisi]
    yabanci_kimlikler = karisik_kimlikler[kayitli_sayisi:]
    return kayitli_kimlikler, yabanci_kimlikler


def galeri_ve_sorgulari_sec(kimlikler, kayitli_kimlikler, yabanci_kimlikler):
    """Galeri, pozitif sorgu ve negatif sorgu indekslerini uretir."""
    galeri_idx = []
    pozitif_sorgu_idx = []

    # Her kayitli kimlik icin ilk fotograf galeriye gider; ayni kimligin kalan
    # fotograflari TPR olcen gercek eslesme sorgulari olur.
    for kimlik in kayitli_kimlikler:
        aday_idx = np.flatnonzero(kimlikler == kimlik)
        if len(aday_idx) < 2:
            # veri_yukle normalde bunu engeller; yine de script baska veriyle
            # cagrilirsa sessizce hatali TPR uretmeyelim.
            continue
        galeri_idx.append(aday_idx[0])
        pozitif_sorgu_idx.extend(aday_idx[1:])

    # Yabanci kimliklerin tum fotograflari FPR olcen negatif sorgudur.
    negatif_sorgu_idx = np.flatnonzero(np.isin(kimlikler, yabanci_kimlikler))

    if not galeri_idx:
        raise ValueError("Galeri olusturulamadi: kayitli kimliklerde yeterli fotograf yok.")
    if not pozitif_sorgu_idx:
        raise ValueError("Pozitif sorgu olusturulamadi: galeri disi kayitli fotograf yok.")
    if len(negatif_sorgu_idx) == 0:
        raise ValueError("Negatif sorgu olusturulamadi: yabanci kimlik fotografi yok.")

    return (
        np.asarray(galeri_idx, dtype=np.int64),
        np.asarray(pozitif_sorgu_idx, dtype=np.int64),
        negatif_sorgu_idx.astype(np.int64),
    )


def vektor_listeye_cevir(vektor):
    """Matcher'in tek-fotograf sozlesmesine uygun Python listesi dondurur."""
    return np.asarray(vektor, dtype=np.float32).tolist()


def motor_sonucu_hesapla(sorgu_vektoru, galeri_vektoru):
    """Gercek urun motorunu kor etiket/konum girdileriyle cagirir."""
    return compute_final_score(
        embeddings_a=vektor_listeye_cevir(sorgu_vektoru),
        embeddings_b=vektor_listeye_cevir(galeri_vektoru),
        labels_a=KOR_ETIKETLER_A,
        labels_b=KOR_ETIKETLER_B,
        distance_km=KOR_MESAFE_KM,
        species_a=KOR_TUR,
        species_b=KOR_TUR,
    )


def tum_galeri_skorlari_hesapla(sorgu_vektoru, galeri_vektorleri):
    """Bir sorguyu galerideki tum adaylara karsi gercek motorla skorlar."""
    ham_gorsel_skorlar = []
    hibrit_skorlar = []

    for galeri_vektoru in galeri_vektorleri:
        sonuc = motor_sonucu_hesapla(sorgu_vektoru, galeri_vektoru)
        ham_gorsel_skorlar.append(float(sonuc["visual"]))
        hibrit_skorlar.append(float(sonuc["score"]))

    return (
        np.asarray(ham_gorsel_skorlar, dtype=np.float32),
        np.asarray(hibrit_skorlar, dtype=np.float32),
    )


def skor_dagilimlarini_hesapla(vektorler, kimlikler, galeri_idx, pozitif_idx, negatif_idx):
    """Pozitif/negatif ham gorsel ve gercek motor hibrit skorlarini hesaplar."""
    galeri_vektorleri = vektorler[galeri_idx]
    galeri_kimlikleri = kimlikler[galeri_idx]
    galeri_sirasi = {kimlik: i for i, kimlik in enumerate(galeri_kimlikleri)}

    pozitif_ham_skorlar = []
    pozitif_hibrit_skorlar = []
    negatif_ham_skorlar = []
    negatif_hibrit_skorlar = []

    # Pozitif sorguda yalnizca galerideki dogru kimlik olculur. Dogru kimligi
    # secmek metrik duzeneginin isidir; motor_sonucu_hesapla bu bilgiyi almaz.
    for sorgu_idx in pozitif_idx:
        kimlik = kimlikler[sorgu_idx]
        if kimlik not in galeri_sirasi:
            raise ValueError(f"Pozitif sorgu icin galeride kimlik bulunamadi: {kimlik}")

        galeri_sira = galeri_sirasi[kimlik]
        sonuc = motor_sonucu_hesapla(vektorler[sorgu_idx], galeri_vektorleri[galeri_sira])
        pozitif_ham_skorlar.append(float(sonuc["visual"]))
        pozitif_hibrit_skorlar.append(float(sonuc["score"]))

    # Negatif sorguda sistemin dusebilecegi en buyuk tuzagi olcuyoruz:
    # yabanci fotografin tum galeriyle skorlarindan maksimum olanlari.
    for sorgu_idx in negatif_idx:
        ham_skorlar, hibrit_skorlar = tum_galeri_skorlari_hesapla(
            vektorler[sorgu_idx], galeri_vektorleri
        )
        negatif_ham_skorlar.append(float(np.max(ham_skorlar)))
        negatif_hibrit_skorlar.append(float(np.max(hibrit_skorlar)))

    return (
        np.asarray(pozitif_ham_skorlar, dtype=np.float32),
        np.asarray(negatif_ham_skorlar, dtype=np.float32),
        np.asarray(pozitif_hibrit_skorlar, dtype=np.float32),
        np.asarray(negatif_hibrit_skorlar, dtype=np.float32),
    )


def tablo_bas(baslik, pozitif_skorlar, negatif_skorlar):
    """Esik taramasi icin FPR ve TPR tablosunu basar."""
    esikler = np.round(np.arange(0.50, 0.951, 0.05), 2)

    # Karsilastirmalari tek seferde matris olarak yapiyoruz. Satir: esik,
    # sutun: sorgu. Ortalama, ilgili oranin dogrudan vektorel hesabidir.
    negatif_gecer = negatif_skorlar[None, :] >= esikler[:, None]
    pozitif_gecer = pozitif_skorlar[None, :] >= esikler[:, None]

    yanlis_alarm_sayilari = np.sum(negatif_gecer, axis=1)
    gercek_yakalama_sayilari = np.sum(pozitif_gecer, axis=1)
    fpr_oranlari = yanlis_alarm_sayilari / len(negatif_skorlar)
    tpr_oranlari = gercek_yakalama_sayilari / len(pozitif_skorlar)

    print(f"\n--- {baslik} ---")
    print("| Esik | FPR (Yanlis Alarm) | TPR (Gercek Yakalama) |")
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
    print("--- ACIK KUME (OPEN-SET) TPR ve FPR OLCUM TESTI ---")

    # 1) Veri kumesini yukle: her kimlikten en fazla FOTO_SAYISI fotograf alinir.
    # En az iki fotograflik kimlikler secildigi icin bir fotografi galeriye,
    # kalanlari pozitif sorguya ayirabiliyoruz.
    df = veri_yukle(
        VERI_ADI,
        ROOT / "data" / VERI_ADI,
        birey_sayisi=BIREY_SAYISI,
        foto_sayisi=FOTO_SAYISI,
        tohum=TOHUM,
    )

    kimlikler = np.asarray(df["identity"].to_numpy())
    yollar = df["tam_yol"].tolist()

    # 2) Kimlikleri acik-kume mantigiyla ikiye bol:
    # kayitli kimlikler galeride yer alir, yabanci kimlikler galeride hic yoktur.
    kayitli_kimlikler, yabanci_kimlikler = kimlikleri_bol(kimlikler, TOHUM)
    galeri_idx, pozitif_idx, negatif_idx = galeri_ve_sorgulari_sec(
        kimlikler, kayitli_kimlikler, yabanci_kimlikler
    )

    print(f"\nVeri kumesi: {VERI_ADI}")
    print(f"Model: {MODEL_ANAHTARI}")
    print(f"Kayitli kimlik: {len(kayitli_kimlikler)}")
    print(f"Yabanci kimlik: {len(yabanci_kimlikler)}")
    print(f"Galeri fotografi: {len(galeri_idx)}")
    print(f"Pozitif sorgu fotografi (TPR): {len(pozitif_idx)}")
    print(f"Negatif sorgu fotografi (FPR): {len(negatif_idx)}")
    print("Gercek motor girdileri: etiket Jaccard 1/6 (0.1667), "
          f"konum mesafesi {KOR_MESAFE_KM:.1f} km; ikisi de kor sabit.")

    # 3) Tum fotograflari bir kez vektore cevir; sonra skorlar urun motorunun
    # compute_final_score fonksiyonuyla hesaplanir.
    print(f"\nModel yukleniyor: {MODEL_ANAHTARI}...")
    gomucu = gomucu_kur(MODEL_ANAHTARI)

    print("Fotograflar vektore cevriliyor...")
    vektorler = gomucu.kodla(yollar)

    (
        pozitif_ham_skorlar,
        negatif_ham_skorlar,
        pozitif_hibrit_skorlar,
        negatif_hibrit_skorlar,
    ) = skor_dagilimlarini_hesapla(vektorler, kimlikler, galeri_idx, pozitif_idx, negatif_idx)

    # 4) Iki ayri esik taramasi: biri yalnizca ham gorsel skor, digeri gercek
    # motorun hibrit skoru. Esikler 0.50-0.95 araliginda 0.05 adimlidir.
    tablo_bas("TABLO 1: SADECE HAM GORSEL SKOR", pozitif_ham_skorlar, negatif_ham_skorlar)
    tablo_bas(
        "TABLO 2: GERCEK MOTOR - HIBRIT SKOR",
        pozitif_hibrit_skorlar,
        negatif_hibrit_skorlar,
    )


if __name__ == "__main__":
    main()
