import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.model_yarisi import gomucu_kur, veri_yukle

def main():
    print("--- AÇIK KÜME (OPEN-SET) 'EŞLEŞME YOK' TESTİ ---")
    
    # 1. VERİ VE MODEL YÜKLEME
    # DogFaceNet yerine CatIndividualImages kullanıyoruz
    df = veri_yukle("CatIndividualImages", ROOT / "data" / "CatIndividualImages", birey_sayisi=100, foto_sayisi=3, tohum=42)
    
    model_anahtari = "google-siglip2"
    print(f"\nŞampiyon Model Yükleniyor: {model_anahtari}...")
    gomucu = gomucu_kur(model_anahtari)
    
    yollar = df["tam_yol"].tolist()
    kimlikler = np.array(df["identity"].tolist())
    
    print("Fotoğraflar Vektöre Çevriliyor (Bu işlem birkaç saniye sürebilir)...")
    vektorler = gomucu.kodla(yollar)
    
    # 2. VERİYİ "GALERİ" VE "YABANCI SORGULAR" OLARAK İKİYE BÖLME
    essiz_kimlikler = np.unique(kimlikler)
    if len(essiz_kimlikler) < 2:
        print(f"HATA: Open-set testi için en az 2 farklı kimlik gerekir (bulunan: {len(essiz_kimlikler)}).")
        return
    np.random.seed(42)
    np.random.shuffle(essiz_kimlikler)
    # Hayvanların yarısı veri tabanında (Galeri) var, yarısı ise tamamen YABANCI
    yari_nokta = len(essiz_kimlikler) // 2
    kayitli_kimlikler = set(essiz_kimlikler[:yari_nokta])
    
    galeri_idx = [i for i, k in enumerate(kimlikler) if k in kayitli_kimlikler]
    sorgu_idx  = [i for i, k in enumerate(kimlikler) if k not in kayitli_kimlikler]
    
    galeri_vektorleri = vektorler[galeri_idx]
    
    yabanci_sorgu_vektorleri = vektorler[sorgu_idx]
    print(f"\nSistemdeki (Galerideki) Hayvan Sayısı: {len(kayitli_kimlikler)} (Toplam {len(galeri_idx)} fotoğraf)")
    print(f"Sisteme Yabancı (Eşi Olmayan) Hayvan Sayısı: {len(essiz_kimlikler) - len(kayitli_kimlikler)} (Toplam {len(sorgu_idx)} fotoğraf)")
    
    # 3. YABANCI SORGULARI GALERİYLE KARŞILAŞTIRMA (TEST AŞAMASI)
    # Nokta çarpımı (cosine similarity) ile tüm yabancı fotoğrafları galeriyle kıyaslıyoruz
    benzerlik_matrisi = yabanci_sorgu_vektorleri @ galeri_vektorleri.T
    
    # Sistemin yabancı bir hayvana verdiği "EN YÜKSEK" benzerlik skorunu buluyoruz
    en_yuksek_skorlar = np.max(benzerlik_matrisi, axis=1)
    
    toplam_yabanci_sorgu = len(en_yuksek_skorlar)

    # Sabit 0.70 yerine ham görsel skorda 0.50-0.95 arası eşik taraması yapıyoruz.
    # Not: Uygulamadaki 0.70 eşiği hibrit final skora uygulanır; bu tablo ham
    # görsel maksimum skorun açık-küme davranışını okumak içindir.
    esikler = np.round(np.arange(0.50, 0.951, 0.05), 2)

    print("\n--- EŞİK TARAMASI SONUÇLARI (Ham Görsel Maksimum Skor) ---")
    print(f"Test Edilen Yabancı Fotoğraf Sayısı: {toplam_yabanci_sorgu}")
    print()
    print(f"{'Eşik':>6} | {'Yanlış Alarm':>13} | {'False Positive Rate':>20} | {'True Negative Rate':>18}")
    print("-" * 68)

    for esik_degeri in esikler:
        # Yabancı bir hayvana eşik ve üzeri skor verilirse bu bir YANLIŞ ALARMDIR.
        yanlis_alarmlar = np.sum(en_yuksek_skorlar >= esik_degeri)
        yanlis_alarm_orani = (yanlis_alarmlar / toplam_yabanci_sorgu) * 100
        true_negative_orani = 100 - yanlis_alarm_orani

        print(
            f"{esik_degeri:>6.2f} | "
            f"{yanlis_alarmlar:>5}/{toplam_yabanci_sorgu:<7} | "
            f"%{yanlis_alarm_orani:>18.1f} | "
            f"%{true_negative_orani:>16.1f}"
        )

if __name__ == "__main__":
    main()
