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
    yabanci_sorgu_kimlikleri = kimlikler[sorgu_idx]
    
    print(f"\nSistemdeki (Galerideki) Hayvan Sayısı: {len(kayitli_kimlikler)} (Toplam {len(galeri_idx)} fotoğraf)")
    print(f"Sisteme Yabancı (Eşi Olmayan) Hayvan Sayısı: {len(essiz_kimlikler) - len(kayitli_kimlikler)} (Toplam {len(sorgu_idx)} fotoğraf)")
    
    # 3. YABANCI SORGULARI GALERİYLE KARŞILAŞTIRMA (TEST AŞAMASI)
    # Nokta çarpımı (cosine similarity) ile tüm yabancı fotoğrafları galeriyle kıyaslıyoruz
    benzerlik_matrisi = yabanci_sorgu_vektorleri @ galeri_vektorleri.T
    
    # Sistemin yabancı bir hayvana verdiği "EN YÜKSEK" benzerlik skorunu buluyoruz
    en_yuksek_skorlar = np.max(benzerlik_matrisi, axis=1)
    
    esik_degeri = 0.70  # İbrahim'in raporundaki bildirim eşiği
    
    # Yabancı bir hayvana %70 ve üzeri skor verirse bu bir YANLIŞ ALARMDIR (False Positive)
    yanlis_alarmlar = np.sum(en_yuksek_skorlar >= esik_degeri)
    toplam_yabanci_sorgu = len(en_yuksek_skorlar)
    yanlis_alarm_orani = (yanlis_alarmlar / toplam_yabanci_sorgu) * 100
    
    print(f"\n--- SONUÇLAR (Eşik Değeri: %{int(esik_degeri*100)}) ---")
    print(f"Test Edilen Yabancı Fotoğraf Sayısı: {toplam_yabanci_sorgu}")
    print(f"Sistemin Başka Bir Hayvana Benzetip 'Eşleşti!' Diyerek Hata Yaptığı: {yanlis_alarmlar}")
    print(f"Yanlış Alarm Oranı (False Positive Rate): %{yanlis_alarm_orani:.1f}")
    
    print("\n>>> SİSTEMİN GÜVENLE 'EŞLEŞME YOK' DİYEBİLME BAŞARISI (True Negative Rate):")
    print(f">>> %{100 - yanlis_alarm_orani:.1f} <<<")

if __name__ == "__main__":
    main()