import sys
import io
import random
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

import os
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.model_yarisi import gomucu_kur

# app modüllerini yüklemeden önce boyut yapılandırmasını ayarla
os.environ["KIMLIK_MODEL"] = "google-siglip2"

from app.matcher import compute_final_score
from app.attributes import AttributeAnalyzer

def main():
    print("--- BOZULMA (DAYANIKLILIK) TESTİ GÜNCELLEMESİ (Canlı Sistem Simülasyonu) ---")

    print("1. Modeller Yükleniyor: google-siglip2 (Kimlik) ve CLIP (Etiket)...")
    gomucu = gomucu_kur("google-siglip2")
    islemci = getattr(gomucu, "_islemci", None)
    model = getattr(gomucu, "_model", None)
    
    analyzer = AttributeAnalyzer()

    if islemci is None or model is None:
        raise TypeError(f"Beklenen HF gömücü bulunamadı: {type(gomucu).__name__} (google-siglip2 hf olmalı).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    def analizi_yap(img):
        # 1. Kimlik Vektörü
        girdi = islemci(images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            ozellik = model.get_image_features(**girdi)
            if not isinstance(ozellik, torch.Tensor):
                havuz = getattr(ozellik, "pooler_output", None)
                ozellik = havuz if havuz is not None else ozellik[0]
            ozellik = ozellik / ozellik.norm(dim=-1, keepdim=True)
            vektor = ozellik.squeeze().cpu().tolist()
            
        # 2. Etiket (Label) Çıkarımı
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        sonuc = analyzer.analyze(image_bytes)
        return vektor, sonuc["labels"], sonuc["species"]

    # 2. Çoklu Fotoğraf Seçimi
    kedi_klasoru = ROOT / "data" / "CatIndividualImages"
    tum_kediler = list(kedi_klasoru.rglob("*.jpg"))

    if len(tum_kediler) < 5:
        print("Uyarı: Kedi verisi bulunamadı. Yedek olarak DogFaceNet kullanılıyor...")
        kopek_klasoru = ROOT / "data" / "DogFaceNet" / "after_4_bis"
        tum_kediler = list(kopek_klasoru.rglob("*.jpg"))
        
        if len(tum_kediler) < 5:
            print("HATA: Ne kedi ne de köpek verisi bulunamadı.")
            return

    random.seed(42)
    secilen_yollar = random.sample(tum_kediler, 5)
    print(f"\n2. Test, seçilen {len(secilen_yollar)} farklı kedi fotoğrafı üzerinde çalışıyor...\n")

    orijinal_resimler = [Image.open(y).convert('RGB') for y in secilen_yollar]
    
    print("Orijinal fotoğraflar analiz ediliyor...")
    orijinal_veriler = [analizi_yap(img) for img in orijinal_resimler]

    # Farklı kedilerin skorlarına bakarak FPR eşiği bulma
    print("Farklı kedilerin çapraz eşleşmeleri (False Positive) test ediliyor...")
    yabanci_skorlar = []
    for i in range(len(orijinal_veriler)):
        for j in range(i+1, len(orijinal_veriler)):
            ori_vektor1, ori_labels1, ori_species1 = orijinal_veriler[i]
            ori_vektor2, ori_labels2, ori_species2 = orijinal_veriler[j]
            res = compute_final_score(
                [ori_vektor1], [ori_vektor2], ori_labels1, ori_labels2, 0.0, ori_species1, ori_species2
            )
            yabanci_skorlar.append(res["score"])
            
    max_yabanci = max(yabanci_skorlar) * 100 if yabanci_skorlar else 0
    print(f">> Yabancı kedilerin birbirleriyle aldığı en yüksek skor: %{max_yabanci:.1f}")
    
    # Eşik değerini yabancı kedilerin alabildiği max skorun biraz üzerine koyuyoruz
    ESIK_DEGERI = round(max_yabanci + 2.0, 1) 
    print(f">> GÜVENLİ ÜRETİM EŞİĞİ (MATCH_THRESHOLD) TAVSİYESİ: %{ESIK_DEGERI}\n")

    def ortalama_bozulma_skoru(bozma_fonksiyonu):
        skorlar = []
        for i, img in enumerate(orijinal_resimler):
            bozuk_img = bozma_fonksiyonu(img)
            bozuk_vektor, bozuk_labels, bozuk_species = analizi_yap(bozuk_img)
            
            ori_vektor, ori_labels, ori_species = orijinal_veriler[i]
            
            result = compute_final_score(
                [ori_vektor], [bozuk_vektor], 
                ori_labels, bozuk_labels, 
                0.0, ori_species, bozuk_species
            )
            skorlar.append(result["score"] * 100)
        return sum(skorlar) / len(skorlar)

    def grafik_kaydet(x_degerleri, y_degerleri, baslik, x_ekseni, renk, dosya_adi):
        plt.figure(figsize=(10, 5))
        plt.plot(x_degerleri, y_degerleri, marker='o', color=renk)
        
        # Grafiklere Dinamik Eşik Çizgisi Ekle
        plt.axhline(y=ESIK_DEGERI, color='black', linestyle='--', label=f'Hesaplanan Güvenli Eşik (%{ESIK_DEGERI})')
        # Eski hatalı eşiği de gösterelim
        plt.axhline(y=95, color='red', linestyle=':', label='Eski Yanlış Eşik (%95)')
        
        plt.title(baslik)
        plt.xlabel(x_ekseni)
        plt.ylabel('Canlı Sistem Skoru (%)')
        plt.grid(True)
        plt.ylim(0, 105)
        plt.legend()
        plt.savefig(ROOT / "degerlendirme" / dosya_adi)
        plt.close()

    # --- TESTLER ---
    print("--- TEST 1: Bulanıklık Testi Başlıyor ---")
    bulaniklik_dereceleri = [1, 5, 15, 35, 65] 
    blur_skorlar = [ortalama_bozulma_skoru(lambda img: img if d == 1 else transforms.GaussianBlur(kernel_size=d)(img)) for d in bulaniklik_dereceleri]
    grafik_kaydet([str(d) for d in bulaniklik_dereceleri], blur_skorlar, 'Bulanıklığa Karşı (Canlı Sistem)', 'Bulanıklık Çekirdek Boyutu', 'blue', "bulaniklik_grafigi.png")

    print("--- TEST 2: Karartma Testi Başlıyor ---")
    parlaklik_dereceleri = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    dark_skorlar = [ortalama_bozulma_skoru(lambda img: TF.adjust_brightness(img, d)) for d in parlaklik_dereceleri]
    grafik_kaydet([f"%{int(d*100)}" for d in parlaklik_dereceleri], dark_skorlar, 'Gece Çekimine Karşı (Canlı Sistem)', 'Fotoğrafta Kalan Işık Miktarı', 'red', "karartma_grafigi.png")

    print("--- TEST 3: Döndürme (Rotasyon) Testi Başlıyor ---")
    aci_dereceleri = [0, 45, 90, 135, 180, 225, 270, 315]
    rot_skorlar = [ortalama_bozulma_skoru(lambda img: TF.rotate(img, a)) for a in aci_dereceleri]
    grafik_kaydet([f"{a}°" for a in aci_dereceleri], rot_skorlar, 'Döndürmeye Karşı (Canlı Sistem)', 'Dönüş Açısı (Derece)', 'green', "rotasyon_grafigi.png")

    print("--- TEST 4: Sıkıştırma (WhatsApp) Testi Başlıyor ---")
    kalite_seviyeleri = [100, 50, 20, 10, 5, 1]
    def sikistir(img, k):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=k)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    comp_skorlar = [ortalama_bozulma_skoru(lambda img: sikistir(img, k)) for k in kalite_seviyeleri]
    grafik_kaydet([f"%{k}" for k in kalite_seviyeleri], comp_skorlar, 'Sıkıştırmaya Karşı (Canlı Sistem)', 'Görüntü Kalitesi (JPEG)', 'orange', "sikistirma_grafigi.png")

    print("\n>>> İŞLEM TAMAM! Tüm grafikler canlı sistem (Matcher) üzerinden yeniden çizildi ve üretim eşiği hesaplandı.")

if __name__ == "__main__":
    main()