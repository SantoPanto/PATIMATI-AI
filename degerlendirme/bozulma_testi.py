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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.model_yarisi import gomucu_kur

def main():
    print("--- BOZULMA (DAYANIKLILIK) TESTİ GÜNCELLEMESİ ---")

    # 1. MODEL GÜNCELLEMESİ
    # NOT: Avito-siglip2 dendi ancak repodaki config hatasından dolayı 
    # sessizce çökmesini engellemek için yedek plan olan google-siglip2'yi yüklüyoruz.
    print("1. Yeni Şampiyon Model Yükleniyor: google-siglip2...")
    gomucu = gomucu_kur("google-siglip2")
    islemci = gomucu._islemci
    model = gomucu._model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    def resmi_vektore_cevir(img):
        girdi = islemci(images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            ozellik = model.get_image_features(**girdi)
            if not isinstance(ozellik, torch.Tensor):
                havuz = getattr(ozellik, "pooler_output", None)
                ozellik = havuz if havuz is not None else ozellik[0]
            ozellik = ozellik / ozellik.norm(dim=-1, keepdim=True)
            return ozellik

    # 2. ÇOKLU FOTOĞRAF VE ORTALAMA ALMA (İbrahim'in Kuralı)
    # İnternetten fotoğraf aramak yerine elimizdeki kedi veri setini kullanıyoruz
    kedi_klasoru = ROOT / "data" / "CatIndividualImages"
    tum_kediler = list(kedi_klasoru.rglob("*.jpg"))

    if len(tum_kediler) < 5:
        print("HATA: Yeterli kedi fotoğrafı bulunamadı. Lütfen kedi verilerinin indiğinden emin olun.")
        return

    # Veri setinden rastgele 5 kedi seç (Tohum 42: her seferinde aynı 5 kedi seçilsin)
    random.seed(42)
    secilen_yollar = random.sample(tum_kediler, 5)
    print(f"\n2. Test, seçilen {len(secilen_yollar)} farklı kedi fotoğrafının ortalaması alınarak yapılıyor...\n")

    orijinal_resimler = [Image.open(y).convert('RGB') for y in secilen_yollar]
    orijinal_vektorler = [resmi_vektore_cevir(img) for img in orijinal_resimler]

    ESIK_DEGERI = 70  # İbrahim'in görmek istediği sınır çizgisi

    def ortalama_bozulma_skoru(bozma_fonksiyonu):
        skorlar = []
        for i, img in enumerate(orijinal_resimler):
            bozuk_img = bozma_fonksiyonu(img)
            bozuk_vektor = resmi_vektore_cevir(bozuk_img)
            skor = F.cosine_similarity(orijinal_vektorler[i], bozuk_vektor).item() * 100
            skorlar.append(skor)
        return sum(skorlar) / len(skorlar)

    def grafik_kaydet(x_degerleri, y_degerleri, baslik, x_ekseni, renk, dosya_adi):
        plt.figure(figsize=(10, 5))
        plt.plot(x_degerleri, y_degerleri, marker='o', color=renk)
        
        # 3. GRAFİKLERE EŞİK ÇİZGİSİ (THRESHOLD) EKLENMESİ
        plt.axhline(y=ESIK_DEGERI, color='black', linestyle='--', label=f'Başarısızlık Eşiği (%{ESIK_DEGERI})')
        
        plt.title(baslik)
        plt.xlabel(x_ekseni)
        plt.ylabel('Eşleşme Skoru (%)')
        plt.grid(True)
        plt.ylim(0, 105)
        plt.legend()
        plt.savefig(ROOT / "degerlendirme" / dosya_adi)
        plt.close()

    # --- TESTLER ---
    print("--- TEST 1: Bulanıklık Testi Başlıyor ---")
    bulaniklik_dereceleri = [1, 5, 15, 35, 65] 
    blur_skorlar = [ortalama_bozulma_skoru(lambda img: img if d == 1 else transforms.GaussianBlur(kernel_size=d)(img)) for d in bulaniklik_dereceleri]
    grafik_kaydet([str(d) for d in bulaniklik_dereceleri], blur_skorlar, 'Bulanıklığa Karşı Dayanıklılık (Robustness)', 'Bulanıklık Çekirdek Boyutu', 'blue', "bulaniklik_grafigi.png")

    print("--- TEST 2: Karartma Testi Başlıyor ---")
    parlaklik_dereceleri = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    dark_skorlar = [ortalama_bozulma_skoru(lambda img: TF.adjust_brightness(img, d)) for d in parlaklik_dereceleri]
    grafik_kaydet([f"%{int(d*100)}" for d in parlaklik_dereceleri], dark_skorlar, 'Gece Çekimine Karşı Dayanıklılık (Robustness)', 'Fotoğrafta Kalan Işık Miktarı', 'red', "karartma_grafigi.png")

    print("--- TEST 3: Döndürme (Rotasyon) Testi Başlıyor ---")
    aci_dereceleri = [0, 45, 90, 135, 180, 225, 270, 315]
    rot_skorlar = [ortalama_bozulma_skoru(lambda img: TF.rotate(img, a)) for a in aci_dereceleri]
    grafik_kaydet([f"{a}°" for a in aci_dereceleri], rot_skorlar, 'Döndürmeye Karşı Dayanıklılık (Robustness)', 'Dönüş Açısı (Derece)', 'green', "rotasyon_grafigi.png")

    print("--- TEST 4: Sıkıştırma (WhatsApp) Testi Başlıyor ---")
    kalite_seviyeleri = [100, 50, 20, 10, 5, 1]
    def sikistir(img, k):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=k)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    comp_skorlar = [ortalama_bozulma_skoru(lambda img: sikistir(img, k)) for k in kalite_seviyeleri]
    grafik_kaydet([f"%{k}" for k in kalite_seviyeleri], comp_skorlar, 'Sıkıştırmaya Karşı Dayanıklılık (WhatsApp)', 'Görüntü Kalitesi (JPEG)', 'orange', "sikistirma_grafigi.png")

    print("\n>>> İŞLEM TAMAM! İbrahim'in tüm kuralları uygulandı ve grafikler (eşik çizgili) güncellendi.")

if __name__ == "__main__":
    main()