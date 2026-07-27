import io
import torch
import timm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

# --- GLOBAL AYARLAR VE MODEL YÜKLEME ---
ROOT = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. AI Modeli Yükleniyor...")
model = timm.create_model('hf_hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
model.eval()
model.to(device)

data_cfg = timm.data.resolve_model_data_config(model)
embed_transform = timm.data.create_transform(**data_cfg, is_training=False)

def resmi_vektore_cevir(img):
    tensor_resim = embed_transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(tensor_resim)

def main():
    # --- TEST HAZIRLIK ---
    orijinal_img = Image.open(ROOT / "test_resim.jpg").convert('RGB')
    orijinal_vektor = resmi_vektore_cevir(orijinal_img)
    
    # ==========================================
    # TEST 1: BULANIKLIK (BLUR) TESTİ
    # ==========================================
    print("\n--- TEST 1: Bulanıklık Testi Başlıyor ---")
    bulaniklik_dereceleri = [1, 5, 15, 35, 65] 
    blur_skorlar = []
    for derece in bulaniklik_dereceleri:
        bozuk_img = orijinal_img if derece == 1 else transforms.GaussianBlur(kernel_size=derece)(orijinal_img)
        skor = F.cosine_similarity(orijinal_vektor, resmi_vektore_cevir(bozuk_img)).item() * 100
        blur_skorlar.append(skor)
        print(f"[Bulanıklık Seviyesi {derece}] -> Skor: %{skor:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot([str(d) for d in bulaniklik_dereceleri], blur_skorlar, marker='o', color='blue')
    plt.title('Yapay Zeka Bulanıklık Dayanıklılık Testi')
    plt.xlabel('Bulanıklık Çekirdek Boyutu')
    plt.ylabel('Eşleşme Skoru (%)')
    plt.grid(True); plt.ylim(0, 105)
    plt.savefig(ROOT / "bulaniklik_grafigi.png")
    plt.close() # Bellek sızıntısı önlendi

    # ==========================================
    # TEST 2: KARANLIK (GECE ÇEKİMİ) TESTİ
    # ==========================================
    print("\n--- TEST 2: Karartma Testi Başlıyor ---")
    parlaklik_dereceleri = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    dark_skorlar = []
    for derece in parlaklik_dereceleri:
        kararan_img = TF.adjust_brightness(orijinal_img, derece)
        skor = F.cosine_similarity(orijinal_vektor, resmi_vektore_cevir(kararan_img)).item() * 100
        dark_skorlar.append(skor)
        print(f"[Parlaklık %{int(derece*100)}] -> Skor: %{skor:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot([f"%{int(d*100)}" for d in parlaklik_dereceleri], dark_skorlar, marker='o', color='red')
    plt.title('Yapay Zeka Gece Çekimi Dayanıklılık Testi')
    plt.xlabel('Fotoğrafta Kalan Işık Miktarı')
    plt.ylabel('Eşleşme Skoru (%)')
    plt.grid(True); plt.ylim(0, 105)
    plt.savefig(ROOT / "karartma_grafigi.png")
    plt.close()

    # ==========================================
    # TEST 3: DÖNDÜRME (ROTASYON) TESTİ
    # ==========================================
    print("\n--- TEST 3: Döndürme (Rotasyon) Testi Başlıyor ---")
    aci_dereceleri = [0, 45, 90, 135, 180, 225, 270, 315]
    rot_skorlar = []
    for aci in aci_dereceleri:
        donen_img = TF.rotate(orijinal_img, aci)
        skor = F.cosine_similarity(orijinal_vektor, resmi_vektore_cevir(donen_img)).item() * 100
        rot_skorlar.append(skor)
        print(f"[Dönüş Açısı {aci}°] -> Skor: %{skor:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot([f"{a}°" for a in aci_dereceleri], rot_skorlar, marker='o', color='green')
    plt.title('Yapay Zeka Döndürme (Rotasyon) Dayanıklılık Testi')
    plt.xlabel('Dönüş Açısı (Derece)')
    plt.ylabel('Eşleşme Skoru (%)')
    plt.grid(True); plt.ylim(0, 105)
    plt.savefig(ROOT / "rotasyon_grafigi.png")
    plt.close()

    # ==========================================
    # TEST 4: SIKIŞTIRMA (WHATSAPP KALİTESİ) TESTİ
    # ==========================================
    print("\n--- TEST 4: JPEG Sıkıştırma Testi Başlıyor ---")
    kalite_seviyeleri = [100, 50, 20, 10, 5, 1]
    comp_skorlar = []
    for kalite in kalite_seviyeleri:
        buffer = io.BytesIO()
        orijinal_img.save(buffer, format="JPEG", quality=kalite)
        buffer.seek(0)
        sikismis_img = Image.open(buffer).convert('RGB')
        
        skor = F.cosine_similarity(orijinal_vektor, resmi_vektore_cevir(sikismis_img)).item() * 100
        comp_skorlar.append(skor)
        print(f"[JPEG Kalitesi %{kalite}] -> Skor: %{skor:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot([f"%{k}" for k in kalite_seviyeleri], comp_skorlar, marker='o', color='orange')
    plt.title('Yapay Zeka Sıkıştırma (WhatsApp) Dayanıklılık Testi')
    plt.xlabel('Görüntü Kalitesi (JPEG)')
    plt.ylabel('Eşleşme Skoru (%)')
    plt.grid(True); plt.ylim(0, 105)
    plt.savefig(ROOT / "sikistirma_grafigi.png")
    plt.close()

    print("\n>>> İŞLEM TAMAM! 4 test de yapıldı ve grafikler 'degerlendirme' klasörüne kaydedildi.")

# Sadece bu dosya doğrudan çalıştırılırsa main() fonksiyonunu tetikle
if __name__ == "__main__":
    main()