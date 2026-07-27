import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import torch
import timm
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parent

print("1. AI Modeli Yükleniyor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('hf_hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
model.eval()
model.to(device)

data_cfg = timm.data.resolve_model_data_config(model)
embed_transform = timm.data.create_transform(**data_cfg, is_training=False)

def resmi_vektore_cevir(img):
    tensor_resim = embed_transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(tensor_resim)

orijinal_img = Image.open("degerlendirme/test_resim.jpg").convert('RGB')
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
plt.savefig("degerlendirme/bulaniklik_grafigi.png")

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
plt.savefig("degerlendirme/karartma_grafigi.png")

print("\n>>> İŞLEM TAMAM! İki test de yapıldı ve grafikler 'degerlendirme' klasörüne kaydedildi.")