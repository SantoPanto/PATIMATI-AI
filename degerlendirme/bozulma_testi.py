import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

print("1. AI Modeli Yükleniyor...")
model = timm.create_model('hf_hub:BVRA/MegaDescriptor-T-224', pretrained=True)
model.eval()

def resmi_vektore_cevir(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_resim = transform(img).unsqueeze(0)
    with torch.no_grad():
        return model(tensor_resim)

orijinal_img = Image.open("degerlendirme/test_resim.jpg").convert('RGB')
orijinal_vektor = resmi_vektore_cevir(orijinal_img)

print("\n2. Zorlu Test Başlıyor: Karartma (Gece Çekimi)...\n")

# 1.0 = Orijinal ışık, 0.05 = Zifiri karanlığa çok yakın
parlaklik_dereceleri = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
skorlar = []

for derece in parlaklik_dereceleri:
    # Resmi verilen dereceye göre karartıyoruz
    kararan_img = TF.adjust_brightness(orijinal_img, derece)
    
    # Kararmış resmin AI skorunu ölçüyoruz
    bozuk_vektor = resmi_vektore_cevir(kararan_img)
    skor = F.cosine_similarity(orijinal_vektor, bozuk_vektor).item() * 100
    skorlar.append(skor)
    
    durum = f"Parlaklık %{int(derece*100)}"
    print(f"[{durum}] -> AI Eşleştirme Skoru: %{skor:.2f}")

print("\n3. Grafik Çizdiriliyor...")
# Topladığımız skorları çizgi grafiğine (plot) döküyoruz
plt.figure(figsize=(10, 5))
x_ekseni_etiketleri = [f"%{int(d*100)}" for d in parlaklik_dereceleri]

plt.plot(x_ekseni_etiketleri, skorlar, marker='o', linestyle='-', color='red', linewidth=2)
plt.title('Yapay Zeka Gece Çekimi (Karanlık) Dayanıklılık Testi')
plt.xlabel('Fotoğrafta Kalan Işık Miktarı')
plt.ylabel('Yapay Zeka Eşleşme Skoru (%)')
plt.grid(True)
plt.ylim(0, 105)

# Grafiği klasöre resim olarak kaydediyoruz
plt.savefig("degerlendirme/karartma_grafigi.png")
print(">>> İŞLEM TAMAM! 'degerlendirme' klasörüne 'karartma_grafigi.png' dosyası eklendi. Hemen kontrol et!")