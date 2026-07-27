import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

print("1. AI Modeli İndiriliyor/Yükleniyor...")
model = timm.create_model('hf_hub:BVRA/MegaDescriptor-T-224', pretrained=True)
model.eval()

def resmi_vektore_cevir(img):
    """Resmi AI'ın anlayacağı 768 haneli matematiksel koda (vektöre) çevirir."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_resim = transform(img).unsqueeze(0)
    with torch.no_grad():
        return model(tensor_resim)

# 1. Orijinal resmi klasörden alıp yapay zekaya tanıtıyoruz
orijinal_img = Image.open("degerlendirme/test_resim.jpg").convert('RGB')
orijinal_vektor = resmi_vektore_cevir(orijinal_img)

print("\n2. Test Başlıyor: Beyaz köpek fotoğrafı kademeli olarak bulanıklaştırılıyor...\n")

# Kademeli bulanıklık seviyeleri (Sayı büyüdükçe resim daha çok bozulur)
bulaniklik_dereceleri = [1, 5, 15, 35, 65] 

for derece in bulaniklik_dereceleri:
    if derece == 1:
        bozuk_img = orijinal_img
        durum = "Seviye 1 (Orijinal, Cam Gibi)"
    else:
        # Resmi burada 'derece' değişkenine göre bulandırıyoruz
        bozucu = transforms.GaussianBlur(kernel_size=derece)
        bozuk_img = bozucu(orijinal_img)
        durum = f"Seviye {derece} (Bulanık)"

    # Bozulmuş resmin vektörünü çıkarıyoruz
    bozuk_vektor = resmi_vektore_cevir(bozuk_img)

    # Orijinal resim ile bozuk resmi karşılaştırıp benzerlik skorunu alıyoruz
    skor = F.cosine_similarity(orijinal_vektor, bozuk_vektor).item() * 100
    print(f"[{durum}] -> AI Eşleştirme Skoru: %{skor:.2f}")

print("\nTest Tamamlandı!")