"""
PatiMati - Paket 4: İlan Metninden Eşleştirme Modülü
Bu modül, iki evcil hayvan ilan açıklaması arasındaki anlamsal benzerliği
Sentence-Transformers (vektörel gömme / text embedding) kullanarak hesaplar.
"""

from sentence_transformers import SentenceTransformer, util
import torch
from .extractor import PatiMatiTextExtractor


class TextMatcher:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Türkçe ve çoklu dil desteği olan anlamsal gömme (semantic embedding) modelini yükler.
        Ayrıca metinlerden yapısal öznitelik çıkarmak için çıkarıcıyı başlatır.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.extractor = PatiMatiTextExtractor()

    def get_embedding(self, text: str):
        """
        Verilen metni vektörel uzaya (embedding) dönüştürür.
        """
        cleaned_text = text.strip().lower()
        if not cleaned_text:
            raise ValueError("Girdi metni boş olamaz!")
        
        return self.model.encode(cleaned_text, convert_to_tensor=True, device=self.device)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        İki ilan açıklaması arasındaki Cosine Similarity skorunu hesaplar.
        Döndürülen değer 0.0 ile 1.0 arasındadır.
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        similarity = util.cos_sim(emb1, emb2)
        skor = float(similarity[0][0])
        
        return max(0.0, min(1.0, skor))

    def extract_text_features(self, text: str) -> dict:
        """
        İlan metninden kural tabanlı öznitelikleri (tür, detay ırk, tasma, kulak durumu vb.) çıkarır.
        """
        return self.extractor.extract_features(text)


if __name__ == "__main__":
    print("🤖 Model yükleniyor ve test ediliyor...")
    matcher = TextMatcher()

    ilan_kayip = "Siyah tekir kedi, sol kulağı çentikli, üzerinde mavi bir tasma var. Odunpazarı civarında kayboldu."
    ilan_bulundu = "Sol kulağında çentik olan siyah renkli tekir kedi bulundu. Mavi tasması bulunuyor."

    skor = matcher.calculate_similarity(ilan_kayip, ilan_bulundu)
    features = matcher.extract_text_features(ilan_kayip)
    
    print(f"\n✅ Benzerlik Skoru: %{skor * 100:.2f}")
    print(f"🔍 Çıkarılan Özellikler: {features}")