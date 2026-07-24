# app/embedder.py
import io
import logging

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class PetEmbedder:
    """
    CLIP ViT-B/32 modeli ile görsel özellik çıkarıcı.
    İlk çağrıda model HuggingFace'den indirilir (~400 MB).
    """
    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self):
        logger.info("CLIP modeli yükleniyor...")
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID)
        self.model.eval()  # Inference moduna al
        logger.info("CLIP modeli hazır.")

    def embed(self, image_path: str) -> list[float]:
        """
        Bir görüntüyü 512 boyutlu L2-normalize embedding'e dönüştürür.
        Dönüş: Python float listesi (JSON serileştirilebilir)
        """
        img = Image.open(image_path).convert("RGB")
        return self._embed_pil(img)

    def embed_bytes(self, image_bytes: bytes) -> list[float]:
        """Byte dizisinden embedding çıkar (API endpoint'i için)."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._embed_pil(img)

    def _embed_pil(self, img: Image.Image) -> list[float]:
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        # transformers 4.x düz tensor, 5.x çıktı nesnesi döndürür
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        # L2 normalizasyon — cosine similarity için ZORUNLU
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()


# Singleton — uygulama başlangıcında bir kez yüklenir
embedder = PetEmbedder()
