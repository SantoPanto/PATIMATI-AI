# app/embedder.py
import io
import logging
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from .hatalar import GecersizGoruntu

logger = logging.getLogger(__name__)

# Bundan küçük görüntüler anlamlı bir vektör üretmez (1x1 bile sessizce
# 512'lik bir vektör döndürüyordu — çöp veriyi veritabanına yazmayalım).
ASGARI_KENAR = 32

# PIL'in varsayılan "decompression bomb" sınırı ~89 megapiksel; bu, RGB olarak
# ~268 MB bellek demek ve küçük bir konteyneri öldürür. 40 MP fazlasıyla yeterli.
Image.MAX_IMAGE_PIXELS = 40_000_000


def goruntu_ac(image_bytes: bytes) -> Image.Image:
    """Ham baytları analize hazır RGB görüntüye çevirir.

    Görüntü açma işi TEK BURADA yapılır — hem embedding hem renk analizi bunu
    kullanır. Ayrı ayrı açılsaydı düzeltmelerden birini diğerine eklemeyi
    unutmak çok kolay olurdu (ör. döndürmeyi ekleyip renkte unutmak).

    Yapılanlar:
      1. EXIF döndürmesini uygular — telefon fotoğrafları çoğu zaman "yan"
         kaydedilir; düzeltilmezse vektör bozulur ve eşleştirme kötüleşir.
      2. Şeffaflığı BEYAZ zemine yerleştirir — düz RGB'ye çevirmek şeffaf
         alanları siyaha çevirip renk analizini bozuyordu.
      3. Çok küçük görüntüleri reddeder.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        raise GecersizGoruntu(f"Görüntü açılamadı: {e}") from e

    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
        zemin = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(zemin, img)

    img = img.convert("RGB")

    if min(img.size) < ASGARI_KENAR:
        raise GecersizGoruntu(
            f"Görüntü çok küçük: {img.size[0]}x{img.size[1]} "
            f"(en az {ASGARI_KENAR}x{ASGARI_KENAR} olmalı)")
    return img


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
        return self.embed_bytes(Path(image_path).read_bytes())

    def embed_bytes(self, image_bytes: bytes) -> list[float]:
        """Byte dizisinden embedding çıkar. Geçersiz görüntüde GecersizGoruntu fırlatır."""
        return self._embed_pil(goruntu_ac(image_bytes))

    def embed_text(self, texts: list[str]) -> torch.Tensor:
        """Metin listesini L2-normalize CLIP embedding matrisine dönüştürür (zero-shot için)."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        return feats / feats.norm(dim=-1, keepdim=True)

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
