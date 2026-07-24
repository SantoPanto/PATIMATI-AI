# app/attributes.py
# Google Vision yerine LOKAL öznitelik çıkarımı (billing engeli nedeniyle, bkz. PR #1):
#   - tür (cat/dog): CLIP zero-shot sınıflandırma
#   - desen (tabby/spotted/solid/bicolor): CLIP zero-shot
#   - dominant renkler: piksel analizi (merkez kırpma + sabit palet)
# Cevap biçimi Vision sürümüyle birebir aynıdır; Vision'a dönüş için app/vision.py duruyor.
import io

import numpy as np
import torch
from PIL import Image

from .embedder import embedder

SPECIES_PROMPTS = {"cat": "a photo of a cat", "dog": "a photo of a dog"}

PATTERN_PROMPTS = {
    "tabby":   "a photo of an animal with tabby striped fur",
    "spotted": "a photo of an animal with spotted fur",
    "solid":   "a photo of an animal with solid single-colored fur",
    "bicolor": "a photo of an animal with two-colored patched fur",
}

# Sabit renk paleti: iki fotoğrafta AYNI kelimeler üretilsin diye kontrollü sözlük
PALETTE = {
    "black":  (20, 20, 20),
    "white":  (240, 240, 240),
    "gray":   (130, 130, 130),
    "brown":  (101, 67, 33),
    "orange": (200, 120, 50),
    "cream":  (235, 215, 175),
    "golden": (190, 150, 60),
}


class AttributeAnalyzer:
    SPECIES_MIN_PROB = 0.8  # altında kalırsa "unknown" (tür filtresi yanlış eleme yapmasın)

    def __init__(self):
        self._species_keys = list(SPECIES_PROMPTS)
        self._species_feats = embedder.embed_text(list(SPECIES_PROMPTS.values()))
        self._pattern_keys = list(PATTERN_PROMPTS)
        self._pattern_feats = embedder.embed_text(list(PATTERN_PROMPTS.values()))

    def analyze(self, image_bytes: bytes, embedding: list[float] | None = None) -> dict:
        """Vision API ile aynı biçimde döner: {labels, species, colors}."""
        if embedding is None:
            embedding = embedder.embed_bytes(image_bytes)
        img_feat = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        species = self._classify(img_feat, self._species_feats, self._species_keys,
                                 min_prob=self.SPECIES_MIN_PROB)
        pattern = self._classify(img_feat, self._pattern_feats, self._pattern_keys)
        colors = self._dominant_colors(image_bytes)

        labels = [c["name"] for c in colors[:2]]
        labels.insert(0, pattern)
        if species != "unknown":
            labels.insert(0, species)
        return {
            "labels": labels,
            "species": species,
            "colors": [{"r": c["r"], "g": c["g"], "b": c["b"], "score": c["score"]}
                       for c in colors],
        }

    @staticmethod
    def _classify(img_feat, text_feats, keys, min_prob=0.0):
        sims = (img_feat @ text_feats.T).squeeze(0)
        probs = torch.softmax(sims * 100, dim=-1)  # CLIP'in öğrenilmiş logit ölçeği ~100
        idx = int(probs.argmax())
        if float(probs[idx]) < min_prob:
            return "unknown"
        return keys[idx]

    @staticmethod
    def _dominant_colors(image_bytes: bytes) -> list[dict]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        # merkez kırpma: arka plan piksellerinin etkisini azalt
        img = img.crop((w // 6, h // 6, w * 5 // 6, h * 5 // 6)).resize((48, 48))
        px = np.asarray(img).reshape(-1, 3).astype(np.float32)
        names = list(PALETTE)
        pal = np.array([PALETTE[n] for n in names], dtype=np.float32)
        nearest = np.argmin(((px[:, None, :] - pal[None, :, :]) ** 2).sum(-1), axis=1)
        counts = np.bincount(nearest, minlength=len(names)) / len(px)
        order = np.argsort(counts)[::-1]
        return [{"name": names[i], "r": int(pal[i][0]), "g": int(pal[i][1]),
                 "b": int(pal[i][2]), "score": round(float(counts[i]), 3)}
                for i in order[:3] if counts[i] > 0.05]


attribute_analyzer = AttributeAnalyzer()
