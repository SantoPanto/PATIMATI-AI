# app/attributes.py
# Google Vision yerine LOKAL öznitelik çıkarımı (billing engeli nedeniyle, bkz. PR #1):
#   - tür (cat/dog): CLIP zero-shot sınıflandırma
#   - cins (37 ırk): CLIP zero-shot, tür tespitiyle daraltılmış aday listesi
#   - desen (tabby/spotted/solid/bicolor): CLIP zero-shot
#   - dominant renkler: piksel analizi (merkez kırpma + sabit palet)
# Cevap biçimi Vision sürümüyle birebir aynıdır; Vision'a dönüş için app/vision.py duruyor.
import io
import os

import numpy as np
import torch
from PIL import Image

from .embedder import embedder

SPECIES_PROMPTS = {"cat": "a photo of a cat", "dog": "a photo of a dog"}

# Oxford-IIIT Pet'in 37 ırkı (12 kedi + 25 köpek). Doğruluk ölçümü bu veri setiyle
# yapıldığı için liste onunla aynı tutuldu — bkz. scripts/measure_breed.py.
# UYARI: Bu ırklar SAFKAN hayvanlardır. Sahadaki kayıp hayvanların çoğu melez olduğu
# için cins tahmini yalnızca bilgi amaçlıdır; eşleştirmede FİLTRE OLARAK KULLANILMAZ
# (bkz. docs/entegrasyon-sozlesmesi.md §7).
BREEDS: list[tuple[str, str]] = [
    ("Abyssinian", "cat"),
    ("Bengal", "cat"),
    ("Birman", "cat"),
    ("Bombay", "cat"),
    ("British Shorthair", "cat"),
    ("Egyptian Mau", "cat"),
    ("Maine Coon", "cat"),
    ("Persian", "cat"),
    ("Ragdoll", "cat"),
    ("Russian Blue", "cat"),
    ("Siamese", "cat"),
    ("Sphynx", "cat"),
    ("American Bulldog", "dog"),
    ("American Pit Bull Terrier", "dog"),
    ("Basset Hound", "dog"),
    ("Beagle", "dog"),
    ("Boxer", "dog"),
    ("Chihuahua", "dog"),
    ("English Cocker Spaniel", "dog"),
    ("English Setter", "dog"),
    ("German Shorthaired Pointer", "dog"),
    ("Great Pyrenees", "dog"),
    ("Havanese", "dog"),
    ("Japanese Chin", "dog"),
    ("Keeshond", "dog"),
    ("Leonberger", "dog"),
    ("Miniature Pinscher", "dog"),
    ("Newfoundland", "dog"),
    ("Pomeranian", "dog"),
    ("Pug", "dog"),
    ("Saint Bernard", "dog"),
    ("Samoyed", "dog"),
    ("Scottish Terrier", "dog"),
    ("Shiba Inu", "dog"),
    ("Soft Coated Wheaten Terrier", "dog"),
    ("Staffordshire Bull Terrier", "dog"),
    ("Yorkshire Terrier", "dog"),
]

# CLIP makalesinin Oxford-IIIT Pet için kullandığı şablon
BREED_PROMPT = "a photo of a {}, a type of pet."

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
    # Cins güveni bunun altındaysa isim döndürülmez (yanlış cins göstermektense boş bırak).
    # 0.70 ölçümle seçildi (scripts/measure_breed.py, 111 fotoğraf):
    #   eşik 0.00 → fotoğrafların %100'üne cins verilir, verilenlerin %78'i doğru
    #   eşik 0.70 → %69'una cins verilir, verilenlerin %90'ı doğru
    #   eşik 0.90 → %49'una cins verilir, verilenlerin %94'ü doğru
    # Bu rakamlar SAFKAN + stüdyo fotoğraflarındandır; sahada daha düşük olacağı için
    # temkinli taraf seçildi.
    BREED_MIN_PROB = float(os.getenv("BREED_MIN_PROB", "0.70"))

    def __init__(self):
        self._species_keys = list(SPECIES_PROMPTS)
        self._species_feats = embedder.embed_text(list(SPECIES_PROMPTS.values()))
        self._pattern_keys = list(PATTERN_PROMPTS)
        self._pattern_feats = embedder.embed_text(list(PATTERN_PROMPTS.values()))

        self._breed_keys = [name for name, _ in BREEDS]
        self._breed_feats = embedder.embed_text(
            [BREED_PROMPT.format(name) for name in self._breed_keys])
        # Tür bilindiğinde aday ırkları daraltmak için indeksler (37 yerine 12 veya 25)
        self._breed_idx = {
            species: torch.tensor([i for i, (_, s) in enumerate(BREEDS) if s == species])
            for species in ("cat", "dog")
        }

    def analyze(self, image_bytes: bytes, embedding: list[float] | None = None) -> dict:
        """{labels, species, species_confidence, breed, breed_confidence, pattern, colors}"""
        if embedding is None:
            embedding = embedder.embed_bytes(image_bytes)
        img_feat = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        species, species_conf = self._classify(
            img_feat, self._species_feats, self._species_keys,
            min_prob=self.SPECIES_MIN_PROB)
        pattern, _ = self._classify(img_feat, self._pattern_feats, self._pattern_keys)
        breed, breed_conf = self.predict_breed(img_feat, species)
        colors = self._dominant_colors(image_bytes)

        # NOT: cins bilerek labels'a KONMUYOR. Etiketler eşleştirme skorunun %30'unu
        # oluşturuyor; cins doğruluğu bunu taşıyacak seviyede değil (bkz. ölçüm raporu).
        labels = [c["name"] for c in colors[:2]]
        labels.insert(0, pattern)
        if species != "unknown":
            labels.insert(0, species)
        return {
            "labels": labels,
            "species": species,
            "species_confidence": round(species_conf, 4),
            "breed": breed if breed_conf >= self.BREED_MIN_PROB else None,
            "breed_confidence": round(breed_conf, 4),
            "pattern": pattern,
            "colors": [{"r": c["r"], "g": c["g"], "b": c["b"], "score": c["score"]}
                       for c in colors],
        }

    def predict_breed(self, img_feat, species: str) -> tuple[str, float]:
        """En olası ırkı ve güvenini döner (eşik UYGULANMAZ — ham tahmin).

        Tür biliniyorsa yalnızca o türün ırkları arasından seçilir; bilinmiyorsa 37'sinden.
        img_feat: (1, 512) L2-normalize tensor ya da 512'lik liste.
        """
        if not isinstance(img_feat, torch.Tensor):
            img_feat = torch.tensor(img_feat, dtype=torch.float32).unsqueeze(0)

        idx = self._breed_idx.get(species)
        if idx is None:  # tür bilinmiyor → tüm ırklar aday
            feats, keys = self._breed_feats, self._breed_keys
        else:
            feats = self._breed_feats[idx]
            keys = [self._breed_keys[i] for i in idx.tolist()]

        sims = (img_feat @ feats.T).squeeze(0)
        probs = torch.softmax(sims * 100, dim=-1)
        best = int(probs.argmax())
        return keys[best], float(probs[best])

    @staticmethod
    def _classify(img_feat, text_feats, keys, min_prob=0.0):
        """(etiket, güven) döner; güven min_prob altındaysa etiket 'unknown' olur."""
        sims = (img_feat @ text_feats.T).squeeze(0)
        probs = torch.softmax(sims * 100, dim=-1)  # CLIP'in öğrenilmiş logit ölçeği ~100
        idx = int(probs.argmax())
        prob = float(probs[idx])
        if prob < min_prob:
            return "unknown", prob
        return keys[idx], prob

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
