# app/attributes.py
# Google Vision yerine LOKAL öznitelik çıkarımı (billing engeli nedeniyle, bkz. PR #1):
#   - tür (cat/dog): CLIP zero-shot sınıflandırma
#   - cins (37 ırk): CLIP zero-shot, tür tespitiyle daraltılmış aday listesi
#   - desen (tabby/spotted/solid/bicolor): CLIP zero-shot
#   - dominant renkler: piksel analizi (merkez kırpma + sabit palet)
# Cevap biçimi Vision sürümüyle birebir aynıdır; Vision'a dönüş için app/vision.py duruyor.
import os

import numpy as np
import torch

from .embedder import embedder, goruntu_ac

SPECIES_PROMPTS = {"cat": "a photo of a cat", "dog": "a photo of a dog"}

# "Hayvan mı?" kapısı.
# Tür seçimi YALNIZCA kedi/köpek arasında yapılırsa, hayvan olmayan bir fotoğraf da
# zorunlu olarak birine atanır — softmax bir kazanan seçmek zorundadır. Güven eşiği
# bunu ancak kısmen engeller (sentetik görüntülerde tuttu, ama insan/tavşan/kurt gibi
# yapılı görüntülerde ikili seçim dengesizleşip eşiği aşabilir).
# Çeldirici seçenekler ikili zorunlu seçimi açık uçlu hâle getirir: kazanan bir
# çeldiriciyse tür "unknown" olur ve is_pet False döner.
CELDIRICI_PROMPTS = [
    "a photo of a person",
    "a photo of a bird",
    "a photo of a rabbit",
    "a photo of a horse",
    "a photo of a car",
    "a photo of food",
    "a photo of a building or a landscape",
    "a photo of furniture or an object",
    "a screenshot of text",
    "a blank or solid color image",
]

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

    # "Hayvan var mı" kapısı: kedi+köpek olasılık toplamı bunun altındaysa is_pet=False.
    # 0.10 ölçümle seçildi (111 gerçek hayvan + sentetik/kırpma negatifleri):
    #   gerçek hayvanlarda en düşük toplam 0.23 (2.3 kat pay), ortanca 0.96
    #   sentetik görüntülerde (düz renk, gürültü, gradyan) 0.02–0.04
    #   0.10'da yanlış reddetme 0/111
    # UYARI: elimizde gerçek "hayvan olmayan fotoğraf" test kümesi YOK; kapının
    # gerçek yakalama oranı henüz ölçülmedi. Bu yüzden temkinli (düşük) seçildi:
    # şüpheli bir fotoğrafı geçirmek, gerçek bir ilanı reddetmekten iyidir.
    PET_GATE_MIN = float(os.getenv("PET_GATE_MIN", "0.10"))
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
        # Tür kapısı: [kedi, köpek, *çeldiriciler] — ilk iki sıra hayvan sınıflarıdır
        self._tur_kapisi_feats = embedder.embed_text(
            list(SPECIES_PROMPTS.values()) + CELDIRICI_PROMPTS)
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
        """{labels, species, species_confidence, is_pet, breed, breed_confidence,
        pattern, colors}"""
        if embedding is None:
            embedding = embedder.embed_bytes(image_bytes)
        img_feat = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        species, species_conf, is_pet = self.predict_species(img_feat)
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
            "is_pet": is_pet,
            # Hayvan değilse cins gösterme — "araba fotoğrafı: Pug" olmasın
            "breed": breed if (is_pet and breed_conf >= self.BREED_MIN_PROB) else None,
            "breed_confidence": round(breed_conf, 4),
            "pattern": pattern,
            "colors": [{"r": c["r"], "g": c["g"], "b": c["b"], "score": c["score"]}
                       for c in colors],
        }

    def predict_species(self, img_feat) -> tuple[str, float, bool]:
        """(tür, güven, hayvan_mı) döner.

        Güven her zaman "en olası hayvan sınıfının olasılığı"dır — çeldirici
        kazansa bile bu değer anlamını korur.
        Tür 'unknown' iki farklı sebeple dönebilir:
          - is_pet=False : kazanan bir çeldirici, yani fotoğrafta kedi/köpek yok
          - is_pet=True  : hayvan var ama kedi/köpek ayrımı yeterince net değil
        """
        if not isinstance(img_feat, torch.Tensor):
            img_feat = torch.tensor(img_feat, dtype=torch.float32).unsqueeze(0)

        sims = (img_feat @ self._tur_kapisi_feats.T).squeeze(0)
        hayvan_sayisi = len(self._species_keys)

        # İKİ AYRI SORU, İKİ AYRI YARIŞMA:
        # 1) "Hayvan var mı?" — kedi+köpek olasılıklarının TOPLAMI bir eşiği aşıyor mu.
        # "Kazanan çeldiriciyse reddet" kuralı fazla keskindi: bahçede yan duran
        # açık renkli bir teriyer %73 ihtimalle "at" sayılıp reddediliyordu.
        # Toplam ölçüt daha sağlam — o fotoğrafta kedi+köpek toplamı yine 0.23.
        hayvan_toplam = float(torch.softmax(sims * 100, dim=-1)[:hayvan_sayisi].sum())
        hayvan_mi = hayvan_toplam >= self.PET_GATE_MIN

        # 2) "Kedi mi köpek mi?" — yalnızca ikisi arasında.
        # Bu ayrım çeldiricilerle BİRLİKTE ölçülürse olasılıklar sulanıyor ve
        # 2 seçenek için ayarlanmış 0.8 eşiği gerçek hayvanları eliyor
        # (ölçümde tür doğruluğu %100'den %97.3'e düşmüştü).
        tur_probs = torch.softmax(sims[:hayvan_sayisi] * 100, dim=-1)
        en_iyi = int(tur_probs.argmax())
        guven = float(tur_probs[en_iyi])

        if not hayvan_mi or guven < self.SPECIES_MIN_PROB:
            return "unknown", guven, hayvan_mi
        return self._species_keys[en_iyi], guven, True

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
        # Görüntüyü embedding ile AYNI şekilde açar (EXIF döndürme + şeffaflık
        # düzeltmesi dahil) — yoksa renk analizi döndürülmemiş görüntüye bakardı
        img = goruntu_ac(image_bytes)
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
