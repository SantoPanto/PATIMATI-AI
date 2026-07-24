# app/vision.py
import logging
import os

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    # Hayvanla ilgili olmayan etiketleri dışla
    EXCLUDED = {"photography", "vertebrate", "organism", "wildlife",
                "terrestrial animal", "adaptation", "whiskers"}

    def __init__(self):
        self.client = None
        # Anahtar yoksa servis çökmesin; /analyze embedding'i yine döner,
        # label/species boş kalır. Anahtar gelince otomatik devreye girer.
        if os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")):
            from google.cloud import vision
            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Vision API istemcisi hazır.")
        else:
            logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS bulunamadı — "
                "Vision API devre dışı, yalnızca CLIP embedding çalışacak.")

    def analyze(self, image_bytes: bytes) -> dict:
        """Görüntüyü analiz eder, label ve renk bilgisi döner."""
        if self.client is None:
            return {"labels": [], "species": "unknown", "colors": []}

        from google.cloud import vision
        image = vision.Image(content=image_bytes)

        # Label tespiti
        label_response = self.client.label_detection(image=image, max_results=15)
        labels = [
            l.description.lower()
            for l in label_response.label_annotations
            if l.score > 0.7 and l.description.lower() not in self.EXCLUDED
        ]

        # Renk analizi (dominant renkler)
        props = self.client.image_properties(image=image)
        colors = []
        for color in props.image_properties_annotation.dominant_colors.colors[:3]:
            c = color.color
            colors.append({
                "r": int(c.red), "g": int(c.green), "b": int(c.blue),
                "score": round(color.score, 3),
            })

        # Tür tahmini (kedi / köpek / bilinmiyor)
        species = "unknown"
        label_set = set(labels)
        if {"cat", "kitten", "felidae"} & label_set:
            species = "cat"
        elif {"dog", "puppy", "canidae"} & label_set:
            species = "dog"

        return {"labels": labels, "species": species, "colors": colors}


vision_analyzer = VisionAnalyzer()
