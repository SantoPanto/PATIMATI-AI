"""
PatiMati - İlan Metninden Öznitelik Çıkarma (Feature Extraction) Modülü
Türkçe eş anlamlıları normalize eder, olumsuzlukları yakalar ve eksik alanlara 'Bilinmiyor' atar.
"""

import re

class PatiMatiTextExtractor:
    def __init__(self):
        # Eş anlamlı kelime normalizasyon sözlüğü (ünsüz yumuşamaları dahil)
        self.synonyms = {
            "kedi": ["kedi", "kedicik", "pisi", "kedisi"],
            "köpek": ["köpek", "köpüş", "cıncık", "köpeği", "köpeğimiz", "köpeğin"],
            "tekir": ["tekir", "bozkır", "çizgili"],
            "tasma": ["tasma", "tasmali", "tasmalar", "tasmasız"],
            "küpe": ["küpe", "küpeli", "çentik", "çentiği", "kulak çentiği"]
        }

    def _normalize_text(self, text: str) -> str:
        # Büyük harf düzeltmeleri lower() fonksiyonundan ÖNCE yapılmalı
        text = text.replace("İ", "i").replace("I", "ı")
        text = text.lower()
        return text

    def extract_features(self, description: str) -> dict:
        desc = self._normalize_text(description)
        
        features = {
            "tur": "Bilinmiyor",
            "detay_irk": "Bilinmiyor",
            "tasma": "Bilinmiyor",
            "kulak": "Bilinmiyor"
        }

        # Tür tespiti
        for standard, variants in self.synonyms.items():
            if standard in ["kedi", "köpek"]:
                if any(v in desc for v in variants):
                    features["tur"] = standard

        # Irk/Desen tespiti (Örn: tekir)
        for variants in self.synonyms.get("tekir", []):
            if variants in desc:
                features["detay_irk"] = "tekir"

        # Tasma analizi (Genişletilmiş olumsuzluk ve ek kontrolleri: "tasmasız", "tasmasından eser yok" vb.)
        if any(t in desc for t in ["tasma", "tasmali", "tasmasız"]):
            if re.search(r'\b(tasma[s]?\s*yok|tasma\s*bulunmu[y]?or|tasma[s]?\s*de[gğ]?il|tasmasız|tasma[s]?ndan\s+eser\s+yok)\b', desc):
                features["tasma"] = "yok"
            else:
                renkler = ["mavi", "kırmızı", "siyah", "yeşil", "pembe"]
                bulunan_renk = next((r for r in renkler if r in desc), "var")
                features["tasma"] = bulunan_renk

        # Kulak / Çentik analizi (Ünsüz yumuşaması dahil: "çentiği")
        if any(k in desc for k in ["çentik", "çentiği", "çentikli", "küpe", "küpeli"]):
            if re.search(r'\b(çentik\s*yok|çentiği\s*yok|küpe\s*yok)\b', desc):
                features["kulak"] = "normal"
            else:
                features["kulak"] = "çentikli/küpeli"

        return features