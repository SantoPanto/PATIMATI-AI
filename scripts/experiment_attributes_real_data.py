# scripts/experiment_attributes_real_data.py
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from app.attributes import attribute_analyzer
from app.embedder import embedder
from app.matcher import compute_final_score
from scripts.model_yarisi import veri_yukle, ROOT

# 1. HARD (Kalıcı) Özellikler (Ceza/Eleyici)
HARD_PROMPTS = {
    "fur_length": {
        "keys": ["short_fur", "long_fur", "hairless", "unknown"],
        "prompts": [
            "a photo of a pet with short fur hair",
            "a photo of a pet with long fluffy fur hair",
            "a photo of a hairless pet",
            "a blurry photo where fur is not visible"
        ]
    },
    "ear_shape": {
        "keys": ["pointy_ears", "floppy_ears", "unknown"],
        "prompts": [
            "a photo of a pet with pointy straight ears",
            "a photo of a pet with floppy folded down ears",
            "a photo of an animal with no ears visible"
        ]
    },
    "tail": {
        "keys": ["long_tail", "short_tail", "unknown"],
        "prompts": [
            "a photo of a pet with a long tail",
            "a photo of a pet with a short or no tail",
            "a photo of a pet's face where tail is not visible"
        ]
    },
    "size": {
        "keys": ["small", "large", "unknown"],
        "prompts": [
            "a photo of a small sized pet",
            "a photo of a large sized pet",
            "a photo where the pet size cannot be determined"
        ]
    }
}

# 2. SOFT (Geçici) Özellikler (Bonus veren, ceza vermeyen)
SOFT_PROMPTS = {
    "collar": {
        "keys": ["collar", "no_collar", "unknown", "unknown2"],
        "prompts": [
            "a photo of a pet wearing a collar around its neck",
            "a photo of a pet with no collar",
            "a photo of an animal's face only, no neck visible",
            "a photo of an unrecognizable object"
        ]
    },
    "ear_tag": {
        "keys": ["tag", "no_tag", "unknown", "unknown2"],
        "prompts": [
            "a photo of a pet with an ear tag",
            "a photo of a pet with no ear tag",
            "a photo of an animal with no ears visible",
            "a photo of an unrecognizable object"
        ]
    }
}

class ExperimentalAttributeAnalyzer:
    def __init__(self):
        self.base_analyzer = attribute_analyzer
        self.hard_features = {}
        for cat, data in HARD_PROMPTS.items():
            self.hard_features[cat] = {
                "keys": data["keys"],
                "feats": embedder.embed_text(data["prompts"])
            }
            
        self.soft_features = {}
        for cat, data in SOFT_PROMPTS.items():
            self.soft_features[cat] = {
                "keys": data["keys"],
                "feats": embedder.embed_text(data["prompts"])
            }

    def _get_feature(self, img_feat, feature_data):
        sims = (img_feat @ feature_data["feats"].T).squeeze(0)
        probs = torch.softmax(sims * 100, dim=-1)
        idx = int(probs.argmax())
        return feature_data["keys"][idx]

    def analyze(self, image_bytes: bytes, embedding: list[float]) -> dict:
        base_result = self.base_analyzer.analyze(image_bytes, embedding)
        img_feat = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        
        ext_hard = {}
        for cat, data in self.hard_features.items():
            val = self._get_feature(img_feat, data)
            if "unknown" not in val:
                ext_hard[cat] = val
                
        ext_soft = {}
        for cat, data in self.soft_features.items():
            val = self._get_feature(img_feat, data)
            if "unknown" not in val:
                ext_soft[cat] = val
                
        base_result["ext_hard"] = ext_hard
        base_result["ext_soft"] = ext_soft
        return base_result

def compute_extended_score(a_emb, b_emb, a_base_labels, b_base_labels, a_hard, b_hard, a_soft, b_soft, a_species, b_species):
    # Hard özellikleri base labellara ekle (Jaccard içinde birleşecekler)
    a_labels = list(a_base_labels) + list(a_hard.values())
    b_labels = list(b_base_labels) + list(b_hard.values())
    
    # Yeni labels ile Jaccard ve Base Skor hesapla
    rb = compute_final_score(a_emb, b_emb, a_labels, b_labels, 2.0, a_species, b_species)
    score = rb["score"]
    
    # Soft özellikler bonus verir
    for cat in a_soft.keys():
        if cat in b_soft:
            if a_soft[cat] == b_soft[cat]:
                score += 0.02  # Bonus (tavana vurmaması için 0.02'ye düşürüldü)
            # Eşleşmezse ceza YOK!
            
    return min(score, 1.0)

def stats(x):
    x = np.array(x)
    if len(x) == 0: return "Yok"
    return (f"n={len(x):<5d} mean={x.mean():.4f} min={x.min():.4f} "
            f"max={x.max():.4f} p10={np.percentile(x, 10):.4f} p90={np.percentile(x, 90):.4f}")

def report_auc(same_scores, diff_scores):
    if len(same_scores) == 0 or len(diff_scores) == 0:
        return 0.0
    y_true = [1]*len(same_scores) + [0]*len(diff_scores)
    y_scores = same_scores + diff_scores
    return roc_auc_score(y_true, y_scores)

def run_experiment_on_dataset(dataset_name, max_individuals):
    print(f"\n{'='*70}\nVERİ SETİ: {dataset_name}\n{'='*70}")
    kok = ROOT / "data" / dataset_name
    
    try:
        df = veri_yukle(dataset_name, kok, max_individuals, 2, 42)
    except Exception as e:
        print(f"HATA: Veri seti yüklenemedi ({e}).")
        return

    print(f"  > Yüklenen: {df['identity'].nunique()} birey, toplam {len(df)} fotoğraf.")
    if len(df) < 2:
        return

    analyzer = ExperimentalAttributeAnalyzer()
    items = []
    print("  > Fotoğraflar CLIP ile analiz ediliyor (Hem Base hem Extended)...")
    for _, row in df.iterrows():
        try:
            image_bytes = Path(row["tam_yol"]).read_bytes()
            emb = embedder.embed_bytes(image_bytes)
            attrs = analyzer.analyze(image_bytes, emb)
            items.append({
                "identity": row["identity"],
                "emb": emb,
                "species": attrs["species"],
                "base_labels": attrs["labels"],
                "ext_hard": attrs["ext_hard"],
                "ext_soft": attrs["ext_soft"]
            })
        except Exception as e:
            continue

    print(f"  > Analiz bitti. Toplam başarılı fotoğraf: {len(items)}")

    DIST_KM = 2.0
    same_base, diff_base = [], []
    same_ext, diff_ext = [], []

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            is_same = (a["identity"] == b["identity"])
            
            # Base score computation
            rb = compute_final_score(a["emb"], b["emb"], a["base_labels"], b["base_labels"], DIST_KM, a["species"], b["species"])
            
            # Extended score computation
            ext_score = compute_extended_score(
                a["emb"], b["emb"], a["base_labels"], b["base_labels"],
                a["ext_hard"], b["ext_hard"], a["ext_soft"], b["ext_soft"],
                a["species"], b["species"]
            )
            
            if is_same:
                same_base.append(rb["score"])
                same_ext.append(ext_score)
            else:
                diff_base.append(rb["score"])
                diff_ext.append(ext_score)

    print("\n--- BASELINE (Species, Pattern, Colors) ---")
    print("Same Animal :", stats(same_base))
    print("Diff Animal :", stats(diff_base))
    base_auc = report_auc(same_base, diff_base)
    print(f"ROC AUC : {base_auc:.4f}")
    
    print("\n--- EXTENDED (+ Collar, Tag, Fur, etc.) ---")
    print("Same Animal :", stats(same_ext))
    print("Diff Animal :", stats(diff_ext))
    ext_auc = report_auc(same_ext, diff_ext)
    print(f"ROC AUC : {ext_auc:.4f}")

def main():
    run_experiment_on_dataset("DogFaceNet", max_individuals=100)
    run_experiment_on_dataset("CatIndividualImages", max_individuals=100)

if __name__ == "__main__":
    main()
