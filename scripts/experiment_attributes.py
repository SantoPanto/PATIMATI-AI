# scripts/experiment_attributes.py

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, ".")
from app.attributes import attribute_analyzer, SPECIES_PROMPTS, PATTERN_PROMPTS
from app.embedder import embedder
from app.matcher import compute_final_score

NEW_PROMPTS = {
    "collar": {
        "with_collar": "a photo of a pet wearing a collar around its neck",
        "no_collar": "a photo of a pet with no collar"
    },
    "ear_tag": {
        "with_tag": "a photo of a pet with an ear tag",
        "no_tag": "a photo of a pet with no ear tag"
    },
    "fur_length": {
        "short_fur": "a photo of a pet with short fur hair",
        "long_fur": "a photo of a pet with long fluffy fur hair"
    },
    "ear_shape": {
        "pointy_ears": "a photo of a pet with pointy straight ears",
        "floppy_ears": "a photo of a pet with floppy folded down ears"
    },
    "tail": {
        "long_tail": "a photo of a pet with a long tail",
        "short_tail": "a photo of a pet with a short or no tail"
    },
    "size": {
        "small": "a photo of a small sized pet",
        "large": "a photo of a large sized pet"
    }
}

class ExperimentalAttributeAnalyzer:
    def __init__(self):
        self.base_analyzer = attribute_analyzer
        
        self.new_features = {}
        for category, prompts in NEW_PROMPTS.items():
            keys = list(prompts.keys())
            feats = embedder.embed_text(list(prompts.values()))
            self.new_features[category] = {"keys": keys, "feats": feats}

    def analyze(self, image_bytes: bytes, embedding: list[float]) -> dict:
        base_result = self.base_analyzer.analyze(image_bytes, embedding)
        extended_labels = list(base_result["labels"])
        
        img_feat = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        
        for category, data in self.new_features.items():
            keys = data["keys"]
            feats = data["feats"]
            sims = (img_feat @ feats.T).squeeze(0)
            probs = torch.softmax(sims * 100, dim=-1)
            idx = int(probs.argmax())
            extended_labels.append(keys[idx])
            
        base_result["extended_labels"] = extended_labels
        return base_result

def stats(x):
    x = np.array(x)
    return (f"n={len(x):3d}  mean={x.mean():.4f}  min={x.min():.4f}  "
            f"max={x.max():.4f}  p10={np.percentile(x, 10):.4f}  p90={np.percentile(x, 90):.4f}")

def main():
    seed = Path("tests/seed_data")
    if not seed.exists():
        print("Error: tests/seed_data not found.")
        return

    analyzer = ExperimentalAttributeAnalyzer()
    classes = sorted(d for d in seed.iterdir() if d.is_dir())
    
    data = {}
    for d in classes:
        items = []
        for p in sorted(d.glob("*.jpg")):
            emb = embedder.embed_bytes(p.read_bytes())
            attrs = analyzer.analyze(p.read_bytes(), emb)
            items.append({
                "emb": emb, 
                "species": attrs["species"],
                "base_labels": attrs["labels"],
                "ext_labels": attrs["extended_labels"]
            })
        data[d.name] = items

    DIST_KM = 2.0
    names = list(data)
    
    same_base, diff_base = [], []
    same_ext, diff_ext = [], []

    for n in names:
        v = data[n]
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                rb = compute_final_score(v[i]["emb"], v[j]["emb"], v[i]["base_labels"], v[j]["base_labels"], DIST_KM, v[i]["species"], v[j]["species"])
                same_base.append(rb["score"])
                
                re = compute_final_score(v[i]["emb"], v[j]["emb"], v[i]["ext_labels"], v[j]["ext_labels"], DIST_KM, v[i]["species"], v[j]["species"])
                same_ext.append(re["score"])

    for i in range(len(names)):
        for k in (1, 5, 11):
            a, b = data[names[i]][0], data[names[(i + k) % len(names)]][0]
            
            rb = compute_final_score(a["emb"], b["emb"], a["base_labels"], b["base_labels"], DIST_KM, a["species"], b["species"])
            diff_base.append(rb["score"])
            
            re = compute_final_score(a["emb"], b["emb"], a["ext_labels"], b["ext_labels"], DIST_KM, a["species"], b["species"])
            diff_ext.append(re["score"])

    print("\n--- BASELINE (Species, Pattern, Colors) ---")
    print("Same Breed :", stats(same_base))
    print("Diff Breed :", stats(diff_base))
    print(f"Diff       : {np.mean(same_base) - np.mean(diff_base):.4f}")
    
    print("\n--- EXTENDED (+ Collar, Tag, Fur, etc.) ---")
    print("Same Breed :", stats(same_ext))
    print("Diff Breed :", stats(diff_ext))
    print(f"Diff       : {np.mean(same_ext) - np.mean(diff_ext):.4f}")

    print("\n--- THRESHOLD COMPARISON ---")
    thresholds = [0.65, 0.70, 0.75]
    for esik in thresholds:
        s_base, d_base = np.array(same_base), np.array(diff_base)
        s_ext, d_ext = np.array(same_ext), np.array(diff_ext)
        
        base_acc = (s_base >= esik).mean() * 100
        base_err = (d_base >= esik).mean() * 100
        ext_acc = (s_ext >= esik).mean() * 100
        ext_err = (d_ext >= esik).mean() * 100
        
        print(f"\nThreshold: {esik:.2f}")
        print(f"  BASELINE -> Accuracy: %{base_acc:.1f} | False Positives: %{base_err:.1f}")
        print(f"  EXTENDED -> Accuracy: %{ext_acc:.1f} | False Positives: %{ext_err:.1f}")

if __name__ == "__main__":
    main()
