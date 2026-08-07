# scripts/experiment_attributes_real_data.py
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from app.attributes import attribute_analyzer
from app.embedder import embedder
from app.matcher import compute_final_score
from scripts.model_yarisi import veri_yukle, ROOT

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

    items = []
    print("  > Fotoğraflar uygulamanın CANLI (app/attributes.py) analizörü ile inceleniyor...")
    for _, row in df.iterrows():
        try:
            image_bytes = Path(row["tam_yol"]).read_bytes()
            emb = embedder.embed_bytes(image_bytes)
            attrs = attribute_analyzer.analyze(image_bytes, emb)
            items.append({
                "identity": row["identity"],
                "emb": emb,
                "species": attrs["species"],
                "labels": attrs["labels"]
            })
        except Exception as e:
            continue

    print(f"  > Analiz bitti. Toplam başarılı fotoğraf: {len(items)}")

    DIST_KM = 2.0
    same_scores, diff_scores = [], []

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            is_same = (a["identity"] == b["identity"])
            
            # Canlı sistemin (app/matcher.py) compute_final_score fonksiyonunu çağır
            rb = compute_final_score(a["emb"], b["emb"], a["labels"], b["labels"], DIST_KM, a["species"], b["species"])
            
            if is_same:
                same_scores.append(rb["score"])
            else:
                diff_scores.append(rb["score"])

    print("\n--- CANLI SİSTEM (Hard/Soft Entegre Edilmiş Hali) ---")
    print("Same Animal :", stats(same_scores))
    print("Diff Animal :", stats(diff_scores))
    auc = report_auc(same_scores, diff_scores)
    print(f"ROC AUC : {auc:.4f}")

def main():
    run_experiment_on_dataset("DogFaceNet", max_individuals=100)
    run_experiment_on_dataset("CatIndividualImages", max_individuals=100)

if __name__ == "__main__":
    main()
