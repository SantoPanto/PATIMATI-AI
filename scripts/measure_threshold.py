# scripts/measure_threshold.py
# Seed dataset üzerinde aynı ırk / farklı ırk benzerlik dağılımını ölçer.
# Rapor bölüm 6.3 beklentisi: aynı ırk 0.75-0.90, farklı ırk daha düşük, fark >= 0.05.
# Çalıştır: venv\Scripts\python.exe scripts/measure_threshold.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from app.embedder import embedder  # noqa: E402

seed = Path("tests/seed_data")
classes = sorted(d for d in seed.iterdir() if d.is_dir())

embs = {}
for d in classes:
    embs[d.name] = [np.array(embedder.embed(str(p))) for p in sorted(d.glob("*.jpg"))]
    print(f"{d.name}: {len(embs[d.name])} embedding", flush=True)

same, diff = [], []
names = list(embs)
for n in names:
    v = embs[n]
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            same.append(float(np.dot(v[i], v[j])))
for i in range(len(names)):
    for k in (1, 5, 11):  # farklı uzaklıktaki sınıflarla eşleştir
        a = embs[names[i]][0]
        b = embs[names[(i + k) % len(names)]][0]
        diff.append(float(np.dot(a, b)))


def stats(x):
    x = np.array(x)
    return (f"n={len(x):3d}  ort={x.mean():.4f}  min={x.min():.4f}  "
            f"max={x.max():.4f}  p10={np.percentile(x, 10):.4f}  p90={np.percentile(x, 90):.4f}")


print("\nAyni irk  :", stats(same))
print("Farkli irk:", stats(diff))
print(f"Ortalama fark: {np.mean(same) - np.mean(diff):.4f} (beklenti: >= 0.05)")
