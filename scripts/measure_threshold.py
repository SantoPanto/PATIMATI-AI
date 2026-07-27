# scripts/measure_threshold.py
# Seed dataset üzerinde eşik ölçümü — iki seviye:
#   1) Görsel-yalnız: aynı ırk / farklı ırk cosine dağılımı (rapor 6.3 doğrulaması)
#   2) Hibrit skor: görsel + etiket (zero-shot) + konum (sabit 2 km) → %70 eşiği analizi
# Çalıştır: venv\Scripts\python.exe scripts/measure_threshold.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from app.attributes import attribute_analyzer  # noqa: E402
from app.embedder import embedder  # noqa: E402
from app.matcher import compute_final_score  # noqa: E402

seed = Path("tests/seed_data")
classes = sorted(d for d in seed.iterdir() if d.is_dir())

data = {}  # sınıf -> [(embedding, labels, species), ...]
for d in classes:
    items = []
    for p in sorted(d.glob("*.jpg")):
        emb = embedder.embed(str(p))
        attrs = attribute_analyzer.analyze(p.read_bytes(), emb)
        items.append((emb, attrs["labels"], attrs["species"]))
    data[d.name] = items
    print(f"{d.name}: {len(items)} fotoğraf analiz edildi", flush=True)

DIST_KM = 2.0  # iki ilan arası varsayılan mesafe

same_vis, diff_vis, same_hyb, diff_hyb = [], [], [], []
names = list(data)
for n in names:
    v = data[n]
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            r = compute_final_score(v[i][0], v[j][0], v[i][1], v[j][1],
                                    DIST_KM, v[i][2], v[j][2])
            same_vis.append(r["visual"]); same_hyb.append(r["score"])
for i in range(len(names)):
    for k in (1, 5, 11):
        a, b = data[names[i]][0], data[names[(i + k) % len(names)]][0]
        r = compute_final_score(a[0], b[0], a[1], b[1], DIST_KM, a[2], b[2])
        diff_vis.append(r["visual"]); diff_hyb.append(r["score"])


def stats(x):
    x = np.array(x)
    return (f"n={len(x):3d}  ort={x.mean():.4f}  min={x.min():.4f}  "
            f"max={x.max():.4f}  p10={np.percentile(x, 10):.4f}  p90={np.percentile(x, 90):.4f}")


def esik_analizi(same, diff, esik):
    s, d = np.array(same), np.array(diff)
    return (f"eşik {esik:.2f} → aynı ırk geçen: %{100 * (s >= esik).mean():.1f}  |  "
            f"farklı ırk YANLIŞ geçen: %{100 * (d >= esik).mean():.1f}")


print("\n--- Görsel-yalnız (cosine) ---")
print("Ayni irk  :", stats(same_vis))
print("Farkli irk:", stats(diff_vis))
print("\n--- Hibrit skor (görsel %55 + etiket %30 + konum %15, mesafe 2 km) ---")
print("Ayni irk  :", stats(same_hyb))
print("Farkli irk:", stats(diff_hyb))
print(f"Ortalama fark: {np.mean(same_hyb) - np.mean(diff_hyb):.4f}")
print()
for esik in (0.65, 0.70, 0.75, 0.80):
    print(esik_analizi(same_hyb, diff_hyb, esik))
print("\nNot: 'aynı ırk' çiftleri FARKLI bireylerdir — aynı bireyin iki fotoğrafı")
print("bunlardan daha yüksek skorlar. 'Farklı ırk yanlış geçen' oranı yanlış")
print("bildirim riskinin üst sınırını gösterir.")
