# scripts/simulate_same_individual.py
# "Aynı bireyin iki farklı fotoğrafı" skor dağılımını SİMÜLE eder:
# her seed fotoğrafına kırpma + döndürme + parlaklık/kontrast varyantları uygulanır
# (farklı çekim açısı/ışığı taklidi) ve orijinaliyle hibrit skoru hesaplanır.
# NOT: gerçek farklı çekimler (poz/arka plan değişimi) bundan biraz daha düşük
# skorlar — sonuçları iyimser üst sınır olarak okuyun.
# Çalıştır: venv\Scripts\python.exe scripts/simulate_same_individual.py
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, ".")
from app.attributes import attribute_analyzer  # noqa: E402
from app.embedder import embedder  # noqa: E402
from app.matcher import compute_final_score  # noqa: E402

DIST_KM = 2.0


def variants(img):
    w, h = img.size
    v1 = img.crop((int(w * 0.10), int(h * 0.10), int(w * 0.95), int(h * 0.95)))
    v1 = ImageEnhance.Brightness(v1.rotate(8, expand=True)).enhance(1.25)
    v2 = img.crop((int(w * 0.05), int(h * 0.15), int(w * 0.90), int(h * 0.98)))
    v2 = v2.transpose(Image.FLIP_LEFT_RIGHT)
    v2 = ImageEnhance.Contrast(ImageEnhance.Brightness(v2).enhance(0.80)).enhance(1.15)
    return [v1, v2]


def to_bytes(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


seed = Path("tests/seed_data")
paths = [next(d.glob("*.jpg")) for d in sorted(seed.iterdir()) if d.is_dir()]

scores, visuals = [], []
for p in paths:
    img = Image.open(p).convert("RGB")
    b0 = to_bytes(img)
    e0 = embedder.embed_bytes(b0)
    a0 = attribute_analyzer.analyze(b0, e0)
    for v in variants(img):
        bv = to_bytes(v)
        ev = embedder.embed_bytes(bv)
        av = attribute_analyzer.analyze(bv, ev)
        r = compute_final_score(e0, ev, a0["labels"], av["labels"],
                                DIST_KM, a0["species"], av["species"])
        scores.append(r["score"])
        visuals.append(r["visual"])
    print(f"{p.parent.name} işlendi", flush=True)

s, v = np.array(scores), np.array(visuals)
print(f"\nAynı birey (simülasyon): n={len(s)}")
print(f"Hibrit : ort={s.mean():.4f}  min={s.min():.4f}  p5={np.percentile(s, 5):.4f}  p10={np.percentile(s, 10):.4f}")
print(f"Görsel : ort={v.mean():.4f}  min={v.min():.4f}  p5={np.percentile(v, 5):.4f}")
print()
for esik in (0.70, 0.75, 0.80):
    print(f"eşik {esik:.2f} → aynı birey bildirimi alan: %{100 * (s >= esik).mean():.1f}")
