# scripts/demo_match.py
# Çalışan servise üç fotoğraf gönderip uçtan uca eşleştirmeyi gösterir.
# 1) Servisi başlat:  venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
# 2) Bu betiği çalıştır:  venv\Scripts\python.exe scripts\demo_match.py
#    (istersen kendi fotoğraflarınla: ... demo_match.py kayip.jpg aday1.jpg aday2.jpg)
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"


def analyze(path):
    with open(path, "rb") as f:
        r = httpx.post(f"{BASE}/analyze",
                       files={"file": (Path(path).name, f, "image/jpeg")}, timeout=120)
    r.raise_for_status()
    return r.json()


if len(sys.argv) == 4:
    kayip, aday1, aday2 = sys.argv[1:4]
else:
    seed = Path("tests/seed_data")
    kayip = str(seed / "class_00" / "0.jpg")   # kayıp ilan: Habeş kedisi
    aday1 = str(seed / "class_00" / "1.jpg")   # aday 1: başka bir Habeş kedisi
    aday2 = str(seed / "class_01" / "0.jpg")   # aday 2: köpek (türü tutmaz, 0 almalı)

print(f"Kayıp ilan fotoğrafı: {kayip}")
a = analyze(kayip)
print(f"  → tür: {a['species']}, etiketler: {a['labels']}")

adaylar = []
for i, p in enumerate([aday1, aday2], 1):
    c = analyze(p)
    print(f"Aday {i}: {p}")
    print(f"  → tür: {c['species']}, etiketler: {c['labels']}")
    adaylar.append({"pet_id": f"aday-{i}", "embedding": c["embedding"],
                    "labels": c["labels"], "species": c["species"], "distance_km": 2.0})

r = httpx.post(f"{BASE}/match",
               json={"embedding": a["embedding"], "labels": a["labels"],
                     "species": a["species"], "candidates": adaylar}, timeout=120)
r.raise_for_status()

print("\nEşleştirme sonucu (yüksekten düşüğe):")
for m in r.json()["matches"]:
    durum = "BİLDİRİM GİDER (>= 0.70)" if m["match"] else "bildirim yok"
    print(f"  {m['pet_id']}: toplam={m['score']:.3f} "
          f"(görsel={m['visual']:.3f}, etiket={m['label']:.3f}, "
          f"konum={m['location']:.3f}) → {durum}")
