# scripts/demo_match.py
# Çalışan servise üç fotoğraf gönderip uçtan uca eşleştirmeyi gösterir.
# 1) Servisi başlat:  venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
# 2) Bu betiği çalıştır:  venv\Scripts\python.exe scripts\demo_match.py
#    (istersen kendi fotoğraflarınla: ... demo_match.py kayip.jpg aday1.jpg aday2.jpg)
import os
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"

# Korumalı uçlar anahtar ister ve anahtar YOKSA DA reddeder (bkz.
# app/main.py `anahtari_dogrula`). Servisi hangi AI_API_KEY ile başlattıysan
# bu betiği de aynı değerle çalıştır, yoksa her istek 401 döner.
BASLIKLAR = {"X-Api-Key": os.getenv("AI_API_KEY", "")}


def analyze(path):
    with open(path, "rb") as f:
        r = httpx.post(f"{BASE}/analyze", headers=BASLIKLAR,
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

def ozet(d):
    cins = f", cins: {d['breed']} ({d['breed_confidence']:.2f})" if d.get("breed") else ""
    return f"tür: {d['species']}{cins}, etiketler: {d['labels']}"


print(f"Kayıp ilan fotoğrafı: {kayip}")
a = analyze(kayip)
print(f"  → {ozet(a)}")

adaylar, adlar = [], {}
for i, p in enumerate([aday1, aday2], 1):
    c = analyze(p)
    print(f"Aday {i}: {p}")
    print(f"  → {ozet(c)}")
    adlar[i] = f"aday-{i}"
    # embeddings LİSTE: bir ilanın birden çok fotoğrafı olabilir, görsel skor
    # en iyi fotoğraf çiftinden alınır. Burada her ilanın tek fotoğrafı var.
    adaylar.append({"ad_id": i, "embeddings": [c["embedding"]], "labels": c["labels"],
                    "species": c["species"], "distance_km": 2.0,
                    # Sürüm gönderilmezse aday atlanır — kasıtlı katı davranış,
                    # eski vektörlerle sessizce kıyaslama yapılmasın diye.
                    "model_version": c["model_version"]})

r = httpx.post(f"{BASE}/match", headers=BASLIKLAR,
               json={"ad_id": 999, "embeddings": [a["embedding"]], "labels": a["labels"],
                     "species": a["species"], "candidates": adaylar}, timeout=120)
r.raise_for_status()
cevap = r.json()

# Eşiği servise soruyoruz, buraya yazmıyoruz: ortam değişkeniyle eziliyor,
# yani bu betiğin konuştuğu servis başka bir eşikle koşuyor olabilir. Sabit
# yazılırsa demo yanlış sayı söyler — bir kez tam olarak bu oldu.
esik = httpx.get(f"{BASE}/health", timeout=30).json()["match_threshold"]

print(f"\nEşleştirme sonucu (yüksekten düşüğe) — servisin bildirim eşiği: {esik}")
for m in cevap["matches"]:
    durum = f"BİLDİRİM GİDER (>= {esik})" if m["match"] else "bildirim yok"
    print(f"  {adlar.get(m['ad_id'], m['ad_id'])}: toplam={m['score']:.3f} "
          f"(görsel={m['visual']:.3f}, etiket={m['label']:.3f}, "
          f"konum={m['location']:.3f}) → {durum}")

atlanan = cevap["skipped_candidates"]
if atlanan["toplam"]:
    print(f"\nAtlanan aday: {atlanan['toplam']} → "
          + ", ".join(f"{k}={v}" for k, v in atlanan.items() if k != "toplam" and v))
