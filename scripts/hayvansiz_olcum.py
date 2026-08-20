# scripts/hayvansiz_olcum.py
"""Hayvansız fotoğraflarda `is_pet` ve `species` davranışını ölçer.

NEDEN VAR: `SPECIES_GATE_MIN` bir sayı ve sayılar bayatlar. Eşiğin hangi
ölçümle seçildiği `app/attributes.py`'de yazıyor; bu betik o ölçümü yeniden
üretir. Model, prompt listesi ya da eşik değiştiğinde tek komutla kontrol
edilebilsin diye ayrı duruyor.

    venv\\Scripts\\python.exe scripts/hayvansiz_olcum.py

GEREKTİRİR: `tests/gercek_veri/` — etiketli fotoğraflar depoda DEĞİL
(gerçek WhatsApp kareleri, ~10 MB). Yoksa betik ne yapılacağını söyleyip
çıkar; testler bundan etkilenmez (bkz. tests/test_tur_kapisi.py, o kararın
kendisini modelden bağımsız ölçüyor).

KİMLİK MODELİ ÖNEMSİZ: öznitelik yolu her hâlükârda CLIP kullanıyor
(`attributes.py` içindeki `embedder`), yani KIMLIK_MODEL bu ölçümü
değiştirmez. CI de `clip` ile koşuyor — buradaki sayı CI'nınkiyle aynı
yapılandırmayı ölçer.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KIMLIK_MODEL", "clip")
os.environ.setdefault("PHOTO_ALLOWED_HOSTS", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.attributes import attribute_analyzer  # noqa: E402
from app.embedder import embedder  # noqa: E402

VERI = Path("tests/gercek_veri")
ETIKET = VERI / "etiketler.json"
NEGATIF = VERI / "negatif"
POZITIF = VERI / "ayni_birey"


def olc(yol: Path) -> dict:
    ham = yol.read_bytes()
    emb = embedder.embed_bytes(ham)
    img_feat = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

    # `predict_species` bu sayıyı dışarı vermiyor (dönüşü sözleşmenin parçası);
    # eşiği ölçmek için burada aynı formülle yeniden hesaplanıyor.
    sims = (img_feat @ attribute_analyzer._tur_kapisi_feats.T).squeeze(0)
    hayvan_sayisi = len(attribute_analyzer._species_keys)
    hayvan_toplam = float(torch.softmax(sims * 100, dim=-1)[:hayvan_sayisi].sum())

    sonuc = attribute_analyzer.analyze(ham, embedding=emb)
    return {"species": sonuc["species"], "is_pet": sonuc["is_pet"],
            "guven": sonuc["species_confidence"], "hayvan_toplam": hayvan_toplam}


def main() -> int:
    if not ETIKET.exists():
        print(f"{ETIKET} yok — etiketli fotoğraf kümesi depoda tutulmuyor.")
        print("Ölçümü yapan kişide `tests/gercek_veri/` klasörü olmalı:")
        print("  etiketler.json -> {'negatif': {'neg_01.jpg': {'hayvan_var': false}, ...}}")
        return 2

    etiketler = json.loads(ETIKET.read_text(encoding="utf-8"))["negatif"]

    print(f"PET_GATE_MIN={attribute_analyzer.PET_GATE_MIN} "
          f"SPECIES_GATE_MIN={attribute_analyzer.SPECIES_GATE_MIN} "
          f"SPECIES_MIN_PROB={attribute_analyzer.SPECIES_MIN_PROB}")
    print()
    print("HAYVANSIZ KÜME (etiket: hayvan_var=false)")
    print(f"{'dosya':14}{'species':10}{'is_pet':8}{'guven':9}hayvan_toplam")

    hayvansiz, tur_atanan = [], []
    for ad in sorted(etiketler):
        yol = NEGATIF / ad
        if not yol.exists():
            continue
        r = olc(yol)
        hayvansiz.append(r)
        print(f"{ad:14}{r['species']:10}{str(r['is_pet']):8}"
              f"{r['guven']:<9}{r['hayvan_toplam']:.4f}")
        if r["species"] != "unknown":
            tur_atanan.append((ad, r["species"], r["hayvan_toplam"]))

    n = len(hayvansiz)
    yanlis_pozitif = [r for r in hayvansiz if r["is_pet"]]
    print("-" * 60)
    print(f"toplam {n} hayvansız fotoğraf")
    print(f"is_pet=True (yanlış pozitif): {len(yanlis_pozitif)}/{n}  "
          f"-> BİLEREK tolere ediliyor, eleme ölçütü değil")
    print(f"tür ATANAN (asıl zarar): {len(tur_atanan)}/{n}  {tur_atanan}")

    # Pozitif kontrol: aynı düzenek gerçek hayvanlarda tür atıyor mu? Atmıyorsa
    # "0 yanlış atama" sonucu iyi haber değil, ölçümün kör olduğunun işaretidir.
    print()
    print("POZİTİF KONTROL — gerçek hayvanlar")
    atanan, en_dusuk = 0, 1.0
    kareler = sorted(POZITIF.glob("*.jpg")) if POZITIF.exists() else []
    # seed_data da katılıyor: `ayni_birey` tek bireyin 7 karesi, tek başına
    # "eşik gerçek ilanları eliyor mu" sorusunu cevaplayacak kadar geniş değil.
    seed = Path("tests/seed_data")
    if seed.exists():
        for sinif in sorted(seed.iterdir()):
            if sinif.is_dir():
                kareler.extend(sorted(sinif.glob("*.jpg"))[:2])
    for yol in kareler:
        r = olc(yol)
        if r["species"] != "unknown":
            atanan += 1
            en_dusuk = min(en_dusuk, r["hayvan_toplam"])
    print(f"tür atanan: {atanan}/{len(kareler)}")
    if atanan:
        print(f"tür atananlar arasında en düşük hayvan_toplam: {en_dusuk:.4f}")
        print(f"  (eşik {attribute_analyzer.SPECIES_GATE_MIN} bunun ALTINDA kalmalı, "
              f"yoksa gerçek ilanlar tür kaybeder)")
    else:
        print("  !! hiç tür atanmadı — ölçüm kör, yukarıdaki sayılar anlamsız")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
