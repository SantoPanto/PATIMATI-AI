"""Cins (breed) tahmininin doğruluğunu seed veri setiyle ölçer.

Veri: tests/seed_data/  → 37 ırk x 3 fotoğraf = 111 görüntü (Oxford-IIIT Pet).
Gerçek etiketler tests/seed_data/siniflar.json içinde (torchvision + annotations/list.txt
çapraz doğrulamasıyla üretildi).

Ölçtükleri:
  1. Tür (kedi/köpek) doğruluğu
  2. Cins top-1 doğruluğu — tür daraltması ile / olmadan / gerçek türle (üst sınır)
  3. Cins top-3 doğruluğu
  4. Güven eşiğine göre doğruluk-kapsam dengesi  → BREED_MIN_PROB'u seçmek için
  5. Farklı prompt şablonlarının karşılaştırması

ÖNEMLİ: Bu veri seti SAFKAN hayvanların stüdyo kalitesindeki fotoğraflarıdır.
Sahadaki melez ve bulanık fotoğraflarda doğruluk BUNUN ALTINDA kalır. Buradaki
rakamlar bir üst sınırdır, saha beklentisi değildir.

Çalıştır:  venv\\Scripts\\python.exe scripts/measure_breed.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attributes import BREED_PROMPT, BREEDS, attribute_analyzer  # noqa: E402
from app.embedder import embedder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "tests" / "seed_data"
CACHE = ROOT / "tmp" / "breed_embed_cache.json"

# Veri setindeki kısa adlar ile bizim kullandığımız tam ırk adları
ALIASES = {
    "German Shorthaired": "German Shorthaired Pointer",
    "Wheaten Terrier": "Soft Coated Wheaten Terrier",
}

BREED_NAMES = [name for name, _ in BREEDS]
BREED_SPECIES = {name: sp for name, sp in BREEDS}


def load_dataset() -> list[dict]:
    """Her fotoğraf için {yol, gercek_irk, gercek_tur, embedding} listesi döner."""
    siniflar = json.loads((SEED / "siniflar.json").read_text(encoding="utf-8"))
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    ornekler, yeni = [], 0
    for klasor, bilgi in sorted(siniflar.items()):
        gercek_irk = ALIASES.get(bilgi["breed"], bilgi["breed"])
        if gercek_irk not in BREED_SPECIES:
            raise SystemExit(f"Irk listesinde yok: {gercek_irk} ({klasor})")
        for foto in sorted((SEED / klasor).glob("*.jpg")):
            anahtar = f"{klasor}/{foto.name}"
            if anahtar not in cache:
                cache[anahtar] = embedder.embed_bytes(foto.read_bytes())
                yeni += 1
                if yeni % 20 == 0:
                    print(f"  ...{yeni} yeni fotoğraf işlendi", flush=True)
            ornekler.append({
                "anahtar": anahtar,
                "gercek_irk": gercek_irk,
                "gercek_tur": bilgi["species"],
                "embedding": cache[anahtar],
            })

    if yeni:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
        print(f"  {yeni} yeni embedding hesaplandı, önbelleğe yazıldı.")
    else:
        print("  Tüm embedding'ler önbellekten okundu.")
    return ornekler


def irk_ozellikleri(sablonlar: list[str]) -> torch.Tensor:
    """Şablon listesini ortalayarak her ırk için tek bir metin vektörü üretir."""
    toplam = None
    for sablon in sablonlar:
        feats = embedder.embed_text([sablon.format(ad) for ad in BREED_NAMES])
        toplam = feats if toplam is None else toplam + feats
    return toplam / toplam.norm(dim=-1, keepdim=True)


def tahmin_et(emb: list[float], feats: torch.Tensor, izinli_tur: str | None, k: int = 3):
    """(top-k ırk adı, en iyinin güveni) döner. izinli_tur verilirse adaylar daraltılır."""
    img = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
    if izinli_tur in ("cat", "dog"):
        idx = [i for i, ad in enumerate(BREED_NAMES) if BREED_SPECIES[ad] == izinli_tur]
    else:
        idx = list(range(len(BREED_NAMES)))
    alt = feats[torch.tensor(idx)]
    adlar = [BREED_NAMES[i] for i in idx]

    probs = torch.softmax((img @ alt.T).squeeze(0) * 100, dim=-1)
    sira = torch.argsort(probs, descending=True)[:k].tolist()
    return [adlar[i] for i in sira], float(probs[sira[0]])


def yuzde(dogru: int, toplam: int) -> str:
    return f"{dogru}/{toplam} = %{100 * dogru / toplam:.1f}" if toplam else "-"


def main() -> None:
    print("Seed veri seti yükleniyor...")
    ornekler = load_dataset()
    print(f"  {len(ornekler)} fotoğraf, {len(BREED_NAMES)} ırk\n")

    # ---- 1. Tür doğruluğu -------------------------------------------------
    print("=" * 68)
    print("1. TÜR (kedi/köpek) DOĞRULUĞU")
    print("=" * 68)
    tur_dogru = tur_bilinmeyen = 0
    for o in ornekler:
        img = torch.tensor(o["embedding"], dtype=torch.float32).unsqueeze(0)
        tahmin, _ = attribute_analyzer._classify(
            img, attribute_analyzer._species_feats, attribute_analyzer._species_keys,
            min_prob=attribute_analyzer.SPECIES_MIN_PROB)
        o["tahmin_tur"] = tahmin
        if tahmin == "unknown":
            tur_bilinmeyen += 1
        elif tahmin == o["gercek_tur"]:
            tur_dogru += 1
    n = len(ornekler)
    print(f"  Doğru      : {yuzde(tur_dogru, n)}")
    print(f"  'unknown'  : {yuzde(tur_bilinmeyen, n)}   (güven eşiği "
          f"{attribute_analyzer.SPECIES_MIN_PROB} altında kalanlar)")
    print(f"  Yanlış     : {yuzde(n - tur_dogru - tur_bilinmeyen, n)}\n")

    # ---- 2-3. Cins doğruluğu ---------------------------------------------
    feats = irk_ozellikleri([BREED_PROMPT])
    print("=" * 68)
    print("2. CİNS DOĞRULUĞU (prompt: CLIP makalesinin şablonu)")
    print("=" * 68)

    senaryolar = {
        "Tür daraltması YOK (37 aday)": lambda o: None,
        "Tahmin edilen türle daraltma": lambda o: o["tahmin_tur"],
        "GERÇEK türle daraltma (üst sınır)": lambda o: o["gercek_tur"],
    }
    for baslik, tur_sec in senaryolar.items():
        t1 = t3 = 0
        for o in ornekler:
            topk, guven = tahmin_et(o["embedding"], feats, tur_sec(o))
            if topk[0] == o["gercek_irk"]:
                t1 += 1
            if o["gercek_irk"] in topk:
                t3 += 1
            if baslik.startswith("Tahmin"):
                o["tahmin_irk"], o["guven"] = topk[0], guven
        print(f"  {baslik:36}  top-1 {yuzde(t1, n):16}  top-3 {yuzde(t3, n)}")

    # ---- 4. Güven eşiği ---------------------------------------------------
    print()
    print("=" * 68)
    print("4. GÜVEN EŞİĞİ — kaçını gösterelim, gösterdiğimizin kaçı doğru?")
    print("=" * 68)
    print(f"  {'eşik':>6}  {'kapsam (gösterilen)':>22}  {'doğruluk (gösterilende)':>24}")
    for esik in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        gosterilen = [o for o in ornekler if o["guven"] >= esik]
        dogru = sum(1 for o in gosterilen if o["tahmin_irk"] == o["gercek_irk"])
        kaps = f"{len(gosterilen)}/{n} = %{100 * len(gosterilen) / n:.0f}"
        dog = yuzde(dogru, len(gosterilen)) if gosterilen else "-"
        print(f"  {esik:>6.1f}  {kaps:>22}  {dog:>24}")

    # ---- 5. Prompt şablonları --------------------------------------------
    print()
    print("=" * 68)
    print("5. PROMPT ŞABLONU KARŞILAŞTIRMASI (gerçek türle daraltılmış)")
    print("=" * 68)
    sablon_setleri = {
        "a photo of a {}, a type of pet.": ["a photo of a {}, a type of pet."],
        "a photo of a {}.": ["a photo of a {}."],
        "3 şablon ortalaması": [
            "a photo of a {}, a type of pet.",
            "a photo of a {}.",
            "a close-up photo of a {}.",
        ],
    }
    for ad, sablonlar in sablon_setleri.items():
        f = irk_ozellikleri(sablonlar)
        t1 = sum(1 for o in ornekler
                 if tahmin_et(o["embedding"], f, o["gercek_tur"])[0][0] == o["gercek_irk"])
        print(f"  {ad:36}  top-1 {yuzde(t1, n)}")

    # ---- En çok karıştırılanlar ------------------------------------------
    print()
    print("=" * 68)
    print("EN ÇOK KARIŞTIRILAN IRKLAR (tahmin edilen türle daraltma)")
    print("=" * 68)
    hata = defaultdict(int)
    for o in ornekler:
        if o["tahmin_irk"] != o["gercek_irk"]:
            hata[(o["gercek_irk"], o["tahmin_irk"])] += 1
    for (gercek, tahmin), adet in sorted(hata.items(), key=lambda x: -x[1])[:10]:
        print(f"  {adet}x  {gercek:32} → {tahmin}")


if __name__ == "__main__":
    main()
