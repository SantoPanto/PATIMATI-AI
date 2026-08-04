"""Aday modeller ETİKET işini de yapabiliyor mu?

NEDEN
-----
Servisimizde gömme modeli iki iş yapıyor:
  1. Kimlik vektörü  → eşleştirme (skorun %55'i)
  2. Zero-shot etiket → tür / desen / cins (skorun %30'u, Jaccard üzerinden)

İkincisi modele METİNLE soru sorarak çalışıyor: fotoğrafın vektörü
"a photo of a cat" cümlesinin vektörüne mi daha yakın, "a photo of a dog"a mı?
Bu, modelin metin kulesinin sağlam olmasını gerektirir.

Model yarışını `avito-siglip2` kazandı ama o model KİMLİK için ince ayarlanmış.
İnce ayar sırasında metin-görüntü hizası bozulmuş olabilir. Eğer bozulduysa
etiket katmanı çöker ve iki ayrı model taşımamız gerekir (etiketçi + kimlikçi),
bu da her fotoğrafta iki kat işlem ve iki kat bellek demek.

Bu betik varsaymak yerine ÖLÇÜYOR: her aday modelle tür sınıflandırması yapıp
seed veri setindeki gerçek etiketlerle karşılaştırıyor.

Ölçüt: tür (kedi/köpek) doğruluğu. Mevcut modelimiz burada %100 alıyor
(bkz. docs/olcum-raporu.md §2). Aday bunu koruyabiliyorsa tek model yeter.

Çalıştır:  venv\\Scripts\\python.exe scripts/etiket_yetenegi.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attributes import BREED_PROMPT, BREEDS, CELDIRICI_PROMPTS, SPECIES_PROMPTS  # noqa: E402
from scripts.model_yarisi import ADAYLAR, yerel_kopya  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "tests" / "seed_data"

ALIASES = {"German Shorthaired": "German Shorthaired Pointer",
           "Wheaten Terrier": "Soft Coated Wheaten Terrier"}
BREED_SPECIES = {ad: tur for ad, tur in BREEDS}


def veri_yukle():
    """seed_data'daki fotoğraflar + gerçek tür/ırk etiketleri."""
    siniflar = json.loads((SEED / "siniflar.json").read_text(encoding="utf-8"))
    ornekler = []
    for klasor, bilgi in sorted(siniflar.items()):
        irk = ALIASES.get(bilgi["breed"], bilgi["breed"])
        for foto in sorted((SEED / klasor).glob("*.jpg")):
            ornekler.append({"yol": foto, "tur": bilgi["species"], "irk": irk})
    return ornekler


class MetinliGomucu:
    """Hem görüntü hem METİN gömen HuggingFace modeli."""

    def __init__(self, kimlik, islemci=None):
        from transformers import AutoModel, AutoProcessor
        self._islemci = AutoProcessor.from_pretrained(islemci or kimlik)
        self._model, bilgi = AutoModel.from_pretrained(kimlik, output_loading_info=True)
        eksik = bilgi.get("missing_keys") or []
        if eksik:
            raise RuntimeError(f"{kimlik}: {len(eksik)} ağırlık eksik yüklendi")
        self._model.eval()

    def _duzelt(self, ozellik):
        if not isinstance(ozellik, torch.Tensor):
            havuz = getattr(ozellik, "pooler_output", None)
            ozellik = havuz if havuz is not None else ozellik[0]
        return ozellik / ozellik.norm(dim=-1, keepdim=True)

    def goruntu(self, yollar, yigin=16):
        parcalar = []
        for i in range(0, len(yollar), yigin):
            grup = [Image.open(y).convert("RGB") for y in yollar[i:i + yigin]]
            girdi = self._islemci(images=grup, return_tensors="pt")
            with torch.no_grad():
                parcalar.append(self._duzelt(self._model.get_image_features(**girdi)).numpy())
        return np.vstack(parcalar)

    def metin(self, cumleler):
        girdi = self._islemci(text=cumleler, return_tensors="pt",
                              padding="max_length", truncation=True)
        with torch.no_grad():
            return self._duzelt(self._model.get_text_features(**girdi)).numpy()


class BizimGomucu(MetinliGomucu):
    """Üretimdeki kod — taban çizgisi gerçekten servisin kullandığı yol olmalı."""

    def __init__(self):
        from app.embedder import embedder
        self._e = embedder

    def goruntu(self, yollar, yigin=16):
        return np.asarray([self._e.embed_bytes(Path(y).read_bytes()) for y in yollar],
                          dtype=np.float32)

    def metin(self, cumleler):
        return self._e.embed_text(cumleler).numpy()


def olc(gomucu, ornekler):
    yollar = [o["yol"] for o in ornekler]
    gorseller = gomucu.goruntu(yollar)

    # --- 1. Tür: sadece kedi/köpek arasında ---
    tur_anahtar = list(SPECIES_PROMPTS)
    tur_metin = gomucu.metin([SPECIES_PROMPTS[k] for k in tur_anahtar])
    tur_tahmin = [tur_anahtar[i] for i in (gorseller @ tur_metin.T).argmax(axis=1)]
    tur_dogru = sum(t == o["tur"] for t, o in zip(tur_tahmin, ornekler))

    # --- 2. Hayvan kapısı: çeldiricilerle birlikte, kedi+köpek toplamı ---
    hepsi = [SPECIES_PROMPTS[k] for k in tur_anahtar] + CELDIRICI_PROMPTS
    kapi_metin = gomucu.metin(hepsi)
    olasilik = torch.softmax(torch.tensor(gorseller @ kapi_metin.T) * 100, dim=-1).numpy()
    hayvan_payi = olasilik[:, :2].sum(axis=1)

    # --- 3. Cins: 37 ırk ---
    irk_adlari = [ad for ad, _ in BREEDS]
    irk_metin = gomucu.metin([BREED_PROMPT.format(ad) for ad in irk_adlari])
    irk_skor = gorseller @ irk_metin.T
    irk_tahmin = [irk_adlari[i] for i in irk_skor.argmax(axis=1)]
    irk_dogru = sum(t == o["irk"] for t, o in zip(irk_tahmin, ornekler))
    ilk3 = np.argsort(-irk_skor, axis=1)[:, :3]
    irk_top3 = sum(o["irk"] in [irk_adlari[j] for j in ilk3[i]]
                   for i, o in enumerate(ornekler))

    n = len(ornekler)
    return {
        "tur_dogruluk": tur_dogru / n,
        "kapi_ortanca": float(np.median(hayvan_payi)),
        "kapi_en_dusuk": float(hayvan_payi.min()),
        "cins_top1": irk_dogru / n,
        "cins_top3": irk_top3 / n,
    }


def main():
    if not (SEED / "siniflar.json").exists():
        print("HATA: tests/seed_data yok. Önce scripts/prepare_seed.py çalıştırın.")
        return 1

    ornekler = veri_yukle()
    print(f"{len(ornekler)} fotoğraf, {len(set(o['irk'] for o in ornekler))} ırk\n")

    adaylar = [("bizim", None), ("avito-siglip2", None), ("google-siglip2", None)]
    tablo = []
    for anahtar, _ in adaylar:
        print(f"--- {anahtar}")
        try:
            if anahtar == "bizim":
                g = BizimGomucu()
            else:
                a = ADAYLAR[anahtar]
                yerel = yerel_kopya(anahtar)
                g = MetinliGomucu(str(yerel) if yerel else a["kimlik"],
                                  islemci=a.get("islemci"))
            s = olc(g, ornekler)
        except Exception as e:
            print(f"    ATLANDI — {type(e).__name__}: {str(e)[:160]}\n")
            continue
        s["aday"] = anahtar
        tablo.append(s)
        print(f"    tür {s['tur_dogruluk']:.1%} · cins top-1 {s['cins_top1']:.1%} · "
              f"top-3 {s['cins_top3']:.1%} · hayvan payı ortanca "
              f"{s['kapi_ortanca']:.2f} (en düşük {s['kapi_en_dusuk']:.2f})\n")

    print("=" * 76)
    print("ETİKET YETENEĞİ — seed veri (safkan, stüdyo)")
    print("=" * 76)
    print(f"  {'aday':16} {'tür':>8} {'cins top-1':>11} {'top-3':>8} {'hayvan payı':>13}")
    for s in tablo:
        print(f"  {s['aday']:16} {s['tur_dogruluk']:>7.1%} {s['cins_top1']:>10.1%} "
              f"{s['cins_top3']:>7.1%} {s['kapi_ortanca']:>12.2f}")
    print()
    print("  Mevcut sistemimizin bilinen değerleri: tür %100 · cins top-1 %78.4 ·")
    print("  top-3 %96.4 (docs/olcum-raporu.md §2)")
    print()
    print("  KARAR: aday tür doğruluğunu koruyorsa TEK MODEL yeter.")
    print("         Belirgin düşüyorsa İKİ MODEL gerekir (etiketçi CLIP + kimlikçi aday).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
