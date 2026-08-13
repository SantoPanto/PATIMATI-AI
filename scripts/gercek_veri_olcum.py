"""Gerçek dünya fotoğraflarıyla ölçüm — seed veri setinin kapatamadığı boşluklar.

Veri: tests/gercek_veri/
  ayni_birey/  aynı sokak kedisinin farklı gün/açı/mesafeyle çekilmiş fotoğrafları
  negatif/     hayvan içermeyen gerçek fotoğraflar (sokak, iç mekân, manzara, nesne)

Neden gerekli: Oxford-IIIT Pet seti safkan hayvanların stüdyo fotoğraflarıydı ve
iki şeyi ÖLÇEMİYORDU:
  1. "Hayvan mı" kapısının gerçek yakalama oranı (elimizde hiç negatif yoktu)
  2. Yürürlükteki eşik (app.matcher.MATCH_THRESHOLD — betik zaten oradan
     okuyor, buraya sayı yazmıyoruz ki değer değişince bu metin yalan olmasın)
     gerçekten aynı hayvanın İKİ AYRI fotoğrafında tutuyor mu
     (önceki doğrulama aynı fotoğrafı bozarak yapılmıştı — çok daha kolay bir test)

Farklı birey karşılaştırması için seed setindeki Bombay (class_07) kullanılıyor:
o da siyah kedi, yani en zor ayrım.

Çalıştır:  venv\\Scripts\\python.exe scripts/gercek_veri_olcum.py
"""
import json
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attributes import attribute_analyzer  # noqa: E402
from app.embedder import embedder  # noqa: E402
from app.matcher import MATCH_THRESHOLD, compute_final_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VERI = ROOT / "tests" / "gercek_veri"
BOMBAY = ROOT / "tests" / "seed_data" / "class_07"   # safkan siyah kedi = farklı birey
MESAFE_KM = 1.0   # aynı mahalle — kayıp hayvan senaryosunda gerçekçi


def analiz_et(yol: Path) -> dict:
    ham = yol.read_bytes()
    emb = embedder.embed_bytes(ham)
    d = attribute_analyzer.analyze(ham, emb)
    d["embedding"] = emb
    d["ad"] = yol.name
    return d


def yuzde(x, n):
    return f"{x}/{n} = %{100 * x / n:.1f}" if n else "-"


def ciftleri_skorla(a_listesi, b_listesi=None):
    """Aynı liste içi tüm çiftler, ya da iki liste arası tüm çiftler."""
    ciftler = combinations(a_listesi, 2) if b_listesi is None \
        else product(a_listesi, b_listesi)
    sonuc = []
    for x, y in ciftler:
        s = compute_final_score(x["embedding"], y["embedding"], x["labels"], y["labels"],
                                MESAFE_KM, x["species"], y["species"])
        sonuc.append({"a": x["ad"], "b": y["ad"], **s})
    return sonuc


def ozet(baslik, skorlar, esik=MATCH_THRESHOLD):
    g = np.array([s["visual"] for s in skorlar])
    t = np.array([s["score"] for s in skorlar])
    gecen = int((t >= esik).sum())
    print(f"  {baslik}")
    print(f"    çift sayısı      : {len(skorlar)}")
    print(f"    görsel benzerlik : min {g.min():.3f} · ortanca {np.median(g):.3f} · max {g.max():.3f}")
    print(f"    toplam skor      : min {t.min():.3f} · ortanca {np.median(t):.3f} · max {t.max():.3f}")
    print(f"    eşiği ({esik}) geçen : {yuzde(gecen, len(skorlar))}")
    return t


def main() -> None:
    print("Fotoğraflar analiz ediliyor...")
    ayni = [analiz_et(p) for p in sorted((VERI / "ayni_birey").glob("*.jpg"))]
    negatif = [analiz_et(p) for p in sorted((VERI / "negatif").glob("*.jpg"))]
    farkli = [analiz_et(p) for p in sorted(BOMBAY.glob("*.jpg"))]
    etiketler = json.loads((VERI / "etiketler.json").read_text(encoding="utf-8"))
    print(f"  aynı birey: {len(ayni)} · negatif: {len(negatif)} · "
          f"farklı birey (Bombay): {len(farkli)}\n")

    # ---- 1. Hayvan kapısı: NEGATİFLER (ilk kez ölçülüyor) ----------------
    print("=" * 70)
    print("1. HAYVAN KAPISI — gerçek negatiflerde yakalama oranı")
    print("=" * 70)
    yakalanan = [d for d in negatif if not d["is_pet"]]
    kacan = [d for d in negatif if d["is_pet"]]
    print(f"  Doğru reddedilen : {yuzde(len(yakalanan), len(negatif))}")
    print(f"  KAÇAN (hayvan sanılan): {yuzde(len(kacan), len(negatif))}")
    for d in kacan:
        print(f"      {d['ad']}  → tür={d['species']}, cins={d['breed']}, "
              f"({etiketler['negatif'][d['ad']]['kaynak'][:40]})")

    # ---- 2. Hayvan kapısı: POZİTİFLER -----------------------------------
    print()
    print("=" * 70)
    print("2. HAYVAN KAPISI — gerçek hayvan fotoğrafları yanlış reddediliyor mu?")
    print("=" * 70)
    for d in ayni:
        mesafe = etiketler["ayni_birey"][d["ad"]]["mesafe"]
        isaret = "" if d["is_pet"] else "   ← YANLIŞ REDDETME"
        print(f"  {d['ad']}  ({mesafe:9}) hayvan={str(d['is_pet']):5} "
              f"tür={d['species']:7} güven={d['species_confidence']:.3f}"
              f" cins={str(d['breed']):20}{isaret}")

    # ---- 3. EŞİK DOĞRULAMASI --------------------------------------------
    print()
    print("=" * 70)
    print("3. EŞİK DOĞRULAMASI — asıl soru")
    print("=" * 70)
    ayni_skor = ozet("AYNI KEDİ (farklı fotoğraflar) — bunlar eşiği GEÇMELİ:",
                     ciftleri_skorla(ayni))
    print()
    farkli_skor = ozet("FARKLI SİYAH KEDİLER (sokak kedisi ↔ Bombay) — GEÇMEMELİ:",
                       ciftleri_skorla(ayni, farkli))

    print()
    print("  AYRIM: aynı bireyin en düşüğü {:.3f} · farklının en yükseği {:.3f}"
          .format(ayni_skor.min(), farkli_skor.max()))
    if ayni_skor.min() > farkli_skor.max():
        print("  → İki küme ÇAKIŞMIYOR, temiz bir eşik seçilebilir.")
    else:
        print("  → İki küme ÇAKIŞIYOR: hiçbir eşik ikisini birden tam ayıramaz.")

    # ---- 4. Eşik taraması ------------------------------------------------
    print()
    print("=" * 70)
    print("4. EŞİK TARAMASI — hangi eşik ne veriyor?")
    print("=" * 70)
    print(f"  {'eşik':>6}  {'yakalanan gerçek eşleşme':>26}  {'yanlış alarm':>18}")
    for esik in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        d = int((ayni_skor >= esik).sum())
        y = int((farkli_skor >= esik).sum())
        isaret = "   ← şu anki" if abs(esik - MATCH_THRESHOLD) < 1e-9 else ""
        print(f"  {esik:>6.2f}  {yuzde(d, len(ayni_skor)):>26}  "
              f"{yuzde(y, len(farkli_skor)):>18}{isaret}")

    # ---- 5. En kötü çiftler ----------------------------------------------
    print()
    print("=" * 70)
    print("5. AYNI KEDİDE EN DÜŞÜK SKORLU ÇİFTLER (neden kaçırıyoruz?)")
    print("=" * 70)
    for s in sorted(ciftleri_skorla(ayni), key=lambda x: x["score"])[:5]:
        ma = etiketler["ayni_birey"][s["a"]]["mesafe"]
        mb = etiketler["ayni_birey"][s["b"]]["mesafe"]
        print(f"  {s['score']:.3f}  {s['a']} ({ma}) ↔ {s['b']} ({mb})"
              f"   görsel={s['visual']:.3f} etiket={s['label']:.3f}")


if __name__ == "__main__":
    main()
