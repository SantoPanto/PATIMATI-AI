#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eşiği/ağırlıkları gevşetmek KAÇ YANLIŞ EŞLEŞME üretir? Çok bireyli veriyle ölçer.

NEDEN VAR: 19.08 ölçümü, aynı hayvanın birebir aynı fotoğrafının 3 km uzaktayken
eşiği geçemediğini gösterdi (cebirsel azami mesafe 432 m, aday arama yarıçapı ise
25 km). Ama "eşiği düşürelim" ya da "konum ağırlığını azaltalım" kararı, ancak
YANLIŞ POZİTİF bedeli bilindiğinde verilebilir — yoksa ürün her hayvanı her
hayvana eşleştirmeye başlar ve kimse fark etmez.

O bedel, yerel veritabanındaki tek bireyle (kedi01) ÖLÇÜLEMEZ. Ama depodaki
model yarışı veri kümeleriyle ölçülebilir: `data/CatIndividualImages` 138 birey,
`data/MPDD` 191, `data/DogFaceNet` 1393. Bu betik onları ilan gibi ele alır:

    POZİTİF çift : aynı bireyin İKİ AYRI fotoğraf kümesi (kayıp ilanı ↔ bulundu
                   ilanı, gerçekten aynı hayvan) — eşiği GEÇMELİ
    NEGATİF çift : iki FARKLI birey — eşiği GEÇMEMELİ

Skor, ürünün kendi yolundan hesaplanır: aynı kimlik gömücüsü, aynı öznitelik
çözümleyicisi, aynı `compute_final_score`. Mesafe veri kümesinde yok; sentetik
olarak taranıyor.

⚠ BU DÜZENEĞİN GÖREMEDİĞİ ŞEY — okumadan sonuç çıkarma:
Bir taramada POZİTİF ve NEGATİF çiftlere AYNI mesafe veriliyor. Yani konum
kanalı her çifte aynı sabiti ekliyor ve sıralamayı hiç değiştirmiyor. Oysa
gerçekte konum kanalının işi tam olarak ayırmaktır: kayıp hayvan çoğunlukla
YAKINDA bulunur, rastgele iki ilan ise birbirinden uzaktır. O ayrım burada
YOKTUR, çünkü veri kümesinde ilanların gerçek konumu yok.

Sonuç olarak bu düzenek şunu ölçebilir:
  - sabit bir eşiğin mesafeyle birlikte fiilen SERTLEŞMESİNİ,
  - BELİRLİ bir mesafede ağırlıkları değiştirmenin duyarlılık/yanlış-pozitif
    dengesine etkisini (o mesafede konum katkısı sabit olduğu için, ona
    verilen ağırlık ayırt eden bir kanaldan alınmış demektir),
şunu ölçEMEZ:
  - konum kanalının GERÇEK ayırt etme değerini,
  - dolayısıyla "konum ağırlığı düşürülsün" türü bir kararı tek başına
    haklı çıkaramaz.

Eksik veri: aynı hayvana ait olduğu bilinen kayıp+bulundu ilan çiftleri VE
aralarındaki gerçek mesafe. İstek metni: yerel-ortam/FATIH-ESLESME-VERISI-NOT.txt

⚠ ÖLÇÜME KİM CEVAP VERİYOR: `KIMLIK_MODEL` varsayılanı "clip"tir, üretimdeki
model ise siglip2-animal. Betik kullandığı modeli çıktının başında ve sonuç
dosyasında YAZAR; modeli belirtmeden koşturulan ölçüm başka bir sistemi ölçer.

KULLANIM
    set KIMLIK_MODEL=siglip2-animal
    set KIMLIK_MODEL_YOLU=data/models/avito-siglip2
    python scripts/yanlis_pozitif_egrisi.py --birey 40 --foto 4

ÇIKIŞ  0 = ölçüm üretildi · 2 = veri ya da model yok (ne yapılacağını söyler)
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))

GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Eşiğin bugünkü değeri 0.80; tarama onu ortalayacak biçimde seçildi.
ESIKLER = [0.80, 0.78, 0.76, 0.75, 0.72, 0.70, 0.68, 0.65]
MESAFELER_KM = [0.0, 1.0, 3.0, 5.0, 10.0, 25.0]


def veri_kumesini_oku(kok: Path, azami_birey: int, azami_foto: int, tohum: int):
    """`kok/<birey>/<foto>` düzenini okur. Birey başına en az 2 fotoğraf şart."""
    if not kok.is_dir():
        return None

    bireyler = []
    for klasor in sorted(p for p in kok.iterdir() if p.is_dir()):
        fotolar = sorted(
            p for p in klasor.iterdir()
            if p.suffix.lower() in GORUNTU_UZANTILARI
        )
        if len(fotolar) >= 2:
            bireyler.append((klasor.name, fotolar[:azami_foto]))

    rastgele = random.Random(tohum)
    rastgele.shuffle(bireyler)
    return bireyler[:azami_birey]


def kodla(bireyler):
    """Her fotoğrafı ÜRÜNÜN YOLUNDAN geçirir: kimlik vektörü + öznitelikler."""
    from app.analiz import attribute_analyzer
    from app.embedder import kimlik_gomucu

    kodlanmis = []
    toplam = sum(len(f) for _, f in bireyler)
    sayac = 0
    baslangic = time.time()

    for birey, fotolar in bireyler:
        kayitlar = []
        for yol in fotolar:
            ham = yol.read_bytes()
            try:
                emb = kimlik_gomucu.embed_bytes(ham)
                onceden = emb if kimlik_gomucu.clip_mi else None
                oznitelik = attribute_analyzer.analyze(ham, embedding=onceden)
            except Exception as e:  # tek bozuk fotoğraf ölçümü düşürmesin
                print("  atlandı %s: %s" % (yol.name, e), file=sys.stderr)
                continue
            kayitlar.append({
                "embedding": emb,
                "labels": oznitelik["labels"],
                "species": oznitelik["species"],
            })
            sayac += 1
            if sayac % 25 == 0:
                gecen = time.time() - baslangic
                print("  %d/%d fotoğraf (%.1f sn, %.2f sn/foto)"
                      % (sayac, toplam, gecen, gecen / sayac), flush=True)
        if len(kayitlar) >= 2:
            kodlanmis.append((birey, kayitlar))
    return kodlanmis


def ilana_bol(kayitlar):
    """Bir bireyin fotoğraflarını iki 'ilan'a böler (kayıp ↔ bulundu)."""
    orta = max(1, len(kayitlar) // 2)
    return kayitlar[:orta], kayitlar[orta:]


def ciftleri_kur(kodlanmis, azami_negatif, tohum):
    pozitif, negatif = [], []

    for _, kayitlar in kodlanmis:
        a, b = ilana_bol(kayitlar)
        if a and b:
            pozitif.append((a, b))

    # Negatifler: farklı bireylerin ilk yarıları. Hepsi alınırsa sayı
    # bireyin karesiyle büyür; tohumlu örnekleme ile sınırlanıyor.
    tum_negatif = [
        (ilana_bol(x[1])[0], ilana_bol(y[1])[0])
        for x, y in itertools.combinations(kodlanmis, 2)
    ]
    rastgele = random.Random(tohum)
    rastgele.shuffle(tum_negatif)
    negatif = tum_negatif[:azami_negatif]

    return pozitif, negatif


def skorla(ciftler, mesafe_km):
    from app.matcher import compute_final_score

    skorlar = []
    for a, b in ciftler:
        sonuc = compute_final_score(
            [k["embedding"] for k in a],
            [k["embedding"] for k in b],
            a[0]["labels"],
            b[0]["labels"],
            mesafe_km,
            a[0]["species"],
            b[0]["species"],
        )
        skorlar.append(sonuc["score"])
    return skorlar


def oran(skorlar, esik):
    if not skorlar:
        return 0.0
    return sum(1 for s in skorlar if s >= esik) / len(skorlar)



# Karsilastirilan kurulumlar. Ilki BUGUNKU davranis; digerleri raporun
# tartistigi iki sik. Uclu de AYNI cift kumesiyle skorlanir, yoksa sayilar
# kiyaslanabilir olmaz.
KURULUMLAR = [
    ("bugun          0.55/0.30/0.15", 0.55, 0.30, 0.15),
    ("konum hafif    0.65/0.30/0.05", 0.65, 0.30, 0.05),
    ("etiket hafif   0.65/0.20/0.15", 0.65, 0.20, 0.15),
    ("konum yok      0.70/0.30/0.00", 0.70, 0.30, 0.00),
]


def agirlik_taramasi(pozitif, negatif):
    """Ayni ciftleri farkli agirliklarla yeniden skorlar.

    `compute_final_score` agirliklari modul duzeyinden okuyor; burada onlar
    gecici olarak degistiriliyor ve sonunda geri konuyor. Fotograflar YENIDEN
    KODLANMIYOR — kiyasin adil olmasi icin ayni vektorler kullaniliyor.
    """
    from app import matcher

    eski = (matcher.GORSEL_AGIRLIK, matcher.ETIKET_AGIRLIK, matcher.KONUM_AGIRLIK)
    cikti = []

    baslik = "kurulum                        | mesafe | eşik | AYNI  | FARKLI"
    print()
    print("=" * len(baslik))
    print("AĞIRLIK TARAMASI — aynı çiftler, farklı ağırlıklar")
    print("=" * len(baslik))
    print(baslik)
    print("-" * len(baslik))

    try:
        for ad, gv, ev, kv in KURULUMLAR:
            matcher.GORSEL_AGIRLIK, matcher.ETIKET_AGIRLIK, matcher.KONUM_AGIRLIK = gv, ev, kv
            for km in (0.0, 5.0, 25.0):
                p = skorla(pozitif, km)
                n = skorla(negatif, km)
                esik = matcher.MATCH_THRESHOLD
                d, y = oran(p, esik), oran(n, esik)
                cikti.append({"kurulum": ad, "mesafe_km": km, "esik": esik,
                              "duyarlilik": round(d, 4),
                              "yanlis_pozitif": round(y, 4)})
                print("%-30s | %5.1f  | %.2f | %5.1f%% | %5.1f%%"
                      % (ad, km, esik, d * 100, y * 100))
            print("-" * len(baslik))
    finally:
        matcher.GORSEL_AGIRLIK, matcher.ETIKET_AGIRLIK, matcher.KONUM_AGIRLIK = eski

    return cikti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", default="CatIndividualImages")
    ap.add_argument("--birey", type=int, default=40)
    ap.add_argument("--foto", type=int, default=4)
    ap.add_argument("--negatif", type=int, default=400)
    ap.add_argument("--tohum", type=int, default=42)
    ap.add_argument("--cikti", default=None)
    ap.add_argument("--agirlik-taramasi", action="store_true",
                    help="Birkaç ağırlık kurulumunu AYNI veriyle karşılaştırır; "
                         "fotoğraflar bir kez kodlanır, skorlama tekrarlanır.")
    args = ap.parse_args()

    kok = KOK / "data" / args.veri
    bireyler = veri_kumesini_oku(kok, args.birey, args.foto, args.tohum)
    if not bireyler:
        print("Veri kümesi yok ya da boş: %s\n"
              "Bu veri git'te DEĞİL (model yarışından kalma). İndirme betiği: "
              "scripts/kedi_verisi_indir.py" % kok, file=sys.stderr)
        return 2

    # ÖLÇÜME KİM CEVAP VERİYOR — modeli kullanım anında doğrula.
    from app.surum import MODEL_SURUMU, SECILEN_KIMLIK, KIMLIK_MODELLERI
    model_yolu = KIMLIK_MODELLERI[SECILEN_KIMLIK]["kimlik"]
    print("=" * 74)
    print("KİMLİK MODELİ : %s  (sürüm %s)" % (SECILEN_KIMLIK, MODEL_SURUMU))
    print("MODEL KAYNAĞI : %s" % model_yolu)
    if SECILEN_KIMLIK == "clip":
        print("⚠ UYARI: 'clip' VARSAYILAN model — üretimdeki model bu DEĞİL.")
        print("  KIMLIK_MODEL=siglip2-animal vermeden alınan sonuç başka bir")
        print("  sistemi ölçer. (Bu tuzak daha önce vurdu: canlı siglip2 ile")
        print("  koşarken ölçüm clip'e soruldu ve her sonuç sahte 0 çıktı.)")
    print("VERİ          : %s — %d birey" % (args.veri, len(bireyler)))
    print("=" * 74)

    from app.matcher import (GORSEL_AGIRLIK, ETIKET_AGIRLIK, KONUM_AGIRLIK,
                             KONUM_YARI_MESAFE_KM, MATCH_THRESHOLD)
    print("AĞIRLIKLAR    : görsel %.2f · etiket %.2f · konum %.2f"
          % (GORSEL_AGIRLIK, ETIKET_AGIRLIK, KONUM_AGIRLIK))
    print("YARI MESAFE   : %.2f km    BUGÜNKÜ EŞİK: %.2f"
          % (KONUM_YARI_MESAFE_KM, MATCH_THRESHOLD))
    print()

    kodlanmis = kodla(bireyler)
    pozitif, negatif = ciftleri_kur(kodlanmis, args.negatif, args.tohum)
    print("\nÇİFTLER: %d pozitif (aynı birey) · %d negatif (farklı birey)\n"
          % (len(pozitif), len(negatif)))
    if not pozitif or not negatif:
        print("Çift kurulamadı — birey başına en az 2 fotoğraf gerekiyor.",
              file=sys.stderr)
        return 2

    sonuc = {
        "model": SECILEN_KIMLIK,
        "model_surumu": MODEL_SURUMU,
        "veri": args.veri,
        "birey": len(kodlanmis),
        "pozitif_cift": len(pozitif),
        "negatif_cift": len(negatif),
        "agirliklar": {"gorsel": GORSEL_AGIRLIK, "etiket": ETIKET_AGIRLIK,
                       "konum": KONUM_AGIRLIK,
                       "yari_mesafe_km": KONUM_YARI_MESAFE_KM},
        "olcum": [],
    }

    baslik = "mesafe | eşik | AYNI HAYVAN geçen | FARKLI HAYVAN geçen"
    print(baslik)
    print("-" * len(baslik))

    for km in MESAFELER_KM:
        p = skorla(pozitif, km)
        n = skorla(negatif, km)
        for esik in ESIKLER:
            duyarlilik = oran(p, esik)
            yanlis = oran(n, esik)
            sonuc["olcum"].append({
                "mesafe_km": km, "esik": esik,
                "duyarlilik": round(duyarlilik, 4),
                "yanlis_pozitif": round(yanlis, 4),
            })
            print("%5.1f  | %.2f | %6.1f%%            | %6.1f%%"
                  % (km, esik, duyarlilik * 100, yanlis * 100))
        print("-" * len(baslik))

    if args.agirlik_taramasi:
        sonuc["agirlik_taramasi"] = agirlik_taramasi(pozitif, negatif)

    if args.cikti:
        Path(args.cikti).write_text(
            json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nYAZILDI: %s" % args.cikti)

    return 0


if __name__ == "__main__":
    sys.exit(main())
