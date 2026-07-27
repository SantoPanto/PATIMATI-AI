"""Teşhis: skorumuz hayvandan mı, arka plandan mı geliyor?

NEDEN BU BETİK VAR
------------------
26 Temmuz'da hayvana kırpmayı "ayrım gücünü düşürüyor" diye reddettik. 27 Temmuz'da
`scripts/gercek_veri_olcum.py` tekrar okununca kararın dayanağının bozuk olduğu
görüldü:

    aynı kedi çiftleri  = sokak fotoğrafı ↔ sokak fotoğrafı  (aynı mahalle)
    farklı kedi çiftleri = sokak fotoğrafı ↔ Oxford STÜDYO fotoğrafı (class_07)

Yani arka plan, cevabın kendisiyle örtüşüyor. Böyle bir kümede arka planı gören
herhangi bir model, hayvana hiç bakmadan iki grubu ayırır. Kırpma bu sahte ipucunu
sildiği için "ayrım" düşmüş olabilir — kırpma kötü olduğu için değil.

Bu betik, animal re-ID literatüründeki teşhis protokolünü uygular
(bkz. "Are We Recognizing the Jaguar or Its Background?", arXiv 2604.09690):

    1. TAM       — fotoğrafın kendisi (bugün yaptığımız şey)
    2. HAYVAN    — dedektör kutusuna kırpılmış (sadece hayvan)
    3. ARKAPLAN  — hayvanın kutusu silinmiş (sadece arka plan)

KARAR KURALI
------------
ARKAPLAN koşulu "aynı kedi"yi "farklı kedi"den hâlâ ayırabiliyorsa, mevcut
skorumuzun ayrımı hayvandan değil ortamdan geliyor demektir — ve ürün ortamında
(KAYIP ilanı evde, BULUNDU ilanı sokakta çekilir) bu ayrım yok olur.

ÖLÇÜT NOTU
----------
"Ayrım gücü" (iki ortalamanın farkı) ölçeğe duyarlıdır ve bir koşul tüm skorları
yukarı ittiğinde yanıltır — bizi yanıltan da buydu. Burada asıl ölçüt **AUC**:
rastgele bir "aynı" çiftinin, rastgele bir "farklı" çiftinden yüksek skor alma
olasılığı. 1.00 = kusursuz ayrım, 0.50 = yazı tura. Ölçekten bağımsızdır.

Çalıştır:  venv\\Scripts\\python.exe scripts/teshis_arkaplan.py
"""
import io
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attributes import attribute_analyzer  # noqa: E402
from app.embedder import embedder, goruntu_ac  # noqa: E402
from app.matcher import compute_final_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AYNI_BIREY = ROOT / "tests" / "gercek_veri" / "ayni_birey"
FARKLI_BIREY = ROOT / "tests" / "seed_data" / "class_07"   # Bombay — stüdyo siyah kedi

MESAFE_KM = 1.0        # gercek_veri_olcum.py ile aynı, kıyaslanabilsin
KUTU_GUVEN = 0.50      # dedektör eşiği; bulamazsa 0.30'a düşülür
KUTU_PAY = 0.10        # kırpmada kutunun her yanına %10 pay
COCO_KEDI, COCO_KOPEK = 17, 18


# --------------------------------------------------------------------------
# Dedektör — ağırlıklar zaten disk önbelleğinde (önceki kırpma denemesinden)
# --------------------------------------------------------------------------
_dedektor = None


def dedektor_yukle():
    """Faster R-CNN'i bir kez yükler (pahalı nesne, singleton)."""
    global _dedektor
    if _dedektor is None:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )
        _dedektor = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        _dedektor.eval()
    return _dedektor


def hayvan_kutusu(img: Image.Image):
    """En yüksek güvenli kedi/köpek kutusunu döndürür; bulamazsa None."""
    model = dedektor_yukle()
    tensor = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    with torch.no_grad():
        cikti = model([tensor])[0]

    for esik in (KUTU_GUVEN, 0.30):
        en_iyi, en_iyi_skor = None, 0.0
        for kutu, etiket, skor in zip(cikti["boxes"], cikti["labels"], cikti["scores"]):
            if int(etiket) in (COCO_KEDI, COCO_KOPEK) and float(skor) >= esik:
                if float(skor) > en_iyi_skor:
                    en_iyi, en_iyi_skor = [float(v) for v in kutu], float(skor)
        if en_iyi is not None:
            return en_iyi, en_iyi_skor
    return None, 0.0


# --------------------------------------------------------------------------
# Üç koşulun görüntülerini üret
# --------------------------------------------------------------------------
def _baytla(img: Image.Image) -> bytes:
    tampon = io.BytesIO()
    img.convert("RGB").save(tampon, format="JPEG", quality=95)
    return tampon.getvalue()


def hayvana_kirp(img: Image.Image, kutu) -> bytes:
    x1, y1, x2, y2 = kutu
    pw, ph = (x2 - x1) * KUTU_PAY, (y2 - y1) * KUTU_PAY
    kutu_payli = (
        max(0, int(x1 - pw)), max(0, int(y1 - ph)),
        min(img.width, int(x2 + pw)), min(img.height, int(y2 + ph)),
    )
    return _baytla(img.crop(kutu_payli))


def hayvani_sil(img: Image.Image, kutu) -> bytes:
    """Hayvanın kutusunu, KUTU DIŞINDAKİ piksellerin ortalama rengiyle doldurur.

    Düz siyah/gri yerine dışarının ortalaması kullanılıyor: yabancı bir renk
    eklemek yerine yamayı arka planın istatistiğine yaklaştırıyor, böylece
    ölçüm "büyük tek renk lekesi" etkisiyle kirlenmiyor.
    """
    dizi = np.asarray(img.convert("RGB")).copy()
    x1, y1, x2, y2 = (int(v) for v in kutu)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)

    maske = np.ones(dizi.shape[:2], dtype=bool)
    maske[y1:y2, x1:x2] = False           # kutu = hayvan bölgesi
    if not maske.any():                   # kutu tüm kareyi kaplıyorsa
        return None
    dis_ortalama = dizi[maske].mean(axis=0)
    dizi[y1:y2, x1:x2] = dis_ortalama.astype(np.uint8)
    return _baytla(Image.fromarray(dizi))


# --------------------------------------------------------------------------
# Analiz
# --------------------------------------------------------------------------
def analiz_et(ham: bytes, ad: str) -> dict:
    emb = embedder.embed_bytes(ham)
    d = attribute_analyzer.analyze(ham, emb)
    d["embedding"] = emb
    d["ad"] = ad
    return d


def kosullari_hazirla(yollar) -> dict:
    """Her fotoğraf için koşulların analizini üretir.

    ADİL KIYAS KURALI: dedektör bir fotoğrafta hayvanı bulamazsa o fotoğraf
    hayvan/arkaplan koşullarına giremez. O yüzden `tam_ortak`, yalnızca dedektörün
    başardığı fotoğrafları içerir — üç koşul BİREBİR aynı kümede kıyaslansın diye.
    `tam_hepsi` ise bugünkü sistemin gerçek hâli (7 fotoğrafın tamamı), bilgi olarak
    ayrıca tutulur. Farklı n'li kümeleri yan yana koymak, bu turda düzeltmeye
    çalıştığımız hatanın ta kendisi.
    """
    kosullar = {"tam_hepsi": [], "tam_ortak": [], "hayvan": [], "arkaplan": []}
    bulunamadi = []

    for yol in yollar:
        ham = yol.read_bytes()
        img = goruntu_ac(ham)                       # EXIF/alfa düzeltmesi app ile aynı
        tam = analiz_et(ham, yol.name)
        kosullar["tam_hepsi"].append(tam)

        kutu, skor = hayvan_kutusu(img)
        if kutu is None:
            bulunamadi.append(yol.name)
            continue
        silinmis = hayvani_sil(img, kutu)
        if silinmis is None:                        # kutu tüm kareyi kaplıyor
            bulunamadi.append(yol.name + " (kutu tüm kare)")
            continue
        kosullar["tam_ortak"].append(tam)
        kosullar["hayvan"].append(analiz_et(hayvana_kirp(img, kutu), yol.name))
        kosullar["arkaplan"].append(analiz_et(silinmis, yol.name))

    return kosullar, bulunamadi


def ciftler(a_listesi, b_listesi=None):
    ikili = combinations(a_listesi, 2) if b_listesi is None else product(a_listesi, b_listesi)
    return [
        compute_final_score(x["embedding"], y["embedding"], x["labels"], y["labels"],
                            MESAFE_KM, x["species"], y["species"])
        for x, y in ikili
    ]


def auc(ayni, farkli) -> float:
    """Mann-Whitney U / (n*m): rastgele bir 'aynı' çiftinin 'farklı'yı geçme olasılığı."""
    if not len(ayni) or not len(farkli):
        return float("nan")
    kazanc = sum((a > f) + 0.5 * (a == f) for a in ayni for f in farkli)
    return kazanc / (len(ayni) * len(farkli))


def rapor(baslik, ayni_skorlar, farkli_skorlar) -> None:
    for alan in ("visual", "score"):
        a = np.array([s[alan] for s in ayni_skorlar])
        f = np.array([s[alan] for s in farkli_skorlar])
        if not len(a) or not len(f):
            print(f"    {alan:7}: veri yok")
            continue
        print(f"    {alan:7}  aynı: ort {a.mean():.3f} (n={len(a):2d})   "
              f"farklı: ort {f.mean():.3f} (n={len(f):2d})   "
              f"fark {a.mean() - f.mean():+.3f}   "
              f"AUC {auc(a, f):.3f}")


def esik_tablosu(ayni_skorlar, farkli_skorlar) -> None:
    a = np.array([s["score"] for s in ayni_skorlar])
    f = np.array([s["score"] for s in farkli_skorlar])
    if not len(a) or not len(f):
        return
    print(f"      {'eşik':>6} {'yakalama':>12} {'yanlış alarm':>14}")
    for esik in (0.60, 0.65, 0.70, 0.75):
        print(f"      {esik:>6.2f} {f'{int((a >= esik).sum())}/{len(a)}':>12} "
              f"{f'{int((f >= esik).sum())}/{len(f)}':>14}")


def main() -> None:
    ayni_yollar = sorted(AYNI_BIREY.glob("*.jpg"))
    farkli_yollar = sorted(FARKLI_BIREY.glob("*.jpg"))
    if not ayni_yollar or not farkli_yollar:
        print("HATA: tests/gercek_veri/ayni_birey ve tests/seed_data/class_07 gerekli.")
        return

    print(f"Aynı birey: {len(ayni_yollar)} foto · Farklı birey (Bombay): "
          f"{len(farkli_yollar)} foto")
    print("Üç koşul hazırlanıyor (dedektör + CLIP)... birkaç dakika sürebilir.\n")

    ayni_k, ayni_yok = kosullari_hazirla(ayni_yollar)
    farkli_k, farkli_yok = kosullari_hazirla(farkli_yollar)

    if ayni_yok or farkli_yok:
        print(f"⚠ Dedektör hayvanı BULAMADI: {len(ayni_yok) + len(farkli_yok)} fotoğraf "
              f"{ayni_yok + farkli_yok}")
        print("  (bu fotoğraflar hayvan/arkaplan koşullarından düşer — tam koşulda kalır)\n")

    print("=" * 78)
    print("TEŞHİS — aynı kedi ile farklı kedi, üç koşulda")
    print("=" * 78)
    print("  AUC 1.00 = kusursuz ayrım · 0.50 = yazı tura\n")

    ozet = {}
    for kosul, aciklama in (
        ("tam_hepsi", "TAM (7 fotoğrafın hepsi) — bugünkü sistemin gerçek hâli"),
        ("tam_ortak", "TAM (ortak alt küme) — üç koşulun ADİL kıyas tabanı"),
        ("hayvan", "HAYVAN — dedektör kutusuna kırpılmış"),
        ("arkaplan", "ARKAPLAN — hayvan silinmiş, sadece ortam"),
    ):
        ayni_s = ciftler(ayni_k[kosul])
        farkli_s = ciftler(ayni_k[kosul], farkli_k[kosul])
        ozet[kosul] = (ayni_s, farkli_s)
        print(f"  {aciklama}")
        rapor(kosul, ayni_s, farkli_s)
        esik_tablosu(ayni_s, farkli_s)
        print()

    # ---- Kararın kendisi -------------------------------------------------
    print("=" * 78)
    print("KARAR")
    print("=" * 78)
    gorsel_auc = {
        k: auc([s["visual"] for s in v[0]], [s["visual"] for s in v[1]])
        for k, v in ozet.items()
    }
    print("  AYNI fotoğraf kümesinde, üç koşulun görsel AUC'si:")
    for k in ("tam_ortak", "hayvan", "arkaplan"):
        print(f"    {k:10}: {gorsel_auc[k]:.3f}")
    print(f"  (bilgi: tam_hepsi = {gorsel_auc['tam_hepsi']:.3f} — 7 fotoğrafın tamamı)")
    print()

    ap, tam, hay = gorsel_auc["arkaplan"], gorsel_auc["tam_ortak"], gorsel_auc["hayvan"]

    if ap >= 0.70:
        print(f"  ⛔ Hayvan SİLİNDİĞİ hâlde ayrım duruyor (AUC {ap:.3f}).")
        print("     Bu kümede ayrımın önemli bölümü ORTAMDAN geliyor.")
        if ap >= tam:
            print(f"     Dahası: sadece arka plan ({ap:.3f}), fotoğrafın tamamından "
                  f"({tam:.3f}) DAHA İYİ ayırıyor.")
            print("     Yani bugünkü sistemimiz hayvana baktığı için değil, ortama")
            print("     baktığı için ayırıyor — hayvan sinyali gürültü gibi davranıyor.")
    elif ap <= 0.60:
        print(f"  ✔ Hayvan silinince ayrım çöküyor (AUC {ap:.3f}).")
        print("     Skor gerçekten hayvandan geliyor; arka plan tek başına ayırmıyor.")
    else:
        print(f"  ~ Ara bölge (AUC {ap:.3f}): arka plan katkı veriyor ama tek başına yetmiyor.")

    print()
    if hay > tam:
        print(f"  ⇒ KIRPMA, aynı kümede ayrımı {tam:.3f} → {hay:.3f} YÜKSELTİYOR.")
        print("     26 Temmuz'da kırpmayı 'ayrım gücü düştü' diye reddetmiştik; o ölçüt")
        print("     ölçeğe duyarlıydı ve kırpma tüm skorları yukarı ittiği için yanılttı.")
        print("     Ölçekten bağımsız ölçütle (AUC) sonuç TERSİNE dönüyor.")
    else:
        print(f"  ⇒ Kırpma bu kümede ayrımı yükseltmiyor ({tam:.3f} → {hay:.3f}).")

    print()
    print("  NOT: 'farklı birey' kümesi hâlâ stüdyo Bombay — yani bu ölçüm de adil")
    print("  değil, sadece TUZAĞI GÖSTERİYOR. Adil karar için aynı kaynaktan farklı")
    print("  bireyler gerekiyor (wildlife-datasets / CatIndividualImages).")


if __name__ == "__main__":
    main()
