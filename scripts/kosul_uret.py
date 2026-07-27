"""Bir veri kümesinin KIRPILMIŞ ve ARKAPLAN sürümlerini üretir.

NEDEN
-----
Model yarışını tek bir koşulda koşturup karar vermek, bu projede üç kez hataya yol
açtı (yanlış ölçüt, tek tür, tek hazırlık). Karar kuralımız artık şu: bir model
ancak BİRDEN ÇOK KOŞULDA tutarlı kazanıyorsa seçilir.

Bu betik ikinci koşul eksenini üretir:

  tam       fotoğrafın kendisi (kaynak klasör, dokunulmaz)
  hayvan    dedektör kutusuna kırpılmış  -> "kırpma sıralamayı değiştiriyor mu?"
  arkaplan  hayvan silinmiş, sadece ortam -> "skor hayvandan mı ortamdan mı?"

ARKAPLAN koşulu bir kontrol deneyidir: hayvan yokken bile yüksek skor çıkıyorsa
o veri kümesinde arka plan sızıntısı var demektir ve mutlak sayılar şişkindir.

Çıktı, model_yarisi.py'nin okuduğu <birey>/<fotoğraf> düzeninde:
  data/<kume>_hayvan/   ve   data/<kume>_arkaplan/

Çalıştır:
  venv\\Scripts\\python.exe scripts/kosul_uret.py --kume CatIndividualImages
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
COCO_KEDI, COCO_KOPEK = 17, 18
KUTU_GUVEN = 0.50
KUTU_PAY = 0.10
# Dedektör büyük fotoğrafta çok yavaş; kutuyu bulmak için küçültmek yeterli,
# kırpma yine ORİJİNAL çözünürlükten yapılıyor (kalite kaybı olmasın).
DEDEKTOR_AZAMI_KENAR = 800

_dedektor = None


def dedektor_yukle():
    global _dedektor
    if _dedektor is None:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )
        _dedektor = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        _dedektor.eval()
    return _dedektor


def hayvan_kutusu(img: Image.Image):
    """Orijinal çözünürlükteki kutuyu döndürür (küçültülmüş kopyada arar)."""
    olcek = min(1.0, DEDEKTOR_AZAMI_KENAR / max(img.size))
    kucuk = (img.resize((int(img.width * olcek), int(img.height * olcek)))
             if olcek < 1.0 else img)

    tensor = torch.from_numpy(np.asarray(kucuk, dtype=np.float32) / 255.0).permute(2, 0, 1)
    with torch.no_grad():
        cikti = dedektor_yukle()([tensor])[0]

    for esik in (KUTU_GUVEN, 0.30):
        en_iyi, en_iyi_skor = None, 0.0
        for kutu, etiket, skor in zip(cikti["boxes"], cikti["labels"], cikti["scores"]):
            if int(etiket) in (COCO_KEDI, COCO_KOPEK) and float(skor) >= esik > 0:
                if float(skor) > en_iyi_skor:
                    en_iyi, en_iyi_skor = [float(v) / olcek for v in kutu], float(skor)
        if en_iyi is not None:
            return en_iyi
    return None


def kirp(img, kutu):
    x1, y1, x2, y2 = kutu
    pw, ph = (x2 - x1) * KUTU_PAY, (y2 - y1) * KUTU_PAY
    return img.crop((max(0, int(x1 - pw)), max(0, int(y1 - ph)),
                     min(img.width, int(x2 + pw)), min(img.height, int(y2 + ph))))


def hayvani_sil(img, kutu):
    """Hayvan kutusunu, kutu DIŞINDAKİ piksellerin ortalama rengiyle doldurur."""
    dizi = np.asarray(img.convert("RGB")).copy()
    x1, y1 = max(0, int(kutu[0])), max(0, int(kutu[1]))
    x2, y2 = min(img.width, int(kutu[2])), min(img.height, int(kutu[3]))
    maske = np.ones(dizi.shape[:2], dtype=bool)
    maske[y1:y2, x1:x2] = False
    if not maske.any():
        return None
    dizi[y1:y2, x1:x2] = dizi[maske].mean(axis=0).astype(np.uint8)
    return Image.fromarray(dizi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", required=True, help="data/ altındaki klasör adı")
    ap.add_argument("--azami-kenar", type=int, default=1024,
                    help="çıktıyı küçült (kodlama hızlansın; model zaten 224'e indiriyor)")
    a = ap.parse_args()

    kaynak = ROOT / "data" / a.kume
    if not kaynak.exists():
        print(f"HATA: {kaynak} yok.")
        return 1

    hedefler = {k: ROOT / "data" / f"{a.kume}_{k}" for k in ("hayvan", "arkaplan")}
    for h in hedefler.values():
        h.mkdir(parents=True, exist_ok=True)

    fotolar = sorted(p for p in kaynak.rglob("*")
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"{len(fotolar)} fotoğraf işlenecek (dedektör CPU'da, sabırlı ol)\n")

    bulunan = bulunamayan = 0
    for i, yol in enumerate(fotolar, 1):
        birey = yol.parent.name
        img = Image.open(yol).convert("RGB")
        kutu = hayvan_kutusu(img)
        if kutu is None:
            bulunamayan += 1
            continue
        bulunan += 1

        for ad, uretici in (("hayvan", kirp), ("arkaplan", hayvani_sil)):
            cikti = uretici(img, kutu)
            if cikti is None:
                continue
            olcek = min(1.0, a.azami_kenar / max(cikti.size))
            if olcek < 1.0:
                cikti = cikti.resize((max(1, int(cikti.width * olcek)),
                                      max(1, int(cikti.height * olcek))))
            klasor = hedefler[ad] / birey
            klasor.mkdir(exist_ok=True)
            cikti.save(klasor / yol.name, format="JPEG", quality=95)

        if i % 50 == 0:
            print(f"  {i}/{len(fotolar)} · bulunan {bulunan} · bulunamayan {bulunamayan}")

    print(f"\nBitti. Hayvan bulunan: {bulunan}/{len(fotolar)} "
          f"(%{100*bulunan/len(fotolar):.1f})")
    if bulunamayan:
        print(f"UYARI: {bulunamayan} fotoğrafta hayvan bulunamadı; bu fotoğraflar "
              f"kırpılmış/arkaplan kümelerinde YOK. Kıyas yaparken bunu hesaba kat — "
              f"dedektörün kaçırması, kırpmalı boru hattının gerçek bir maliyetidir.")
    for ad, h in hedefler.items():
        # Windows'ta dosya sistemi büyük/küçük harf ayırmadığı için *.JPG ve *.jpg
        # aynı dosyaları iki kez sayıyordu — uzantıya göre tek seferde say
        n = sum(1 for p in h.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"  {ad:9}: {len([d for d in h.iterdir() if d.is_dir()])} birey, {n} fotoğraf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
