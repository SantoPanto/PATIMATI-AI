# scripts/tasma_dogruluk.py
"""Tasma tespitinin DOĞRULUĞUNU ölçer — etiketli küme üzerinde.

NEDEN VAR: `scripts/tasma_olcum.py` yalnız "ne dendiğini" yazar, doğru mu
söyleyemez. Bu betik gözle etiketlenmiş bir küme kullandığı için iki yönü de
ölçer: yanlış pozitif (tasmasıza "var" demek) VE yanlış negatif (tasmalıyı
kaçırmak).

    venv\\Scripts\\python.exe scripts/tasma_dogruluk.py

NEDEN İKİ YÖN DE GEREKLİ: kedi kümesinde ölçüm tek yönlüydü. Barınak
kedilerinin çoğunda zaten tasma yok, yani "her şeye tasma yok" diyen bir model
orada kusursuz görünür. 21.08'de tam bu oldu: yanlış pozitif 15/30 → 0/30
düştü ama köpeklere bakılınca gerçek tasmaların da kaçtığı görüldü.

ETİKETLER: `scripts/tasma_etiketleri.json` — 21 tasmalı + 22 tasmasız,
`data/MPDD` üzerinden. Etiketler tek kişinin gözüne dayanıyor; şüpheli bir
satırı silip gerekçesini dosyaya yazmak serbest.

GEREKTİRİR: `data/MPDD` — depoda DEĞİL. Yoksa betik ne yapılacağını söyleyip
çıkar.

KİMLİK MODELİ ÖNEMSİZ: öznitelik yolu her hâlükârda CLIP kullanıyor.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PHOTO_ALLOWED_HOSTS", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.attributes import attribute_analyzer  # noqa: E402

VERI = Path("data/MPDD")
ETIKETLER = Path("scripts/tasma_etiketleri.json")
VAR = "bonus:collar_collar"
YOK = "bonus:collar_no_collar"


def ilk_foto(birey: str) -> Path | None:
    klasor = VERI / birey
    if not klasor.is_dir():
        return None
    fotolar = sorted(klasor.glob("*.jpg"))
    return fotolar[0] if fotolar else None


def karar(yol: Path) -> str:
    """Modelin bu fotoğraf için verdiği karar: VAR / YOK / ETIKETSIZ."""
    etiketler = attribute_analyzer.analyze(yol.read_bytes()).get("labels", [])
    if VAR in etiketler:
        return "VAR"
    if YOK in etiketler:
        return "YOK"
    return "ETIKETSIZ"


def main() -> int:
    if not VERI.is_dir():
        print("Veri yok: %s" % VERI, file=sys.stderr)
        print("MPDD depoda değil (Mendeley, CC BY 4.0).", file=sys.stderr)
        return 2

    etiket = json.loads(ETIKETLER.read_text(encoding="utf-8"))
    tasmali = etiket["tasmali"]
    tasmasiz = etiket["tasmasiz"]

    print("=" * 70)
    print("TASMA DOĞRULUK ÖLÇÜMÜ — %d tasmalı, %d tasmasız"
          % (len(tasmali), len(tasmasiz)))
    print("=" * 70)

    sonuc = {"tasmali": {}, "tasmasiz": {}}
    for kova, bireyler in (("tasmali", tasmali), ("tasmasiz", tasmasiz)):
        print("\n--- GERÇEKTE %s ---" % kova.upper())
        for birey in bireyler:
            yol = ilk_foto(birey)
            if yol is None:
                print("  #%-4s (fotoğraf yok, atlandı)" % birey)
                continue
            k = karar(yol)
            sonuc[kova][birey] = k
            isaret = ""
            if kova == "tasmali" and k != "VAR":
                isaret = "  <== KAÇIRDI"
            if kova == "tasmasiz" and k == "VAR":
                isaret = "  <== YANLIŞ ALARM"
            print("  #%-4s -> %-10s%s" % (birey, k, isaret))

    dp = sum(1 for k in sonuc["tasmali"].values() if k == "VAR")
    yn = len(sonuc["tasmali"]) - dp
    yp = sum(1 for k in sonuc["tasmasiz"].values() if k == "VAR")
    dn = len(sonuc["tasmasiz"]) - yp

    print()
    print("=" * 70)
    print("Doğru pozitif (tasmalıya 'var')   : %d/%d" % (dp, len(sonuc["tasmali"])))
    print("KAÇIRILAN   (tasmalıya 'var' değil): %d" % yn)
    print("Yanlış alarm (tasmasıza 'var')     : %d/%d" % (yp, len(sonuc["tasmasiz"])))
    print("Doğru negatif                      : %d" % dn)
    print()
    if dp + yp:
        print("Kesinlik (var dediklerinin doğruluğu): %.0f%%"
              % (100.0 * dp / (dp + yp)))
    else:
        print("Kesinlik: HİÇ 'var' demedi — bu ölçülemez.")
    if dp + yn:
        print("Duyarlılık (gerçek tasmaları yakalama): %.0f%%"
              % (100.0 * dp / (dp + yn)))
    print()
    print("Not: 'ETIKETSIZ' hiç etiket üretilmemesi demek (unknown dalları).")
    print("     Form o alanı boş bırakır — yanlış bilgiden iyidir ama tespit değildir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
