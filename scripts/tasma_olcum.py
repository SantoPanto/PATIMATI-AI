# scripts/tasma_olcum.py
"""Tasma (collar) etiketinin yanlış pozitif oranını ve güven dağılımını ölçer.

NEDEN VAR: ilan formu `bonus:collar_collar` etiketinden "Tasma: Var" diye
ön-doldurma yapıyor. 20.08'de 30 gerçek kedi fotoğrafında ölçüldü: 15'inde
"tasma var" dendi, gözle bakılan 7'sinin 7'sinde de tasma YOKTU. Canlıda da
doğrulandı — tasmasız bir kedi fotoğrafı yüklendiğinde form "Var" geliyor.

    venv\\Scripts\\python.exe scripts/tasma_olcum.py

BETİK KARAR VERMEZ. Elimizde tasması BİLİNEN etiketli küme yok, o yüzden
"doğru mu" sorusunu makine cevaplayamıyor. Betik yalnız hangi dosyada ne
dendiğini ve hangi güvenle dendiğini yazar; doğrulama gözle yapılır.

KİMLİK MODELİ ÖNEMSİZ: öznitelik yolu her hâlükârda CLIP kullanıyor
(`attributes.py` içindeki `embedder` = `PetEmbedder`; kimlik vektörü ayrı
nesne, `kimlik_gomucu`). Yani `KIMLIK_MODEL` bu ölçümü değiştirmez.

GEREKTİRİR: `data/CatIndividualImages` — depoda DEĞİL, model yarışından kalma
yerel küme (`scripts/kedi_verisi_indir.py` ile inebilir). Yoksa betik ne
yapılacağını söyleyip çıkar.

EŞİK DENENDİ, ÇÖZMÜYOR: yanlış pozitifleri elemek için gereken eşik doğru
pozitifleri de eliyor — en yüksek güvenli üç fotoğraf da yanlış çıktı. Sorun
eşikte değil, yönlendirmelerin (SOFT_PROMPTS["collar"]) ayrım gücünde.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PHOTO_ALLOWED_HOSTS", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.attributes import attribute_analyzer  # noqa: E402
from app.embedder import embedder  # noqa: E402

VERI = Path("data/CatIndividualImages")
AZAMI = 30

# 20.08'de gözle bakılıp "tasma YOK" diye doğrulanan dosyalar. Betik bunları
# yalnız işaretler; yeni bir ölçümde hâlâ "var" diyorlarsa kusur duruyordur.
GOZLE_TASMASIZ = {"0001", "0003", "0009", "0015", "0019", "0029", "0031"}


def fotograflari_topla() -> list[Path]:
    fotolar: list[Path] = []
    for klasor in sorted(p for p in VERI.iterdir() if p.is_dir()):
        for f in sorted(klasor.glob("*.JPG"))[:1]:
            fotolar.append(f)
        if len(fotolar) >= AZAMI:
            break
    return fotolar


def main() -> int:
    if not VERI.is_dir():
        print("Veri yok: %s" % VERI, file=sys.stderr)
        print("Bu küme depoda değil (model yarışından kalma).", file=sys.stderr)
        print("İndirmek için: venv\\Scripts\\python.exe scripts/kedi_verisi_indir.py",
              file=sys.stderr)
        return 2

    tasma = attribute_analyzer._soft_features["collar"]
    anahtarlar = tasma["keys"]

    print("=" * 74)
    print("TASMA ÖLÇÜMÜ — öznitelik yolu CLIP kullanıyor")
    print("anahtarlar: %s" % anahtarlar)
    print("=" * 74)
    print()
    print("%-16s %-11s %8s %10s  %s" % ("DOSYA", "KAZANAN", "GÜVEN", "collar_p", "GÖZLE"))
    print("-" * 74)

    satirlar = []
    for yol in fotograflari_topla():
        gomme = embedder.embed_bytes(yol.read_bytes())
        img_feat = torch.tensor(gomme, dtype=torch.float32).unsqueeze(0)
        sims = (img_feat @ tasma["feats"].T).squeeze(0)
        probs = torch.softmax(sims * 100, dim=-1)
        idx = int(probs.argmax())
        kazanan = anahtarlar[idx]
        guven = float(probs[idx])
        collar_p = float(probs[anahtarlar.index("collar")])

        birey = yol.parent.name
        gozle = "tasma YOK" if birey in GOZLE_TASMASIZ else ""
        satirlar.append((birey, kazanan, guven, collar_p, gozle))
        print("%-16s %-11s %8.4f %10.4f  %s"
              % (yol.parent.name + "/" + yol.stem, kazanan, guven, collar_p, gozle))

    var = [s for s in satirlar if s[1] == "collar"]
    yok = [s for s in satirlar if s[1] == "no_collar"]
    diger = [s for s in satirlar if s[1] not in ("collar", "no_collar")]

    print()
    print('"TASMA VAR" DENEN : %d/%d' % (len(var), len(satirlar)))
    print('"TASMA YOK" DENEN : %d' % len(yok))
    print("BELİRSİZ          : %d" % len(diger))

    yanlis = [s for s in var if s[4]]
    if yanlis:
        print()
        print("⚠ GÖZLE TASMASIZ OLDUĞU BİLİNEN AMA 'VAR' DENEN: %d" % len(yanlis))
        for birey, _, guven, _, _ in yanlis:
            print("   %s (güven %.4f)" % (birey, guven))

    if var:
        guvenler = sorted(s[2] for s in var)
        print()
        print("güven aralığı: %.4f - %.4f  ·  ortanca: %.4f"
              % (guvenler[0], guvenler[-1], guvenler[len(guvenler) // 2]))
        print()
        print("EŞİK DENEMESİ (kaç 'var' etiketi kalırdı):")
        for esik in (0.50, 0.60, 0.70, 0.80):
            kalan = [s for s in var if s[2] >= esik]
            kalan_yanlis = [s for s in kalan if s[4]]
            print("  eşik %.2f -> %2d etiket kalır, bunların %d'i gözle YANLIŞ"
                  % (esik, len(kalan), len(kalan_yanlis)))
        print()
        print("Not: eşik yükseldikçe yanlışlar da kalıyorsa sorun eşikte değil,")
        print("     yönlendirmelerin ayrım gücünde demektir.")

    print()
    print("--> 'Var' denen dosyaları GÖZLE aç ve gerçekten tasma var mı bak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
