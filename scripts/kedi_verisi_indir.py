"""CatIndividualImages'ten dengeli bir ALT KÜME indirir (11 GB'ın tamamını değil).

NEDEN
-----
Veri kümesi 518 kedi / 13.536 fotoğraf ve toplam ~11 GB. Model yarışı için bu
kadarına gerek yok: kedi başına 4 fotoğrafla 150 kedi, sıralama kararı vermeye
fazlasıyla yeter (köpek tarafında da böyle yaptık). Kaggle tek tek dosya
indirmeye izin verdiği için 11 GB yerine ~500 MB iniyor.

Kümenin yapısı: cat_individuals_dataset/<kedi>/<kedi>_<sira>.JPG
Çıktı yapısı  : data/CatIndividualImages/<kedi>/<dosya>.JPG
                (model_yarisi.py'nin beklediği <birey>/<foto> düzeni)

ÖN KOŞUL
--------
Kaggle API anahtarı: kaggle.com > Settings > API > "Create Legacy API Key"
İnen kaggle.json dosyası ~/.kaggle/kaggle.json konumuna konur.

Çalıştır:
  venv\\Scripts\\python.exe scripts/kedi_verisi_indir.py
  venv\\Scripts\\python.exe scripts/kedi_verisi_indir.py --kedi 300 --foto 4
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KUME = "timost1234/cat-individuals"
HEDEF = ROOT / "data" / "CatIndividualImages"


def isleri_uret(kedi_sayisi, foto_sayisi):
    """İndirilecek dosya yollarını LİSTELEMEDEN üretir.

    Neden listelemiyoruz: Kaggle'ın dosya listeleme uçları sayfa başına yalnızca
    20 kayıt veriyor. 13.536 dosya için ~677 istek gerekir ve Kaggle 81. istekte
    `429 Too Many Requests` ile kesiyor. Neyse ki isimlendirme tamamen düzenli:

        cat_individuals_dataset/<4 haneli kedi>/<kedi>_<3 haneli sıra>.JPG

    Bu yüzden yolları doğrudan kuruyoruz. Var olmayan bir dosya istenirse indirme
    hata verir, o dosya atlanır — kedi başına fotoğraf sayısı değişebildiği için
    bu normaldir ve akışı bozmaz.
    """
    return [(f"{k:04d}", f"cat_individuals_dataset/{k:04d}/{k:04d}_{f:03d}.JPG")
            for k in range(1, kedi_sayisi + 1)
            for f in range(foto_sayisi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kedi", type=int, default=150, help="kaç birey")
    ap.add_argument("--foto", type=int, default=4, help="birey başına kaç fotoğraf")
    a = ap.parse_args()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("HATA: kaggle paketi yok. pip install kaggle")
        return 1

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"HATA: Kaggle kimlik doğrulaması başarısız — {e}")
        print("~/.kaggle/kaggle.json dosyasını kontrol edin (bkz. betiğin başı).")
        return 1

    isler = isleri_uret(a.kedi, a.foto)
    print(f"İndirilecek: {a.kedi} kedi × {a.foto} foto = {len(isler)} dosya")
    print(f"Tahmini süre: ~{len(isler) * 1.9 / 60:.0f} dakika")
    print("(listeleme yapılmıyor — Kaggle hız sınırı yüzünden yollar doğrudan kuruluyor)\n")

    HEDEF.mkdir(parents=True, exist_ok=True)
    basarili = atlanan = hata = 0
    basla = time.time()

    for i, (kedi, uzak) in enumerate(isler, 1):
        klasor = HEDEF / kedi
        klasor.mkdir(exist_ok=True)
        yerel = klasor / Path(uzak).name
        if yerel.exists() and yerel.stat().st_size > 0:
            atlanan += 1
            continue

        for deneme in range(4):
            try:
                api.dataset_download_file(KUME, file_name=uzak, path=str(klasor),
                                          force=True, quiet=True)
                # Kaggle bazen alt klasör yapısını koruyarak indiriyor — düzelt
                if not yerel.exists():
                    for bulunan in klasor.rglob(Path(uzak).name):
                        if bulunan != yerel:
                            shutil.move(str(bulunan), str(yerel))
                            break
                if yerel.exists():
                    basarili += 1
                break
            except Exception as e:
                # 429 = hız sınırı. Bu bir hata değil, "yavaşla" demek; uzun bekle.
                # Diğer hatalar genelde "dosya yok" (kedinin o kadar fotoğrafı
                # yok) — kısa bekleyip pes et, akışı durdurmasın.
                hiz_siniri = "429" in str(e) or "Too Many Requests" in str(e)
                if deneme == 3:
                    if hiz_siniri:
                        print(f"  HIZ SINIRI aşılamadı: {uzak}")
                    hata += 1
                else:
                    time.sleep(30 if hiz_siniri else 1)

        if i % 50 == 0:
            gecen = time.time() - basla
            print(f"  {i}/{len(isler)} · {gecen/60:.1f} dk · kalan ~"
                  f"{(len(isler)-i) * gecen/i/60:.1f} dk")

    # Boş bırakılmış ara klasörleri temizle
    for p in HEDEF.rglob("*"):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()

    toplam = sum(f.stat().st_size for f in HEDEF.rglob("*") if f.is_file())
    print(f"\nBitti: {basarili} indi · {atlanan} zaten vardı · {hata} hata")
    print(f"Toplam: {toplam/1024/1024:.0f} MB · {HEDEF}")
    kediler = [d for d in HEDEF.iterdir() if d.is_dir()]
    print(f"Klasörde {len(kediler)} kedi, "
          f"{sum(1 for _ in HEDEF.rglob('*.JPG'))} fotoğraf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
