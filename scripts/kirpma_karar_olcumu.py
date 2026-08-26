"""Kırpma kararı için ADİL yeniden ölçüm — çok bireyli, konfonde OLMAYAN veri.

NEDEN BU BETİK VAR
------------------
`scripts/teshis_arkaplan.py` kırpmanın (26 Temmuz'daki "ayrım gücü düştü"
kararının tersine) ayrımı YÜKSELTTİĞİNİ gösterdi -- ama kendi NOT'unda da
itiraf ettiği gibi "farklı birey" kümesi hâlâ stüdyo Bombay (tests/seed_data/
class_07), yani "aynı kedi" grubu sokak↔sokak, "farklı kedi" grubu sokak↔
STÜDYO karşılaştırıyor. Bu, §4 DÜZELTME'de geçersiz ilan edilen ORİJİNAL
hatanın ta kendisi: arka plan tipi cevapla örtüşüyor. O yüzden
teshis_arkaplan.py'nin kendi sonucu bile "tuzağı gösteriyor, kararı vermiyor"
diye işaretlenmişti (docs/olcum-raporu.md §4 sonu, §6, §7).

Bu betik AYNI protokolü (AUC, ölçekten bağımsız -- bkz. teshis_arkaplan.py'nin
"ÖLÇÜT NOTU"), ama ADİL bir veri kümesiyle uygular: çok bireyli, AYNI kaynaktan
(ör. CatIndividualImages -- telefonla çekilmiş, tam sahne, ev içi fotoğraflar,
138+ farklı kedi) "aynı kedi" ve "farklı kedi" çiftleri, ikisi de AYNI TÜR
ortamdan geliyor. Böylece "kırpma ayrımı yükseltiyor mu" sorusunun cevabı,
arka plan tipinin etikete sızmasından bağımsız hale gelir.

app/kirpma.py'deki GERÇEK üretim kırpma fonksiyonunu (HayvanDedektoru/
hayvana_kirp) kullanır -- teshis_arkaplan.py'nin kendi (ayrı, kopya) dedektör
kodunu DEĞİL. Gerekçe: bu betiğin ölçtüğü şey, canlıya alınacak KOD olmalı;
ikinci bir elle yazılmış kopya, ölçümü koddan bağımsızlaştırıp sessizce
farklı bir şey ölçme riski taşır.

GERÇEK VERİYLE ÇALIŞTIRILDI (2026-08-22) — MPDD (Mendeley, "Multi-pose dog
dataset", 191 köpek, CC BY 4.0 -- Kaggle gerektirmez, indirme adresi
`https://data.mendeley.com/datasets/v5j6m8dzhv/1`) ile 40 ve 80 bireylik iki
örneklemde: TAM/HAYVAN görsel AUC farkı ±0.01 içinde ve YÖN ÖRNEKLEMLE
DEĞİŞİYOR (40 birey: kırpma +0.003, 80 birey: kırpma −0.008) -- yani bu
ölçümde kırpmanın görünür/güvenilir bir kazancı YOK. Tam tablo ve sınırlamalar
(CLIP ile ölçüldü, üretim modeliyle değil; yalnızca köpek, kedi ile
tekrarlanmadı) için bkz. docs/olcum-raporu.md §8.1. **Sonuç: CROP_TO_ANIMAL
şimdilik AÇILMAMALI** (karar kuralı §7: "birden çok koşulda tutarlı
kazanmalı" -- bu ölçüm bunu sağlamıyor).

191 bireyin TAMAMIYLA (foto_sayisi_birey=4) çalıştırılması ~30 dakikada
bitmedi (embedding maliyeti beklenenden yüksek çıktı) ve yarıda durduruldu;
80 birey/3 foto ~14 dakikada tamamlandı. Daha büyük bir örneklem denemek
isteyen süreyi göze almalı:

  venv\\Scripts\\python.exe scripts/kirpma_karar_olcumu.py --kok data/MPDD_bireyler --birey-sayisi 80 --foto-sayisi-birey 3

(MPDD ham verisi `<birey_id>_c<kamera>s<sekans>_<kare>.jpg` adlandırmasıyla
gelir, `<kök>/<birey>/<foto>` düzeninde DEĞİL -- indirilen zip'i bu düzene
sokan tek seferlik dönüştürme adımı için bkz. docs/olcum-raporu.md §8.1.)

KIMLIK_MODEL: üretim varsayılanı `siglip2-animal` bu ortamda `sentencepiece`
paketi kurulu olmadığı ve ağırlık (~1,4 GB) cache'de olmadığı için
yüklenemedi -- ölçüm `KIMLIK_MODEL=clip` ortam değişkeniyle (testlerin/CI'ın
zaten kullandığı, ağ bağımsız model) çalıştırıldı. Üretim modeliyle
tekrarlamak isteyen `pip install sentencepiece` sonrası bu betiği
`KIMLIK_MODEL` ayarlamadan (varsayılanı kullanarak) çalıştırabilir.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attributes import attribute_analyzer  # noqa: E402
from app.embedder import embedder  # noqa: E402
from app.kirpma import _dedektoru_al, hayvana_kirp  # noqa: E402
from app.matcher import compute_final_score  # noqa: E402

MESAFE_KM = 1.0  # teshis_arkaplan.py/gercek_veri_olcum.py ile aynı, kıyaslanabilsin
TOHUM = 0  # herkes aynı bireyleri/çiftleri seçsin


def _tabloyu_kur(kok: Path) -> list[tuple[str, bytes]]:
    """<kök>/<birey>/<fotoğraf> düzenini (foto_yolu -> ham bayt) listesine çevirir.

    model_yarisi.py'nin _tabloyu_kur'uyla AYNI düz-klasör sözleşmesi -- ayrı
    bir indirme/ayrıştırma mantığı icat etmiyoruz, ekip zaten bu düzeni
    kullanıyor (kedi_verisi_indir.py bunu üretir).

    model_yarisi.py'nin kendi _cok_kucukleri_ele'siyle AYNI gerekçeyle,
    app/embedder.py'nin ASGARI_KENAR sınırının altındaki fotoğraflar burada
    da elenir: "MPDD kümesinde bu sınırın altında fotoğraflar var" (bkz. o
    fonksiyonun docstring'i) -- elenmezse goruntu_ac() GecersizGoruntu
    fırlatıp TÜM ölçümü çökertir. Gerçek MPDD verisiyle ilk çalıştırmada
    tam olarak bu oldu (27x57 bir fotoğraf)."""
    from app.gorsel_sabitleri import ASGARI_KENAR
    from PIL import Image

    satirlar: list[tuple[str, bytes]] = []
    elenen = 0
    for birey_dizin in sorted(p for p in kok.iterdir() if p.is_dir()):
        for foto in sorted(birey_dizin.iterdir()):
            if foto.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            ham = foto.read_bytes()
            try:
                with Image.open(foto) as im:
                    if min(im.size) < ASGARI_KENAR:
                        elenen += 1
                        continue
            except Exception:
                elenen += 1
                continue
            satirlar.append((birey_dizin.name, ham))
    if elenen:
        print(f"  {elenen} fotoğraf elendi (servisimizin asgari {ASGARI_KENAR}px "
              f"sınırının altında ya da okunamıyor)")
    return satirlar


def _bireylere_grupla(satirlar: list[tuple[str, bytes]]) -> dict[str, list[bytes]]:
    gruplar: dict[str, list[bytes]] = {}
    for birey, ham in satirlar:
        gruplar.setdefault(birey, []).append(ham)
    return gruplar


def _analiz_et(ham: bytes, *, kirp: bool) -> dict:
    """Bir fotoğrafı TAM ya da HAYVANA-KIRPILMIŞ olarak analiz eder.

    `kirp=True` iken app/kirpma.py'nin GERÇEK üretim fonksiyonu çağrılır --
    CROP_TO_ANIMAL env bayrağından BAĞIMSIZ olarak (bu betik iki koşulu da
    aynı çalıştırmada kıyaslamak zorunda, tek bir global bayrakla değil)."""
    from app.embedder import goruntu_ac

    img = goruntu_ac(ham)
    if kirp:
        img = hayvana_kirp(img)
        import io
        tampon = io.BytesIO()
        img.convert("RGB").save(tampon, format="JPEG", quality=95)
        ham = tampon.getvalue()

    emb = embedder.embed_bytes(ham)
    d = attribute_analyzer.analyze(ham, emb)
    d["embedding"] = emb
    return d


def _ciftler(a_listesi, b_listesi=None) -> list[dict]:
    ikili = combinations(a_listesi, 2) if b_listesi is None else product(a_listesi, b_listesi)
    return [
        compute_final_score(x["embedding"], y["embedding"], x["labels"], y["labels"],
                            MESAFE_KM, x["species"], y["species"])
        for x, y in ikili
    ]


def _farkli_indeksleri_ornekle(
    bireyler: dict[str, list[dict]], rng: np.random.Generator, hedef_sayi: int,
) -> list[tuple[str, str, int, int]]:
    """`hedef_sayi` kadar rastgele (birey_a, birey_b, foto_indeks_a,
    foto_indeks_b) örneği üretir -- tekrar (aynı çiftin iki kez seçilmesi)
    istatistiksel bir AUC tahmini için sorun değildir, standart örnekleme
    pratiğidir. `bireyler`in sadece ANAHTARLARI ve fotoğraf SAYILARI
    kullanılır -- TAM ve HAYVAN koşulu aynı bireyler/fotoğraf sayılarını
    paylaştığı için (olcumu_calistir'de aynı `fotolar` listesinden üretilir)
    bu indeksler ikisine de uygulanabilir."""
    isimler = list(bireyler)
    sonuc = []
    for _ in range(hedef_sayi):
        a, b = rng.choice(isimler, size=2, replace=False)
        i = int(rng.integers(len(bireyler[a])))
        j = int(rng.integers(len(bireyler[b])))
        sonuc.append((str(a), str(b), i, j))
    return sonuc


def auc(ayni: list[float], farkli: list[float]) -> float:
    """Mann-Whitney U / (n*m) -- teshis_arkaplan.py'yle AYNI, ölçekten
    bağımsız ölçüt (bkz. o betiğin 'ÖLÇÜT NOTU')."""
    if len(ayni) == 0 or len(farkli) == 0:
        return float("nan")
    kazanc = sum((a > f) + 0.5 * (a == f) for a in ayni for f in farkli)
    return kazanc / (len(ayni) * len(farkli))


# "Farklı birey" çiftlerinin sayısı, birey sayısıyla KARESEL büyür --
# C(191,2)=18.145 birey-çifti × foto_sayisi_birey² foto-kombosu, gerçek MPDD
# verisiyle ilk denemede ~290.000 karşılaştırmaya çıktı ve ölçüm 20+ dakikada
# bitmedi (bkz. bu betiğin commit tarihçesi). AUC zaten bir ORANDIR --
# tüm çiftleri değil, rastgele bir örneklemini kullanmak istatistiksel olarak
# geçerli ve süreyi birey sayısından BAĞIMSIZ kılıyor.
MAKS_FARKLI_CIFT = 3000


def olcumu_calistir(kok: Path, birey_sayisi: int, foto_sayisi_birey: int) -> dict:
    """Ana ölçüm: TAM vs HAYVAN koşulunda aynı-birey/farklı-birey AUC'si.

    ADİL KIYAS: her iki koşul da AYNI fotoğraf çiftlerinden (ve AYNI örneklenmiş
    farklı-birey indekslerinden -- bkz. `_farkli_indeksleri_ornekle`) üretilir,
    yalnızca ön işleme -- kırpılmış mı değil mi -- değişir. AUC farkı bu yüzden
    örnekleme şansından değil, kırpmanın kendisinden gelir. AYNI kaynaktan
    (tek veri kümesi) farklı bireyler kullanılır -- teshis_arkaplan.py'nin
    stüdyo/sokak konfondu burada YOK, çünkü hem "aynı birey" hem "farklı
    birey" grubu AYNI ortamdan (MPDD/CatIndividualImages'ın telefonla çekilmiş
    ev/sokak fotoğrafları) geliyor.
    """
    satirlar = _tabloyu_kur(kok)
    if not satirlar:
        raise SystemExit(f"HATA: {kok} altında <birey>/<fotoğraf> yapısı bulunamadı.")

    gruplar = _bireylere_grupla(satirlar)
    uygun = sorted(b for b, fotolar in gruplar.items() if len(fotolar) >= 2)
    if len(uygun) < 2:
        raise SystemExit("HATA: en az 2 fotoğraflı en az 2 birey gerekiyor.")

    rng = np.random.default_rng(TOHUM)
    secilen = rng.choice(np.array(uygun), size=min(birey_sayisi, len(uygun)), replace=False)

    tam_bireyler: dict[str, list[dict]] = {}
    hayvan_bireyler: dict[str, list[dict]] = {}
    dedektor = _dedektoru_al()  # bir kez yükle, her fotoğrafta yeniden yükleme
    for birey in secilen:
        fotolar = gruplar[birey][:foto_sayisi_birey]
        tam_bireyler[birey] = [_analiz_et(h, kirp=False) for h in fotolar]
        hayvan_bireyler[birey] = [_analiz_et(h, kirp=True) for h in fotolar]

    hedef_farkli = min(MAKS_FARKLI_CIFT, len(secilen) * (len(secilen) - 1) // 2)
    farkli_indeksleri = _farkli_indeksleri_ornekle(tam_bireyler, rng, hedef_farkli)

    def _auc_hesapla(bireyler: dict[str, list[dict]]) -> float:
        ayni: list[float] = []
        for isim in bireyler:
            ayni.extend(s["visual"] for s in _ciftler(bireyler[isim]))
        farkli = [
            compute_final_score(
                bireyler[a][i]["embedding"], bireyler[b][j]["embedding"],
                bireyler[a][i]["labels"], bireyler[b][j]["labels"],
                MESAFE_KM, bireyler[a][i]["species"], bireyler[b][j]["species"],
            )["visual"]
            for a, b, i, j in farkli_indeksleri
        ]
        return auc(np.array(ayni), np.array(farkli))

    return {
        "birey_sayisi": len(secilen),
        "tam_auc": _auc_hesapla(tam_bireyler),
        "hayvan_auc": _auc_hesapla(hayvan_bireyler),
        "_dedektor_kullanildi": dedektor is not None,
    }


def main() -> None:
    # Windows konsolunun varsayılan kod sayfası (ör. cp1254) "⇒" gibi
    # karakterleri kodlayamayıp UnicodeEncodeError ile ÇÖKÜYORDU -- gerçek
    # bir çalıştırmada tam da kararın kendisini yazdıracağı satırda oldu
    # (bkz. bu betiğin PR/commit tarihçesi). reconfigure Python 3.7+'ta var
    # ve mevcut olmadığı (bazı gömülü/eski ortamlar) durumlarda sessizce
    # atlanır -- o zaman en kötü ihtimalle eski davranışa (olası hata) geri
    # dönülür, yeni bir çökme türü eklenmez.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kok", required=True, type=Path,
                    help="<kök>/<birey>/<fotoğraf> düzenindeki veri klasörü "
                         "(ör. data/CatIndividualImages, bkz. scripts/kedi_verisi_indir.py)")
    ap.add_argument("--birey-sayisi", type=int, default=40)
    ap.add_argument("--foto-sayisi-birey", type=int, default=3)
    args = ap.parse_args()

    if not args.kok.exists():
        raise SystemExit(
            f"HATA: {args.kok} yok. Önce indirin: "
            f"venv\\Scripts\\python.exe scripts/kedi_verisi_indir.py")

    print(f"Veri: {args.kok} -- {args.birey_sayisi} birey, birey başına en fazla "
          f"{args.foto_sayisi_birey} fotoğraf")
    print("TAM ve HAYVAN koşulları hazırlanıyor (dedektör + CLIP)... birkaç dakika sürebilir.\n")

    sonuc = olcumu_calistir(args.kok, args.birey_sayisi, args.foto_sayisi_birey)

    print("=" * 78)
    print(f"KARAR — {sonuc['birey_sayisi']} birey, ADİL küme (tek kaynak, konfonde yok)")
    print("=" * 78)
    print(f"  TAM (bugünkü sistem)      görsel AUC: {sonuc['tam_auc']:.3f}")
    print(f"  HAYVAN (kırpılmış)        görsel AUC: {sonuc['hayvan_auc']:.3f}")
    print()
    if sonuc["hayvan_auc"] > sonuc["tam_auc"]:
        print(f"  ⇒ Kırpma bu ADİL kümede ({args.kok.name}) ayrımı YÜKSELTİYOR "
              f"({sonuc['tam_auc']:.3f} → {sonuc['hayvan_auc']:.3f}).")
        print("     docs/olcum-raporu.md §7'nin 'birden çok koşulda tutarlı kazanmalı'")
        print(f"     kuralına göre: aynı ölçümü BAŞKA bir kaynaktan veriyle (ör. "
              f"{args.kok.name} değilse CatIndividualImages, oysa buysa MPDD) tekrarlayıp")
        print("     iki kümede de aynı yönde çıktığını doğrulamadan CROP_TO_ANIMAL=true")
        print("     YAPILMAMALI.")
    else:
        print(f"  ⇒ Kırpma bu kümede ({args.kok.name}) ayrımı yükseltmiyor "
              f"({sonuc['tam_auc']:.3f} → {sonuc['hayvan_auc']:.3f}).")
        print("     CROP_TO_ANIMAL açılmamalı; app/kirpma.py yetenek olarak kalmalı.")


if __name__ == "__main__":
    main()
