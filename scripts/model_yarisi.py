"""Model yarışı: hangi gömme modeli BİREYİ en iyi ayırıyor?

NEDEN BU BETİK VAR
------------------
Bugüne kadar yaptığımız her ölçüm tek bir soruyu sordu: "CLIP'i tüm sahneye
uygularsak nasıl ayarlarım?" Hiçbiri "CLIP'i tüm sahneye uygulamak doğru mu?"
diye sormadı. Tek bir alternatif model bile denemediğimiz için "%24 iyi mi kötü
mü" sorusunun cevabı elimizde yok. Eksik olan bir model değil, KIYASLAMA DÜZENEĞİ.

Bu betik o düzenek. Girdi: birey etiketli bir veri kümesi + bir gömme modeli.
Çıktı: aynı veride, aynı ölçütlerle karşılaştırılabilir tek bir satır.

ÖLÇÜTLER — ve neden bunlar
--------------------------
Eski raporlarımızda "ayrım gücü" (iki ortalamanın farkı) kullanılmıştı. O ölçüt
ÖLÇEĞE DUYARLI: bir koşul tüm skorları yukarı ittiğinde fark daralır, oysa
sıralama kalitesi artmış olabilir. Kırpma kararını bu yüzden yanlış verdik
(bkz. docs/olcum-raporu.md §4 düzeltmesi). Burada alanda standart olanlar var:

  Top-1  : sorguya en benzeyen fotoğraf aynı hayvana mı ait? (ürünün asıl sorusu)
  Top-5  : ilk beşte doğru hayvan var mı? (sıralı aday listesi tasarımımızın karşılığı)
  mAP    : doğru eşleşmeler listenin ne kadar üstünde toplanıyor
  ROC AUC: rastgele bir "aynı" çifti, rastgele bir "farklı" çiftinden yüksek
           puan alma olasılığı — eşikten ve ölçekten tamamen bağımsız
  EER    : kaçırma ve yanlış alarm oranlarının eşitlendiği nokta (düşük = iyi)

DOĞRULUK KONTROLÜ
-----------------
AvitoTech kendi modelini DogFaceNet üzerinde ölçüp yayınladı (Top-1 %59.25,
ROC AUC 0.9776, EER 0.0672). Aynı modeli aynı kümede koşturunca benzer sayıları
görüyorsak DÜZENEĞİMİZ DOĞRU demektir. Bu, kendi ölçüm aracımızı doğrulamanın
tek yolu — daha önce hiç yapamadığımız şey.

Kullanım:
  venv\\Scripts\\python.exe scripts/model_yarisi.py --liste
  venv\\Scripts\\python.exe scripts/model_yarisi.py --veri DogFaceNet --model bizim
  venv\\Scripts\\python.exe scripts/model_yarisi.py --veri DogFaceNet --model hepsi
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
# Sonuçlar depoya GİRER: dosyalar minik ve ölçüm raporunun ham kanıtı.
# (data/ klasörü veri kümeleri için git dışında; sonuçları oraya koymak
#  onları ekipten gizlerdi.)
SONUC_DIZIN = ROOT / "docs" / "olcum-sonuclari"

# --------------------------------------------------------------------------
# Adaylar
# --------------------------------------------------------------------------
ADAYLAR = {
    "bizim": {
        "tur": "app",
        "kimlik": "openai/clip-vit-base-patch32",
        "aciklama": "MEVCUT SİSTEMİMİZ — app/embedder.py'nin kendisi (taban çizgisi)",
    },
    # AvitoTech depolarında YALNIZCA config.json + model.safetensors var;
    # preprocessor_config.json yayınlanmamış. Yani ağırlıklar elimizde ama
    # "fotoğrafı modele nasıl hazırlayacaksın" tarifi yok. Ön işlemeyi taban
    # modelden alıyoruz (config'teki mimari ve girdi boyutuyla eşleşiyor).
    # DİKKAT: ön işleme yanlış seçilirse model SESSİZCE kötü skor verir ve
    # haksız yere elenir. Bu yüzden tam kümedeki sayıyı AvitoTech'in yayınladığı
    # DogFaceNet sonucuyla (Top-1 %59.25) karşılaştırmak şart.
    "avito-clip": {
        "tur": "hf",
        "kimlik": "AvitoTech/CLIP-ViT-base-for-animal-identification",
        "islemci": "openai/clip-vit-base-patch32",   # config: clip, 224px, patch 32
        "aciklama": "Aynı CLIP mimarisi, hayvan kimliği için ince ayarlı → en ucuz geçiş",
    },
    "avito-siglip2": {
        "tur": "hf",
        "kimlik": "AvitoTech/SigLIP2-Base-for-animal-identification",
        "islemci": "google/siglip-base-patch16-224",  # config: siglip, 224px, patch 16
        "aciklama": "SigLIP2, 768 boyut, 1.9M foto/695k birey ile eğitilmiş",
    },
    "megadescriptor": {
        "tur": "timm",
        "kimlik": "hf-hub:BVRA/MegaDescriptor-L-384",
        "aciklama": "Hayvan re-ID uzmanı — DİKKAT: CC-BY-NC-4.0, ticari kullanım YASAK",
    },
    # Ticari kullanıma açık tek güçlü aday ailesi (Apache 2.0). AvitoTech bunları
    # ince ayarlayıp yayınlamış ama kendi ağırlıklarına lisans koymamış; taban
    # modeller ise net biçimde serbest. Ürünleşme ihtimali varsa karar bunlar
    # arasından çıkmak zorunda.
    "google-siglip2": {
        "tur": "hf",
        "kimlik": "google/siglip2-base-patch16-224",
        "aciklama": "Google SigLIP2 tabanı — Apache 2.0, TİCARİ KULLANIMA AÇIK",
    },
    "google-siglip1": {
        "tur": "hf",
        "kimlik": "google/siglip-base-patch16-224",
        "aciklama": "Google SigLIP1 tabanı — Apache 2.0, ticari kullanıma açık",
    },
}


# --------------------------------------------------------------------------
# Gömücü arayüzü — her aday bu sözleşmeyi uygular
# --------------------------------------------------------------------------
class Gomucu:
    """Fotoğraf YOLLARININ listesini (n, d) boyutlu L2-normalize matrise çevirir.

    Girdi neden PIL görüntüsü değil de yol? İki sebep:
    1) Tam veri kümesinde 8.363 fotoğrafın hepsini birden bellekte açık tutmak
       gerekmesin — her gömücü kendi yığını kadarını açsın.
    2) Her model dosyayı KENDİ istediği gibi okusun. Önceden ortak bir PIL
       nesnesi geçiriliyordu ve kendi modelimiz onu yeniden JPEG'e çevirmek
       zorunda kalıyordu; bu, yalnız bize uygulanan bir kalite kaybıydı ve
       kıyası kendi aleyhimize bozuyordu.
    """

    ad = "?"
    boyut = 0

    def kodla(self, yollar):
        raise NotImplementedError


class AppGomucu(Gomucu):
    """Üretimdeki kodun kendisi. Taban çizgisi 'yeniden yazılmış CLIP' olmamalı;
    gerçekten servisin kullandığı yol olmalı, yoksa kıyas dürüst olmaz."""

    def __init__(self):
        from app.embedder import embedder
        from app.surum import MODEL_SURUMU
        self._e = embedder
        self.ad = f"bizim ({MODEL_SURUMU})"
        self.boyut = 512

    def kodla(self, yollar):
        # Servisin gerçek girdisi ham bayt — üretimde de böyle geliyor
        return np.asarray(
            [self._e.embed_bytes(Path(y).read_bytes()) for y in yollar],
            dtype=np.float32)


class HFGomucu(Gomucu):
    """transformers ile yüklenen her görüntü-metin modeli (CLIP, SigLIP2...)."""

    def __init__(self, kimlik, islemci=None, yigin=16):
        from transformers import AutoModel, AutoProcessor
        self.ad = kimlik
        self.yigin = yigin
        # Ön işleme kaynağı ağırlıklardan farklı olabilir (bkz. ADAYLAR notu)
        self._islemci = AutoProcessor.from_pretrained(islemci or kimlik)
        self._model, bilgi = AutoModel.from_pretrained(kimlik, output_loading_info=True)

        # Ağırlıklar tutmadıysa önek sorunu olabilir — onarmayı dene, olmuyorsa çök.
        if bilgi.get("missing_keys") or bilgi.get("mismatched_keys"):
            onarilmis = self._onekli_agirliklari_onar(kimlik)
            if onarilmis is None:
                self._agirliklari_dogrula(kimlik, bilgi)   # tanıdık değil → hata fırlatır
            self._model = onarilmis
        self._model.eval()

    @staticmethod
    def _onekli_agirliklari_onar(kimlik):
        """Tüm anahtarları ortak bir önek taşıyan checkpoint'i yükleyebilir hâle getirir.

        AvitoTech'in SigLIP2 deposunda 408 anahtarın hepsi `clip.` önekli ve
        `text_config.vocab_size` eksik. transformers yalnızca hedef sınıfın kendi
        önekini soyduğu için (SiglipModel'de `siglip`) hiçbir anahtar tutmuyor ve
        model RASTGELE ağırlıkla yükleniyor.

        Bunu daha önce elle düzeltip yerel klasöre koymuştuk — ama o klasör depoda
        olmadığı için ekipten kimse modeli kullanamıyordu (Yusuf PR #6'da bildirdi).
        Artık onarım kodda: önek soyuluyor, eksik vocab_size çıkarılıyor.
        """
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        yol = Path(kimlik)
        dosya = (yol / "model.safetensors") if yol.exists() else Path(
            hf_hub_download(kimlik, "model.safetensors"))
        sd = load_file(str(dosya))

        onek = "clip."
        if not all(k.startswith(onek) for k in sd):
            return None                      # tanıdık bir onarım değil
        sd = {k[len(onek):]: v for k, v in sd.items()}

        cfg = AutoConfig.from_pretrained(kimlik)
        # Metin kulesinin sözlük boyutu config'te yoksa varsayılan (32000) kullanılır
        # ama checkpoint 256000 taşıyor; boyutu ağırlıktan okuyup düzeltiyoruz.
        gomme = sd.get("text_model.embeddings.token_embedding.weight")
        if gomme is not None and hasattr(cfg, "text_config"):
            cfg.text_config.vocab_size = gomme.shape[0]

        model = AutoModel.from_config(cfg)
        eksik, fazla = model.load_state_dict(sd, strict=False)
        if eksik:
            raise RuntimeError(f"{kimlik}: onarımdan sonra da {len(eksik)} anahtar eksik")
        print(f"    (ağırlıklar onarıldı: '{onek}' öneki soyuldu, "
              f"vocab_size={getattr(cfg.text_config, 'vocab_size', '?')})")
        return model

    @staticmethod
    def _agirliklari_dogrula(kimlik, bilgi):
        """Ağırlıklar gerçekten yüklendi mi? Yüklenmediyse GÜRÜLTÜLÜ çök.

        27 Temmuz'da yaşanan gerçek hata: AvitoTech'in SigLIP2 deposundaki 408
        anahtarın hepsi `clip.` önekiyle kaydedilmiş. transformers yalnızca model
        sınıfının kendi önekini (SiglipModel için `siglip`) soyduğu için hiçbiri
        eşleşmedi ve model TAMAMEN RASTGELE ağırlıklarla yüklendi — sadece bir
        uyarı basarak. Ölçüm sessizce çalıştı ve Top-1 %20.2 verdi; en umut
        verici adayı "kötü çıktı" diye eleyecektik.

        Bir gömme modelinin ağırlıkları eksikse ölçüm geçersizdir. Sessiz yanlış
        sonuç yerine açık hata verilmesi şart.
        """
        eksik = bilgi.get("missing_keys") or []
        uyusmayan = bilgi.get("mismatched_keys") or []
        if eksik or uyusmayan:
            raise RuntimeError(
                f"{kimlik}: ağırlıklar tam yüklenmedi "
                f"({len(eksik)} eksik, {len(uyusmayan)} uyuşmayan anahtar). "
                f"Ölçüm geçersiz olurdu. İlk eksikler: {eksik[:3]}"
            )

    def kodla(self, yollar):
        parcalar = []
        for i in range(0, len(yollar), self.yigin):
            grup = [Image.open(y).convert("RGB") for y in yollar[i:i + self.yigin]]
            girdi = self._islemci(images=grup, return_tensors="pt")
            with torch.no_grad():
                ozellik = self._model.get_image_features(**girdi)
            # transformers 5.x bazı modellerde düz tensör yerine nesne döndürüyor
            # (bkz. docs 6.1 dersi) — vektörü içinden al.
            # NOT: burada `x or y` KULLANILMAZ; `or` soldakinin doğruluğuna bakar,
            # tensörün doğruluğu ise tanımsızdır ("ambiguous") ve çöker.
            if not isinstance(ozellik, torch.Tensor):
                havuz = getattr(ozellik, "pooler_output", None)
                ozellik = havuz if havuz is not None else ozellik[0]
            ozellik = ozellik / ozellik.norm(dim=-1, keepdim=True)
            parcalar.append(ozellik.cpu().numpy().astype(np.float32))
        sonuc = np.vstack(parcalar)
        self.boyut = sonuc.shape[1]
        return sonuc


class TimmGomucu(Gomucu):
    """MegaDescriptor ailesi — timm üzerinden."""

    def __init__(self, kimlik, yigin=8):
        import timm
        self.ad = kimlik
        self.yigin = yigin
        self._model = timm.create_model(kimlik, pretrained=True, num_classes=0)
        self._model.eval()
        yapilandirma = timm.data.resolve_model_data_config(self._model)
        self._donustur = timm.data.create_transform(**yapilandirma, is_training=False)

    def kodla(self, yollar):
        parcalar = []
        for i in range(0, len(yollar), self.yigin):
            grup = torch.stack([self._donustur(Image.open(y).convert("RGB"))
                                for y in yollar[i:i + self.yigin]])
            with torch.no_grad():
                ozellik = self._model(grup)
            ozellik = ozellik / ozellik.norm(dim=-1, keepdim=True)
            parcalar.append(ozellik.cpu().numpy().astype(np.float32))
        sonuc = np.vstack(parcalar)
        self.boyut = sonuc.shape[1]
        return sonuc


def yerel_kopya(anahtar):
    """data/models/<anahtar> altında elle indirilmiş bir kopya varsa onu kullan.

    Neden: HuggingFace indiricisi yavaş/kesintili bağlantılarda yeniden denemiyor,
    yarım dosyada saatlerce asılı kalabiliyor. Büyük modelleri `curl -C -` ile
    (kaldığı yerden devam + yeniden deneme) indirip buraya koymak daha güvenilir.
    """
    yol = ROOT / "data" / "models" / anahtar
    return yol if (yol / "config.json").exists() else None


def gomucu_kur(anahtar) -> Gomucu:
    aday = ADAYLAR[anahtar]
    if aday["tur"] == "app":
        return AppGomucu()
    if aday["tur"] == "hf":
        yerel = yerel_kopya(anahtar)
        if yerel:
            print(f"    (yerel kopya kullanılıyor: {yerel})")
        return HFGomucu(str(yerel) if yerel else aday["kimlik"],
                        islemci=aday.get("islemci"))
    if aday["tur"] == "timm":
        return TimmGomucu(aday["kimlik"])
    raise ValueError(aday["tur"])


# --------------------------------------------------------------------------
# Veri
# --------------------------------------------------------------------------
def _tabloyu_kur(ad, kok):
    """(tam_yol, identity) tablosunu üretir. İki kaynak destekleniyor.

    1) wildlife-datasets sınıfı — kütüphanenin kendi indirdiği kümeler için.
    2) DÜZ KLASÖR: <kök>/<birey>/<fotoğraf>. Bu şart, çünkü (a) Kaggle'dan
       parça parça indirdiğimiz kedi kümesi kütüphanenin beklediği arşiv
       yapısında değil, (b) ileride kendi gerçek ilan fotoğraflarımızı da aynı
       düzenekte ölçmek isteyeceğiz — onlar hiçbir kütüphanede yok.
    """
    import pandas as pd
    kok = Path(kok)

    try:
        from wildlife_datasets import datasets
        sinif = getattr(datasets, ad, None)
        if sinif is not None:
            df = sinif(str(kok)).df[["path", "identity"]].copy()
            df["tam_yol"] = df["path"].apply(lambda p: str(kok / p))
            return df
    except Exception as e:
        print(f"  (wildlife-datasets okuyamadı: {type(e).__name__} — düz klasör "
              f"olarak deneniyor)")

    satirlar = []
    for birey_dizin in sorted(p for p in kok.iterdir() if p.is_dir()):
        for foto in sorted(birey_dizin.iterdir()):
            if foto.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                satirlar.append({"tam_yol": str(foto), "identity": birey_dizin.name,
                                 "path": foto.name})
    if not satirlar:
        raise SystemExit(f"HATA: {kok} altında <birey>/<fotoğraf> yapısı bulunamadı.")
    return pd.DataFrame(satirlar)


def _cok_kucukleri_ele(df, asgari_kenar=32):
    """Servisimizin reddettiği boyuttaki fotoğrafları TÜM adaylardan çıkarır.

    app/embedder.py'de 32x32 altındaki görüntüler `GecersizGoruntu` ile
    reddediliyor (bozuk/çöp veriden gelen anlamsız vektörlere karşı konmuş bir
    koruma). MPDD kümesinde bu sınırın altında fotoğraflar var.

    Böyle bir fotoğrafı sadece bizim modelimizin reddetmesi, diğerlerinin
    işlemesi kıyası bozar — modeller farklı kümelerde sınanmış olur. Adil kıyas
    için hepsi BİREBİR aynı fotoğrafları görmeli, o yüzden burada topluca eliyoruz.
    """
    from PIL import Image
    tutulan, elenen = [], 0
    for _, satir in df.iterrows():
        try:
            with Image.open(satir["tam_yol"]) as im:
                if min(im.size) >= asgari_kenar:
                    tutulan.append(satir)
                else:
                    elenen += 1
        except Exception:
            elenen += 1
    if elenen:
        print(f"  {elenen} fotoğraf elendi (servisimizin asgari {asgari_kenar}px "
              f"sınırının altında ya da okunamıyor) — adil kıyas için tüm "
              f"adaylardan çıkarıldı")
    import pandas as pd
    return pd.DataFrame(tutulan).reset_index(drop=True)


def veri_yukle(ad, kok, birey_sayisi, foto_sayisi, tohum):
    """wildlife-datasets kümesinden dengeli bir alt küme seçer.

    Alt küme şart: CPU'da 8.363 fotoğraf × 4 model günler sürer. Sıralama kararı
    için birkaç yüz fotoğraf fazlasıyla yeter; kazanan sonra tam kümede doğrulanır.
    Tohum sabit — herkes aynı alt kümeyi alsın, kıyas anlamlı olsun.
    """
    df = _tabloyu_kur(ad, kok)
    df = _cok_kucukleri_ele(df)

    sayim = df.groupby("identity").size()
    uygun = sayim[sayim >= 2].index          # tek fotoğraflı birey sorgu olamaz
    rng = np.random.default_rng(tohum)
    secilen = rng.choice(np.array(sorted(uygun)), size=min(birey_sayisi, len(uygun)),
                         replace=False)

    satirlar = []
    for kimlik in secilen:
        grup = df[df["identity"] == kimlik].sort_values("path")
        if len(grup) > foto_sayisi:
            idx = rng.choice(len(grup), size=foto_sayisi, replace=False)
            grup = grup.iloc[sorted(idx)]
        satirlar.append(grup)

    import pandas as pd
    alt = pd.concat(satirlar).reset_index(drop=True)
    return alt


# --------------------------------------------------------------------------
# Ölçütler
# --------------------------------------------------------------------------
def olc(vektorler, kimlikler):
    """Her fotoğraf sırayla sorgu; kendisi galeriden çıkarılır."""
    benzerlik = vektorler @ vektorler.T
    n = len(kimlikler)
    np.fill_diagonal(benzerlik, -np.inf)          # kendinle eşleşme yok

    kimlikler = np.asarray(kimlikler)
    ayni = kimlikler[:, None] == kimlikler[None, :]
    np.fill_diagonal(ayni, False)

    top1, top5, ap_listesi = [], [], []
    for i in range(n):
        if not ayni[i].any():                     # galeride eşi kalmadıysa atla
            continue
        sira = np.argsort(-benzerlik[i])
        dogru = ayni[i][sira]
        top1.append(bool(dogru[0]))
        top5.append(bool(dogru[:5].any()))
        kesif = np.cumsum(dogru)
        kesinlik = kesif / (np.arange(len(dogru)) + 1)
        ap_listesi.append(float((kesinlik * dogru).sum() / dogru.sum()))

    # eşikten bağımsız ikili ölçütler
    ust = np.triu_indices(n, k=1)
    puanlar, etiketler = benzerlik[ust], ayni[ust]
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = float(roc_auc_score(etiketler, puanlar))
    fpr, tpr, _ = roc_curve(etiketler, puanlar)
    eer = float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])

    return {
        "top1": float(np.mean(top1)),
        "top5": float(np.mean(top5)),
        "map": float(np.mean(ap_listesi)),
        "roc_auc": auc,
        "eer": eer,
        "sorgu_sayisi": len(top1),
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", default="DogFaceNet", help="wildlife-datasets sınıf adı")
    ap.add_argument("--kok", default=None, help="veri klasörü (varsayılan data/<veri>)")
    ap.add_argument("--model", default="bizim", help="aday anahtarı ya da 'hepsi'")
    ap.add_argument("--birey", type=int, default=200, help="kaç birey")
    ap.add_argument("--foto", type=int, default=4, help="birey başına en fazla foto")
    ap.add_argument("--tohum", type=int, default=42)
    ap.add_argument("--liste", action="store_true", help="adayları listele ve çık")
    a = ap.parse_args()

    if a.liste:
        print("Adaylar:")
        for k, v in ADAYLAR.items():
            print(f"  {k:16} {v['aciklama']}")
        return

    kok = Path(a.kok) if a.kok else ROOT / "data" / a.veri
    if not kok.exists():
        print(f"HATA: {kok} yok. Önce veri kümesini indirin.")
        return

    print(f"Veri: {a.veri} · {a.birey} birey × en fazla {a.foto} foto (tohum {a.tohum})")
    alt = veri_yukle(a.veri, kok, a.birey, a.foto, a.tohum)
    print(f"  seçilen: {len(alt)} fotoğraf, {alt['identity'].nunique()} birey\n")

    yollar = alt["tam_yol"].tolist()
    kimlikler = alt["identity"].tolist()

    secilenler = list(ADAYLAR) if a.model == "hepsi" else [a.model]
    SONUC_DIZIN.mkdir(parents=True, exist_ok=True)
    tablo = []

    for anahtar in secilenler:
        print(f"--- {anahtar} · {ADAYLAR[anahtar]['aciklama']}")
        try:
            g = gomucu_kur(anahtar)
        except ImportError as e:
            print(f"    ATLANDI — kütüphane eksik: {e}\n")
            continue
        except Exception as e:
            print(f"    ATLANDI — model yüklenemedi: {type(e).__name__}: {e}\n")
            continue

        basla = time.time()
        vektorler = g.kodla(yollar)
        sure = time.time() - basla
        sonuc = olc(vektorler, kimlikler)
        sonuc.update({
            "aday": anahtar, "model": g.ad, "boyut": int(vektorler.shape[1]),
            "veri": a.veri, "birey": int(alt["identity"].nunique()),
            "fotograf": len(alt), "tohum": a.tohum,
            "saniye_foto": round(sure / len(yollar), 3),
        })
        tablo.append(sonuc)
        print(f"    Top-1 {sonuc['top1']:.1%} · Top-5 {sonuc['top5']:.1%} · "
              f"mAP {sonuc['map']:.3f} · AUC {sonuc['roc_auc']:.4f} · "
              f"EER {sonuc['eer']:.4f} · {sonuc['saniye_foto']}sn/foto\n")

        # Dosya adında ÖLÇEK de var: aynı modelin 150 bireylik ve tam kümedeki
        # sonuçları ayrı ayrı saklansın. (Önce yoktu ve tam küme sonucu, sonradan
        # yapılan küçük bir doğrulama koşumu tarafından silinmişti.)
        (SONUC_DIZIN / f"{a.veri}_{anahtar}_{sonuc['birey']}birey.json").write_text(
            json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8")

    if not tablo:
        print("Hiçbir aday koşmadı.")
        return

    print("=" * 84)
    print(f"SONUÇ TABLOSU — {a.veri}, {alt['identity'].nunique()} birey / "
          f"{len(alt)} fotoğraf")
    print("=" * 84)
    print(f"  {'aday':16} {'boyut':>6} {'Top-1':>7} {'Top-5':>7} {'mAP':>7} "
          f"{'AUC':>8} {'EER':>7} {'sn/foto':>8}")
    for s in sorted(tablo, key=lambda x: -x["top1"]):
        print(f"  {s['aday']:16} {s['boyut']:>6} {s['top1']:>6.1%} {s['top5']:>6.1%} "
              f"{s['map']:>7.3f} {s['roc_auc']:>8.4f} {s['eer']:>7.4f} "
              f"{s['saniye_foto']:>8.3f}")
    print()
    birey_sayisi = alt["identity"].nunique()
    if birey_sayisi < 500:
        print("  ⚠ DİKKAT — bu sayılar İYİMSER, dışarıyla kıyaslamayın.")
        print(f"     Galeride yalnızca {birey_sayisi} birey var. Aday havuzu küçüldükçe")
        print("     doğru cevabı bulmak kolaylaşır; Top-1 yapay olarak yükselir.")
        print("     Bu koşum yalnızca ADAYLARI BİRBİRİYLE kıyaslamak için geçerlidir —")
        print("     hepsi aynı zorlukla karşılaştığı için sıralama anlamlıdır.")
        print("     Yayınlanmış sayılarla kıyas için tam kümede koşun (--birey 99999).")
    else:
        print("  Karşılaştırma noktası — AvitoTech'in DogFaceNet (tam küme) için")
        print("  yayınladığı: avito-siglip2 → Top-1 %59.25 · ROC AUC 0.9776 · EER 0.0672")
        print("  Bizim sayımız aynı mertebedeyse DÜZENEĞİMİZ DOĞRU çalışıyor demektir.")


if __name__ == "__main__":
    main()
