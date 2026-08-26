# app/surum.py
"""Kimlik modeli seçimi ve model sürümü.

Servis İKİ model kullanır ve ikisi farklı iş yapar:

  ETİKETÇİ (CLIP, sabit)   tür / desen / renk / cins / "hayvan mı" kapısı
  KİMLİKÇİ (seçilebilir)   eşleştirmede kullanılan vektör

Neden ayrıldılar: 2026-07-28'de ölçtük (`scripts/etiket_yetenegi.py`, 111 fotoğraf).
Kimlik için ince ayarlanmış modeller etiket işini yapamıyor —
`avito-siglip2` cins tahmininde %3.6 aldı (rastgele %2.7), tür ayrımında %64.9.
CLIP ise cinste %78.4, türde %100. Tek model iki işi birden yapamıyor.

MODEL_SURUMU her sonuç mesajıyla döner ve Java tarafında ilanın yanında
`ai_model_version` sütununda saklanır (bkz. docs/entegrasyon-sozlesmesi.md).

SÜRÜMÜ ARTIRMAYI GEREKTİREN DEĞİŞİKLİKLER:
  - kimlik modelinin değişmesi
  - görüntü ön işleme (EXIF döndürme, alfa birleştirme, boyut sınırı)
  - embedding'in üretilme biçimi (normalizasyon, katman seçimi)

Sürüm arttığında eski vektörler yenileriyle KIYASLANAMAZ. Eşleştirmede farklı
sürümlü adaylar atlanır ve `skipped_candidates` içinde raporlanır; o ilanların
yeniden analiz edilmesi gerekir.

Sürümü artırmadan ön işlemeyi değiştirmek en tehlikeli hatadır: hata vermez,
sadece sessizce yanlış benzerlik üretir. Bu yüzden sürüm etiketi burada
modelle BİRLİKTE tanımlanıyor — modeli değiştirip sürümü unutmak mümkün olmasın.
"""
import os

# Kimlik modeli seçenekleri. Anahtar = KIMLIK_MODEL ortam değişkenine yazılan ad.
KIMLIK_MODELLERI = {
    # Ayrı bir model yüklenmez, etiketçi CLIP kimlik için de kullanılır: ek bellek
    # maliyeti sıfır ve indirilecek ağırlık yok. Ölçümde EN ZAYIF seçenek
    # (EER 0.113), o yüzden ÜRETİMİN varsayılanı değil — ama her makinede ağ
    # olmadan çalıştığı için TESTLERİN ve CI'ın seçimi budur
    # (tests/conftest.py sabitliyor, .github/workflows/ci.yml açıkça veriyor).
    "clip": {
        "kimlik": "openai/clip-vit-base-patch32",
        "islemci": None,
        "surum": "clip-vit-base-patch32/v1",
        "boyut": 512,
    },
    # >>> VARSAYILAN (2026-08-19) <<<
    # Model yarışının kazananı (docs/olcum-raporu.md §7). Üç veri kümesinde de
    # en düşük hata oranı: EER 0.052, CLIP'in 0.113'üne karşı.
    #
    # ÜÇ UYARI — varsayılan olduğu için üçü de artık her kurulumu ilgilendiriyor:
    # 1) Ağırlıklar ~1.4 GB ve depoda değil. KIMLIK_MODEL_YOLU ile yerel bir
    #    klasör gösterilebilir; yoksa HuggingFace'ten iner. Dockerfile bunu
    #    BUILD sırasında indirir ki çalışma zamanında ağ bağımlılığı kalmasın.
    # 2) LİSANS: Apache 2.0 — 22.08.2026'da model yazarı (Kirill Borodin,
    #    MTUCI) e-postayla doğruladı ve depoya açıkça işlendi; ticari
    #    kullanım serbest. (19-22.08 arası lisans belirsizdi ve bilinçli
    #    risk olarak taşınıyordu; google-siglip2'ye dönme yedek planı
    #    geçersizleşti.) ⚠ Depo aynı gün güncellendi (yükleme düzeltmesi):
    #    bir SONRAKİ imaj build'i YENİ ağırlıkları indirir — build öncesi
    #    aynı fotoyla eski/yeni embedding karşılaştırılmalı; değerler
    #    değiştiyse toplu yeniden analiz gerekir (zaten açık iş).
    # 3) CI bu modelle KOŞMUYOR (ağırlık 1,4 GB). CI'ın yeşili kodun
    #    sağlamlığını gösterir, bu modelin doğruluğunu değil.
    "siglip2-animal": {
        "kimlik": os.getenv("KIMLIK_MODEL_YOLU",
                            "AvitoTech/SigLIP2-Base-for-animal-identification"),
        # Depoda preprocessor_config.json yok; ön işleme taban modelden alınıyor
        # (config: siglip, 224px, patch16 — SigLIP 1 ve 2 görüntü hazırlığı aynı).
        "islemci": "google/siglip-base-patch16-224",
        "surum": "siglip2-animal/v2",
        "boyut": 768,
    },
    # Yedek model (Apache 2.0). Ölçümde EER 0.091 — kazananın iki katı hata.
    # 22.08.2026'ya kadar "lisansı temiz tek seçenek" olduğu için duruyordu;
    # artık varsayılanın lisansı da temiz, bu yalnız teknik alternatif.
    "google-siglip2": {
        "kimlik": "google/siglip2-base-patch16-224",
        "islemci": None,
        "surum": "google-siglip2-base/v1",
        "boyut": 768,
    },
}

# Varsayılan buradan okunur. Değiştirirken .env.example ve Dockerfile'daki ARG
# da değişmeli — tests/test_varsayilan_model_tutarliligi.py üçünün kaymasını
# engelliyor (eşik için aynı işi test_esik_tutarliligi.py yapıyor).
SECILEN_KIMLIK = os.getenv("KIMLIK_MODEL", "siglip2-animal")
if SECILEN_KIMLIK not in KIMLIK_MODELLERI:
    raise ValueError(
        f"KIMLIK_MODEL='{SECILEN_KIMLIK}' tanınmıyor. "
        f"Seçenekler: {', '.join(KIMLIK_MODELLERI)}")

_SECIM = KIMLIK_MODELLERI[SECILEN_KIMLIK]

MODEL_SURUMU = _SECIM["surum"]
# Embedding boyutu — gelen adayların doğrulanmasında kullanılır
VEKTOR_BOYUTU = _SECIM["boyut"]
