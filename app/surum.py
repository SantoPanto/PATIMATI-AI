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
    # Varsayılan. Ayrı bir model yüklenmez, etiketçi CLIP kimlik için de kullanılır.
    # Ek bellek maliyeti sıfır; ölçümde en zayıf seçenek ama her yerde çalışır.
    "clip": {
        "kimlik": "openai/clip-vit-base-patch32",
        "islemci": None,
        "surum": "clip-vit-base-patch32/v1",
        "boyut": 512,
    },
    # Model yarışının kazananı (docs/olcum-raporu.md §7). Üç veri kümesinde de
    # en düşük hata oranı: EER 0.052, CLIP'in 0.113'üne karşı.
    #
    # İKİ UYARI:
    # 1) Ağırlıklar ~1.4 GB ve depoda değil. KIMLIK_MODEL_YOLU ile yerel bir
    #    klasör gösterilebilir; yoksa HuggingFace'ten iner.
    # 2) LİSANSI BELİRTİLMEMİŞ. Staj kapsamında dağıtım olmadığı için
    #    kullanılıyor; ticarileşme hâlinde yeniden değerlendirilmeli.
    "siglip2-animal": {
        "kimlik": os.getenv("KIMLIK_MODEL_YOLU",
                            "AvitoTech/SigLIP2-Base-for-animal-identification"),
        # Depoda preprocessor_config.json yok; ön işleme taban modelden alınıyor
        # (config: siglip, 224px, patch16 — SigLIP 1 ve 2 görüntü hazırlığı aynı).
        "islemci": "google/siglip-base-patch16-224",
        "surum": "siglip2-animal/v2",
        "boyut": 768,
    },
    # Lisansı temiz yedek (Apache 2.0). Ölçümde EER 0.091 — kazananın iki katı
    # hata, ama ticari kullanım gerekirse sorunsuz.
    "google-siglip2": {
        "kimlik": "google/siglip2-base-patch16-224",
        "islemci": None,
        "surum": "google-siglip2-base/v1",
        "boyut": 768,
    },
}

SECILEN_KIMLIK = os.getenv("KIMLIK_MODEL", "clip")
if SECILEN_KIMLIK not in KIMLIK_MODELLERI:
    raise ValueError(
        f"KIMLIK_MODEL='{SECILEN_KIMLIK}' tanınmıyor. "
        f"Seçenekler: {', '.join(KIMLIK_MODELLERI)}")

_SECIM = KIMLIK_MODELLERI[SECILEN_KIMLIK]

MODEL_SURUMU = _SECIM["surum"]
# Embedding boyutu — gelen adayların doğrulanmasında kullanılır
VEKTOR_BOYUTU = _SECIM["boyut"]
