# app/surum.py
"""Model sürümü — embedding'in hangi model ve ön işlemeyle üretildiğini belirtir.

Bu değer HER sonuç mesajıyla döner ve Java tarafında ilanın yanında
`ai_model_version` sütununda saklanır (bkz. docs/entegrasyon-sozlesmesi.md).

AŞAĞIDAKİLERDEN BİRİ DEĞİŞİRSE SÜRÜMÜ ARTIR:
  - CLIP model kimliği (embedder.PetEmbedder.MODEL_ID)
  - görüntü ön işleme (EXIF döndürme, kırpma, alfa birleştirme, yeniden boyutlandırma)
  - embedding'in üretilme biçimi (normalizasyon, katman seçimi)

Sürüm arttığında eski vektörler yenileriyle KIYASLANAMAZ. Eşleştirmede farklı
sürümlü adaylar atlanır ve `skipped_candidates` içinde raporlanır; o ilanların
yeniden analiz edilmesi gerekir.

Sürümü artırmadan ön işlemeyi değiştirmek en tehlikeli hatadır: hata vermez,
sadece sessizce yanlış benzerlik üretir.
"""

MODEL_SURUMU = "clip-vit-base-patch32/v1"

# Embedding boyutu — gelen adayların doğrulanmasında kullanılır
VEKTOR_BOYUTU = 512
