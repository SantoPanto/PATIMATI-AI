# app/hatalar.py
"""Sözleşmedeki `error.code` değerlerine birebir karşılık gelen hata sınıfları.

Kuyruk tüketicisi yakaladığı hatanın `KOD` alanını doğrudan sonuç mesajına
yazar; böylece hata kodu tek yerde tanımlı kalır ve kod ile belge birbirinden
kaymaz (bkz. docs/entegrasyon-sozlesmesi.md §4).
"""


class AIHatasi(Exception):
    """Tüm AI servisi hatalarının tabanı."""
    KOD = "INTERNAL"


class FotografYok(AIHatasi):
    """İstekte hiç fotoğraf adresi gelmedi."""
    KOD = "NO_PHOTOS"


class FotografIndirilemedi(AIHatasi):
    """Adres indirilemedi (404, zaman aşımı, boyut sınırı)."""
    KOD = "PHOTO_DOWNLOAD_FAILED"


class GecersizGoruntu(AIHatasi):
    """İndirildi ama görüntü olarak açılamadı ya da kullanılamayacak kadar bozuk."""
    KOD = "INVALID_IMAGE"


class DesteklenmeyenSema(AIHatasi):
    """Mesajın `schema_version` değeri tanınmıyor."""
    KOD = "UNSUPPORTED_SCHEMA"


class ModelHatasi(AIHatasi):
    """Model çalışırken hata verdi."""
    KOD = "MODEL_ERROR"


class GecersizEmbedding(ValueError):
    """Aday embedding'i kullanılamaz (yanlış boyut, NaN, sonsuz).

    Sözleşmede karşılığı olan bir hata kodu YOKTUR: bu, tüm mesajı düşüren bir
    hata değil, tek bir adayı atlatan bir durumdur. Atlananların sayısı sonuçta
    `skipped_candidates` altında raporlanır.
    """
