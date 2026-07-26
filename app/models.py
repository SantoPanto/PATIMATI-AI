# app/models.py
# Alan adları docs/entegrasyon-sozlesmesi.md ile hizalıdır (ad_id, model_version...).
from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    embedding: list[float]          # 512 boyutlu CLIP vektörü
    labels: list[str]               # Eşleştirme skorunda kullanılan etiketler
    species: str                    # "cat" | "dog" | "unknown"
    species_confidence: float = 0.0
    is_pet: bool = True             # False → fotoğrafta kedi/köpek görünmüyor
    breed: str | None = None        # Güven eşiğinin altındaysa None
    breed_confidence: float = 0.0
    pattern: str | None = None      # tabby | spotted | solid | bicolor
    colors: list[dict]              # Dominant renkler
    model_version: str              # Hangi model/ön işlemeyle üretildi


class AnalyzeUrlRequest(BaseModel):
    """Kuyruk mesajının taşıdığı biçim: dosya değil adres.

    Üretim akışında fotoğraflar S3'te durur ve mesajda yalnızca adresleri
    taşınır — büyük dosyaları kuyruktan geçirmemek için.
    """
    photo_urls: list[str] = Field(min_length=1, max_length=5)


class MatchCandidate(BaseModel):
    ad_id: int
    # İlanın HER fotoğrafı için bir vektör. Tek fotoğrafla eşleşme oranı gerçek
    # veride %24'te kaldığı için çoklu fotoğraf zorunlu hale geldi — görsel skor
    # en iyi fotoğraf çiftinden alınır (bkz. docs/olcum-raporu.md).
    embeddings: list[list[float]] = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)
    species: str = "unknown"
    distance_km: float = 0.0
    # Adayın vektörü hangi model sürümüyle üretildi. Farklıysa aday ATLANIR:
    # farklı sürümlerin vektörleri kıyaslanamaz ve sessizce yanlış sonuç üretir.
    # None → sürüm bilinmiyor, eski kayıt sayılır ve yine atlanır.
    model_version: str | None = None


class MatchRequest(BaseModel):
    embeddings: list[list[float]] = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)
    species: str = "unknown"
    candidates: list[MatchCandidate] = Field(default_factory=list)
    # Verilirse aday listesindeki aynı kimlikli kayıt elenir (ilan kendisiyle
    # eşleşmesin). Java yanlışlıkla gönderirse %100 skorla eşleşme çıkardı.
    ad_id: int | None = None


class MatchResult(BaseModel):
    ad_id: int
    score: float
    visual: float
    label: float
    location: float
    match: bool
    photo_a: int | None = None   # eşleşen fotoğraf çiftinin indeksleri
    photo_b: int | None = None
