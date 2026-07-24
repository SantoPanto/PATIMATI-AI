# PATIMATI-AI

PatiMati (PawFinder) kayıp hayvan eşleştirme platformunun yapay zeka servisi.

Aşama A yaklaşımı: model eğitimi YOK — önceden eğitilmiş modellerle özellik
çıkarma (feature extraction) ve hibrit skorlama.

## Mimari

| Bileşen | Görev |
|---|---|
| CLIP ViT-B/32 | Fotoğraftan 512 boyutlu embedding (görsel parmak izi) |
| Google Vision API | Etiket + tür tespiti (cat/dog) + dominant renk |
| Hibrit skor | %55 görsel + %30 etiket + %15 konum → eşik 0.70 |
| FastAPI | `/health`, `/analyze`, `/match` endpoint'leri (port 8000) |

Spring Boot backend bu servise HTTP ile bağlanır (bkz. teknik rapor bölüm 10).

## Kurulum

```bash
# Python 3.11 gerekli
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Ortam değişkenleri
copy .env.example .env         # sonra .env içini doldur
```

Google Vision API anahtarı için: Google Cloud Console → Cloud Vision API etkinleştir
→ Service Account oluştur → JSON anahtarı `google_key.json` olarak proje köküne koy.
(`google_key.json` ve `.env` .gitignore'dadır, asla commit'lenmez.)

## Çalıştırma

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

## Docker

```bash
docker build -t patimati-ai .
docker run -p 8000:8000 patimati-ai
```

Model imajın içine build sırasında gömülür (~600 MB); container ağ erişimi olmadan
da çalışır. Railway'de `PORT` ortam değişkeni otomatik kullanılır.

## Test

```bash
python scripts/prepare_seed.py   # Oxford-IIIT Pet seed dataset'i indirir (bir kez)
pytest tests/
```

## Klasör Yapısı

```
app/
├── main.py            # FastAPI giriş noktası
├── embedder.py        # CLIP feature extraction
├── vision.py          # Google Vision API
├── matcher.py         # Hibrit skor hesaplama
└── models.py          # Pydantic şemaları
tests/                 # Birim testler + seed_data/
scripts/               # Yardımcı betikler (seed dataset vb.)
```
