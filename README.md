# PATIMATI-AI

PatiMati (PawFinder) kayıp hayvan eşleştirme platformunun yapay zeka servisi.

Aşama A yaklaşımı: model eğitimi YOK — önceden eğitilmiş modellerle özellik
çıkarma (feature extraction) ve hibrit skorlama.

## Mimari

| Bileşen | Görev |
|---|---|
| CLIP ViT-B/32 | Fotoğraftan 512 boyutlu embedding (görsel parmak izi) |
| CLIP zero-shot + piksel analizi | Tür (cat/dog) + desen + dominant renk etiketleri — tamamen lokal, dış API'siz |
| Hibrit skor | %55 görsel + %30 etiket + %15 konum → eşik 0.70 (`MATCH_THRESHOLD` ile ayarlanabilir) |
| FastAPI | `/health`, `/analyze`, `/match` endpoint'leri (port 8000) |

Not: Plandaki Google Vision API, faturalandırma şartı nedeniyle CLIP zero-shot ile
değiştirildi (PR #1 yorumlarında gerekçe ve ölçümler). `app/vision.py` dönüş yolu
olarak duruyor; `google_key.json` + faturalandırmalı proje varsa tekrar bağlanabilir.

Spring Boot backend bu servise **RabbitMQ üzerinden asenkron** bağlanır; HTTP uçları
elle deneme ve hata ayıklama için duruyor. Mesaj biçimleri ve kurallar:
`docs/entegrasyon-sozlesmesi.md`.

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

Etiket çıkarımı lokal çalıştığı için hiçbir API anahtarı GEREKMEZ.
(Opsiyonel: Vision API'ye dönülecekse `google_key.json` proje köküne konur —
`google_key.json` ve `.env` .gitignore'dadır, asla commit'lenmez.)

## Çalıştırma

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

## Kuyruk köprüsü (RabbitMQ)

Tüketici **ayrı bir süreçtir**, FastAPI'nin içinde değil: model çıkarımı CPU'yu
saniyelerce meşgul eder ve aynı süreçte olsaydı HTTP isteklerini aç bırakırdı.

```bash
venv\Scripts\python.exe -m app.kuyruk
```

Ayarlar `.env` içinde (`AMQP_URL`, `AMQP_PREFETCH`, `AMQP_HEARTBEAT`).
Topolojiyi tüketici kendisi ilan eder — elle kuyruk oluşturmaya gerek yok.

### Lokal RabbitMQ (Windows, yönetici yetkisi gerekmez)

Windows servisine kaydolmadan, zip'ten çalıştırılabilir:

```powershell
# Bir kez: Erlang 27.x (RabbitMQ 4.3 Erlang 28/29 ile ÇALIŞMAZ)
winget install --id Erlang.ErlangOTP --version 27.3.4.13

# Her açılışta
$env:ERLANG_HOME = "C:\Program Files\Erlang OTP"
& "C:\erp-backup\tools\rabbitmq\rabbitmq_server-4.3.4\sbin\rabbitmq-server.bat"
```

Yönetim arayüzü (kuyrukları gözle görmek için):

```powershell
& "...\sbin\rabbitmq-plugins.bat" enable rabbitmq_management
# http://localhost:15672  — kullanıcı/parola: guest / guest
```

### Uçtan uca kanıt

Java tarafı henüz yok; yerine geçen betik istek yayınlar, sonucu dinler ve
sözleşmeye göre denetler:

```bash
venv\Scripts\python.exe scripts/sahte_java.py --dosya degerlendirme\test_resim.jpg
```

Yerel dosya kullanılıyorsa `.env` içinde `PHOTO_ALLOWED_HOSTS=127.0.0.1`
olmalı — SSRF koruması aksi hâlde kendi makinemizi de reddeder (bilerek).

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
├── main.py            # FastAPI giriş noktası (HTTP — deneme/hata ayıklama)
├── kuyruk.py          # RabbitMQ köprüsü (üretim akışı) — ayrı süreç
├── analiz.py          # Fotoğrafları uçtan uca analiz eder; HTTP ve kuyruk aynı yeri çağırır
├── indirici.py        # Adresten fotoğraf indirme + SSRF koruması
├── embedder.py        # CLIP (etiketçi) + seçilebilir kimlik modeli
├── surum.py           # Kimlik modeli seçimi ve model sürümü
├── attributes.py      # Zero-shot tür/desen/cins + piksel renk analizi
├── vision.py          # (opsiyonel/yedek) Google Vision — şu an kullanılmıyor
├── matcher.py         # Hibrit skor hesaplama
├── hatalar.py         # Sözleşmedeki error.code karşılıkları
└── models.py          # Pydantic şemaları
tests/                 # Birim testler + seed_data/
scripts/               # Ölçüm ve yardımcı betikler (sahte_java.py = uçtan uca kanıt)
degerlendirme/         # Dayanıklılık testleri (Paket 2)
docs/                  # Entegrasyon sözleşmesi + ölçüm raporu + ölçüm sonuçları
```
