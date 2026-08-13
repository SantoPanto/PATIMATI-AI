# PATIMATI-AI

PatiMati (PawFinder) kayıp hayvan eşleştirme platformunun yapay zeka servisi.

Aşama A yaklaşımı: üretim serve akışında model eğitimi YOK — önceden eğitilmiş
(ya da elle fine-tune edilmiş) modellerle özellik çıkarma (feature extraction) ve
hibrit skorlama. `scripts/train.py` ile SigLIP2 backbone'unu fine-tune etmek ayrı,
elle çalıştırılan bir offline pipeline'dır — bkz. [Model Eğitimi](#model-eğitimi-fine-tuning-opsiyonel).

## Mimari

| Bileşen | Görev |
|---|---|
| CLIP ViT-B/32 | Fotoğraftan 512 boyutlu embedding (görsel parmak izi) |
| CLIP zero-shot + piksel analizi | Tür (cat/dog) + desen + dominant renk etiketleri — tamamen lokal, dış API'siz |
| Hibrit skor | %55 görsel + %30 etiket + %15 konum → bildirim eşiği `MATCH_THRESHOLD` (varsayılanı ve neden tartışmalı olduğu `app/matcher.py`'nin başında; sayıyı burada tekrarlamıyoruz ki kayması mümkün olmasın) |
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

Hata yollarını sınamak için (fotoğraf gerekmez):

```bash
venv\Scripts\python.exe scripts/sahte_java.py --hata-yollari
```

Bu, bozuk bir mesajın gerçekten DLQ'ya düştüğünü ve tanınmayan şema sürümünün
DLQ yerine `status: error` cevabı ürettiğini **broker üzerinde** doğrular.
Birim testler topolojinin doğru ilan edildiğini gösterir; bu ise RabbitMQ'nun
öyle yönlendirdiğini. DLQ yanlış anahtarla bağlanmış olsaydı ilan yine
başarılı olur, mesajlar ise hata vermeden yok olurdu.

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

## Model Eğitimi (fine-tuning, opsiyonel)

SigLIP2 görüntü backbone'unu kendi verimizle ArcFace kaybıyla fine-tune eden
ayrı bir pipeline. Üretim servisinde ÇALIŞMAZ; yalnızca elle, model
geliştirirken kullanılır. `wildlife-tools` gibi ağır bağımlılıklar bu yüzden
`requirements.txt`'te DEĞİL, ayrı bir dosyada:

```bash
pip install -r requirements-train.txt
```

Veri, her klasörü bir hayvanın kimliği (`animal_id`) olan bir yapı bekler
(`<animal_id>/<foto>.jpg`, bkz. `app/services/image_service.py`). Uçtan uca akış:

```bash
# 1) Ham (animal_id klasörlü) veriyi train/val/test'e böl
python scripts/prepare_dataset.py --source-dir data/raw --output-dir data \
    --val-ratio 0.15 --test-ratio 0.15

# 2) Fine-tuning öncesi referans (baseline) doğruluğunu ölç — train.py bu
#    dosyanın var olmasını ZORUNLU tutar, körlemesine fine-tune'u engeller
python scripts/evaluate_baseline.py --data-dir data/test \
    --output-dir outputs/evaluations/baseline

# 3) Fine-tuning
python scripts/train.py --train-dir data/train --val-dir data/val \
    --output-dir outputs/training/run_001 --epochs 10
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
├── models.py          # Pydantic şemaları
├── device.py          # torch cihaz seçimi (yalnızca scripts/train.py kullanır)
└── services/
    └── image_service.py   # animal_id klasör tarama + görüntü yükleme (eğitim pipeline'ı)
tests/                 # Birim testler + seed_data/
scripts/               # Ölçüm ve yardımcı betikler (sahte_java.py = uçtan uca kanıt)
                        # + eğitim pipeline'ı: prepare_dataset.py, evaluate_baseline.py, train.py
degerlendirme/         # Dayanıklılık testleri (Paket 2)
docs/                  # Entegrasyon sözleşmesi + ölçüm raporu + ölçüm sonuçları
```
