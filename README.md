# Pet AI — Kayıp/Bulunan Kedi ve Köpek Eşleştirme Servisi

## 1. Projenin amacı

Kayıp ve bulunan kedi/köpek fotoğraflarını karşılaştırıp en olası eşleşmeleri bulan bir
yapay zekâ servisi. Hazır **MegaDescriptor-S-224** modeli (görüntüden hayvan kimliği
"embedding"i çıkaran, bir Swin Transformer tabanlı model) kullanılarak:

1. Bir fotoğraftan embedding çıkarılır.
2. Aynı hayvana ait farklı fotoğrafların embedding'leri birbirine yakın olur.
3. Bir sorgu fotoğrafı için en benzer ilk 5 hayvan bulunur.
4. Top-1 / Top-5 başarı oranları ölçülür ve baseline olarak kaydedilir.
5. Gerektiğinde model kendi verimizle fine-tune edilebilir.
6. En iyi model FastAPI üzerinden servis edilir.

Model sıfırdan eğitilmez; yalnızca gerekiyorsa hazır MegaDescriptor ağırlıkları
başlangıç noktası alınarak fine-tune edilir.

## 2. Sistem mimarisi

```
app/
├── api.py                 FastAPI uygulaması (/health, /embedding, /index, /match, /index/reload)
├── config.py               .env tabanlı ayarlar (Settings)
├── device.py                MPS/CPU cihaz seçimi
├── schemas.py                Pydantic istek/cevap modelleri
└── services/
    ├── embedding_service.py   MegaDescriptor modelini yükler, embedding çıkarır (tek instance)
    ├── image_service.py        Görsel keşif, güvenli yükleme, veri seti tarama
    ├── index_service.py         NumPy tabanlı yerel vektör indeksi (VectorIndex arayüzü)
    └── matching_service.py       Embedding + indeksi birleştirip arama yapar

scripts/                    CLI araçları (aşağıda tek tek anlatılmıştır)
tests/                       pytest test paketi (sahte/mock embedding servisiyle, hızlı ve çevrimdışı)
```

`index_service.py`, ileride pgvector gibi harici bir vektör veritabanına geçilebilmesi
için soyut bir `VectorIndex` arayüzü ile somut `LocalNumpyIndex` uygulamasını ayırır.
`matching_service.py` ve `app/api.py`, yalnızca bu soyut arayüzle konuşur.

### Veri gizliliği

**`data/raw`, `data/train`, `data/val`, `data/test` ve `outputs/` klasörleri Git'e
eklenmez** (bkz. `.gitignore`). Bu klasörlerdeki fotoğraflar ve üretilen embedding/model
dosyaları yalnızca yerel makinede kalır.

## 3. Kurulum (macOS)

Python 3.11 ve sanal ortamın zaten kurulu olduğu varsayılır (bu proje için hazır).

```bash
cd pet-ai
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"   # ruff, mypy, pytest, httpx (geliştirme bağımlılıkları)
```

`.env.example` dosyasını kopyalayıp gerekirse düzenleyin:

```bash
cp .env.example .env
```

## 4. Sanal ortamın etkinleştirilmesi

Her terminal oturumunda:

```bash
cd pet-ai
source .venv/bin/activate
```

Doğrulama:

```bash
python test_model.py
```

Çıktı `Model başarıyla çalıştı.` ile bitmeli ve cihaz olarak Apple Silicon'da `mps`,
desteklenmiyorsa `cpu` görünmeli.

## 5. Veri klasörünün hazırlanması

Ham fotoğraflarınızı `data/raw/` altına, her hayvan için ayrı bir alt klasörle
(**klasör adı = `animal_id`**) yerleştirin:

```
data/raw/
├── kedi_001/
│   ├── onden.jpg
│   ├── yandan.jpg
│   └── uzaktan.jpg
├── kedi_002/
│   ├── foto1.jpg
│   └── foto2.jpg
└── kopek_001/
    ├── foto1.jpg
    ├── foto2.jpg
    └── foto3.jpg
```

Desteklenen formatlar: `.jpg`, `.jpeg`, `.png`, `.webp`. Bozuk veya okunamayan
dosyalar programı durdurmaz; script'ler bunları raporlayıp devam eder.

## 6. Veri setinin incelenmesi

```bash
python scripts/inspect_dataset.py --data-dir data/raw
```

Hayvan sayısı, toplam/hayvan başına fotoğraf sayısı, okunamayan dosyalar, az
fotoğraflı hayvanlar ve desteklenmeyen formatlar raporlanır.

### Train/val/test bölmesi oluşturma

```bash
python scripts/prepare_dataset.py \
  --source data/raw \
  --output data \
  --mode closed-set \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

- **closed-set**: aynı hayvanın fotoğrafları train/val/test'e dağıtılır (sistemin
  kayıtlı bir hayvanı yeni bir fotoğraftan tanıma yeteneğini ölçer).
- **open-set**: hayvan kimlikleri train/val/test arasında tamamen ayrılır (modelin
  hiç görmediği hayvanlar üzerindeki genelleme gücünü ölçer). `--mode open-set` ile
  çalıştırın.

Dosyalar **kopyalanır**, orijinal veri asla taşınmaz veya silinmez.
`data/metadata/dataset.csv` üretilir (`image_path,animal_id,split,species`).

## 7. Baseline değerlendirmesi

```bash
python scripts/evaluate_baseline.py \
  --data-dir data/test \
  --output-dir outputs/evaluations/baseline \
  --top-k 5
```

Her fotoğraf sırayla sorgu kabul edilir, kendisi aday listesinden çıkarılır, kalan
fotoğraflarla cosine similarity hesaplanır. Top-1/Top-5 doğruluk, MRR, ortalama
doğru/yanlış eşleşme benzerliği, gecikme ve işlenen/atlanan sorgu sayısı hesaplanıp
şu dosyalara yazılır:

- `results.json` — özet metrikler
- `top_k_results.csv` — her sorgunun ilk 5 sonucu
- `false_positives.csv` — Top-1'i yanlış olan sorgular
- `similarity_distribution.png` — doğru/yanlış eşleşme benzerlik dağılımı grafiği

**Veri yoksa sahte sonuç üretilmez.** `data/test` boşsa veya hiçbir hayvanın havuzda
ikinci bir fotoğrafı yoksa script açık bir hata mesajıyla durur.

## 8. İndeks oluşturma

```bash
python scripts/build_index.py \
  --data-dir data/raw \
  --output outputs/embeddings/pet_index.npz
```

Klasördeki tüm fotoğraflardan embedding çıkarıp `LocalNumpyIndex` biçiminde
`.npz` dosyasına kaydeder. Bu dosya `INDEX_PATH` ayarıyla API tarafından okunur.

## 9. API'yi çalıştırma

```bash
uvicorn app.api:app --reload
```

Swagger dokümantasyonu: http://127.0.0.1:8000/docs

Uygulama başlarken model bir kez yüklenir ve `INDEX_PATH` dosyası varsa (`build_index.py`
ile oluşturulmuşsa) indeks belleğe okunur; yoksa boş bir indeksle başlanır.

## 10. Curl ile örnek istekler

```bash
# Sağlık kontrolü
curl http://127.0.0.1:8000/health

# Embedding çıkarma
curl -X POST "http://127.0.0.1:8000/embedding?include_vector=false" \
  -F "image=@test.jpg"

# Fotoğrafı indekse ekleme
curl -X POST http://127.0.0.1:8000/index \
  -F "image=@test.jpg" \
  -F "image_id=img_001" \
  -F "animal_id=kedi_001" \
  -F "species=cat"

# Benzer hayvan arama
curl -X POST "http://127.0.0.1:8000/match?top_k=5" \
  -F "image=@test.jpg"

# İndeksi diskten yeniden yükleme
curl -X POST http://127.0.0.1:8000/index/reload
```

## 11. Fine-tuning

**Önce baseline değerlendirmesi kaydedilmeden eğitim başlatılmaz** — `train.py`,
`outputs/evaluations/baseline/results.json` dosyasının var ve anlamlı olduğunu kontrol
eder, yoksa açık bir hata ile durur.

```bash
python scripts/train.py \
  --train-dir data/train \
  --val-dir data/val \
  --output-dir outputs/training/run_001 \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --seed 42
```

Özellikler:

- Backbone, hazır MegaDescriptor ağırlıklarından başlar; `wildlife-tools` kütüphanesinin
  `ArcFaceLoss` (metric-learning) ve `BasicTrainer` bileşenleri kullanılır.
- **İki aşamalı fine-tuning**: 1. aşamada (`--warmup-epochs`) parametrelerin son
  `--unfreeze-fraction` oranı açık, geri kalanı donuk eğitilir; 2. aşamada tüm ağ
  `--stage2-lr-factor` ile düşürülmüş bir learning-rate'te açılır. (MegaDescriptor'ın
  Swin Transformer mimarisinde katmanları isimle güvenilir biçimde ayırmak kırılgan
  olacağından, parametre sırasına dayanan basit ve mimariden bağımsız bir yöntem
  kullanılır.)
- AdamW optimizer, gradient clipping, sabit seed, early stopping (`--patience`).
- Doğrulama: `data/train` galeri, `data/val` sorgu kabul edilerek Top-1/Top-5 hesaplanır
  (closed-set bölmesinde aynı hayvan train'de de bulunduğu için anlamlıdır).
- `best_model.pth` (en iyi val_top1) ve `last_model.pth` (son epoch) ayrı kaydedilir.
- `--resume <checkpoint>` ile kaldığı yerden devam edebilir.
- `training_history.json` ve `training_curves.png` (kayıp + doğruluk grafikleri) üretilir.

**Veri yetersizse eğitim başlamaz**: en az 2 farklı hayvan ve en az 4 eğitim
fotoğrafı gerekir; script bunu kontrol edip açık bir mesajla durur.

## 12. Baseline ve fine-tuned model karşılaştırması

```bash
python scripts/evaluate_model.py \
  --data-dir data/test \
  --checkpoint outputs/training/run_001/best_model.pth \
  --output-dir outputs/evaluations/comparison \
  --top-k 5
```

Hem hazır modeli hem fine-tune edilmiş checkpoint'i **aynı test kümesinde**
değerlendirip bir karşılaştırma tablosu (Top-1, Top-5, MRR, gecikme, yanlış pozitif
oranı) basar ve `comparison.json` olarak kaydeder. **Fine-tune edilmiş modelin daha
iyi olduğu varsayılmaz** — sonuç daha kötüyse bu açıkça raporlanır.

En iyi modeli API'ye seçtirmek için `.env`:

```
MODEL_SOURCE=finetuned
MODEL_CHECKPOINT=outputs/training/run_001/best_model.pth
```

(`MODEL_SOURCE=pretrained` hazır modeli kullanır; varsayılan budur.)

## 13. MPS ve CPU kullanımı

`app/device.py`, Apple Silicon'da otomatik olarak `mps`, desteklenmiyorsa `cpu`
seçer. Kod hiçbir yerde CUDA'ya bağımlı değildir. `extract_embeddings.py` ve
`scripts/_common.py`, MPS/bellek hatası oluşursa `batch_size`'ı otomatik olarak
yarıya indirip yeniden dener.

## 14. Bilinen sınırlamalar

- Yerel indeks (`LocalNumpyIndex`) bellek içi NumPy tabanlıdır; çok büyük veri
  setlerinde (onbinlerce fotoğraf) yavaşlayabilir. Böyle durumlarda `VectorIndex`
  arayüzünü uygulayan bir pgvector/FAISS backend'ine geçilmesi önerilir.
- İki aşamalı fine-tuning'deki katman dondurma stratejisi mimariden bağımsız, kaba
  bir sezgi kullanır (parametre sırasına göre); belirli blokları hedeflemez.
- `/index` endpoint'i yüklenen fotoğrafları `outputs/uploaded_images/` altına kaydeder
  (bu klasör Git'e eklenmez); büyük ölçekte kalıcı depolama için harici bir dosya
  deposu düşünülmelidir.
- Değerlendirme script'leri, bir hayvanın havuzda ikinci bir fotoğrafı olmasını
  gerektirir; tek fotoğraflı hayvanlar doğruluk hesaplamasından hariç tutulur.

## 15. Veri gizliliği

Kullanıcıların yüklediği/verdiği hayvan fotoğrafları kişisel/hassas veri
sayılabilir. `data/raw`, `data/train`, `data/val`, `data/test` ve `outputs/`
klasörleri `.gitignore` ile Git'ten hariç tutulmuştur; bu klasörlerdeki hiçbir
dosya commit edilmemelidir. `data/metadata/dataset.csv` dosya yolları ve
`animal_id` içerir — paylaşmadan önce içeriğini gözden geçirin.

## 16. Model lisansı hakkında uyarı

`BVRA/MegaDescriptor-S-224` modeli MIT lisansı ile Hugging Face Hub üzerinden
dağıtılmaktadır (bkz. model kartı). Modeli kullanmadan/dağıtmadan önce güncel
lisans ve kullanım koşullarını Hugging Face üzerinden kontrol edin. Bu proje
modeli olduğu gibi kullanır; ağırlıkları üzerinde bir yeniden lisanslama iddiası
taşımaz.

## Hızlı başlangıç (özet)

```bash
source .venv/bin/activate

python scripts/inspect_dataset.py --data-dir data/raw

python scripts/prepare_dataset.py \
  --source data/raw --output data --mode closed-set

python scripts/evaluate_baseline.py \
  --data-dir data/test --output-dir outputs/evaluations/baseline

python scripts/build_index.py \
  --data-dir data/raw --output outputs/embeddings/pet_index.npz

uvicorn app.api:app --reload
```

## Geliştirme: kod kalitesi ve testler

```bash
ruff check .
ruff format --check .
mypy app scripts
pytest -q
```

Testler, gerçek MegaDescriptor modelini indirmeden/yüklemeden çalışır — `tests/conftest.py`
içindeki `tiny_embedding_service` fixture'ı, `EmbeddingService`'in ağır bağımlılıklarını
(`timm.create_model`, `resolve_data_config`, `create_transform`) küçük, sahte ve tamamen
çevrimdışı bir backbone ile değiştirir; böylece asıl sınıfın mantığı (normalizasyon,
batch işleme, hata yönetimi) gerçekten test edilir.
