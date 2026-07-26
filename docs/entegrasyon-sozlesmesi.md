# AI Servisi ↔ Backend Entegrasyon Sözleşmesi

**Sürüm:** 1 · **Durum:** taslak, ekip onayı bekliyor · **Son güncelleme:** 2026-07-26

Bu belge, Java backend ile Python AI servisinin birbirine ne göndereceğini tanımlar.
İki taraf ayrı repolarda ve ayrı dillerde olduğu için derleyici bizi korumuyor —
sözleşme bu belgedir. Değişiklik gerekirse `schema_version` artırılır ve iki taraf
birlikte güncellenir.

Yol haritası karşılığı: **KISIM 4, madde 1 — Asenkron Yapay Zeka Entegrasyonu (RabbitMQ).**

---

## 1. Akış

```
Kullanıcı ilan oluşturur (fotoğraflı)
        │
        ▼
  Java: ilanı kaydeder  ─────────────────► ilan anında görünür (AI beklenmez)
        │                                   ai_status = PENDING
        │  adayları PostGIS ile süzer
        ▼
  Java ──[ ai.analysis.request ]──► RabbitMQ ──► Python AI
                                                    │ fotoğrafı indirir
                                                    │ embedding + tür/cins/renk
                                                    │ adaylarla skorlar
                                                    ▼
  Java ◄──[ ai.analysis.result ]── RabbitMQ ◄───────┘
        │
        ├─ ilanı günceller (ai_* alanları, ai_status = DONE)
        └─ eşleşme varsa ilgili kullanıcılara bildirim
```

AI çağrısı **asenkrondur**: kullanıcı ilan verirken beklemez. Fotoğraf analizi
1-3 saniye sürdüğü için senkron çağrı kullanıcı deneyimini bozardı.

---

## 2. RabbitMQ topolojisi

| Nesne | Ad | Tip |
|---|---|---|
| Exchange | `patimati.ai` | direct, durable |
| Kuyruk (istek) | `ai.analysis.request` | durable, routing key `analysis.request` |
| Kuyruk (sonuç) | `ai.analysis.result` | durable, routing key `analysis.result` |
| Dead-letter exchange | `patimati.ai.dlx` | direct |
| Dead-letter kuyruğu | `ai.analysis.request.dlq` | durable |

- **Java** → `patimati.ai` exchange'ine `analysis.request` anahtarıyla yayınlar,
  `ai.analysis.result` kuyruğunu dinler.
- **Python** → `ai.analysis.request` kuyruğunu dinler,
  `patimati.ai` exchange'ine `analysis.result` anahtarıyla yayınlar.
- Mesaj gövdesi **UTF-8 JSON**, `content_type: application/json`.

---

## 3. İstek mesajı — `ai.analysis.request`

Java → Python.

```json
{
  "schema_version": 1,
  "request_id": "9f1c2b7e-3a44-4a1e-9d55-0c2f6b1a77de",
  "ad_id": 123,
  "ad_type": "LOST",
  "declared_species": "cat",
  "photo_urls": [
    "https://patimati-media.s3.eu-central-1.amazonaws.com/ads/123/1.jpg"
  ],
  "candidates": [
    {
      "ad_id": 98,
      "embedding": [0.0123, -0.0456, "... 512 adet float ..."],
      "labels": ["cat", "tabby", "brown", "white"],
      "species": "cat",
      "distance_km": 1.2
    }
  ]
}
```

| Alan | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| `schema_version` | int | ✅ | Şu an `1`. Tanınmayan sürümde AI hata sonucu döner. |
| `request_id` | string (UUID) | ✅ | İzleme ve günlükleme için. Sonuçta aynen geri döner. |
| `ad_id` | long | ✅ | İşlenen ilanın kimliği. |
| `ad_type` | enum | ✅ | `LOST` \| `FOUND`. **`ADOPTION` gönderilmez** (eşleştirmeye girmez). |
| `declared_species` | string \| null | ❌ | Kullanıcının beyan ettiği tür: `cat` \| `dog` \| `null`. |
| `photo_urls` | string[] | ✅ | En az 1 adres. AI hepsini indirir, en iyi eşleşmeyi kullanır. Üst sınır 5. |
| `candidates` | object[] | ✅ | Boş dizi olabilir (ilk ilan). Üst sınır 100. |

**Aday alanları** (`candidates[]`):

| Alan | Tip | Kaynak |
|---|---|---|
| `ad_id` | long | aday ilanın kimliği |
| `embedding` | float[512] | adayın `ai_embedding` sütunu |
| `labels` | string[] | adayın `ai_labels` sütunu |
| `species` | string | adayın `ai_species` sütunu |
| `distance_km` | double | PostGIS ile hesaplanan gerçek mesafe |

> `embedding` mesajda **JSON dizisi** olmalı, dizi içeren bir metin değil.
> Java tarafında `float[]` olarak tutulursa Jackson bunu doğru üretir (bkz. §6).

---

## 4. Sonuç mesajı — `ai.analysis.result`

Python → Java.

### Başarılı

```json
{
  "schema_version": 1,
  "request_id": "9f1c2b7e-3a44-4a1e-9d55-0c2f6b1a77de",
  "ad_id": 123,
  "status": "ok",
  "model_version": "clip-vit-base-patch32/attrs-v2",
  "analysis": {
    "embedding": [0.0123, -0.0456, "... 512 adet float ..."],
    "species": "cat",
    "species_confidence": 0.99,
    "breed": "British Shorthair",
    "breed_confidence": 0.42,
    "pattern": "tabby",
    "colors": [{ "name": "brown", "score": 0.41 }],
    "labels": ["cat", "tabby", "brown", "white"]
  },
  "matches": [
    {
      "ad_id": 98,
      "score": 0.81,
      "visual": 0.79,
      "label": 0.66,
      "location": 0.80,
      "match": true
    }
  ],
  "processed_at": "2026-07-26T12:00:00Z"
}
```

| Alan | Açıklama |
|---|---|
| `model_version` | **Kritik.** Hangi model/ön işleme ile üretildiğini söyler. Java bunu `ai_model_version` sütununa yazar. Model değişirse eski vektörler kıyaslanamaz hâle gelir; bu alan olmadan hangilerinin bayat olduğu anlaşılamaz. |
| `analysis.embedding` | 512 boyutlu, L2-normalize. `ai_embedding` sütununa yazılır. |
| `analysis.species` | `cat` \| `dog` \| `unknown`. Güveni düşükse `unknown` döner. |
| `analysis.breed` | Bilgi amaçlı. **Filtre olarak kullanılmaz** (bkz. §7). |
| `matches` | Skora göre azalan sıralı, en fazla 20 kayıt. Aday yoksa boş dizi. |
| `matches[].match` | `score >= eşik` ise `true`. Eşik AI tarafında ortam değişkeniyle ayarlanır. |
| `matches[].visual/label/location` | Skorun bileşenleri — hata ayıklama ve arayüzde gerekçe göstermek için. |

### Hatalı

```json
{
  "schema_version": 1,
  "request_id": "9f1c2b7e-3a44-4a1e-9d55-0c2f6b1a77de",
  "ad_id": 123,
  "status": "error",
  "error": {
    "code": "PHOTO_DOWNLOAD_FAILED",
    "message": "404 for https://.../123/1.jpg"
  },
  "processed_at": "2026-07-26T12:00:00Z"
}
```

Java bu durumda `ai_status = FAILED` yazar; ilan normal şekilde yayında kalır.

| `error.code` | Anlamı |
|---|---|
| `NO_PHOTOS` | `photo_urls` boş geldi |
| `PHOTO_DOWNLOAD_FAILED` | Adres indirilemedi (404, zaman aşımı, çok büyük dosya) |
| `INVALID_IMAGE` | İndirildi ama görüntü olarak açılamadı |
| `UNSUPPORTED_SCHEMA` | `schema_version` tanınmıyor |
| `MODEL_ERROR` | Model çalışırken hata verdi |
| `INTERNAL` | Beklenmeyen hata |

---

## 5. Adayları kim süzer, hangi kurallarla?

**Java süzer.** PostGIS, ilan tipi ve tarih bilgisi onda; AI'ın veritabanına
erişimi yok. AI yalnızca görsel işi yapar.

Süzme kuralları:

| Kural | Değer | Gerekçe |
|---|---|---|
| Karşıt ilan tipi | `LOST` → `FOUND` adayları, `FOUND` → `LOST` adayları | Kayıp ilanını başka kayıp ilanıyla eşleştirmek anlamsız |
| `ADOPTION` | Hiç gönderilmez | Sahiplendirme eşleştirmeye girmez |
| `active` | `true` | Kapanmış ilanlar aday değil |
| AI durumu | `ai_status = 'DONE'` ve `ai_embedding IS NOT NULL` | Vektörü olmayan aday kıyaslanamaz |
| **Yarıçap** | **25 km** | ⚠️ Bildirim yarıçapı (5 km) ile **aynı değildir** — kaybolan hayvan yürür, günlerce uzaklaşabilir |
| Zaman penceresi | Son 90 gün | Eski ilanlar gürültü yaratır |
| Üst sınır | 100 aday, mesafeye göre yakından uzağa | Mesaj boyutu ve işlem süresi |

> **Not:** Aday başına ~5 KB (512 float). 100 aday ≈ 500 KB mesaj. RabbitMQ için
> sorun değil ama gereksiz büyümesin diye üst sınır konuldu. Aday sayısı binleri
> bulursa çözüm `pgvector`'a geçmektir (Faz 2).

---

## 6. Veritabanı alanları (`ads` tablosu)

AI sonucunun yazılacağı alanlar. **Bu alanları AI sorumlusu (İbrahim) ekler**,
ilan CRUD'una karışmaz.

```java
// entity/Ad.java — eklenecek alanlar

@JdbcTypeCode(SqlTypes.JSON)
@Column(name = "ai_embedding", columnDefinition = "jsonb")
private float[] aiEmbedding;          // 512 boyutlu CLIP vektörü

@JdbcTypeCode(SqlTypes.JSON)
@Column(name = "ai_labels", columnDefinition = "jsonb")
private String[] aiLabels;            // ["cat","tabby","brown"]

@Column(name = "ai_species", length = 16)
private String aiSpecies;             // cat | dog | unknown

@Column(name = "ai_breed", length = 64)
private String aiBreed;

@Column(name = "ai_breed_confidence")
private Float aiBreedConfidence;

@Column(name = "ai_model_version", length = 64)
private String aiModelVersion;

@Enumerated(EnumType.STRING)
@Column(name = "ai_status", length = 16)
private AiStatus aiStatus = AiStatus.PENDING;   // PENDING | DONE | FAILED

@Column(name = "ai_processed_at")
private Instant aiProcessedAt;
```

`@JdbcTypeCode(SqlTypes.JSON)` Hibernate 6+ ile gelir; `float[]` alanı `jsonb`
sütuna yazar ve Jackson mesajda doğru JSON dizisi üretir. Ayrıca dönüştürücü
yazmaya gerek yok.

**İlanın kendisinde gerekenler** (KISIM 2 — ilan sorumlusunda):

| Alan | Neden |
|---|---|
| `photoUrls` (`List<String>`) | AI'ın girdisi. Bu olmadan entegrasyon çalışmaz. |
| `species` | Kullanıcının beyanı. AI'ınkinden **ayrı tutulur** (bkz. §7). |
| `createdAt` / `updatedAt` | Aday zaman penceresi ve sıralama için |

---

## 7. Kurallar

1. **Cins asla filtre değildir.** Türkiye'de kayıp hayvanların çoğu melez;
   yanlış cins tahmini gerçek eşleşmeyi eler. Cins yalnızca bilgi olarak
   gösterilir, tercihen güven oranıyla birlikte.
2. **Kullanıcı beyanı, AI tahmininden önceliklidir.** Tür filtresi
   `declared_species` varsa ona göre uygulanır; yoksa AI'ın `species`
   tahmini kullanılır. AI `unknown` derse tür filtresi **uygulanmaz**
   (yanlış eleme yapmamak için).
3. **AI kesin eşleşme üretmez, sıralı aday listesi üretir.** Son onay her
   zaman kullanıcıdadır. Arayüz "eşleşti" değil "olası eşleşme" dilini
   kullanmalıdır.
4. **Bildirim tekrarı önlenmelidir.** KISIM 3 zaten yeni ilanda 5 km'deki
   herkese bildirim atıyor. Eşleşme bildirimi ayrı bir olaydır ve yalnızca
   eşleşen ilanların sahiplerine gitmelidir.

---

## 8. Teslim garantisi, tekrar ve hata

- RabbitMQ **en az bir kez** teslim eder; aynı mesaj iki kez gelebilir.
- **AI tarafı saftır:** aynı girdi → aynı çıktı. İki kez işlemek yalnızca
  boşa iş demektir, veriyi bozmaz.
- **Java tarafı `ad_id` üzerinden idempotenttir:** aynı sonucu iki kez
  yazmak aynı satırı aynı değerlerle günceller.
- Bu yüzden ek bir tekrar-koruması gerekmez; `request_id` yalnızca izleme
  ve günlük eşlemesi içindir.
- AI, işleyemediği mesajı **kuyruğa geri koymaz**; `status: "error"` ile
  cevaplar. Böylece sonsuz döngü oluşmaz.
- Mesaj hiç ayrıştırılamazsa (bozuk JSON) DLQ'ya düşer.

---

## 9. Sürümleme

- `schema_version` her iki mesajda da zorunludur.
- Alan **eklemek** kırıcı değildir; iki taraf da tanımadığı alanları yok sayar
  (Java: `FAIL_ON_UNKNOWN_PROPERTIES = false`).
- Alan **silmek** veya anlamını değiştirmek kırıcıdır → `schema_version` artar,
  iki taraf birlikte güncellenir.
- `model_version` şema sürümünden bağımsızdır; model değişince artar ve
  o andan sonraki kayıtlara yazılır.

---

## 10. Güvenlik

- AI servisi **iç ağda** kalmalı, internete açık olmamalıdır. Zorunlu olarak
  açılacaksa paylaşılan bir gizli anahtar başlığı istenir.
- S3 fotoğrafları için **süreli özel adres (presigned URL)** önerilir; böylece
  AWS anahtarı AI servisine hiç girmez. Süre en az 10 dakika olmalı (kuyruk
  gecikmesi payı).
- Kuyruk mesajlarında kişisel veri taşınmaz: ad, e-posta, telefon gönderilmez.
  Yalnızca ilan kimliği, fotoğraf adresi ve türetilmiş vektörler taşınır.

---

## 11. Açık sorular (ekip kararı bekliyor)

1. **Eşleşme bildirimi kime gidiyor?** Yeni ilanın sahibine mi, eşleşen eski
   ilanın sahibine mi, ikisine birden mi?
2. **Eşleşme sonrası akış nedir?** Kullanıcı onaylarsa ilan kapanıyor mu?
3. **Eşleşme yarıçapı 25 km uygun mu?** Bildirim yarıçapından farklı olması
   kabul ediliyor mu?
4. **S3 adresleri public mi, presigned mi olacak?**
5. **RabbitMQ'yu kim ayağa kaldırıyor** (lokal + dağıtım ortamı)?

---

## 12. Uygulama durumu

| Taraf | Durum |
|---|---|
| Python AI — analiz (`/analyze`) | ✅ Çalışıyor (HTTP) |
| Python AI — eşleştirme (`/match`) | ✅ Çalışıyor (HTTP), alan adları bu sözleşmeye göre güncellenecek |
| Python AI — cins (`breed`) | ⬜ Yapılacak |
| Python AI — URL'den indirme | ⬜ Yapılacak |
| Python AI — RabbitMQ tüketici/üretici | ⬜ Yapılacak |
| Java — `Ad.photoUrls` | ⬜ KISIM 2'de |
| Java — `ai_*` alanları | ⬜ AI sorumlusunda |
| Java — kuyruk config + publisher + listener | ⬜ AI sorumlusunda |
| Ortak — `docker-compose` (PostGIS + RabbitMQ + AI) | ⬜ AI sorumlusunda |
