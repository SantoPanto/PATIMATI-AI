# AI Servisi ↔ Backend Entegrasyon Sözleşmesi

**Sürüm:** 1 · **Durum:** Python ayağı çalışıyor ve broker üzerinde doğrulandı;
açık soruların 4'ü karara bağlandı (§11) · **Son güncelleme:** 2026-07-28

> `schema_version` hâlâ **1**: verilen kararların hiçbiri mesaj biçimini
> değiştirmedi, yalnızca kuralları netleştirdi. Java tarafı bu belgeye göre
> yazılabilir.

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
- Sonuç mesajları **kalıcı** yayınlanır (`delivery_mode=2`). Kuyruk `durable`
  olsa bile mesaj kalıcı değilse broker yeniden başladığında sonuç kaybolur ve
  ilan sonsuza kadar `PENDING` kalır — ikisi birlikte gerekir.

> ⚠️ **İki taraf da aynı nesneleri ilan edecek — argümanlar BİREBİR aynı olmalı.**
> Aynı kuyruğu farklı argümanlarla ilan etmek `PRECONDITION_FAILED` verir ve
> kanalı kapatır. Özellikle `ai.analysis.request` iki tarafta da
> `x-dead-letter-exchange: patimati.ai.dlx` argümanıyla ilan edilmelidir.
> Spring AMQP'de bu `QueueBuilder.durable("ai.analysis.request").deadLetterExchange("patimati.ai.dlx").build()` demektir.
>
> Ölü mektup kuyruğu (`ai.analysis.request.dlq`), DLX'e **kuyruk adıyla değil
> `analysis.request` anahtarıyla** bağlanır: RabbitMQ mesajı ölü mektuba
> düşürürken orijinal yönlendirme anahtarını korur. Yanlış bağlanırsa mesajlar
> hata vermeden yok olur.
>
> Referans uygulama Python tarafında: `app/kuyruk.py: topolojiyi_kur()`.

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
      "embeddings": [
        [0.0123, -0.0456, "... 768 adet float ..."],
        [0.0311, -0.0122, "... ilanın ikinci fotoğrafı ..."]
      ],
      "labels": ["cat", "tabby", "brown", "white"],
      "species": "cat",
      "distance_km": 1.2,
      "model_version": "siglip2-animal/v2"
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
| `embeddings` | float[768][] | ⚠️ **Liste — ilanın her fotoğrafı için bir vektör.** Görsel skor tüm fotoğraf çiftlerinin **en iyisinden** alınır. Tek fotoğrafla eşleşme oranı gerçek veride %24'te kaldığı için çoklu fotoğraf zorunludur (bkz. `docs/olcum-raporu.md` §4). En az 1, en fazla 5. |
| `labels` | string[] | adayın `ai_labels` sütunu |
| `species` | string | adayın `ai_species` sütunu |
| `distance_km` | double | PostGIS ile hesaplanan gerçek mesafe |
| `model_version` | string | ⚠️ **Zorunlu.** Adayın `ai_model_version` sütunu. Servisin güncel sürümüyle aynı değilse aday **atlanır** — farklı sürümle üretilmiş vektörler kıyaslanamaz ve hata vermeden yanlış benzerlik üretir. Atlananlar `skipped_candidates` içinde raporlanır. |

> `embedding` mesajda **JSON dizisi** olmalı, dizi içeren bir metin değil.
> Java tarafında `float[]` olarak tutulursa Jackson bunu doğru üretir (bkz. §6).
>
> **Aday listesi ön elemesi:** AI tarafı ilanın kendisini (aynı `ad_id`), tekrar eden
> adayları ve sürümü tutmayanları eler; 100'den fazlası kırpılır. Hepsi raporlanır.

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
  "model_version": "siglip2-animal/v2",
  "analysis": {
    "embeddings": [
      [0.0123, -0.0456, "... ilk fotoğrafın 768 float'ı ..."],
      [0.0311, -0.0122, "... ikinci fotoğrafın ..."]
    ],
    "species": "cat",
    "species_confidence": 0.99,
    "is_pet": true,
    "breed": "British Shorthair",
    "breed_confidence": 0.88,
    "pattern": "tabby",
    "colors": [{ "name": "brown", "score": 0.41 }],
    "labels": ["cat", "tabby", "brown", "white"]
  },
  "nlp_attributes": {},
  "extracted_features": [],
  "matches": [
    {
      "ad_id": 98,
      "score": 0.81,
      "visual": 0.79,
      "label": 0.66,
      "location": 0.80,
      "match": true,
      "photo_a": 0,
      "photo_b": 2
    }
  ],
  "match_threshold": 0.80,
  "skipped_candidates": {
    "toplam": 2,
    "kendisi": 0,
    "tekrar_eden": 0,
    "model_surumu_uyusmuyor": 2,
    "gecersiz_embedding": 0,
    "aday_siniri_asildi": 0
  },
  "processed_at": "2026-07-26T12:00:00Z"
}
```

| Alan | Açıklama |
|---|---|
| `model_version` | **Kritik.** Hangi model/ön işleme ile üretildiğini söyler. Java bunu `ai_model_version` sütununa yazar. Model değişirse eski vektörler kıyaslanamaz hâle gelir; bu alan olmadan hangilerinin bayat olduğu anlaşılamaz. |
| `analysis.embeddings` | Her fotoğraf için 768 boyutlu, L2-normalize vektör. `ai_embeddings` sütununa yazılır. Sırası `photo_urls` ile aynıdır. |
| `matches[].photo_a` / `photo_b` | Hangi fotoğraf çiftinin eşleştiği (0 tabanlı indeks). Arayüzde "bu iki fotoğraf benziyor" diye gösterilebilir; hata ayıklamada hangi karenin tuttuğunu söyler. Tür uyuşmazlığında `null`. |
| `analysis.species` | `cat` \| `dog` \| `unknown`. Güveni düşükse `unknown` döner. |
| `analysis.is_pet` | `false` → fotoğrafta kedi/köpek görünmüyor (ekran görüntüsü, insan, nesne...). Java bunu `ai_is_pet` sütununa yazar ve `AdResponse` ile arayüze verir; arayüz kullanıcıdan başka bir fotoğraf isteyebilir. **Eleme ölçütü DEĞİLDİR** — aday süzme sorgusuna girmez: kapının yanlış reddetme oranı ölçüldü (111 gerçek hayvan fotoğrafında **0**), ama gerçek "hayvan olmayan fotoğraf" kümesi henüz olmadığı için yakalama oranı ölçülmedi. Ölçülmemiş bir kapıyı eleyici yapmak gerçek bir kayıp hayvan ilanını sessizce havuz dışında bırakabilir. İlan birden çok fotoğraf taşıyorsa **en az biri** hayvan içerdiğinde `true` döner. |
| `analysis.breed` | Bilgi amaçlı. **Filtre olarak kullanılmaz** (bkz. §7). `is_pet` false ise her zaman `null`. |
| `nlp_attributes` | NLP çıkarımları için yer tutucu nesne. Şimdilik boş (`{}`) döner; NLP modülü entegre edildiğinde niyet/tasma durumu gibi anahtarlar buraya yazılır. |
| `extracted_features` | NLP tarafından çıkarılan özelliklerin listesi. Şimdilik boş dizi (`[]`) döner; entegrasyon sonrası metinden türetilen etiketler burada taşınır. |
| `matches` | Skora göre azalan sıralı, en fazla 20 kayıt. Aday yoksa boş dizi. |
| `skipped_candidates` | Elenen adayların gerekçeli sayımı. "Hiç eşleşme çıkmadı" durumunun sebebi görünür olsun diye vardır — özellikle `model_surumu_uyusmuyor` sıfırdan büyükse ilgili ilanların yeniden analiz edilmesi gerekir. |
| `matches[].match` | `score >= eşik` ise `true`. Eşik AI tarafında ortam değişkeniyle ayarlanır. |
| `match_threshold` | **Bu koşumda kullanılan eşiğin kendisi.** `matches[].match` bunun sonucudur ve eşik zamanla değişir (0.70 → 0.80, ölçüm sonucu). Java bunu `ad_match.threshold_at_time` sütununa yazar: kayıt "bu eşleşme üretilirken eşik neydi" sorusuna sonradan doğru cevap verebilsin diye. Değer `MATCH_THRESHOLD`'dan gelir; **yukarıdaki örnekteki sayı yalnızca gösterimdir**, tek kaynak koddur. |
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
| `INVALID_REQUEST` | Mesaj geçerli JSON ama sözleşmedeki alanları taşımıyor (eksik `ad_id`, yanlış tip...). **Gönderen taraftaki hatadır.** |
| `MODEL_ERROR` | Model çalışırken hata verdi |
| `INTERNAL` | Beklenmeyen hata |

> `INVALID_REQUEST` neden `INTERNAL`den ayrıldı: `INTERNAL` "AI servisinde bir
> şey patladı" demektir ve hatayı arayan kişiyi yanlış repoda arama yapmaya
> gönderir. Eksik bir alan gönderen taraftaki hatadır; kodun bunu söylemesi
> saatler kazandırır. (Kod eklemek kırıcı değildir, bkz. §9.)

**Hata mesajında da `request_id` ve `ad_id` döner** — mesaj şemaya hiç uymasa
bile, okunabildiği kadarıyla. Dönmezse Java hangi ilanın başarısız olduğunu
bilemez ve o ilan sonsuza kadar `PENDING` kalır.

Başarılı sonuçta ayrıca `failed_photos` alanı gelir (sözleşmeye sonradan
eklendi, kırıcı değil): indirilemeyen fotoğrafların adresi ve sebebi. Kullanıcı
3 fotoğraf yükleyip 1'iyle analiz edildiyse bunun bir izi kalsın diye.

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

> **Not:** Aday başına ~7 KB (768 float). 100 aday ≈ 700 KB mesaj. RabbitMQ için
> sorun değil ama gereksiz büyümesin diye üst sınır konuldu. Aday sayısı binleri
> bulursa çözüm `pgvector`'a geçmektir (Faz 2).

---

## 6. Veritabanı alanları (`ads` tablosu)

AI sonucunun yazılacağı alanlar. **Bu alanları AI sorumlusu (İbrahim) ekler**,
ilan CRUD'una karışmaz.

```java
// entity/Ad.java — eklenecek alanlar

// İlanın HER fotoğrafı için bir vektör. Tek fotoğrafla eşleşme oranı gerçek
// veride %24'te kaldığı için çoklu fotoğraf zorunlu (docs/olcum-raporu.md §4).
// Kendi kayıt tipiyle saklamak, fotoğraf sırası değişse bile eşlemeyi korur.
@JdbcTypeCode(SqlTypes.JSON)
@Column(name = "ai_embeddings", columnDefinition = "jsonb")
private List<AiFotoVektoru> aiEmbeddings;

public record AiFotoVektoru(String photoUrl, float[] embedding) {}

@JdbcTypeCode(SqlTypes.JSON)
@Column(name = "ai_labels", columnDefinition = "jsonb")
private String[] aiLabels;            // ["cat","tabby","brown"]

@Column(name = "ai_species", length = 16)
private String aiSpecies;             // cat | dog | unknown

// Sarmalayıcı Boolean, ilkel boolean DEĞİL: üç durum var —
//   null  analiz yapılmadı (PENDING) ya da yapılamadı (FAILED)
//   true  fotoğrafta hayvan görüldü
//   false bakıldı, hayvan görünmüyor
// İlkel tipte, alan mesajda hiç gelmezse Jackson sessizce false yazar ve
// "AI söylemedi" ile "AI hayvan görmedi" aynı değere düşer.
@Column(name = "ai_is_pet")
private Boolean aiIsPet;

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
   *Bu bir tercih değil, ölçüm sonucudur:* gerçek fotoğraflarda "aynı hayvan"
   ve "farklı hayvan" skor dağılımları çakışıyor — hiçbir eşik ikisini temiz
   ayırmıyor (`docs/olcum-raporu.md` §4). `match: true` "kesin aynı hayvan"
   değil, **"bildirim gönderilecek kadar eminiz"** demektir. Daha düşük skorlu
   adaylar da listede gösterilmeli, sadece bildirim tetiklememelidir.
5. **İlan başına birden çok fotoğraf istenmelidir.** Tek fotoğrafla gerçek
   eşleşmelerin yalnızca %24'ü yakalanıyor; üç fotoğrafla eşik aşılıyor.
   Arayüz kullanıcıyı birden fazla fotoğraf yüklemeye teşvik etmelidir —
   bu, eşleştirme başarısını en çok artıran tek şey.
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

  ✅ **Uygulandı (A1).** `/analyze`, `/analyze_url`, `/compare` ve `/match`
  uçları **`X-Api-Key`** başlığı ister; eşleşmezse `401 UNAUTHORIZED`.
  Karşılaştırma sabit zamanlıdır. `/health` bilerek anahtarsızdır — canlılık
  yoklaması kimlik isteseydi çalışan servis "ölü" görünürdü.

  > 🔒 **`AI_API_KEY` ZORUNLUDUR; verilmezse korumalı uçların hepsi 401 döner.**
  > *(20.08.2026'da değişti — öncesinde tam tersiydi: anahtar verilmezse kapı
  > tamamen açılıyordu. Gerekçesi dağıtım sırasıydı, servis tarayıcıdan da
  > çağrılıyordu; o engel kalktı — ön yüz artık `/api/ai/analyze` üzerinden
  > backend'e gidiyor ve backend `X-Api-Key` gönderiyor.)*
  > **Dağıtımda iki tarafa da aynı değer verilmeli**: AI'da `AI_API_KEY`,
  > backend'de `ai.service.api-key`. Yalnız birinde varsa eşleştirme çalışmaz.
  > **Durum sessiz değildir:** açılışta HATA günlüğe yazılır ve `/health`
  > `api_anahtari_yapilandirildi` alanıyla gerçeği söyler. Değişkeni yazmayı
  > unutan bir dağıtım, `PHOTO_ALLOWED_HOSTS`'ta bir kez yaşandığı gibi,
  > korumanın açık olduğunu sanarak kapalı çalışmasın.
- **Fotoğraflar kendi alan adımızın alt alan adından sunulacak** (karar: Fatih,
  2026-07-29). Ham S3 adresi (`patimati-media.s3.eu-central-1.amazonaws.com`)
  yerine `cdn.<alanadi>` / `media.<alanadi>` gibi bir adres belirlenecek; DNS'te
  CNAME ile S3'e (ya da önüne konacak CloudFront'a) yönlendirilecek.

  Neden AI tarafı için de doğru karar: beyaz listeye **tek ve kalıcı** bir ad
  yazılıyor. Yarın S3'ten başka bir sağlayıcıya geçilse, CloudFront eklense ya
  da kova adı değişse AI tarafında hiçbir şey değişmez — kodda da, ayarda da.

  > ⚠️ **CNAME olmalı, HTTP yönlendirmesi DEĞİL.** AI indiricisi yönlendirmeleri
  > bilerek takip etmiyor (her sıçramayı yeniden doğrulayamayacağı için — §10
  > SSRF korumaları). `cdn.<alanadi>` DNS seviyesinde S3'e işaret ederse sorun
  > yok; ama sunucu `301/302` ile ham S3 adresine yönlendirirse **her indirme
  > başarısız olur** (`PHOTO_DOWNLOAD_FAILED`). Kurulumu yapan kişi bunu bilsin.

  Somut değer belli olunca yapılacak tek şey: `.env` içinde
  `PHOTO_ALLOWED_HOSTS=cdn.<alanadi>`. Kod değişmiyor.

- **Adresler public** (karar: Fatih, 2026-07-28).
  Gerekçe: ilan görselleri herkese hızlıca açılabilmeli. AI servisi hiçbir
  token, yetkilendirme başlığı ya da imzalı adres parametresiyle uğraşmaz;
  düz adresten indirir. Presigned adres seçilseydi süresinin en az 10 dakika
  olması gerekirdi (kuyruk gecikmesi payı) — public olduğu için bu kısıt düştü,
  gecikmiş bir mesaj artık süresi dolmuş adres yüzünden başarısız olmaz.

  > İki sonucu bilerek kabul ediyoruz:
  > 1. Adresi bilen herkes fotoğrafa erişir ve adres **ilan silinse bile
  >    çalışmaya devam eder** — nesne S3'ten ayrıca silinmedikçe. İlan silme
  >    akışı S3 nesnesini de silmeli, yoksa "sildim" diyen kullanıcının
  >    fotoğrafı ortada kalır.
  > 2. Public olması beyaz listeyi GEREKSİZ KILMAZ. Beyaz liste, adresi
  >    verenin (kuyruk mesajının) bizi başka bir yere yönlendirmesini
  >    engellemek içindir; fotoğrafın kendisinin gizli olup olmamasıyla
  >    ilgisi yoktur.
- **Fotoğraf adresleri beyaz listeye alınmalıdır** (`PHOTO_ALLOWED_HOSTS`).
  AI servisi kendisine verilen adresi indirdiği için, korunmazsa iç ağ
  adreslerine istek attırılabilir (SSRF). Uygulanan korumalar:
  yalnızca listedeki alan adları; yerel/özel ağ adresleri ancak listede
  yazılıysa; bağlantı-yerel adresler (169.254.x — bulut kimlik bilgisi ucu)
  **listede yazsa bile her koşulda reddedilir**; yönlendirme takip edilmez;
  indirme sırasında boyut sınırı ve zaman aşımı uygulanır.
  ⚠️ Backend hangi alan adından servis edeceğini bildirmeli ki listeye eklensin.
- Kuyruk mesajlarında kişisel veri taşınmaz: ad, e-posta, telefon gönderilmez.
  Yalnızca ilan kimliği, fotoğraf adresi ve türetilmiş vektörler taşınır.

---

## 11. Kararlar ve kalan sorular

### Verilen kararlar (Fatih, 2026-07-28)

| Soru | Karar | AI tarafına etkisi |
|---|---|---|
| Eşleşme bildirimi kime gider? | **Her iki ilanın sahibine de** (2026-07-29'da netleşti) | Yok — bildirimi Java gönderiyor, AI yalnızca sıralı liste üretiyor |
| Onaylayınca ilan kapanır mı? | **Hayır.** İlanı kapatmak ilan sahibinin elle yapacağı ayrı bir iştir | Yok, ama aşağıdaki nota bak |
| Eşleşme yarıçapı 25 km | Uygun | Yok — süzme Java'da (§5) |
| S3 adresleri | **Public URL** | Kimlik doğrulama kodu gerekmiyor; beyaz liste yine şart (§10) |
| RabbitMQ'yu dağıtım ortamında kim kurar? | **Fatih** — sunucuyu kendisi kuracağını söyledi (2026-07-29) | Lokal taraf çözüldü (zip'ten, yönetici yetkisi gerekmeden — bkz. README). Dağıtımda kuyruk adlarının ve argümanlarının §2'deki gibi olması şart. AI tarafı kurulum ayarlarını hazır tarif olarak teslim edecek |

> ⚠️ **"Onay ilanı kapatmıyor" kararının bir sonucu var.** İlan aktif kaldığı
> için aday havuzunda kalmaya devam eder. Aynı ilan yeniden analiz edilirse
> (fotoğraf değişikliği ya da model sürümü yükseltmesi sonrası toplu yeniden
> analiz) aynı çift yeniden eşleşir ve bildirim TEKRAR gider. §7 kural 4 zaten
> "bildirim tekrarı önlenmelidir" diyor; bunun karşılığı Java tarafında
> "bu ilan çifti için bildirim gönderildi" kaydıdır. AI tarafı bunu bilemez —
> saf bir fonksiyon, geçmişi yok.

### Hâlâ açık

1. **Bildirim hangi ilanın sahibine gidiyor?** Cevap "ilanın sahibine" idi ama
   bir eşleşmede İKİ ilan var: yeni verilen ve eşleşen eski ilan. Kaybettiği
   hayvanı arayan da, bulduğu hayvanı bildiren de haber almak ister — bu
   yüzden büyük ihtimalle cevap "ikisine de", ama netleşmeli.
2. **Alt alan adı ne olacak?** Yaklaşım karara bağlandı (aşağıya bak), sadece
   somut değer bekleniyor: `cdn.<alanadi>` mı `media.<alanadi>` mı.
3. ~~Boş `PHOTO_ALLOWED_HOSTS` ne yapmalı?~~ **Karara bağlandı (2026-07-29):
   kapalıya düşer.** Liste boşsa hiçbir adres indirilmez; hata mesajı ne
   yazılması gerektiğini söyler. Öncesinde boş liste dış adresleri serbest
   bırakıyordu — değişkeni yazmayı unutan bir dağıtım, korumanın açık olduğunu
   sanarak kapalı çalışırdı. ⚠️ **Dağıtımda bu değişken doldurulmalı**, yoksa
   servis hiçbir fotoğrafı indiremez (§10).

---

## 12. Uygulama durumu

| Taraf | Durum |
|---|---|
| Python AI — analiz (`/analyze`) | ✅ Çalışıyor (HTTP) |
| Python AI — eşleştirme (`/match`) | ✅ Çalışıyor (HTTP), alan adları sözleşmeyle hizalandı (`ad_id`, `model_version`) |
| Python AI — uç durum sağlamlaştırması | ✅ Negatif mesafe, NaN/bozuk vektör, kendisiyle eşleşme, tekrar eden aday, aday sınırı, çok küçük/bozuk görüntü, EXIF döndürme, şeffaf PNG, "hayvan mı" kapısı — 24 gerileme testi |
| Python AI — çoklu fotoğraf eşleştirme | ✅ Görsel skor en iyi fotoğraf çiftinden; gerçek veriyle doğrulandı (`docs/olcum-raporu.md`) |
| Python AI — gerçek veriyle ölçüm | ✅ Eşik ve hayvan kapısı gerçek fotoğraflarla sınandı; bulgular tasarıma işlendi |
| Python AI — URL'den indirme + çoklu fotoğraf | ✅ `POST /analyze_url` — kuyruk akışıyla aynı kodu çağırır, RabbitMQ olmadan da denenebilir |
| Python AI — SSRF koruması | ✅ Beyaz liste, yerel ağ engeli, bağlantı-yerel mutlak yasak, yönlendirme yok, boyut/zaman sınırı |
| Python AI — cins (`breed`) | ✅ Yapıldı — 37 ırk zero-shot; top-1 %78, güven eşiği 0.70 üstünde %90 (ölçüm: `scripts/measure_breed.py`) |
| Python AI — RabbitMQ tüketici/üretici | ✅ `app/kuyruk.py` — topoloji, tüketici, üretici, DLQ, yeniden bağlanma. Ayrı süreç: `python -m app.kuyruk`. 21 test broker olmadan koşuyor (`tests/test_kuyruk.py`) |
| Python AI — tüketicinin imajda çalışması | ✅ **Düzeltildi.** İmajın `CMD`'i yalnız uvicorn'u başlatıyordu; tüketici üretimde HİÇ çalışmıyor, ilanlar sonsuza kadar `ai_status=PENDING` kalıyordu. Artık `entrypoint.sh` iki süreci de kaldırıyor (`SERVIS_ROLU=hepsi\|http\|kuyruk`, varsayılan `hepsi`); biri ölürse konteyner sıfırdan farklı kodla kapanıyor. `docker stop` artık düzgün kapanıyor (SIGTERM işleyicisi). Her PR'da ölçülüyor: `ci.yml` → `docker` işi imajı kurup `/health` + kuyruk akışını gerçek broker üzerinde sınıyor |
| Python AI — uçtan uca kanıt | ✅ **Gerçek broker üzerinde koşturuldu** (2026-07-28, RabbitMQ 4.3.4 + Erlang 27.3.4.13). `scripts/sahte_java.py` Java'nın yerine geçip istek yayınladı, sonuç 2,4 sn'de döndü ve sözleşme denetimini geçti — **`siglip2-animal/v2`, 768 boyutlu vektör**, yani bu belgedeki boyutla birebir. `--hata-yollari` kipiyle DLQ ve hata cevabı da doğrulandı |
| Java — `Ad.photoUrls` | ⬜ KISIM 2'de |
| Java — `ai_*` alanları | ⬜ AI sorumlusunda |
| Java — kuyruk config + publisher + listener | ⬜ AI sorumlusunda |
| Ortak — `docker-compose` (PostGIS + RabbitMQ + AI) | ⬜ AI sorumlusunda |
