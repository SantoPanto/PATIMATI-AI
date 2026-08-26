# Ölçüm Raporu — AI Eşleştirme Servisi

**Tarih:** 2026-07-26 · **Güncel ürün modeli:** `siglip2-animal/v2`

Bu belge, servisin doğruluğuna dair yapılan tüm ölçümleri ve bu ölçümlerin
tasarımı nasıl değiştirdiğini kaydeder. Buradaki her sayı çalıştırılabilir bir
betikten gelir; hiçbiri tahmin değildir.

| Betik | Ne ölçer |
|---|---|
| `scripts/measure_breed.py` | Tür ve cins doğruluğu (Oxford-IIIT Pet, 111 fotoğraf) |
| `scripts/measure_threshold.py` | Eşik taraması (sentetik "aynı birey" simülasyonu) |
| `scripts/gercek_veri_olcum.py` | **Gerçek dünya fotoğraflarıyla eşik ve hayvan kapısı doğrulaması** |
| `scripts/teshis_arkaplan.py` | **Skor hayvandan mı arka plandan mı geliyor?** (tam / kırpılmış / hayvansız, AUC ile) |
| `scripts/eslesme_yok_testi.py` | **Açık Küme (Open-Set) yanlış alarm ve eşik taraması** |

> **Veri notu:** Gerçek dünya fotoğrafları (`tests/gercek_veri/`) KVKK gereği
> repoya dâhil edilmez — sokak ve iç mekân kareleri uzaktan da olsa insan
> içerebiliyor. Ölçümü tekrarlamak için kendi fotoğraflarınızı aynı klasör
> yapısında toplayın (bkz. §6).

---

## 1. Veri kümeleri

| Küme | İçerik | Kaynak |
|---|---|---|
| Seed | 37 ırk × 3 fotoğraf = 111 | Oxford-IIIT Pet — safkan, stüdyo kalitesi |
| Aynı birey | 1 sokak kedisi × 7 fotoğraf | Gerçek, farklı gün/açı/mesafe, siyah kedi |
| Negatif | 26 fotoğraf | Gerçek: sokak, iç mekân, manzara, nesne — hayvan yok |
| Farklı birey | 3 fotoğraf (Bombay) | Seed setinden, siyah kedi = en zor ayrım |

---

## 2. Tür ve cins doğruluğu (seed veri)

| Ölçüm | Sonuç |
|---|---|
| Tür (kedi/köpek) doğruluğu | **%100** (111/111) |
| Cins top-1 | **%78.4** (87/111) |
| Cins top-3 | %96.4 |
| Rastgele tahmin taban çizgisi | %2.7 |

**Cins güven eşiği seçimi:**

| Eşik | Cins gösterilen fotoğraf | Gösterilenin doğruluğu |
|---|---|---|
| 0.00 | %100 | %78 |
| 0.50 | %91 | %81 |
| **0.70 (seçildi)** | **%69** | **%90** |
| 0.90 | %49 | %94 |

**Denenip reddedilenler:**
- *Tür daraltması* (kedi fotoğrafında yalnız 12 kedi ırkına bakmak): doğruluğu
  hiç değiştirmedi — CLIP kediyle köpeği ırk seviyesinde bile karıştırmıyor.
  Yine de tutuldu: bozuk görüntülerde garanti sağlıyor, maliyeti sıfır.
- *Çoklu prompt şablonu ortalaması* (literatürde standart iyileştirme):
  %78.4 → %76.6, yani **düşürdü**. Alınmadı.

**Hataların doğası:** En çok karışanlar British Shorthair↔Russian Blue,
Cocker Spaniel↔Setter, Chihuahua↔Miniature Pinscher — hepsi gerçekten benzeyen
ırklar. Model saçmalamıyor, insanın da zorlanacağı yerlerde zorlanıyor.

---

## 3. "Hayvan var mı" kapısı

Tür sınıflandırması yalnızca kedi/köpek arasında yapılırsa, hayvan olmayan bir
fotoğraf da zorunlu olarak birine atanır. Kapı, 10 çeldirici seçenek (insan,
araba, kuş, tavşan, at, yemek, manzara, eşya, ekran görüntüsü, düz renk)
ekleyerek bu zorunlu seçimi açık uçlu hâle getirir.

**Ölçüt seçimi:** "Kazanan çeldiriciyse reddet" kuralı fazla keskin çıktı —
bahçede yan duran açık renkli bir teriyer %73 ihtimalle "at" sayılıp
reddediliyordu. Onun yerine **kedi+köpek olasılıklarının toplamı** kullanıldı.

| Küme | Kedi+köpek toplamı |
|---|---|
| Gerçek hayvan (111) | min **0.23** · ortanca 0.96 |
| Sentetik görüntü (düz renk, gürültü, gradyan) | 0.02 – 0.04 |

Eşik **0.10** seçildi (en kötü gerçek fotoğrafa 2.3 kat pay). Sonuç: seed
verisinde **0/111 yanlış reddetme**, tür doğruluğu %100'de kaldı.

### ⚠️ Gerçek negatiflerde sonuç çok daha kötü

| Küme | Doğru reddedilen |
|---|---|
| Sentetik negatifler | 4/4 = **%100** |
| **Gerçek negatifler (26 fotoğraf)** | **19/26 = %73** |

7 gerçek fotoğraf "hayvan var" sanıldı, **3'üne "köpek" dendi** (boş kaldırım,
bina cephesi). **Sentetik test yanıltıcıydı.** Kapı bir uyarı mekanizmasıdır,
filtre değildir — bu oranla ilan reddetmek için kullanılamaz.

---

## 4. 🔴 Eşik doğrulaması — en önemli bulgu

Aynı kedinin 7 fotoğrafı → 21 çift. Aynı kedi ↔ farklı siyah kediler → 21 çift.

| | Aynı kedi | Farklı kedi |
|---|---|---|
| Görsel benzerlik (ortanca) | 0.739 | 0.690 |
| Toplam skor (ortanca) | 0.611 | 0.604 |
| **Eşiği (0.70) geçen** | **5/21 = %24** | 0/21 = %0 |

**İki dağılım çakışıyor.** Aynı bireyin en düşüğü 0.446, farklının en yükseği
0.699 — hiçbir eşik ikisini temiz ayırmıyor:

| Eşik | Gerçek eşleşme yakalama | Yanlış alarm |
|---|---|---|
| 0.60 | %52 | %52 |
| 0.65 | %38 | %14 |
| **0.70** | **%24** | **%0** |

Yani mevcut eşik "dediğinde doğru diyor" ama **dört gerçek eşleşmeden üçünü
kaçırıyor**. Önceki doğrulama (aynı fotoğrafı bozarak) gerçeğe göre çok
iyimserdi.

### Denenen çözüm: hayvana kırpma — **reddedildi**

Hipotez: CLIP tüm sahneyi gömüyor; hayvana kırparsak vektör hayvanı anlatır.
Hazır bir COCO detektörüyle test edildi.

| | Ayrım gücü | 0.70'te yakalama | 0.70'te yanlış alarm |
|---|---|---|---|
| Kırpmasız | **+0.063** | 5/21 | **0/21** |
| Kırpmalı | +0.016 ⬇ | 15/21 | 8/21 ⬆ |

Kırpma ayrımı **kötüleştirdi**: hayvanı izole edince bütün siyah kediler
birbirine benziyor — arka plan (aynı mahalle, aynı kaldırım) ayırt edici bilgi
taşıyormuş. Dedektör ayrıca 7 fotoğrafın 2'sinde hayvanı hiç bulamadı.
**Uygulanmadı.**

---

### 🔁 DÜZELTME (2026-07-27): yukarıdaki karar geçersiz

Yukarıdaki tablo silinmedi, çünkü asıl ders onda: **yanlış ölçüt, doğru veriden
yanlış karar üretir.** Karar iki ayrı kusura dayanıyordu.

**Kusur 1 — ölçüt ölçeğe duyarlıydı.** "Ayrım gücü" iki ortalamanın farkıdır.
Kırpma tüm skorları yukarı ittiğinde bu fark daralır, ama *sıralama kalitesi*
artmış olabilir — nitekim artmıştı. Animal re-ID literatürünün kullandığı ölçütler
ölçekten bağımsızdır: Rank-1, mAP, CMC, AUC.

**Kusur 2 — negatif küme cevapla örtüşüyordu.** `scripts/gercek_veri_olcum.py`
"farklı birey" grubunu `tests/seed_data/class_07`ten (Oxford stüdyo Bombay) alıyor:

```
aynı kedi  çiftleri = sokak fotoğrafı ↔ sokak fotoğrafı   (aynı mahalle)
farklı kedi çiftleri = sokak fotoğrafı ↔ STÜDYO fotoğrafı
```

Arka plan tipi, etiketin kendisiyle birebir örtüşüyor. Arka planı gören herhangi
bir model bu iki grubu hayvana hiç bakmadan ayırır. §6'da bu kümenin adil olmadığı
zaten not edilmişti — ama karar yine o kümeyle verildi.

**Yeniden ölçüm** (`scripts/teshis_arkaplan.py`, jaguar teşhis protokolü —
arXiv 2604.09690). Üç koşul, **birebir aynı fotoğraf kümesi**, görsel AUC:

| Koşul | Görsel AUC | Ne anlama geliyor |
|---|---|---|
| **TAM** — fotoğrafın kendisi (bugünkü sistem) | 0.793 | taban çizgisi |
| **HAYVAN** — dedektör kutusuna kırpılmış | **1.000** | kırpma ayrımı yükseltiyor |
| **ARKAPLAN** — hayvan silinmiş, sadece ortam | **0.800** | ⛔ hayvansız hâli, tam kareden **iyi** ayırıyor |

Yani mevcut skorumuzun ayrımı hayvandan değil **ortamdan** geliyor; hayvan sinyali
gürültü gibi davranıyor. Üründe bu ölümcül: KAYIP ilanı evde çekilir (koltuk,
parke), BULUNDU ilanı sokakta. Arka planlar örtüşmez, ayrım da yok olur.

**Bu düzeltmenin sınırları — kırpma "doğru" ilan edilmiyor:**
- n küçük (10 aynı-çift, 15 farklı-çift). AUC 1.000 bu kümede gerçek, genel değil.
- Negatif küme **hâlâ** stüdyo Bombay. Bu ölçüm tuzağı gösterir, kararı vermez.
- Dedektör 7 fotoğrafın 2'sinde (%29) hayvanı bulamadı — kırpmaya geçmenin gerçek
  bir işletme bedeli var, "bulamazsa ne olur" tasarlanmalı.

**Durum:** kırpma kararı **yeniden açıldı**. Adil karar, aynı kaynaktan farklı
bireyler içeren bir kümeyle verilecek (`wildlife-datasets` → `CatIndividualImages`,
518 kedi / 13.536 fotoğraf). Bkz. §7.

### Uygulanan çözüm: çoklu fotoğraf

İlan başına 3 fotoğraf, görsel skor **en iyi fotoğraf çiftinden**:

| | Skor | 0.70 eşiği |
|---|---|---|
| Aynı kedi, tek fotoğraf (ortalama) | 0.647 | ✗ |
| **Aynı kedi, 3 fotoğraf (en iyi çift)** | **0.748** | **✓** |
| Farklı kedi, 3 fotoğraf (en iyi çift) | 0.640 | ✓ temiz |

Çoklu fotoğraf kaybedilen duyarlılığı geri getiriyor ve yanlış alarm üretmiyor.
**Bu yüzden `MatchCandidate.embeddings` bir listedir ve sözleşme ilan başına
birden çok fotoğraf zorunlu kılar.**

### 4.1. Açık Küme Eşik Taraması (Gerçek Motor ve "Data Leakage" Önlemi)

**Amaç:** Açık küme TPR/FPR analizini yapay varsayımlardan arındırarak, üretimdeki
`siglip2-animal` kimlik modeli ve canlı eşleştirme motoru
`app.matcher.compute_final_score` ile kör (blind) olarak ölçmek.

**Metodoloji:**
* Testte "Data Leakage" (Veri sızıntısı) engellenir. Motor, fotoğrafların aynı
  hayvana ait olup olmadığını (ground truth) bilmez.
* Her fotoğrafın etiketleri ve türü, ham fotoğraf baytları `attribute_analyzer`
  servisine verilerek üretilir. Kimlik modelinin 768 boyutlu vektörü etiketçiye
  verilmez; etiketçi üretimdeki gibi kendi CLIP vektörünü kullanır.
* `CatIndividualImages` doğrulanmış konum verisi taşımadığından, konum katmanını
  denetimli tutmak için her karşılaştırmada 5.0 km kullanılır. Bu yalnızca konum
  bileşenini kontrol eder; etiket ve tür girdileri fotoğrafa özeldir.

**Bulgular:** Önceki tablodaki sayılar `google-siglip2` ve sabit etiketlerle
üretilmişti. Modelin skor dağılımı/EER'i ve dinamik etiketlerin katkısı değiştiği
için bu sayılar `siglip2-animal` için geçerli değildir. Eski sayılar geçersizdi, aşağıdaki tablo sızıntı kapatıldıktan sonraki yeni koşumdan elde edilmiştir.

| Ürün eşiği | Ham Görsel Skor (TPR) | Hibrit Skor - app.matcher (TPR) |
|---|---|---|
| **0.80  (`MATCH_THRESHOLD`)** | %96.9 (FPR: %85.9) | %94.9 (FPR: %57.7) |

**Sonuç ve kritik çıkarım:**

1. Ürünün bildirim eşiği `MATCH_THRESHOLD = 0.80`'dir. Önceki taslaktan kalan
   eşik değeri artık bu raporda ürün eşiği olarak kullanılmaz.
2. Eski sabit-etiket sonuçlarıyla 0.80 için duyarlılık, yanlış alarm veya hibrit
   skor tavanı hakkında çıkarım yapılamaz.
3. Betik, üretimdeki güncel `siglip2-animal` kimlik modeli ve dinamik `attribute_analyzer` etiketleriyle yeniden çalıştırılmış olup, elde edilen taze sonuçlar (FPR: %57.7, TPR: %94.9) tabloya işlenmiştir.

### Eşik Kararının Gerekçesi (Tam Tarama Tablosu)
Artık eşik karara bağlandığına göre, bu kararın gerekçesini oluşturan tam tarama verileri aşağıdadır:

| Eşik | Yanlış alarm (FPR) | Yakalama (TPR) |
| :--- | :--- | :--- |
| 0.65 | %80,5 | %96,9 |
| 0.70 | %57,7 | %94,9 |
| **0.80** | **%13,4** | **%68,4** |
| 0.85 | %3,4 | %40,8 |

### Ölçüm Metodolojisi ve Kısıtlar

1. **FPR Ölçüm Kriteri (Galeri Maksimumu):** Testler sırasında negatifler galeri maksimumu olarak puanlanmıştır. Bu kurgu, ürünün çalışma mantığıyla (adayın tamamına bakılması) örtüşmektedir. Rapordaki FPR değeri basit bir "iki fotoğraf karşılaştırmasındaki yanılma oranı" değil; "bir sorgunun, 50 kimlikli bir galeride en az bir yanlış eşleşme üretme oranıdır". Gerçek üründe galeri boyutu büyüdükçe bu oranın da matematiksel olarak artma potansiyeli bulunmaktadır.
2. **Konum Sabiti:** Kullanılan `CatIndividualImages` veri setinde konum verisi bulunmadığı için (başka bir seçenek olmadığından), mesafe her karşılaştırmada sabit **5,0 km** kabul edilmiştir (`location_score(5.0) = 0.50` ve ağırlığı `0.15`). Bu durum her skora eşit olarak sabit **+0.075** eklemektedir. Değer her iki tarafa da eşit eklendiği için eşikler arası kıyası bozmamakta, ancak rapordaki mutlak TPR/FPR sayılarının bu sabite bağlı olduğu unutulmamalıdır.

## 5. Bulguların tasarıma etkisi

| Bulgu | Karar |
|---|---|
| Dağılımlar çakışıyor | İkili "eşleşti/eşleşmedi" kararı **bırakıldı**. Sistem sıralı aday listesi üretir, son onay kullanıcıdadır. |
| Tek fotoğraf yetersiz | İlan başına çoklu fotoğraf **zorunlu**; skor en iyi çiftten |
| Eşik 0.70 = düşük duyarlılık, sıfır yanlış alarm | Eşiğin anlamı değişti: "eşleşme" değil, **"bildirim gönderilecek kadar eminiz"**. Daha düşük skorlu adaylar listede görünür ama bildirim tetiklemez. |
| ~~Kırpma zarar veriyor~~ → **karar geçersiz (27.07)** | Yeniden açıldı. Ölçüt ölçeğe duyarlıydı, negatif küme cevapla örtüşüyordu. Adil kümeyle tekrar ölçülecek |
| Ayrımın kaynağı hayvan değil ortam (27.07) | Ölçüt olarak **AUC/Rank-1/mAP** kullanılacak; "ayrım gücü" bırakıldı |
| Kapı gerçek negatiflerde %73 | Uyarı olarak kullanılır, ilan reddetmek için kullanılmaz |
| Cins doğruluğu %78, melezlerde daha düşük | Cins **asla filtre değil**, yalnızca bilgi |

---

## 6. Hâlâ ölçemediklerimiz

| Boşluk | Neden önemli | Ne gerekiyor |
|---|---|---|
| **Tek bir birey ölçüldü** | 21 çiftin tamamı aynı kediden; genelleme zayıf | 10+ farklı hayvan, her biri 3+ fotoğraf |
| **Farklı birey kümesi adil değil** | Sokak kedisi ↔ stüdyo Bombay: kimlik dışında ortam da farklı. Gerçek "farklı sokak kedileri" karşılaştırması daha zor olacaktır, yani ölçtüğümüz ayrım **iyimser** olabilir | Aynı ortamda çekilmiş farklı bireyler |
| Köpek verisi yok | Gerçek veride hiç köpek yok | Köpek fotoğrafları |
| Kapının gerçek yakalama oranı dar örneklem | 26 negatif azdır; oyuncak/çizim hiç denenmedi | 50+ negatif, peluş ve çizim dâhil |
| Zaman içindeki değişim | Hayvan büyür, tüyü değişir | Aylar arayla çekilmiş çiftler |
| Geri bildirim döngüsü | Kullanıcı "evet bu benim kedim" dediğinde kaydetmiyoruz — eşiği gerçek kullanımla iyileştirmenin tek yolu bu | Onay ekranı + kayıt |

**Ölçümü tekrarlamak için** `tests/gercek_veri/` altında şu yapıyı kurun:

```
ayni_birey/   kedi01_01.jpg, kedi01_02.jpg, ...   (aynı hayvanın fotoğrafları)
negatif/      neg_01.jpg, ...                     (hayvan içermeyen fotoğraflar)
etiketler.json
```

sonra `venv\Scripts\python.exe scripts/gercek_veri_olcum.py` çalıştırın.

> **27.07 notu:** Bu tablodaki ilk iki satır (tek birey, adil olmayan negatif küme)
> sanıldığından kolay kapanıyor. Birey etiketli **halka açık** veri kümeleri var —
> §7'ye bakın. "Kendi fotoğraflarımızı toplamalıyız" varsayımı, bakmadığımız için
> doğru sanılmıştı.

---

## 7. Model yarışı — altı aday, üç veri kümesi (2026-07-27/28)

### Neden yapıldı

Bu rapordaki §2–§6 arası her ölçüm tek bir soruyu soruyor: *"CLIP'i tüm sahneye
uygularsak nasıl ayarlarım?"* Hiçbiri *"CLIP'i tüm sahneye uygulamak doğru mu?"*
diye sormuyor. Tek bir alternatif model bile denenmediği için "%24 iyi mi kötü mü"
sorusunun cevabı yoktu. **Eksik olan bir model değil, kıyaslama düzeneğiydi.**

Düzenek: `scripts/model_yarisi.py`. Ham sonuçlar: `docs/olcum-sonuclari/*.json`.

### Karar kuralı

> Bir model, ancak **birden çok koşulda tutarlı** kazanıyorsa seçilir.

Bu kural, §4 düzeltmesindeki hatanın tekrarını engellemek için kondu: tek koşulda
ölçüp genellemek bu projede üç kez yanlış karar üretti.

### Veri (hepsi halka açık, `wildlife-datasets` / Kaggle / Mendeley)

| Küme | İçerik | Hazırlık | Lisans |
|---|---|---|---|
| DogFaceNet | 1.393 köpek / 5.327 foto | sıkı yüz kırpması 224×224 | CC BY 4.0 |
| MPDD | 191 köpek / 762 foto | gevşek kırpma | CC BY 4.0 |
| CatIndividualImages | 138 kedi / 540 foto | **tam sahne**, 3264×2448 telefon | CC BY 4.0 |

### Sonuçlar

**EER — bildirim kararında hata oranı (düşük iyi). Ürün için en önemli ölçüt.**

| Model | DogFaceNet | MPDD | Kedi | Ortalama |
|---|---|---|---|---|
| **avito-siglip2** | **0.0654** | **0.0625** | **0.0285** | **0.052** |
| avito-clip | 0.0708 | 0.1140 | 0.0642 | 0.083 |
| google-siglip2 | 0.1262 | 0.0999 | 0.0463 | 0.091 |
| google-siglip1 | 0.1247 | 0.1004 | 0.0466 | 0.091 |
| **bizim (mevcut)** | 0.1888 | 0.1508 | 0.0603 | **0.113** |
| megadescriptor | — (yavaş) | 0.1245 | 0.0548 | — |

**Top-1 — sorguya en benzeyen fotoğraf doğru hayvan mı (yüksek iyi)**

| Model | DogFaceNet | MPDD | Kedi |
|---|---|---|---|
| google-siglip1 | **%62.6** | %82.8 | %96.9 |
| google-siglip2 | %60.5 | **%83.3** | **%97.0** |
| avito-siglip2 | %55.5 | %76.8 | %94.8 |
| avito-clip | %50.0 | %64.0 | %88.1 |
| bizim | %48.5 | %69.9 | %94.4 |

`avito-siglip2` **üç kümede de EER'de birinci**; Google modelleri Top-1'de
birinci ama farkları küçük ve EER'de iki katı hata yapıyorlar. İkisi farklı
yeteneği ölçüyor: Top-1 sıralamayı, EER skorların eşiğe göre ayrılabilirliğini.
Ürünümüz bildirim eşiği kullandığı için **EER belirleyicidir**.

### ⚠️ Arka plan ablasyonu — bu turun en önemli bulgusu

Kedi fotoğraflarından hayvan silinip **yalnız arka planla** ölçüldü
(`scripts/kosul_uret.py`). Rastgele taban ≈ %0.6.

| Model | Tam sahne | Sadece arka plan | Hayvanın katkısı |
|---|---|---|---|
| google-siglip1 | %96.9 | %89.6 | 7.3 puan |
| google-siglip2 | %97.0 | %88.3 | 8.7 puan |
| megadescriptor | %95.9 | %83.9 | 12.0 puan |
| **bizim** | %94.4 | **%79.4** | **15.0 puan** |
| avito-clip | %88.1 | %50.6 | 37.5 puan |
| **avito-siglip2** | %94.8 | **%53.4** | **41.4 puan** |

Kedi kümesinde aynı hayvanın fotoğrafları aynı evde çekilmiş; arka plan kimliği
ele veriyor. **Kedi tablosundaki mutlak sayılar bu yüzden şişkindir.**

Yalnız `avito-siglip2` (ve daha zayıf hâliyle `avito-clip`) arka plan gidince
gerçekten çöküyor — yani asıl bilgiyi hayvandan alıyorlar. Bu şaşırtıcı değil:
AvitoTech modelleri 1.9M kayıp hayvan ilanıyla eğitilmiş, o veride aynı hayvan
farklı ortamlarda göründüğü için model arka planı kullanmayı öğrenemiyor.

**Yanlış çıkan bir çıkarım, kayda geçsin:** "arka plana yaslanan model, arka plan
olmayınca çöker" diye öngörmüştük. DogFaceNet'te (arka plan yok) Google modelleri
Top-1'de yine önde çıktı — öngörü yanlıştı. Ablasyon, modelin arka planı ne kadar
*kodladığını* ölçüyor; ondan "hayvana bakmıyor" sonucu çıkmıyor. Doğru okuma:
Google modelleri hem sahneyi hem hayvanı kodluyor, `avito-siglip2` ise ağırlıklı
olarak hayvanı.

### Kırpmanın etkisi

Kedi verisinde dedektörle kırpıp tekrar ölçüldü: **fark gürültü mertebesinde**
(±1 puan, yönü bile tutarsız). Sebep: bu fotoğraflar zaten yakın çekim, hayvan
kareyi dolduruyor. §4'teki bulgu ise geniş sokak karelerindeydi.

⇒ Doğru ifade "kırpma her zaman iyidir" değil, **"hayvan kareyi doldurmuyorsa
kırpma önemlidir"**. Dedektör kedi fotoğraflarının %99.6'sında hayvanı buldu.

### Lisanslar (HuggingFace API'den doğrulandı)

| Model | Lisans | Ticari kullanım |
|---|---|---|
| `google/siglip2-base-patch16-224` | apache-2.0 | ✅ |
| `google/siglip-base-patch16-224` | apache-2.0 | ✅ |
| `openai/clip-vit-base-patch32` | kartta yok; orijinal MIT | ✅ |
| `BVRA/MegaDescriptor-L-384` | **cc-by-nc-4.0** | ❌ ticari YASAK |
| `AvitoTech/*` (ikisi de) | **hiç alan yok** | ❌ izin belirsiz |

AvitoTech için üç kaynak da boş: model kartı README'si, HF API `cardData`, depo
dosya listesi. Bir kullanıcı (`kh1nt`) 2026-07 başında aynı soruyu tartışma #3'te
sormuş, **cevap gelmemiş**. Model Ocak 2026'dan beri güncellenmemiş.

### Karar

**Seçilen: `avito-siglip2`.** Gerekçe: üç veri kümesinin üçünde de en düşük hata
oranı; mevcut modelimize göre bildirim kararında hata ~%11'den ~%5'e iniyor.

**Lisans notu — bilerek alınmış risk.** İzin belirsiz. Staj kapsamında dağıtım
yapılmadığı için kullanılıyor. **Bu servis ticarileşecekse model yeniden
değerlendirilmelidir.** Yedek plan: `google/siglip2-base-patch16-224` (Apache 2.0,
EER 0.091) ya da o tabandan kendi ince ayarımız — eğitim verisi olarak
kullandığımız üç küme de CC BY 4.0, yani buna hukuken engel yok.

### Bilinen engel (uygulamaya geçerken)

`app/attributes.py` zero-shot etiketler için `embedder.embed_text()` çağırıyor.
Gömme modelini değiştirmek **skorun %30'unu oluşturan etiket katmanını kırar**
⇒ iki model gerekecek: etiketçi (CLIP, tür doğruluğunda %100) + kimlikçi
(`avito-siglip2`). `app/surum.py:VEKTOR_BOYUTU` ve
`app/matcher.py:dogrula_embedding` 512'ye sabit; SigLIP2 **768** üretiyor.
Java tarafında `ai_*` kolonları henüz yazılmadı, yani boyut değişikliği şu an bedava.

Ayrıca `AvitoTech` ağırlıkları olduğu gibi yüklenmiyor: 408 anahtarın hepsi
`clip.` önekli ve `text_config.vocab_size` eksik. Düzeltilmezse model **rastgele
ağırlıklarla, sessizce** yükleniyor (ilk ölçümümüzde Top-1 %20.2 çıktı).
`scripts/model_yarisi.py` artık eksik ağırlıkta hata fırlatıyor.

### Hâlâ ölçemediğimiz

Halka açık köpek kümelerinin hepsi kırpık; **tam sahne köpek verisi yok**. Ve
hiçbir küme gerçek kayıp/bulundu çifti değil — yani "kayıp fotoğrafı evde, bulundu
fotoğrafı sokakta" senaryosu ölçülmedi. Bunun tek çözümü gerçek "sahibine kavuştu"
ilanlarından çift toplamak.

---

## 8. Kırpma yeteneği + tasarım/afiş sinyali (2026-08-22)

İki kullanıcı şikayetinin araştırılması sırasında eklendi: (1) farklı açıdan
çekilmiş fotoğraflar eşleşmiyor, (2) poster/afiş görselleri hiç eşleşmiyor.
İkisi de bu raporun §4/§7'sindeki "arka plan, ayrımın büyük bölümünü taşıyor"
bulgusuyla aynı kökten geliyor — kırpma bunu düzeltme adayı olarak §4'te
"yeniden açıldı" denip kapanmamıştı.

**Eklenen, CANLI DAVRANIŞI DEĞİŞTİRMEYEN iki şey:**

1. **`app/kirpma.py`** — torchvision'ın hazır COCO dedektörüyle (ek indirme/
   lisans riski yok, ~74 MB, torch/torchvision zaten bağımlılık) kedi/köpek
   bölgesine kırpma yeteneği. `CROP_TO_ANIMAL` env bayrağıyla açılır,
   **varsayılan KAPALI**. Hayvan bulunamazsa (ölçümde 7 fotoğrafın 2'sinde
   olduğu gibi) ya da kırpılan bölge çok küçük çıkarsa tam görüntüye geri
   döner. `app/embedder.py:goruntu_ac`'a min-boyut kontrolünden SONRA
   eklendi.

   ⚠ **Açılması ayrı bir karar** — `KIMLIK_MODEL` gibi embedding'in üretilme
   biçimini değiştirir, `MODEL_SURUMU` artışı ve mevcut ilanların yeniden
   analizini gerektirir (bkz. app/surum.py, .env.example). Bu, backend #121
   canlıya alınıp toplu yeniden-analiz çalıştırılana kadar zaten mümkün
   değil.

   **`scripts/kirpma_karar_olcumu.py`** eklendi: §4/§7'nin "adil değil, sadece
   tuzağı gösteriyor" dediği stüdyo/sokak konfondu OLMADAN (çok bireyli, tek
   kaynaktan veri), gerçek üretim kırpma fonksiyonuyla (`app/kirpma.py`) TAM
   vs HAYVAN AUC'sini kıyaslar. Mantığı sentetik veriyle sınandı (`tests/
   test_kirpma_karar_olcumu_betik_mantigi.py`).

   ### 8.1. Sonuç — MPDD ile GERÇEK ölçüm (2026-08-22)

   CatIndividualImages ile denenmedi (Kaggle kimlik doğrulaması bu ortamda
   yok). Bunun yerine **MPDD** (Mendeley, "Multi-pose dog dataset", 191 köpek/
   1657 foto, CC BY 4.0, `https://data.mendeley.com/datasets/v5j6m8dzhv/1` —
   Kaggle GEREKTİRMEZ) indirilip `<birey>/<foto>` düzenine dönüştürüldü
   (`data/MPDD_bireyler/`, depoda değil — data/ gitignore'da).

   | Örneklem | TAM (bugünkü) görsel AUC | HAYVAN (kırpılmış) görsel AUC | Fark |
   |---|---|---|---|
   | 40 birey × 3 foto | 0.899 | 0.902 | +0.003 |
   | 80 birey × 3 foto | 0.907 | 0.899 | **−0.008** |

   **Sonuç: TUTARSIZ.** Yön, örneklem büyüklüğüyle DEĞİŞİYOR — iki ölçüm de
   ±0.01 içinde, yani fark gürültü seviyesinde. Bu raporun kendi karar kuralı
   (§7: *"bir model/değişiklik, ancak birden çok koşulda tutarlı kazanıyorsa
   seçilir"*) burada net bir HAYIR üretiyor: **kırpma bu ölçümde CLIP kimlik
   modeliyle görünür/güvenilir bir kazanç sağlamıyor.**

   ⚠ **Sınırlamalar:**
   - Ölçüm üretim kimlik modeliyle (`siglip2-animal`/`avito-siglip2`) DEĞİL,
     `KIMLIK_MODEL=clip` ile yapıldı — bu ortamda `sentencepiece` paketi kurulu
     değil ve avito-siglip2 ağırlığı (~1,4 GB) cache'de yok. §7'nin arka plan
     ablasyonu, modellerin arka plana bağımlılığının ÇOK FARKLI olduğunu
     gösteriyor (CLIP'e en yakın `avito-clip` arka plandan 37,5 puan
     etkileniyor, `avito-siglip2` 41,4 puan) — yani bu sonuç CLIP'e özgü
     olabilir, **üretim modeliyle yeniden ölçülmeden genellenemez.**
   - Yalnızca MPDD (köpek) ölçüldü, kedi verisiyle (CatIndividualImages)
     tekrarlanmadı — karar kuralının "birden çok koşul" şartı tam
     karşılanmıyor.

   **Karar önerisi: `CROP_TO_ANIMAL` şimdilik AÇILMAMALI.** Gelecekte
   yeniden değerlendirmek için: `pip install sentencepiece` + üretim
   ağırlığını indirip aynı betiği (`KIMLIK_MODEL` ayarlamadan, varsayılanı
   kullanarak) MPDD ve mümkünse CatIndividualImages'la tekrarlayın.

2. **`app/attributes.py`: `is_designed_graphic`/`graphic_confidence`** —
   mevcut zero-shot kapı mimarisiyle (is_pet kapısıyla aynı yöntem) "bu görsel
   düz bir fotoğraf mı, tasarlanmış bir poster/afiş mi" sinyali. **BİLGİ
   AMAÇLI** — is_pet/species/tür eşleştirmesini etkilemez, yalnızca
   `AnalyzeResponse`'a eklendi. ⚠ ÖLÇÜLMEDİ: gerçek poster/afiş test kümesi
   yok (§6'nın "oyuncak/çizim hiç denenmedi" boşluğuyla aynı kategori).
   Ekranda gösterme ya da eşleştirmede kullanma (ör. tespit edilirse zorunlu
   kırpma) kararı, alanın doğruluğu gerçek verilerle ölçüldükten SONRA
   ayrıca alınacak.
