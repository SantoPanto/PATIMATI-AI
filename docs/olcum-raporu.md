# Ölçüm Raporu — AI Eşleştirme Servisi

**Tarih:** 2026-07-26 · **Model sürümü:** `clip-vit-base-patch32/v1`

Bu belge, servisin doğruluğuna dair yapılan tüm ölçümleri ve bu ölçümlerin
tasarımı nasıl değiştirdiğini kaydeder. Buradaki her sayı çalıştırılabilir bir
betikten gelir; hiçbiri tahmin değildir.

| Betik | Ne ölçer |
|---|---|
| `scripts/measure_breed.py` | Tür ve cins doğruluğu (Oxford-IIIT Pet, 111 fotoğraf) |
| `scripts/measure_threshold.py` | Eşik taraması (sentetik "aynı birey" simülasyonu) |
| `scripts/gercek_veri_olcum.py` | **Gerçek dünya fotoğraflarıyla eşik ve hayvan kapısı doğrulaması** |

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

---

## 5. Bulguların tasarıma etkisi

| Bulgu | Karar |
|---|---|
| Dağılımlar çakışıyor | İkili "eşleşti/eşleşmedi" kararı **bırakıldı**. Sistem sıralı aday listesi üretir, son onay kullanıcıdadır. |
| Tek fotoğraf yetersiz | İlan başına çoklu fotoğraf **zorunlu**; skor en iyi çiftten |
| Eşik 0.70 = düşük duyarlılık, sıfır yanlış alarm | Eşiğin anlamı değişti: "eşleşme" değil, **"bildirim gönderilecek kadar eminiz"**. Daha düşük skorlu adaylar listede görünür ama bildirim tetiklemez. |
| Kırpma zarar veriyor | Uygulanmadı, gerekçesi kayıtlı |
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
