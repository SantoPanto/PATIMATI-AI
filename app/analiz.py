# app/analiz.py
"""Bir ilanın fotoğraflarını uçtan uca analiz eder.

Sözleşmedeki `analysis` bloğunu üreten yer burası. HTTP uç noktası ve
(gelecekte) kuyruk tüketicisi aynı fonksiyonu çağırır — iş mantığı tek yerde
durur, sadece giriş kapısı değişir.
"""
import logging

from .attributes import attribute_analyzer
from .embedder import kimlik_gomucu
from .hatalar import GecersizGoruntu
from .indirici import hepsini_indir
from .surum import MODEL_SURUMU

logger = logging.getLogger(__name__)


def baytlari_analiz_et(fotograflar: list[bytes],
                       basarisizlar: list[dict] | None = None) -> dict:
    """Fotoğraf baytlarından sözleşmedeki `analysis` bloğunu üretir.

    Bozuk bir fotoğraf diğerlerini düşürmez; atlanır ve raporlanır. Hepsi
    bozuksa hata fırlatılır — o zaman analiz edilecek bir şey kalmamıştır.
    """
    basarisizlar = list(basarisizlar or [])
    embeddings, oznitelikler = [], []

    for i, ham in enumerate(fotograflar):
        try:
            emb = kimlik_gomucu.embed_bytes(ham)
        except GecersizGoruntu as e:
            logger.warning("Fotoğraf %d atlandı: %s", i, e)
            basarisizlar.append({"index": i, "error": str(e)})
            continue
        embeddings.append(emb)
        try:
            # Kimlik vektörü YALNIZCA kimlik modeli CLIP'in kendisiyse etiket
            # analizine geçirilir: o zaman ikisi aynı uzaydadır ve CLIP'i aynı
            # fotoğraf için ikinci kez çalıştırmak anlamsız olur. Kimlik modeli
            # farklıysa (ör. SigLIP2) vektör başka bir uzayda olur ve zero-shot
            # metin karşılaştırması anlamsız sonuç verir — bu durumda
            # attribute_analyzer kendi CLIP vektörünü hesaplar (embedding=None).
            onceden_hesaplanan = emb if kimlik_gomucu.clip_mi else None
            oznitelikler.append(attribute_analyzer.analyze(ham, embedding=onceden_hesaplanan))
        except Exception as e:
            # Öznitelik hatası embedding'i çöpe atmamalı: eşleştirme etiketsiz
            # de çalışır, sadece skorun etiket bileşeni sıfırlanır.
            logger.error("Öznitelik çıkarma hatası (etiketsiz devam): %s", e)
            oznitelikler.append({"labels": [], "species": "unknown",
                                 "species_confidence": 0.0, "is_pet": True,
                                 "breed": None, "breed_confidence": 0.0,
                                 "pattern": None, "colors": []})

    if not embeddings:
        raise GecersizGoruntu("Hiçbir fotoğraf işlenemedi")

    birincil = _birincil_sec(oznitelikler)
    return {
        "embeddings": embeddings,
        "species": birincil["species"],
        "species_confidence": birincil["species_confidence"],
        # İlanda tek bir hayvan fotoğrafı bile varsa hayvan var sayılır;
        # kullanıcı 3 fotoğraf yüklerken birine yanlışlıkla manzara koyabilir.
        "is_pet": any(o["is_pet"] for o in oznitelikler),
        "breed": birincil["breed"],
        "breed_confidence": birincil["breed_confidence"],
        "pattern": birincil["pattern"],
        "colors": birincil["colors"],
        "labels": birincil["labels"],
        "model_version": MODEL_SURUMU,
        "photo_count": len(embeddings),
        "failed_photos": basarisizlar,
    }


def _birincil_sec(oznitelikler: list[dict]) -> dict:
    """Etiketleri hangi fotoğraftan alacağımızı seçer.

    İlanın birden çok fotoğrafı var ama sözleşmede tek bir tür/cins/etiket
    alanı dönüyor. En NET fotoğrafı seçiyoruz: hayvan görünen fotoğraflar
    arasından tür güveni en yüksek olanı. Uzaktan çekilmiş bulanık kareye
    bakıp "unknown" demek yerine, kullanıcının koyduğu net kareyi kullanır.

    SIRALAMA ÖLÇÜTÜ İKİ BASAMAKLI (20.08.2026'da düzeltildi): önce "türü
    atanabildi mi", sonra tür güveni. Tek başına `species_confidence`
    yanlış ölçüttü — o sayı "fotoğrafta hayvan var mı" sorusunu HİÇ ölçmez,
    yalnız "hayvansa kedi mi köpek mi" sorusunu ölçer ve iki seçenek
    üzerinden hesaplandığı için hayvansız bir karede bile %83'e çıkabiliyor.

    Ölçüldü: gerçek kedi karesi (güven 0.8177) + hayvansız kare (güven
    0.8293) aynı ilana konduğunda BİRİNCİL hayvansız kare seçiliyordu ⇒
    ilanın türü `cat` yerine `unknown` oluyor, etiketleri/deseni de o
    kareden geliyordu. `is_pet` filtresi bunu yakalayamıyor çünkü hayvan
    kapısı o karede zaten yanlış pozitif vermişti (26 hayvansız fotoğrafta
    7 kez).

    Türü atanmış bir fotoğraf, tanım gereği hayvan kanıtı eşiği geçmiş
    fotoğraftır (bkz. attributes.py SPECIES_GATE_MIN) — yani bu basamak
    yeni bir sinyal uydurmuyor, var olan kararı yeniden kullanıyor.
    """
    hayvanlilar = [o for o in oznitelikler if o["is_pet"]]
    aday = hayvanlilar or oznitelikler
    return max(aday, key=lambda o: (o["species"] != "unknown",
                                    o["species_confidence"]))


def urlleri_analiz_et(photo_urls: list[str]) -> dict:
    """Adresleri indirip analiz eder — kuyruk akışının yaptığı iş."""
    baytlar, basarisizlar = hepsini_indir(photo_urls)
    return baytlari_analiz_et(baytlar, basarisizlar)
