# app/matcher.py
import logging
import os

import numpy as np

from .hatalar import GecersizEmbedding
from .surum import MODEL_SURUMU, VEKTOR_BOYUTU

logger = logging.getLogger(__name__)

# Bildirim eşiği — ortam değişkeninden ayarlanabilir (varsayılan 0.70,
# 111 fotoğrafla ölçülerek doğrulandı; bkz. scripts/measure_threshold.py)
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.70"))

# Sözleşmede aday üst sınırı 100; burada da zorluyoruz ki gelen liste büyükse
# sessizce boğulmak yerine kırpıp raporlayalım (bkz. sözleşme §5).
AZAMI_ADAY = int(os.getenv("MAX_CANDIDATES", "100"))
AZAMI_SONUC = 20


def dogrula_embedding(vektor, ad: str = "embedding") -> np.ndarray:
    """Vektörü doğrular ve numpy dizisine çevirir; kullanılamazsa hata fırlatır.

    Neden gerekli: doğrulama olmadan NaN içeren bir vektör skoru NaN yapıyordu.
    NaN sessiz değil ama YANLIŞ YERDE patlıyor — JSON standardı NaN'ı desteklemediği
    için mesaj Java tarafında ayrıştırılamıyor ve sonuç tamamen kayboluyordu.
    Yanlış uzunluktaki vektör de yakalanmayan bir ValueError'a yol açıyordu.
    """
    if vektor is None or not hasattr(vektor, "__len__"):
        raise GecersizEmbedding(f"{ad}: liste bekleniyordu, {type(vektor).__name__} geldi")
    if len(vektor) != VEKTOR_BOYUTU:
        raise GecersizEmbedding(
            f"{ad}: {VEKTOR_BOYUTU} boyut bekleniyordu, {len(vektor)} geldi")
    dizi = np.asarray(vektor, dtype=np.float32)
    if not np.all(np.isfinite(dizi)):
        raise GecersizEmbedding(f"{ad}: NaN veya sonsuz değer içeriyor")
    return dizi


def cosine_similarity(a: list, b: list) -> float:
    """İki embedding vektörü arasında cosine similarity hesaplar (asla NaN dönmez)."""
    va = dogrula_embedding(a, "embedding_a")
    vb = dogrula_embedding(b, "embedding_b")
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm == 0.0:
        return 0.0
    return float(np.clip(float(np.dot(va, vb)) / norm, -1.0, 1.0))


def _vektor_listesi(x) -> list:
    """Tek vektörü de vektör listesini de kabul eder; her zaman liste döner.

    İlan başına birden çok fotoğraf olabildiği için eşleştirme liste üzerinden
    çalışır. /compare gibi tek fotoğraflı yollar tek vektör göndermeye devam
    edebilsin diye bu esneklik var; tel üzerindeki biçimi Pydantic zorluyor.
    """
    if not x:
        return []
    return [x] if isinstance(x[0], (int, float)) else list(x)


def en_iyi_gorsel(a_listesi: list, b_listesi: list) -> tuple:
    """İki ilanın TÜM fotoğraf çiftleri arasındaki en yüksek benzerliği bulur.

    Neden en iyi çift: gerçek veriyle ölçüldü — aynı kedinin iki ayrı
    fotoğrafında eşleşme oranı %24'te kalıyordu (dağılımlar çakışıyor).
    İlan başına 3 fotoğraf koyup en iyi çifti almak skoru 0.647'den 0.748'e
    çıkardı ve yanlış alarm üretmedi. Tek fotoğrafla bu iş yürümüyor.

    Dönüş: (en_yuksek_benzerlik, a_foto_indeksi, b_foto_indeksi)
    """
    en_iyi, ia, ib = -1.0, -1, -1
    for i, va in enumerate(a_listesi):
        for j, vb in enumerate(b_listesi):
            s = cosine_similarity(va, vb)
            if s > en_iyi:
                en_iyi, ia, ib = s, i, j
    return en_iyi, ia, ib


def jaccard_score(labels_a: list, labels_b: list) -> float:
    """İki etiket kümesinin Jaccard benzerliği."""
    if not labels_a or not labels_b:
        return 0.0
    sa, sb = set(labels_a), set(labels_b)
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0


def location_score(distance_km: float) -> float:
    """
    Mesafe bazlı skor: yakınsa yüksek, uzaksa düşük.
    0 km → 1.00, 5 km → 0.50, 20 km → 0.20, 50 km → 0.09

    Negatif mesafe fiziksel olarak anlamsızdır ama gelirse iki ayrı hataya yol
    açıyordu: -5 km ZeroDivisionError ile servisi çökertiyor, -1 km ise 1.25
    döndürüp skoru üst sınırın üstüne çıkarıyordu. Sıfıra kırpıyoruz.
    """
    if not np.isfinite(distance_km):
        return 0.0
    return 1.0 / (1.0 + max(0.0, float(distance_km)) / 5.0)


def compute_final_score(
    embeddings_a,
    embeddings_b,
    labels_a: list,
    labels_b: list,
    distance_km: float,
    species_a: str = "unknown",
    species_b: str = "unknown",
) -> dict:
    """
    Hibrit eşleşme skoru hesaplar. Her iki taraf da birden çok fotoğraf
    taşıyabilir; görsel benzerlik en iyi fotoğraf çiftinden alınır.
    Farklı türler (kedi vs köpek) için skor otomatik sıfırlanır.
    """
    # Tür uyumsuzluğu — erken çıkış.
    # Cevap şekli normal yolla BİREBİR aynı olmalı: eksik anahtar Spring
    # tarafındaki DTO'yu kırıyordu (demo sırasında bulunan gerçek bir hataydı).
    if (species_a != "unknown" and species_b != "unknown"
            and species_a != species_b):
        return {"score": 0.0, "visual": 0.0, "label": 0.0, "location": 0.0,
                "match": False, "photo_a": None, "photo_b": None,
                "blocked_reason": "species_mismatch"}

    a_listesi, b_listesi = _vektor_listesi(embeddings_a), _vektor_listesi(embeddings_b)
    if not a_listesi or not b_listesi:
        raise GecersizEmbedding("her iki tarafta da en az bir fotoğraf olmalı")

    visual, foto_a, foto_b = en_iyi_gorsel(a_listesi, b_listesi)
    label = jaccard_score(labels_a, labels_b)
    location = location_score(distance_km)

    # Kosinüs teorik olarak negatif olabildiği için toplam da [0,1] dışına
    # çıkabilir; skor her zaman yorumlanabilir bir aralıkta kalsın.
    score = (0.55 * visual) + (0.30 * label) + (0.15 * location)
    score = round(min(1.0, max(0.0, score)), 4)

    return {
        "score": score,
        "visual": round(visual, 4),
        "label": round(label, 4),
        "location": round(location, 4),
        "match": score >= MATCH_THRESHOLD,
        # Hangi fotoğraf çifti eşleşti — arayüzde "bu ikisi benziyor" diye
        # gösterilebilir, hata ayıklarken de hangi karenin tuttuğunu söyler.
        "photo_a": foto_a,
        "photo_b": foto_b,
    }


def adaylari_eslestir(embeddings, labels, species, candidates,
                      ad_id=None, model_version=MODEL_SURUMU):
    """Adayları skorlar, sıralar; eleyip atladıklarını sayarak raporlar.

    Tek bir bozuk aday tüm isteği düşürmemeli — o yüzden hatalı aday atlanır,
    ama SESSİZCE atlanmaz: kaç tanesinin neden elendiği dönüş değerinde yer alır.
    Sessiz eleme, "neden hiç eşleşme çıkmadı?" sorusunu cevapsız bırakır.

    model_version: None verilirse sürüm denetimi yapılmaz. Varsayılan davranış
    KATIDIR — sürümü tutmayan aday atlanır, çünkü farklı sürümle üretilmiş
    vektörler kıyaslanamaz ve hata vermeden yanlış benzerlik üretir.

    Dönüş: (eşleşmeler, atlananlar)
    """
    atlanan = {"toplam": 0, "kendisi": 0, "tekrar_eden": 0,
               "model_surumu_uyusmuyor": 0, "gecersiz_embedding": 0,
               "aday_siniri_asildi": 0}

    if len(candidates) > AZAMI_ADAY:
        atlanan["aday_siniri_asildi"] = len(candidates) - AZAMI_ADAY
        logger.warning("Aday sayısı %d, sınır %d — fazlası kırpıldı.",
                       len(candidates), AZAMI_ADAY)
        candidates = candidates[:AZAMI_ADAY]

    gorulen: set = set()
    sonuclar = []
    for aday in candidates:
        if ad_id is not None and aday.ad_id == ad_id:
            atlanan["kendisi"] += 1
            continue
        if aday.ad_id in gorulen:
            atlanan["tekrar_eden"] += 1
            continue
        gorulen.add(aday.ad_id)

        if model_version is not None and aday.model_version != model_version:
            atlanan["model_surumu_uyusmuyor"] += 1
            continue

        try:
            sonuc = compute_final_score(
                embeddings_a=embeddings, embeddings_b=aday.embeddings,
                labels_a=labels, labels_b=aday.labels,
                distance_km=aday.distance_km,
                species_a=species, species_b=aday.species,
            )
        except GecersizEmbedding as e:
            logger.warning("Aday %s atlandı: %s", aday.ad_id, e)
            atlanan["gecersiz_embedding"] += 1
            continue

        sonuclar.append({"ad_id": aday.ad_id, **sonuc})

    atlanan["toplam"] = sum(v for k, v in atlanan.items() if k != "toplam")
    sonuclar.sort(key=lambda x: x["score"], reverse=True)
    return sonuclar[:AZAMI_SONUC], atlanan
