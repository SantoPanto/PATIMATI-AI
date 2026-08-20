# app/matcher.py
import logging
import os

import numpy as np

from .hatalar import GecersizEmbedding
from .surum import MODEL_SURUMU, VEKTOR_BOYUTU

logger = logging.getLogger(__name__)

# Bildirim eşiği — ortam değişkeninden ayarlanabilir.
#
# ÖLÇÜT (asıl mesele bu; sayı bunun sonucu):
#   Sözleşme §7 gereği eşiği geçemeyen adaylar arayüzde LİSTELENMEYE DEVAM
#   EDER, yalnızca bildirim tetiklenmez. Yani eşiği yükseltmek eşleşme
#   KAYBETTİRMEZ — sadece telefonu daha az titretir. Buradan:
#     - kaçırmanın (bildirim gitmemesi) bedeli DÜŞÜK: eşleşme listede duruyor
#     - yalanın (yanlış bildirim) bedeli YÜKSEK: kullanıcı yanlış hayvanın
#       ilanına gidiyor ve bir daha bildirimlere güvenmiyor
#   ⇒ eşik, isabet (precision) tarafına yaslanmalı.
#
# Kısa geçmişi, çünkü buraya bakan bir sonraki kişi "hangi sayı doğru" diye soracak:
#   - Uzun süre 0.70'ti (o günkü gerekçe: scripts/measure_threshold.py).
#   - 2026-08-13'te 0.65'e indirildi. O kalibrasyon 5 fotoğrafın 10 YABANCI
#     çiftinden türetilmişti: kümede aynı hayvana ait tek bir çift bile yoktu,
#     yani "aynı hayvan bu eşiği hâlâ geçiyor mu" hiç ölçülmemişti. Ayrıca
#     üretimdekinden farklı bir kimlik modeliyle (google-siglip2) koşulmuştu.
#   - Üretim modeli (siglip2-animal) ve bu dosyadaki compute_final_score ile
#     İKİ TARAFLI ölçüldüğünde 0.65 belirgin biçimde daha kötü çıktı:
#     yanlış alarm %57.7 -> %80.5, buna karşılık yakalama %94.9 -> %96.9.
#     Yani 22.8 puan yanlış alarmın karşılığı 2 puan yakalama.
#   - Aynı ölçümün tam taraması (149 negatif, 98 pozitif sorgu):
#         eşik   yanlış alarm   yakalama
#         0.65      %80.5         %96.9
#         0.70      %57.7         %94.9
#         0.75      %32.9         %90.8
#         0.80      %13.4         %68.4   <- seçilen
#         0.85       %3.4         %40.8
#   ⇒ Yukarıdaki ölçüte göre 0.80. 0.70'te yabancıların YARISINDAN FAZLASI
#     eşiği geçiyordu; o değerle bildirim özelliği açılsa kullanıcıların
#     çoğuna yanlış hayvan bildirilirdi. 0.80'de bildirim gitmeyen gerçek
#     eşleşmeler kayıp değil: "Eşleşmelerim" listesinde görünmeye devam
#     ediyorlar (bkz. yukarıdaki ölçüt).
#
#   Ölçümün iki sınırı, ikisi de 0.80'i ZAYIFLATMIYOR:
#     - Galeri 50 kimlik; üretimin aday üst sınırı 100 ⇒ ölçtüğümüz boyut
#       gerçeğe yakın. Galeri büyüdükçe yanlış alarm ARTAR, yani daha yüksek
#       eşik gerekir — ters yön değil.
#     - Konum her karşılaştırmada sabit 5 km (veri kümesinde konum yok).
#       Gerçekte adaylar 25 km'ye kadar dağılıyor ve uzaklık skoru DÜŞÜRÜYOR
#       ⇒ gerçek yanlış alarm ölçtüğümüzden biraz daha iyi çıkar.
#   Açık iş: gerçek mesafeler ve daha geniş galeriyle bir kez daha ölçmek.
#   Ölçüm tabloları docs/olcum-raporu.md §4.1'de.
#
# Eşiğin ne demek olduğu için sözleşme §7: `match: true` "kesin aynı hayvan"
# değil, "bildirim gönderecek kadar eminiz" demektir. Eşiği geçmeyen adaylar
# arayüzde listelenmeye devam eder — bu yüzden eşiği yükseltmek eşleşmeleri
# KAYBETTİRMEZ, yalnızca bildirim gönderilenleri azaltır.
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.80"))

# --- Skor kanallarının ağırlıkları ve konum cezasının sertliği ---------------
#
# NEDEN ORTAMDAN: 19.08 ölçümü, aynı hayvanın birebir aynı fotoğrafının
# 3 km uzaktayken eşiği geçemediğini gösterdi (görsel 0.9417, etiket 0.4800,
# skor 0.7558 < 0.80); cebirsel azami mesafe 432 m çıktı — oysa aday arama
# yarıçapı 25 km. Ama hangi sayının doğru olduğu ÖLÇÜLMEDEN bilinemez:
# bir sayıyı gevşetmek yanlış-pozitif üretir ve o etki ancak çok bireyli
# veriyle ölçülür (scripts/yanlis_pozitif_egrisi.py bunu üretiyor).
#
# Bu yüzden burada hiçbir sayı DEĞİŞTİRİLMİYOR; yalnız ölçüm sonucuna göre
# dağıtım yapmadan denenebilir hâle getiriliyor. Varsayılanlar eski
# davranışın birebir aynısı.
GORSEL_AGIRLIK = float(os.getenv("GORSEL_AGIRLIK", "0.55"))
ETIKET_AGIRLIK = float(os.getenv("ETIKET_AGIRLIK", "0.30"))
KONUM_AGIRLIK = float(os.getenv("KONUM_AGIRLIK", "0.15"))

# konum = 1 / (1 + km / KONUM_YARI_MESAFE_KM) — bu mesafede konum skoru 0.50.
KONUM_YARI_MESAFE_KM = float(os.getenv("KONUM_YARI_MESAFE_KM", "5.0"))

# Ağırlıklar toplamı 1 değilse skor artık [0,1] aralığında DEĞİLDİR ve eşik
# başka bir şey ölçmeye başlar. Sessizce yanlış çalışmaktansa açılmıyoruz:
# yanlış ayar dağıtımda değil, ilk saniyede görünsün.
_AGIRLIK_TOPLAMI = GORSEL_AGIRLIK + ETIKET_AGIRLIK + KONUM_AGIRLIK
if abs(_AGIRLIK_TOPLAMI - 1.0) > 1e-9:
    raise ValueError(
        "Skor ağırlıklarının toplamı 1.0 olmalı; şu an %.4f "
        "(GORSEL_AGIRLIK=%.4f, ETIKET_AGIRLIK=%.4f, KONUM_AGIRLIK=%.4f). "
        "Toplam 1 değilse eşik (%.2f) artık aynı şeyi ölçmez."
        % (_AGIRLIK_TOPLAMI, GORSEL_AGIRLIK, ETIKET_AGIRLIK, KONUM_AGIRLIK,
           MATCH_THRESHOLD))

if KONUM_YARI_MESAFE_KM <= 0:
    raise ValueError(
        "KONUM_YARI_MESAFE_KM pozitif olmalı; şu an %.4f. "
        "Sıfır ya da negatif değer konum skorunu tanımsız yapar."
        % KONUM_YARI_MESAFE_KM)

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


def location_score(distance_km: float | None) -> float:
    """
    Mesafe bazlı skor: yakınsa yüksek, uzaksa düşük.
    Varsayılan yarı-mesafeyle (5 km):
    0 km → 1.00, 5 km → 0.50, 20 km → 0.20, 50 km → 0.09
    Eğri KONUM_YARI_MESAFE_KM ile ayarlanır; yukarıdaki sayılar
    varsayılana aittir, sabit değildir.

    Negatif mesafe fiziksel olarak anlamsızdır ama gelirse iki ayrı hataya yol
    açıyordu: -5 km ZeroDivisionError ile servisi çökertiyor, -1 km ise 1.25
    döndürüp skoru üst sınırın üstüne çıkarıyordu. Sıfıra kırpıyoruz.

    None = mesafe BİLİNMİYOR ve GEÇERSİZ mesafeyle (NaN/inf) aynı davranır.
    Sebebi: konum skoru toplamın %15'i ve 0 km maksimum puan demek. Alanı hiç
    göndermeyen bir çağıran, "bilmiyorum"un karşılığı olarak sessizce EN İYİ
    puanı alıyordu — eşiğin 0.80 olduğu bir sistemde 0.15'lik sessiz bonus,
    0.65 ile 0.80 arasındaki tüm farktan büyük.

    ⚠ Açıkça verilen 0.0 hâlâ meşru "aynı noktada" demektir ve 1.0 döndürür;
    değişen yalnızca bilginin HİÇ olmadığı durum.
    """
    if distance_km is None or not np.isfinite(distance_km):
        return 0.0
    return 1.0 / (1.0 + max(0.0, float(distance_km))
                  / KONUM_YARI_MESAFE_KM)


def compute_final_score(
    embeddings_a,
    embeddings_b,
    labels_a: list,
    labels_b: list,
    distance_km: float | None,
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
    #
    # Tür değerleri BÜYÜK/KÜÇÜK HARF duyarsız karşılaştırılır. Sözleşme küçük
    # harf diyor ("cat"), ama Java tarafındaki enum büyük harf üretiyor ("CAT")
    # ve @Enumerated(EnumType.STRING) bunu olduğu gibi JSON'a koyuyor. Düz metin
    # karşılaştırmasında "CAT" != "cat" olduğu için HER aday sıfırlanırdı —
    # hata vermeden, sadece hiç eşleşme bulunmayarak. Gönderene sıkı, alana
    # hoşgörülü olmak bu sınıf hatayı tümden kapatıyor.
    species_a = (species_a or "unknown").strip().lower()
    species_b = (species_b or "unknown").strip().lower()
    if (species_a != "unknown" and species_b != "unknown"
            and species_a != species_b):
        return {"score": 0.0, "visual": 0.0, "label": 0.0, "location": 0.0,
                "match": False, "photo_a": None, "photo_b": None,
                "blocked_reason": "species_mismatch"}

    a_listesi, b_listesi = _vektor_listesi(embeddings_a), _vektor_listesi(embeddings_b)
    if not a_listesi or not b_listesi:
        raise GecersizEmbedding("her iki tarafta da en az bir fotoğraf olmalı")

    visual, foto_a, foto_b = en_iyi_gorsel(a_listesi, b_listesi)
    
    # Geriye dönük uyumluluk (Backward Compatibility): Eski öneksiz etiketleri (örn. "cat", "tabby") yeni formata çevir.
    def _migrate(lbls):
        migrated = []
        for l in lbls:
            if ":" in l:
                migrated.append(l)
            elif l in {"cat", "dog"}:
                migrated.append(f"hard:species_{l}")
            elif l in {"tabby", "spotted", "solid", "bicolor"}:
                migrated.append(f"soft:pattern_{l}")
            elif l in {"black", "white", "gray", "brown", "orange", "cream", "golden"}:
                migrated.append(f"soft:color_{l}")
            else:
                migrated.append(l) # Bilinmeyen eski etiketleri aynen bırak
        return migrated

    labels_a = _migrate(labels_a)
    labels_b = _migrate(labels_b)
    
    # Özellik ayrımı
    hard_a = [l for l in labels_a if l.startswith("hard:")]
    hard_b = [l for l in labels_b if l.startswith("hard:")]
    soft_a = [l for l in labels_a if l.startswith("soft:")]
    soft_b = [l for l in labels_b if l.startswith("soft:")]
    bonus_a = [l for l in labels_a if l.startswith("bonus:")]
    bonus_b = [l for l in labels_b if l.startswith("bonus:")]
    
    hard_score = jaccard_score(hard_a, hard_b)
    soft_score = jaccard_score(soft_a, soft_b)
    
    if not soft_a or not soft_b:
        label = hard_score
    else:
        # Hard özellikler (tür, kürk, kulak) daha önemlidir, soft özellikler (renk, desen) daha az etkilidir.
        label = (0.7 * hard_score) + (0.3 * soft_score)
        
    location = location_score(distance_km)

    # Kosinüs teorik olarak negatif olabildiği için toplam da [0,1] dışına
    # çıkabilir; skor her zaman yorumlanabilir bir aralıkta kalsın.
    score = ((GORSEL_AGIRLIK * visual)
             + (ETIKET_AGIRLIK * label)
             + (KONUM_AGIRLIK * location))
    
    # Bonus özellikler uyuşmazsa ceza vermez, eşleşirse +0.02 bonus verir
    for ba in bonus_a:
        if ba in bonus_b:
            score += 0.02
            
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

    SORGUNUN kendi embedding'i (aday değil, `embeddings` parametresi) burada,
    döngüden ÖNCE doğrulanır. Doğrulanmazsa `compute_final_score` her aday
    için aynı hatayla patlar ve döngüdeki try/except bunu "bu aday bozuk" diye
    yorumlayıp her adayı `gecersiz_embedding` altında sessizce eler — çağıran
    tarafın KENDİ isteği bozuk olsa bile sonuç "eşleşme yok" gibi görünür.
    Burada erken ve açıkça fırlatmak, hatanın doğru yere (çağıran) gitmesini
    sağlar (bkz. `app/main.py`'deki `/match` ucu, bunu 400'e çeviriyor).

    Dönüş: (eşleşmeler, atlananlar)
    """
    for vektor in _vektor_listesi(embeddings):
        dogrula_embedding(vektor, "embeddings")

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
