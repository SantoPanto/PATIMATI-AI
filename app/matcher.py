# app/matcher.py
import numpy as np


def cosine_similarity(a: list, b: list) -> float:
    """İki embedding vektörü arasında cosine similarity hesaplar."""
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.clip(dot / norm, -1.0, 1.0))


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
    """
    return 1.0 / (1.0 + distance_km / 5.0)


def compute_final_score(
    embedding_a: list,
    embedding_b: list,
    labels_a: list,
    labels_b: list,
    distance_km: float,
    species_a: str = "unknown",
    species_b: str = "unknown",
) -> dict:
    """
    Hibrit eşleşme skoru hesaplar.
    Farklı türler (kedi vs köpek) için skor otomatik sıfırlanır.
    """
    # Tür uyumsuzluğu — erken çıkış
    if (species_a != "unknown" and species_b != "unknown"
            and species_a != species_b):
        return {"score": 0.0, "visual": 0.0, "label": 0.0, "location": 0.0,
                "blocked_reason": "species_mismatch"}

    visual = cosine_similarity(embedding_a, embedding_b)
    label = jaccard_score(labels_a, labels_b)
    location = location_score(distance_km)

    score = (0.55 * visual) + (0.30 * label) + (0.15 * location)
    score = round(score, 4)

    return {
        "score": score,
        "visual": round(visual, 4),
        "label": round(label, 4),
        "location": round(location, 4),
        "match": score >= 0.70,  # Bildirim eşiği
    }
