# app/main.py
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .attributes import attribute_analyzer
from .embedder import embedder
from .matcher import compute_final_score
from .models import AnalyzeResponse, MatchRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI servisi başlatılıyor...")
    yield
    logger.info("AI servisi kapatılıyor.")


app = FastAPI(title="PatiMati AI Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "*")],
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "clip-vit-base-patch32"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Fotoğrafı analiz et: embedding çıkar + label al.
    Hem kayıp hem buldum ilanı oluşturulurken çağrılır.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Yalnızca görüntü dosyaları kabul edilir.")

    img_bytes = await file.read()
    if len(img_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(413, "Dosya boyutu 10 MB'ı aşıyor.")

    try:
        embedding = embedder.embed_bytes(img_bytes)
    except Exception as e:
        logger.error(f"Embedding hatası: {e}")
        raise HTTPException(500, "Analiz sırasında hata oluştu.")

    # Öznitelik hatası analizi engellemez: embedding döner, etiketler boş kalır
    try:
        vision = attribute_analyzer.analyze(img_bytes, embedding)
    except Exception as e:
        logger.error(f"Öznitelik çıkarma hatası (etiketsiz devam ediliyor): {e}")
        vision = {"labels": [], "species": "unknown", "colors": []}

    return AnalyzeResponse(
        embedding=embedding,
        labels=vision["labels"],
        species=vision["species"],
        colors=vision["colors"],
    )


@app.post("/compare")
async def compare(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    distance_km: float = Form(2.0),
):
    """
    İki fotoğrafı doğrudan karşılaştır — demo ve elle test için.
    Swagger'dan iki fotoğraf yükle, skor dökümünü gör.
    (Üretim akışı /analyze + /match kullanır; bu endpoint kolay deneme içindir.)
    """
    analyses = []
    for f in (file1, file2):
        if not f.content_type.startswith("image/"):
            raise HTTPException(400, "Yalnızca görüntü dosyaları kabul edilir.")
        img_bytes = await f.read()
        if len(img_bytes) > 10 * 1024 * 1024:
            raise HTTPException(413, "Dosya boyutu 10 MB'ı aşıyor.")
        embedding = embedder.embed_bytes(img_bytes)
        try:
            attrs = attribute_analyzer.analyze(img_bytes, embedding)
        except Exception as e:
            logger.error(f"Öznitelik çıkarma hatası (etiketsiz devam ediliyor): {e}")
            attrs = {"labels": [], "species": "unknown", "colors": []}
        analyses.append((embedding, attrs))

    (emb1, a1), (emb2, a2) = analyses
    result = compute_final_score(emb1, emb2, a1["labels"], a2["labels"],
                                 distance_km, a1["species"], a2["species"])
    return {
        "foto1": {"species": a1["species"], "labels": a1["labels"]},
        "foto2": {"species": a2["species"], "labels": a2["labels"]},
        "sonuc": result,
        "aciklama": ("ESLESME: bildirim giderdi (skor >= esik)" if result["match"]
                     else "eslesme yok (0.50+ ise aday listesinde yine gorunurdu)"),
    }


@app.post("/match")
async def match(req: MatchRequest):
    """
    Yeni ilan ile mevcut ilanları karşılaştır, skorla sırala.
    Spring Boot bu endpoint'i yeni ilan oluşturulduğunda çağırır.
    """
    results = []
    for candidate in req.candidates:
        result = compute_final_score(
            embedding_a=req.embedding,
            embedding_b=candidate.embedding,
            labels_a=req.labels,
            labels_b=candidate.labels,
            distance_km=candidate.distance_km,
            species_a=req.species,
            species_b=candidate.species,
        )
        results.append({"pet_id": candidate.pet_id, **result})

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"matches": results[:20]}  # En iyi 20 eşleşme


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0",
                port=int(os.getenv("AI_SERVICE_PORT", 8000)), reload=True)
