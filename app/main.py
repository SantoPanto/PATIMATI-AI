# app/main.py
import logging
import os
import secrets
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

load_dotenv()

from .analiz import urlleri_analiz_et
from .attributes import attribute_analyzer
from .embedder import PetEmbedder, embedder, kimlik_gomucu
from .hatalar import AIHatasi, FotografIndirilemedi, GecersizGoruntu
from .matcher import MATCH_THRESHOLD, adaylari_eslestir, compute_final_score
from .models import AnalyzeResponse, AnalyzeUrlRequest, MatchRequest
from .surum import MODEL_SURUMU, SECILEN_KIMLIK, VEKTOR_BOYUTU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ANAHTAR_BASLIGI = "X-Api-Key"


def api_anahtari() -> str:
    """Yapılandırılmış paylaşılan sır; boşsa korumalı uçlar HİÇBİR isteği kabul etmez.

    Her çağrıda okunur, modül yüklenirken bir kez değil: değeri sabitlemek
    testlerin ortamı değiştirmesini imkânsız kılar ve "anahtar verildi mi"
    sorusunun cevabı yalnızca yeniden başlatmayla değişebilirdi.
    """
    return os.getenv("AI_API_KEY", "").strip()


async def anahtari_dogrula(
    x_api_key: str | None = Header(default=None, alias=ANAHTAR_BASLIGI),
) -> None:
    """Uçları paylaşılan bir sırla korur (A1). VARSAYILAN DAVRANIŞ: REDDET.

    ANAHTAR YAPILANDIRILMAMIŞSA hiçbir istek geçmez — 401. Eskiden tersiydi:
    anahtar yoksa kapı tamamen açılıyordu. Gerekçesi dağıtım sırasıydı, bu
    servis tarayıcıdan da çağrılıyordu (`AddListingPage`) ve anahtarı bir anda
    zorunlu kılmak ilan oluşturma ekranını kırardı. **O engel kalktı:** ön yüz
    artık AI'ı doğrudan çağırmıyor (`/api/ai/analyze` üzerinden backend'e
    gidiyor) ve backend `X-Api-Key` başlığını gönderiyor
    (`AiMatchService`, `ai.service.api-key`).

    Bir yapılandırma eksiği yüzünden kapının AÇILMASI yanlış varsayılandır:
    unutulan bir ortam değişkeni sessizce "herkese açık AI servisi" üretir ve
    canlıda bu, kaynak sömürüsü demektir. Unutulan değişkenin cezası
    "çalışmıyor" olmalı, "korumasız çalışıyor" değil.

    ⚠ Anahtarsız kurulum sessiz DEĞİLDİR: açılışta hata seviyesinde kayıt
    düşer ve `/health` `api_anahtari_yapilandirildi: false` der. `/health`
    bilerek anahtarsız kalır, yoksa canlılık yoklaması servisi ölü gösterirdi.

    Karşılaştırma sabit zamanlıdır: sıradan `==` ilk farklı bayta kadar geçen
    süreyi sızdırır ve anahtar bayt bayt tahmin edilebilir hâle gelir.
    """
    beklenen = api_anahtari()
    if not beklenen:
        # Anahtar yokken "geçerli anahtar" diye bir şey yoktur; karşılaştırmaya
        # girmeden reddedilir. Boş sırla compare_digest yapmak, boş başlık
        # gönderen herkesi içeri alırdı.
        raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED",
                                                     "message": "Gecersiz veya eksik API anahtari"})

    gelen = x_api_key or ""
    if not secrets.compare_digest(gelen.encode("utf-8"), beklenen.encode("utf-8")):
        # Eksik ile yanlış anahtar AYNI cevabı alır: hangisinin olduğunu
        # söylemek, saldırgana anahtarın var olup olmadığını öğretirdi.
        raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED",
                                                     "message": "Gecersiz veya eksik API anahtari"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI servisi başlatılıyor...")
    if not api_anahtari():
        # Uyarı değil HATA: bu hâlde servis ayakta ama İŞE YARAMAZ durumdadır,
        # /analyze de /match de 401 döner. Kayıt seviyesi bunu söylemeli ki
        # "kalktı demek ki çalışıyor" sanılmasın.
        logger.error(
            "AI_API_KEY tanımlı DEĞİL: /analyze, /analyze_url, /compare ve /match "
            "uçlarının HEPSİ 401 dönecek. Aynı değer backend'in "
            "ai.service.api-key ayarına da verilmeli, yoksa eşleştirme çalışmaz.")
    yield
    logger.info("AI servisi kapatılıyor.")


app = FastAPI(title="PatiMati AI Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Virgülle ayrılmış birden fazla origin desteklenir (ör. prod + staging).
    # Tek elemanlı liste yazılsaydı CORSMiddleware tüm string'i TEK bir origin
    # sanır, virgülden sonraki hiçbir origin asla eşleşmezdi.
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
async def health():
    # İki model birden çalışıyor; hangisinin ne yaptığı buradan görünsün ki
    # yanlış yapılandırmayla ayağa kalkan bir servis fark edilebilsin.
    # Eşik de aynı sebeple burada: ortam değişkeniyle eziliyor, ve bir kez
    # kod ile belgeler farklı değer söyler hâle geldi. Servisin GERÇEKTE
    # hangi eşikle karar verdiği dışarıdan görünsün.
    # Anahtarın YAPILANDIRILIP yapılandırılmadığı da aynı sebeple burada:
    # anahtarı vermeyi unutmuş bir dağıtım, dışarıdan bakınca çalışan bir
    # servisten ayırt edilemez — /health 200 döner ama korumalı uçların hepsi
    # 401'dir. Anahtarın KENDİSİ değil, yalnız verilip verilmediği yazılır.
    # (Alan eskiden `api_anahtari_zorunlu` idi; kilit artık her hâlükârda
    # zorunlu olduğu için o soru tek cevaplı hâle geldi ve bilgi taşımıyordu.
    # Dağıtımın cevaplanması gereken sorusu artık "anahtar verildi mi".)
    # /health bilerek anahtarsız kalır: canlılık yoklaması kimlik isteseydi,
    # servis "ölü" görünürdü.
    return {"status": "ok",
            "etiket_modeli": PetEmbedder.MODEL_ID,
            "kimlik_modeli": SECILEN_KIMLIK,
            "vektor_boyutu": VEKTOR_BOYUTU,
            "model_version": MODEL_SURUMU,
            "match_threshold": MATCH_THRESHOLD,
            "api_anahtari_yapilandirildi": bool(api_anahtari())}


def _oznitelik_cikar(img_bytes: bytes, embedding: list[float] | None) -> dict:
    """Öznitelikleri çıkarır; hata olursa boş etiketlerle devam eder.

    Embedding zaten hesaplandığı için eşleştirme etiketsiz de çalışır —
    öznitelik hatası tüm analizi düşürmemeli.

    `embedding` YALNIZCA kimlik modeli CLIP'in kendisiyse geçirilir (bkz.
    `_goruntuyu_isle`): o zaman ikisi aynı uzaydadır ve CLIP'i aynı fotoğraf
    için ikinci kez çalıştırmak anlamsızdır. Aksi halde None geçirilir ve
    attribute_analyzer kendi CLIP vektörünü hesaplar — kimlik modeli farklıysa
    (ör. SigLIP2) vektör başka bir uzayda olur ve zero-shot metin
    karşılaştırması sessizce yanlış etiket üretir.
    """
    try:
        return attribute_analyzer.analyze(img_bytes, embedding=embedding)
    except Exception as e:
        logger.error(f"Öznitelik çıkarma hatası (etiketsiz devam ediliyor): {e}")
        return {"labels": [], "species": "unknown", "species_confidence": 0.0,
                "is_pet": True, "breed": None, "breed_confidence": 0.0,
                "pattern": None, "colors": [],
                "is_designed_graphic": False, "graphic_confidence": 0.0}


async def _goruntuyu_isle(img_bytes: bytes) -> tuple[list[float], dict]:
    """Embedding + öznitelik çıkarımını iş parçacığı havuzunda yapar.

    CLIP çıkarımı CPU'yu saniyelerce meşgul eden senkron bir iştir. Doğrudan
    `async def` içinde çağrılırsa olay döngüsünü (event loop) bloklar: o sırada
    servis BAŞKA HİÇBİR isteğe cevap veremez, /health bile yanıtsız kalır.
    Havuza taşıyınca sunucu cevap verebilir durumda kalıyor.
    """
    try:
        embedding = await run_in_threadpool(kimlik_gomucu.embed_bytes, img_bytes)
    except GecersizGoruntu as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Embedding hatası: {e}")
        raise HTTPException(500, "Analiz sırasında hata oluştu.")

    onceden_hesaplanan = embedding if kimlik_gomucu.clip_mi else None
    vision = await run_in_threadpool(_oznitelik_cikar, img_bytes, onceden_hesaplanan)
    return embedding, vision


@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(anahtari_dogrula)])
async def analyze(file: UploadFile = File(...)):
    """
    Fotoğrafı analiz et: embedding çıkar + label al.
    Hem kayıp hem buldum ilanı oluşturulurken çağrılır.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Yalnızca görüntü dosyaları kabul edilir.")

    img_bytes = await file.read()
    if len(img_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(413, "Dosya boyutu 10 MB'ı aşıyor.")

    embedding, vision = await _goruntuyu_isle(img_bytes)

    return AnalyzeResponse(
        embedding=embedding,
        labels=vision["labels"],
        species=vision["species"],
        species_confidence=vision["species_confidence"],
        is_pet=vision["is_pet"],
        breed=vision["breed"],
        breed_confidence=vision["breed_confidence"],
        pattern=vision["pattern"],
        colors=vision["colors"],
        model_version=MODEL_SURUMU,
        is_designed_graphic=vision["is_designed_graphic"],
        graphic_confidence=vision["graphic_confidence"],
    )


@app.post("/analyze_url", dependencies=[Depends(anahtari_dogrula)])
async def analyze_url(req: AnalyzeUrlRequest):
    """
    Fotoğrafları ADRESLERİNDEN indirip analiz et — üretim akışının yaptığı iş.

    Kuyruk mesajı dosya değil URL taşıyor (büyük dosyaları kuyruktan geçirmemek
    için). Bu endpoint, kuyruk tüketicisiyle aynı kodu çağırır; böylece akış
    RabbitMQ kurulmadan da Swagger'dan denenebilir.

    Bir fotoğraf indirilemezse diğerleriyle devam eder ve `failed_photos`
    altında bildirir. Hiçbiri indirilemezse hata döner.
    """
    try:
        sonuc = await run_in_threadpool(urlleri_analiz_et, req.photo_urls)
        # `photo_bytes` yalnızca kuyruk.py'nin (metin analizine görsel geçirmek
        # için) kullandığı bir ara değer, sözleşmede yok ve JSON'a çevrilemez
        # (ham bayt listesi) -- burada bırakılırsa bu uç HER fotoğraflı
        # çağrıda 500 verir.
        sonuc.pop("photo_bytes", None)
        return sonuc
    except FotografIndirilemedi as e:
        # Kaynak sunucu kaynaklı: 502 (bizim değil, dış servisin sorunu)
        raise HTTPException(502, {"code": e.KOD, "message": str(e)})
    except AIHatasi as e:
        raise HTTPException(400, {"code": e.KOD, "message": str(e)})
    except Exception as e:
        logger.error(f"Analiz hatası: {e}")
        raise HTTPException(500, {"code": "INTERNAL", "message": "Analiz sırasında hata"})


@app.post("/compare", dependencies=[Depends(anahtari_dogrula)])
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
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(400, "Yalnızca görüntü dosyaları kabul edilir.")
        img_bytes = await f.read()
        if len(img_bytes) > 10 * 1024 * 1024:
            raise HTTPException(413, "Dosya boyutu 10 MB'ı aşıyor.")
        analyses.append(await _goruntuyu_isle(img_bytes))

    (emb1, a1), (emb2, a2) = analyses
    result = compute_final_score(emb1, emb2, a1["labels"], a2["labels"],
                                 distance_km, a1["species"], a2["species"])

    def _ozet(a):
        return {"species": a["species"], "is_pet": a["is_pet"], "breed": a["breed"],
                "breed_confidence": a["breed_confidence"], "labels": a["labels"]}

    return {
        "foto1": _ozet(a1),
        "foto2": _ozet(a2),
        "sonuc": result,
        "aciklama": ("ESLESME: bildirim giderdi (skor >= esik)" if result["match"]
                     else "eslesme yok (0.50+ ise aday listesinde yine gorunurdu)"),
    }


@app.post("/match", dependencies=[Depends(anahtari_dogrula)])
async def match(req: MatchRequest):
    """
    Yeni ilan/external kayıt ile mevcut adayları karşılaştır, skorla sırala.

    Elenen adaylar `skipped_candidates` altında gerekçesiyle raporlanır —
    "hiç eşleşme çıkmadı" durumunun sebebi görünür olsun diye.
    DİKKAT: adayların `model_version` alanı bu servisin sürümüyle aynı değilse
    aday ATLANIR (farklı sürümlerin vektörleri kıyaslanamaz).

    Faz 2 (Instagram entegrasyonu) — bu uç, external_pet_records için "Aşama 2"
    eşleştirme çağrısının hedefidir: Java, Aşama 1'de (ai.analysis.request
    kuyruğu üzerinden, adaysız) zaten üretilmiş embedding/etiket/tür bilgisini
    buraya senkron olarak gönderir. `req.external_record_id` sorgunun kendi
    kimliğidir (ad_id ile aynı anda dolu olamaz — bkz. models.py); adaylar da
    kendi `ad_id`/`external_record_id` çiftini taşır. `req.match_threshold`
    verilmezse bu servisin ortam değişkeni varsayılanı (MATCH_THRESHOLD)
    kullanılır — native davranış hiç değişmez.
    """
    try:
        # run_in_threadpool: adaylari_eslestir CPU'ya bağlı (yüzlerce adayla
        # cosine similarity) senkron bir iştir -- /analyze ve /compare aynı
        # sebeple havuza atıyor (bkz. _goruntuyu_isle), burası da tutarlı
        # olmalı; aksi hâlde yoğun bir /match isteği olay döngüsünü bloklar,
        # /health bile o sırada yanıtsız kalır.
        matches, atlanan = await run_in_threadpool(
            adaylari_eslestir,
            embeddings=req.embeddings, labels=req.labels, species=req.species,
            candidates=req.candidates, ad_id=req.ad_id,
            external_record_id=req.external_record_id,
            threshold=req.match_threshold,
        )
    except Exception as e:  # sorgu vektörünün kendisi bozuksa
        logger.error(f"Eşleştirme hatası: {e}")
        raise HTTPException(400, f"Eşleştirme yapılamadı: {e}")

    return {"matches": matches, "skipped_candidates": atlanan,
            "model_version": MODEL_SURUMU}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0",
                port=int(os.getenv("AI_SERVICE_PORT", 8000)), reload=True)
