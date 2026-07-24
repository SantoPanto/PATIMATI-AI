"""Kayıp/bulunan kedi-köpek eşleştirme servisi için FastAPI uygulaması.

Çalıştırma:
    uvicorn app.api:app --reload
"""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.config import PROJECT_ROOT, Settings, get_settings
from app.schemas import (
    EmbeddingResponse,
    HealthResponse,
    IndexAddResponse,
    IndexReloadResponse,
    MatchItem,
    MatchQueryInfo,
    MatchResponse,
)
from app.services.embedding_service import EmbeddingError, EmbeddingService, get_embedding_service
from app.services.image_service import SUPPORTED_EXTENSIONS, is_supported_extension
from app.services.index_service import IndexError_, IndexRecord, LocalNumpyIndex, VectorIndex
from app.services.matching_service import MatchingService

logger = logging.getLogger("pet_ai")

_settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    embedding_service = get_embedding_service(
        model_name=settings.model_name, checkpoint_path=settings.active_checkpoint
    )

    index: VectorIndex = LocalNumpyIndex()
    if settings.index_path.exists():
        try:
            index.load(settings.index_path)
            logger.info("İndeks yüklendi: %s (%d kayıt)", settings.index_path, len(index))
        except Exception:  # noqa: BLE001 - bozuk indeks dosyasında da servis ayağa kalkmalı
            logger.warning(
                "İndeks yüklenemedi (%s); boş indeksle başlatılıyor.", settings.index_path
            )
    else:
        logger.info("İndeks dosyası yok (%s); boş indeksle başlanıyor.", settings.index_path)

    app.state.settings = settings
    app.state.embedding_service = embedding_service
    app.state.index = index
    app.state.matching_service = MatchingService(embedding_service, index)
    yield


app = FastAPI(
    title="Pet AI - Kayıp/Bulunan Hayvan Eşleştirme Servisi",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Beklenmeyen sunucu hatası: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500, content={"detail": "Beklenmeyen bir sunucu hatası oluştu."}
    )


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_app_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service


def get_app_index(request: Request) -> VectorIndex:
    return request.app.state.index


def get_app_matching_service(request: Request) -> MatchingService:
    return request.app.state.matching_service


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    if file.filename is None or not is_supported_extension(Path(file.filename)):
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı. İzin verilenler: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Yüklenen dosya boş.")
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Dosya çok büyük (maksimum {settings.max_upload_size_bytes} byte).",
        )
    return contents


def _decode_image(contents: bytes, filename: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(contents)) as raw:
            raw.load()
            return raw.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Görsel okunamadı ({filename}): {exc}"
        ) from exc


@app.get("/health", response_model=HealthResponse)
async def health(
    embedding_service: EmbeddingService = Depends(get_app_embedding_service),
    index: VectorIndex = Depends(get_app_index),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=str(embedding_service.device),
        index_size=len(index),
    )


@app.post("/embedding", response_model=EmbeddingResponse)
async def create_embedding(
    image: UploadFile = File(...),
    include_vector: bool = Query(False),
    settings: Settings = Depends(get_app_settings),
    embedding_service: EmbeddingService = Depends(get_app_embedding_service),
) -> EmbeddingResponse:
    contents = await _read_upload(image, settings)
    pil_image = _decode_image(contents, image.filename or "upload")

    start = time.perf_counter()
    try:
        vector = embedding_service.embed_image(pil_image)
    except EmbeddingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    return EmbeddingResponse(
        filename=image.filename or "upload",
        embedding_dim=int(vector.shape[0]),
        processing_time_ms=round(elapsed_ms, 2),
        vector=vector.tolist() if include_vector else None,
    )


@app.post("/index", response_model=IndexAddResponse)
async def add_to_index(
    image: UploadFile = File(...),
    image_id: str = Form(...),
    animal_id: str = Form(...),
    species: str | None = Form(None),
    metadata: str | None = Form(None),
    settings: Settings = Depends(get_app_settings),
    embedding_service: EmbeddingService = Depends(get_app_embedding_service),
    index: VectorIndex = Depends(get_app_index),
) -> IndexAddResponse:
    if image_id in index:
        raise HTTPException(status_code=409, detail=f"image_id zaten indekste mevcut: {image_id}")

    contents = await _read_upload(image, settings)
    pil_image = _decode_image(contents, image.filename or "upload")

    parsed_metadata: dict[str, Any] = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"metadata geçerli bir JSON değil: {exc}"
            ) from exc

    try:
        vector = embedding_service.embed_image(pil_image)
    except EmbeddingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    extension = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    saved_path = settings.upload_dir / animal_id / f"{image_id}{extension}"
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(contents)
    try:
        relative_path = saved_path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative_path = saved_path

    record = IndexRecord(
        embedding=vector,
        image_id=image_id,
        image_path=str(relative_path),
        animal_id=animal_id,
        species=species,
        metadata=parsed_metadata,
    )
    index.add(record)
    try:
        index.save(settings.index_path)
    except IndexError_ as exc:
        raise HTTPException(status_code=500, detail=f"İndeks diske kaydedilemedi: {exc}") from exc

    return IndexAddResponse(
        image_id=image_id, animal_id=animal_id, species=species, index_size=len(index)
    )


@app.post("/match", response_model=MatchResponse)
async def match_animal(
    image: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50),
    species: str | None = Query(None),
    min_similarity: float | None = Query(None, ge=-1.0, le=1.0),
    exclude_image_id: str | None = Query(None),
    settings: Settings = Depends(get_app_settings),
    matching_service: MatchingService = Depends(get_app_matching_service),
) -> MatchResponse:
    contents = await _read_upload(image, settings)
    pil_image = _decode_image(contents, image.filename or "upload")

    start = time.perf_counter()
    try:
        results = matching_service.match(
            pil_image,
            top_k=top_k,
            exclude_image_id=exclude_image_id,
            species=species,
            min_similarity=min_similarity,
        )
    except (EmbeddingError, IndexError_) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    return MatchResponse(
        query=MatchQueryInfo(
            filename=image.filename or "upload", processing_time_ms=round(elapsed_ms, 2)
        ),
        matches=[
            MatchItem(
                rank=result.rank,
                image_id=result.record.image_id,
                animal_id=result.record.animal_id,
                species=result.record.species,
                similarity=round(result.similarity, 4),
                image_path=result.record.image_path,
            )
            for result in results
        ],
    )


@app.post("/index/reload", response_model=IndexReloadResponse)
async def reload_index(
    settings: Settings = Depends(get_app_settings),
    index: VectorIndex = Depends(get_app_index),
) -> IndexReloadResponse:
    if not settings.index_path.exists():
        raise HTTPException(
            status_code=404, detail=f"İndeks dosyası bulunamadı: {settings.index_path}"
        )
    try:
        index.load(settings.index_path)
    except IndexError_ as exc:
        raise HTTPException(status_code=500, detail=f"İndeks yüklenemedi: {exc}") from exc
    return IndexReloadResponse(status="reloaded", index_size=len(index))
