"""Uygulama ayarları. Değerler `.env` dosyasından (varsa) ve ortam değişkenlerinden okunur."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _split_origins(raw: str) -> list[str]:
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


@dataclass(frozen=True)
class Settings:
    """Ortam değişkenlerinden türetilen tekil ayar nesnesi."""

    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "hf-hub:BVRA/MegaDescriptor-S-224")
    )
    model_source: str = field(default_factory=lambda: os.getenv("MODEL_SOURCE", "pretrained"))
    model_checkpoint: Path = field(
        default_factory=lambda: (
            PROJECT_ROOT / os.getenv("MODEL_CHECKPOINT", "models/best_model.pth")
        )
    )
    index_path: Path = field(
        default_factory=lambda: (
            PROJECT_ROOT / os.getenv("INDEX_PATH", "outputs/embeddings/pet_index.npz")
        )
    )
    upload_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("UPLOAD_DIR", "outputs/uploaded_images")
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _split_origins(os.getenv("CORS_ORIGINS", "*"))
    )
    max_upload_size_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))
    )

    @property
    def active_checkpoint(self) -> Path | None:
        """model_source 'finetuned' ise checkpoint yolunu, aksi halde None döndürür."""
        if self.model_source == "finetuned":
            return self.model_checkpoint
        return None


def get_settings() -> Settings:
    """Her çağrıda ortam değişkenlerinden taze bir Settings üretir."""
    return Settings()
