"""MegaDescriptor tabanlı görsel embedding servisi.

Model süreç başına yalnızca bir kez yüklenir, `eval()` modunda çalışır ve
MPS mevcutsa MPS'i, değilse CPU'yu kullanır. Girdi boyutu ve normalizasyon
değerleri elle yazılmaz; modelin kendi `timm` yapılandırmasından okunur.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config

from app.device import get_device

DEFAULT_MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-S-224"


class EmbeddingError(RuntimeError):
    """Embedding çıkarımı sırasında oluşan hatalar için yükseltilir."""


def _extract_output_tensor(output: object) -> torch.Tensor:
    """Modelin tensor, tuple/list veya dict biçimindeki çıktısından öznitelik tensörünü çıkarır."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        if not output:
            raise EmbeddingError("Model boş bir çıktı döndürdü.")
        return _extract_output_tensor(output[0])
    if isinstance(output, dict):
        for key in ("embedding", "embeddings", "features", "logits"):
            if key in output:
                return _extract_output_tensor(output[key])
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
        raise EmbeddingError(f"Model çıktısı beklenmeyen bir dict yapısında: {list(output.keys())}")
    raise EmbeddingError(f"Model çıktısı desteklenmeyen bir türde: {type(output)}")


class EmbeddingService:
    """MegaDescriptor modelini yükler ve görsellerden L2-normalize embedding üretir."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        checkpoint_path: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or get_device()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.model = self.model.eval().to(self.device)

        data_config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**data_config)
        self.input_size: tuple[int, int, int] = tuple(data_config["input_size"])
        self.embedding_dim = self._infer_embedding_dim()

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise EmbeddingError(f"Checkpoint dosyası bulunamadı: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise EmbeddingError(f"Checkpoint, model mimarisiyle uyuşmuyor: {exc}") from exc

    @torch.inference_mode()
    def _infer_embedding_dim(self) -> int:
        dummy = torch.zeros(1, *self.input_size, device=self.device)
        output = _extract_output_tensor(self.model(dummy))
        return int(output.shape[-1])

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise EmbeddingError(f"Beklenmeyen görsel türü: {type(image)}")
        return self.transform(image.convert("RGB"))

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Tek bir PIL görseli için L2-normalize edilmiş embedding döndürür."""
        return self.embed_images([image])[0]

    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 8) -> np.ndarray:
        """Bir dizi PIL görseli için toplu, L2-normalize edilmiş embedding matrisi döndürür.

        Sonuç, `images` ile aynı sırada bir `(N, embedding_dim)` `numpy.ndarray`'dir.
        """
        if not images:
            raise EmbeddingError("Embedding çıkarmak için en az bir görsel gerekli.")
        if batch_size < 1:
            raise EmbeddingError("batch_size en az 1 olmalı.")

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack([self._preprocess(img) for img in batch]).to(self.device)
            try:
                with torch.inference_mode():
                    output = self.model(tensors)
            except RuntimeError as exc:
                raise EmbeddingError(
                    f"Model çıkarımı başarısız oldu (batch_size={len(batch)}): {exc}"
                ) from exc

            features = _extract_output_tensor(output)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_embeddings.append(features.detach().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)


@lru_cache(maxsize=1)
def _cached_service(model_name: str, checkpoint_str: str | None) -> EmbeddingService:
    checkpoint_path = Path(checkpoint_str) if checkpoint_str else None
    return EmbeddingService(model_name=model_name, checkpoint_path=checkpoint_path)


def get_embedding_service(
    model_name: str = DEFAULT_MODEL_NAME, checkpoint_path: Path | None = None
) -> EmbeddingService:
    """Süreç boyunca (model_name, checkpoint) kombinasyonu başına tek bir servis örneği döndürür."""
    checkpoint_str = str(checkpoint_path) if checkpoint_path else None
    return _cached_service(model_name, checkpoint_str)
