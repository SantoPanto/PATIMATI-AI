"""Testler için paylaşılan fixture'lar.

Gerçek MegaDescriptor modelinin her test çalıştırmasında indirilip yüklenmesini
önlemek için `tiny_embedding_service`, `EmbeddingService`'in ağır bağımlılıklarını
(`timm.create_model`, `resolve_data_config`, `create_transform`) sahte, küçük ve
tamamen çevrimdışı çalışan bir backbone ile değiştirir. Böylece asıl sınıfın
(EmbeddingService) mantığı gerçekten test edilirken ağ erişimi gerekmez.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

TEST_INPUT_SIZE = (3, 32, 32)
TEST_EMBEDDING_DIM = 16


class TinyBackbone(nn.Module):
    """Gerçek MegaDescriptor yerine kullanılan, indirme gerektirmeyen minik bir backbone."""

    def __init__(self, out_dim: int = TEST_EMBEDDING_DIM) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(3, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return self.proj(pooled)  # type: ignore[no-any-return]


@pytest.fixture
def tiny_embedding_service(monkeypatch: pytest.MonkeyPatch) -> object:
    from app.services import embedding_service as es_module

    tiny_model = TinyBackbone()
    fake_transform = transforms.Compose(
        [transforms.Resize(TEST_INPUT_SIZE[1:]), transforms.ToTensor()]
    )

    def fake_create_model(name: str, pretrained: bool = True, num_classes: int = 0) -> nn.Module:
        return tiny_model

    def fake_resolve_data_config(cfg: dict, model: nn.Module) -> dict:
        return {"input_size": TEST_INPUT_SIZE}

    def fake_create_transform(**kwargs: object) -> transforms.Compose:
        return fake_transform

    monkeypatch.setattr(es_module.timm, "create_model", fake_create_model)
    monkeypatch.setattr(es_module, "resolve_data_config", fake_resolve_data_config)
    monkeypatch.setattr(es_module, "create_transform", fake_create_transform)

    return es_module.EmbeddingService(model_name="tiny-test-model", device=torch.device("cpu"))


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (48, 48), color=(120, 30, 200)).save(path)
    return path


@pytest.fixture
def corrupt_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not an image")
    return path


@pytest.fixture
def sample_image_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), color=(10, 200, 50)).save(buffer, format="JPEG")
    return buffer.getvalue()
