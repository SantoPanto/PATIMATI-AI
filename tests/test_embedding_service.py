"""app/device.py ve app/services/embedding_service.py için testler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.device import get_device
from app.services.embedding_service import EmbeddingError
from app.services.image_service import ImageLoadError, load_rgb_image


def test_get_device_returns_mps_or_cpu() -> None:
    device = get_device()
    assert device.type in {"mps", "cpu"}


def test_embed_image_returns_l2_normalized_vector(
    tiny_embedding_service, sample_image_path: Path
) -> None:
    image = load_rgb_image(sample_image_path)
    vector = tiny_embedding_service.embed_image(image)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (tiny_embedding_service.embedding_dim,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_embed_images_batch_matches_single(tiny_embedding_service, sample_image_path: Path) -> None:
    image = load_rgb_image(sample_image_path)
    batch = tiny_embedding_service.embed_images([image, image], batch_size=2)

    assert batch.shape == (2, tiny_embedding_service.embedding_dim)
    np.testing.assert_allclose(batch[0], batch[1], atol=1e-5)


def test_embed_images_empty_list_raises(tiny_embedding_service) -> None:
    with pytest.raises(EmbeddingError):
        tiny_embedding_service.embed_images([])


def test_load_rgb_image_rejects_corrupt_file(corrupt_image_path: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_rgb_image(corrupt_image_path)


def test_load_rgb_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_rgb_image(tmp_path / "does_not_exist.jpg")
