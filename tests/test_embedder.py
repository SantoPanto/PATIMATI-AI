# tests/test_embedder.py
# Not: ilk çalıştırmada CLIP modeli (~400 MB) indirilir, birkaç dakika sürebilir.
import io

import numpy as np
from PIL import Image

from app.embedder import embedder


def make_image_bytes(color, size=(224, 224)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_embedding_shape_and_norm():
    emb = embedder.embed_bytes(make_image_bytes((255, 128, 0)))
    assert len(emb) == 512
    # L2-normalize edilmiş vektörün normu 1 olmalı
    assert abs(np.linalg.norm(np.array(emb)) - 1.0) < 1e-3


def test_identical_images_similarity_one():
    b = make_image_bytes((100, 150, 200))
    e1 = embedder.embed_bytes(b)
    e2 = embedder.embed_bytes(b)
    sim = float(np.dot(np.array(e1), np.array(e2)))
    assert sim > 0.999
