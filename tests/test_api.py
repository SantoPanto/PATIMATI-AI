"""app/api.py için uçtan uca testler.

`get_embedding_service` ve `get_settings`, gerçek MegaDescriptor modelinin indirilmesini
ve gerçek `outputs/` klasörlerinin kullanılmasını önlemek için sahte/izole sürümlerle
değiştirilir (bkz. tests/conftest.py: tiny_embedding_service).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.config import Settings


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, tiny_embedding_service: object, tmp_path: Path
) -> Iterator[TestClient]:
    test_settings = Settings(index_path=tmp_path / "pet_index.npz", upload_dir=tmp_path / "uploads")
    monkeypatch.setattr(api_module, "get_settings", lambda: test_settings)
    monkeypatch.setattr(
        api_module, "get_embedding_service", lambda **kwargs: tiny_embedding_service
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_endpoint_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["index_size"] == 0


def test_embedding_endpoint_returns_dimension(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    response = client.post(
        "/embedding", files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["embedding_dim"] > 0
    assert body["vector"] is None


def test_embedding_endpoint_include_vector(client: TestClient, sample_image_bytes: bytes) -> None:
    response = client.post(
        "/embedding?include_vector=true",
        files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )

    body = response.json()
    assert isinstance(body["vector"], list)
    assert len(body["vector"]) == body["embedding_dim"]


def test_embedding_endpoint_rejects_unsupported_format(client: TestClient) -> None:
    response = client.post(
        "/embedding", files={"image": ("notes.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_index_and_match_roundtrip(client: TestClient, sample_image_bytes: bytes) -> None:
    add_response = client.post(
        "/index",
        files={"image": ("cat.jpg", sample_image_bytes, "image/jpeg")},
        data={"image_id": "img_1", "animal_id": "kedi_001", "species": "cat"},
    )
    assert add_response.status_code == 200
    assert add_response.json()["index_size"] == 1

    match_response = client.post(
        "/match",
        params={"top_k": 5},
        files={"image": ("query.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert match_response.status_code == 200
    matches = match_response.json()["matches"]
    assert len(matches) == 1
    assert matches[0]["animal_id"] == "kedi_001"
    assert matches[0]["similarity"] > 0.99


def test_match_excludes_given_image_id(client: TestClient, sample_image_bytes: bytes) -> None:
    client.post(
        "/index",
        files={"image": ("cat.jpg", sample_image_bytes, "image/jpeg")},
        data={"image_id": "img_1", "animal_id": "kedi_001"},
    )

    response = client.post(
        "/match",
        params={"top_k": 5, "exclude_image_id": "img_1"},
        files={"image": ("query.jpg", sample_image_bytes, "image/jpeg")},
    )

    assert response.json()["matches"] == []


def test_match_on_empty_index_returns_empty_list(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    response = client.post(
        "/match",
        params={"top_k": 5},
        files={"image": ("query.jpg", sample_image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["matches"] == []


def test_index_duplicate_image_id_returns_conflict(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    data = {"image_id": "dup", "animal_id": "kedi_001"}
    first = client.post(
        "/index", files={"image": ("a.jpg", sample_image_bytes, "image/jpeg")}, data=data
    )
    assert first.status_code == 200

    second = client.post(
        "/index", files={"image": ("a.jpg", sample_image_bytes, "image/jpeg")}, data=data
    )
    assert second.status_code == 409


def test_index_reload_without_file_returns_404(client: TestClient) -> None:
    response = client.post("/index/reload")
    assert response.status_code == 404
