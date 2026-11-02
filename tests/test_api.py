"""API 형식 테스트. 모델 가중치 없이 돌도록 임베더를 가짜로 바꾼다."""

import io
import math

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main


class FakeEmbedder:
    model_name = "ViT-B-32"

    def __init__(self, ready: bool = True, error: Exception | None = None) -> None:
        self.ready = ready
        self.error = error
        self.received: Image.Image | None = None

    def embed(self, image: Image.Image) -> list[float]:
        if self.error:
            raise self.error
        self.received = image
        return [0.6, 0.8]


def image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (8, 8), exif=None) -> bytes:
    buffer = io.BytesIO()
    kwargs = {"exif": exif} if exif else {}
    Image.new("RGB", size, "red").save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    # with 문을 쓰지 않아 lifespan의 모델 로딩이 돌지 않는다.
    return TestClient(main.app)


def post_image(client: TestClient, data: bytes):
    return client.post("/embed", files={"image": ("photo.jpg", data, "image/jpeg")})


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_embed_returns_vector(client, monkeypatch, fmt):
    monkeypatch.setattr(main, "embedder", FakeEmbedder())

    response = post_image(client, image_bytes(fmt))

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "ViT-B-32"
    assert body["dim"] == len(body["embedding"]) == 2
    assert math.isclose(sum(x * x for x in body["embedding"]), 1.0)


def test_embed_applies_exif_orientation(client, monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(main, "embedder", fake)
    exif = Image.Exif()
    exif[0x0112] = 6  # 시계 방향 90도 회전

    response = post_image(client, image_bytes(size=(16, 8), exif=exif))

    assert response.status_code == 200
    assert fake.received.size == (8, 16)


@pytest.mark.parametrize(
    "data",
    [b"not an image", b"", image_bytes()[:40]],
    ids=["garbage", "empty", "truncated"],
)
def test_embed_rejects_broken_image(client, monkeypatch, data):
    monkeypatch.setattr(main, "embedder", FakeEmbedder())

    response = post_image(client, data)

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_IMAGE"


def test_embed_rejects_too_large_image(client, monkeypatch):
    monkeypatch.setattr(main, "embedder", FakeEmbedder())

    response = post_image(client, b"0" * (main.MAX_IMAGE_BYTES + 1))

    assert response.status_code == 400
    assert response.json()["code"] == "IMAGE_TOO_LARGE"


def test_embed_returns_500_when_inference_fails(client, monkeypatch):
    monkeypatch.setattr(main, "embedder", FakeEmbedder(error=RuntimeError("boom")))

    response = post_image(client, image_bytes())

    assert response.status_code == 500
    assert response.json()["code"] == "INFERENCE_FAILED"


def test_returns_503_while_model_is_loading(client, monkeypatch):
    monkeypatch.setattr(main, "embedder", FakeEmbedder(ready=False))

    assert client.get("/health").status_code == 503
    assert post_image(client, image_bytes()).status_code == 503


def test_health_returns_model_when_ready(client, monkeypatch):
    monkeypatch.setattr(main, "embedder", FakeEmbedder())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "ViT-B-32"}
