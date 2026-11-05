"""임베딩 서버. BE에서만 호출하며 외부에 공개하지 않는다."""

import io
import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from app.embedder import Embedder

MAX_IMAGE_BYTES = 10 * 1024 * 1024

logger = logging.getLogger(__name__)

embedder = Embedder(
    model_name=os.getenv("EMBED_MODEL", "ViT-B-32"),
    pretrained=os.getenv("EMBED_PRETRAINED", "openai"),
    num_threads=int(os.getenv("EMBED_NUM_THREADS", "2")),
)


def _load_model() -> None:
    try:
        embedder.load()
    except Exception:
        # 로딩에 실패하면 /health가 계속 503을 반환하므로 서버는 띄워 둔다.
        logger.exception("모델 로딩 실패")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # 로딩 중에도 /health가 503으로 응답할 수 있도록 별도 스레드에서 로딩한다.
    threading.Thread(target=_load_model, daemon=True).start()
    yield


app = FastAPI(title="그리움 퀘스트 임베딩 서버", lifespan=lifespan)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "message": message})


def _not_ready() -> JSONResponse:
    return _error(503, "MODEL_LOADING", "모델을 아직 사용할 수 없습니다.")


def _decode_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        # 휴대폰 사진은 회전 정보가 EXIF에 있어서, 적용하지 않으면 누운 채로 임베딩된다.
        return ImageOps.exif_transpose(image).convert("RGB")


# 임베딩 호출은 CPU를 오래 쓰므로 스레드풀에서 돌도록 async를 쓰지 않는다.
@app.post("/embed")
def embed(image: UploadFile = File(...)) -> JSONResponse:
    if not embedder.ready:
        return _not_ready()

    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        return _error(400, "IMAGE_TOO_LARGE", "이미지는 최대 10MB입니다.")

    try:
        decoded = _decode_image(data)
    except Exception:
        # Pillow는 손상된 파일의 종류에 따라 여러 예외를 던진다.
        logger.info("이미지 해석 실패: %s", image.filename, exc_info=True)
        return _error(400, "INVALID_IMAGE", "이미지를 해석할 수 없습니다.")

    try:
        embedding = embedder.embed(decoded)
    except Exception:
        logger.exception("추론 실패")
        return _error(500, "INFERENCE_FAILED", "임베딩 추론에 실패했습니다.")

    return JSONResponse({"model": embedder.model_name, "dim": len(embedding), "embedding": embedding})


@app.get("/health")
def health() -> JSONResponse:
    if not embedder.ready:
        return _not_ready()
    return JSONResponse({"status": "ok", "model": embedder.model_name})
