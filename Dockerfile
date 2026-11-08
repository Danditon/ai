FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    EMBED_MODEL=ViT-B-32 \
    EMBED_PRETRAINED=openai

WORKDIR /srv

# CPU 전용 PyTorch를 받아 이미지 크기를 줄인다.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 모델 가중치를 이미지에 포함한다. 실행 중에는 외부에서 내려받지 않는다.
# 모델을 만들지 않고 파일만 받아서 RAM 2GB 서버에서도 빌드 메모리가 적게 든다.
RUN python -c "import os, open_clip; open_clip.download_pretrained(open_clip.get_pretrained_cfg(os.environ['EMBED_MODEL'], os.environ['EMBED_PRETRAINED']))"
ENV HF_HUB_OFFLINE=1

COPY app ./app

RUN useradd --system --no-create-home appuser
USER appuser

EXPOSE 8000

# 모델 로딩이 끝나야 /health가 200을 반환한다.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
