"""CLIP 이미지 임베딩."""

import logging
import threading

import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, pretrained: str, num_threads: int) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_threads = num_threads
        self._model = None
        self._preprocess = None
        self._ready = threading.Event()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def load(self) -> None:
        # BE와 같은 인스턴스에서 돌기 때문에 CPU를 다 쓰지 않도록 스레드 수를 제한한다.
        torch.set_num_threads(self.num_threads)
        # openai 가중치는 QuickGELU로 학습되어, 맞춰 주지 않으면 벡터가 틀어진다.
        pretrained_cfg = open_clip.get_pretrained_cfg(self.model_name, self.pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            force_quick_gelu=pretrained_cfg.get("quick_gelu", False),
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._ready.set()
        logger.info("모델 로딩 완료: %s (%s)", self.model_name, self.pretrained)

    def embed(self, image: Image.Image) -> list[float]:
        """L2 정규화된 벡터를 반환한다. BE에서는 내적이 곧 코사인 유사도다."""
        batch = self._preprocess(image).unsqueeze(0)
        with torch.inference_mode():
            features = self._model.encode_image(batch)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features[0].tolist()
