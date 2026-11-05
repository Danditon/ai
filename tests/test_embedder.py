"""모델 로딩 설정 테스트. 가중치를 내려받지 않는다."""

from unittest.mock import MagicMock

import open_clip
import pytest

from app.embedder import Embedder


@pytest.mark.parametrize(
    ("pretrained", "quick_gelu"),
    [("openai", True), ("laion2b_s34b_b79k", False)],
)
def test_load_matches_activation_to_weights(monkeypatch, pretrained, quick_gelu):
    create = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    monkeypatch.setattr(open_clip, "create_model_and_transforms", create)
    embedder = Embedder("ViT-B-32", pretrained, num_threads=1)

    embedder.load()

    assert create.call_args.kwargs["force_quick_gelu"] is quick_gelu
    assert embedder.ready
