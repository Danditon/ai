"""임계값 분석의 쌍 구성과 임계값 추천 테스트. 모델은 불러오지 않는다."""

from threshold.analyze import build_pairs, recommend_threshold


def touch(root, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_pairs_by_number_and_cross_pairs(tmp_path):
    touch(tmp_path, "ref/01.jpg", "ref/02.jpg", "same/01.jpg", "same/02.jpg", "angle/01.jpg", "nearby/02.png", "same/.DS_Store")

    pairs, warnings = build_pairs(tmp_path)

    got = sorted((p["group"], p["pair"]) for p in pairs)
    assert got == [("angle", "01"), ("cross", "01-02"), ("cross", "02-01"), ("nearby", "02"), ("same", "01"), ("same", "02")]
    cross = next(p for p in pairs if p["pair"] == "01-02")
    assert cross["ref"].name == "01.jpg" and cross["photo"] == tmp_path / "same" / "02.jpg"
    assert warnings == []


def test_sets_of_the_same_place_are_not_cross_pairs(tmp_path):
    touch(tmp_path, "ref/03a.jpg", "ref/03b.jpg", "ref/04.jpg", "same/03a.jpg", "same/03b.jpg", "same/04.jpg")

    pairs, _ = build_pairs(tmp_path)

    cross = sorted(p["pair"] for p in pairs if p["group"] == "cross")
    assert cross == ["03a-04", "03b-04", "04-03a", "04-03b"]


def test_photo_without_ref_is_skipped_with_warning(tmp_path):
    touch(tmp_path, "ref/01.jpg", "nearby/03.jpg")

    pairs, warnings = build_pairs(tmp_path)

    assert pairs == []
    assert any("nearby/03.jpg" in w for w in warnings)


def test_duplicate_numbers_warn(tmp_path):
    touch(tmp_path, "ref/01.jpg", "ref/01.png", "same/01.jpg")

    pairs, warnings = build_pairs(tmp_path)

    assert [p["ref"].name for p in pairs] == ["01.jpg"]
    assert any("01.png" in w for w in warnings)


def test_separable_scores_pick_the_middle_of_the_gap():
    result = recommend_threshold([0.9, 0.95], {"nearby": [0.7, 0.8], "cross": [0.5, 0.6]})

    assert result["threshold"] == 0.85
    assert result["balanced_accuracy"] == 1.0
    assert result["passed"] == {"count": 2, "total": 2}
    assert result["rejected"]["nearby"] == {"count": 2, "total": 2}
    assert result["gap"] == 0.1


def test_nearby_failures_are_not_hidden_by_many_cross_pairs():
    # 교차 쌍이 많아도 200m 안 쌍을 놓치는 임계값은 고르지 않아야 한다.
    positives = [0.86, 0.9, 0.92, 0.95]
    result = recommend_threshold(positives, {"nearby": [0.84, 0.85], "cross": [0.5] * 50})

    assert 0.85 < result["threshold"] <= 0.86
    assert result["rejected"]["nearby"]["count"] == 2


def test_overlapping_scores_report_negative_gap():
    result = recommend_threshold([0.9, 0.75], {"nearby": [0.8, 0.6]})

    assert result["gap"] < 0
    assert result["balanced_accuracy"] < 1.0


def test_missing_groups_return_none():
    assert recommend_threshold([], {"nearby": [0.5]}) is None
    assert recommend_threshold([0.9], {"nearby": [], "cross": []}) is None
