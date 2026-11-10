"""퀘스트 정형화 출력 검사 테스트. LLM은 호출하지 않는다."""

import pytest

from llm.check_quest import SAMPLES, SCHEMA, build_input, check

GENERAL = {"questType": "GENERAL", "placeName": "삼덕동 떡볶이집", "expectedDifficulty": 2}
IMAGE = {"questType": "IMAGE", "placeName": "영일대 해수욕장", "expectedDifficulty": 3}


def valid_output(**overrides):
    data = {
        "quest_title": "삼덕동 떡볶이집 안부 확인하기",
        "quest_description": "아직 장사하는지 확인해 주세요.",
        "difficulty": 2,
        "difficulty_reason": "영업 여부 확인이 필요해요.",
        "keywords": ["학창시절", "친구", "분식"],
    }
    data.update(overrides)
    return data


def test_valid_output_has_no_errors_or_notes():
    assert check(GENERAL, valid_output()) == ([], [])


@pytest.mark.parametrize(
    "overrides",
    [
        {"difficulty": 4},
        {"difficulty": "2"},
        {"difficulty": True},
        {"keywords": []},
        {"keywords": ["a", "b", "c", "d", "e", "f"]},
        {"keywords": ["학창시절", " "]},
        {"quest_title": "  "},
        {"extra": "x"},
    ],
)
def test_format_errors(overrides):
    errors, _ = check(GENERAL, valid_output(**overrides))
    assert errors


def test_missing_field_is_error():
    data = valid_output()
    del data["keywords"]
    errors, _ = check(GENERAL, data)
    assert any("keywords" in e for e in errors)


def test_image_quest_without_composition_is_noted():
    data = valid_output(difficulty=3, quest_description="해 뜰 때 다녀와 주세요.")
    errors, notes = check(IMAGE, data)
    assert not errors
    assert any("구도" in n for n in notes)


def test_difficulty_mismatch_is_noted_not_error():
    errors, notes = check(GENERAL, valid_output(difficulty=1))
    assert not errors
    assert any("예상" in n for n in notes)


def test_long_title_is_noted():
    _, notes = check(GENERAL, valid_output(quest_title="가" * 31))
    assert any("제목" in n for n in notes)


def test_build_input_contains_every_field():
    text = build_input(SAMPLES[0])
    for key in ("questType", "placeName", "address", "longingText"):
        assert SAMPLES[0][key] in text


def test_samples_and_schema_agree():
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"])
    for sample in SAMPLES:
        assert sample["questType"] in ("GENERAL", "IMAGE")
        assert sample["expectedDifficulty"] in (1, 2, 3)
