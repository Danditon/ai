"""퀘스트 정형화 프롬프트를 Gemini로 돌려 JSON 형식과 난이도를 확인한다.

usage: python llm/check_quest.py [--model MODEL] [--repeat N]
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "quest_prompt.md").read_text(encoding="utf-8")
SCHEMA = json.loads((HERE / "quest_schema.json").read_text(encoding="utf-8"))
SAMPLES = json.loads((HERE / "samples.json").read_text(encoding="utf-8"))

DEFAULT_MODEL = "gemini-3.5-flash-lite"
TITLE_MAX = 30
DESCRIPTION_MAX = 100
# Gemini는 minLength 같은 일부 스키마 키워드를 강제하지 않아서 직접 검사한다.
FIELD_TYPES = {
    "quest_title": str,
    "quest_description": str,
    "difficulty": int,
    "difficulty_reason": str,
    "keywords": list,
}
IMAGE_HINTS = ("구도", "각도", "같은 자리", "같은 위치")
MAX_RETRIES = 2


def build_input(sample: dict) -> str:
    return (
        f"퀘스트 종류: {sample['questType']}\n"
        f"장소명: {sample['placeName']}\n"
        f"주소: {sample['address']}\n"
        f"그리운 이유: {sample['longingText']}"
    )


def check(sample: dict, data: dict) -> tuple[list[str], list[str]]:
    """(형식 오류, 사람이 볼 점)을 반환한다. 형식 오류가 있으면 BE가 그대로 쓸 수 없다."""
    errors = []
    for field, kind in FIELD_TYPES.items():
        value = data.get(field)
        # bool은 int의 하위 타입이라 따로 막는다.
        if not isinstance(value, kind) or isinstance(value, bool):
            errors.append(f"{field}: {kind.__name__}가 아님 ({value!r})")
    extra = set(data) - set(FIELD_TYPES)
    if extra:
        errors.append(f"정의되지 않은 필드: {sorted(extra)}")
    if errors:
        return errors, []

    if data["difficulty"] not in (1, 2, 3):
        errors.append(f"difficulty가 1~3이 아님 ({data['difficulty']})")
    keywords = data["keywords"]
    if not 1 <= len(keywords) <= 5 or not all(isinstance(k, str) and k.strip() for k in keywords):
        errors.append(f"keywords는 빈 값 없이 1~5개여야 함 ({keywords!r})")
    for field in ("quest_title", "quest_description", "difficulty_reason"):
        if not data[field].strip():
            errors.append(f"{field}가 비어 있음")

    notes = []
    if len(data["quest_title"]) > TITLE_MAX:
        notes.append(f"제목이 {TITLE_MAX}자를 넘음 ({len(data['quest_title'])}자)")
    if len(data["quest_description"]) > DESCRIPTION_MAX:
        notes.append(f"설명이 {DESCRIPTION_MAX}자를 넘음 ({len(data['quest_description'])}자)")
    if sample["questType"] == "IMAGE" and not any(h in data["quest_description"] for h in IMAGE_HINTS):
        notes.append("이미지 퀘스트인데 같은 구도로 찍어 달라는 말이 없음")
    expected = sample.get("expectedDifficulty")
    if expected is not None and data["difficulty"] != expected:
        notes.append(f"난이도가 예상과 다름 (예상 {expected}, 결과 {data['difficulty']})")
    return errors, notes


def make_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def call(client, model: str, sample: dict) -> dict:
    from google.genai import errors as genai_errors
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=PROMPT,
        response_mime_type="application/json",
        response_json_schema=SCHEMA,
    )
    # 503(과부하) 같은 서버 오류는 잠깐 뒤 다시 시도하고, 몇 번 재시도했는지 남긴다.
    retries = 0
    while True:
        started = time.perf_counter()
        try:
            response = client.models.generate_content(model=model, contents=build_input(sample), config=config)
            break
        except genai_errors.ServerError as e:
            if retries == MAX_RETRIES:
                return {"error": f"API 오류: {e}", "retries": retries}
            retries += 1
            time.sleep(2 * retries)
        except genai_errors.APIError as e:
            return {"error": f"API 오류: {e}", "retries": retries}
    result = {"seconds": round(time.perf_counter() - started, 2), "retries": retries, "raw": response.text}
    usage = response.usage_metadata
    if usage is not None:
        result["tokens"] = {
            "input": usage.prompt_token_count,
            "output": usage.candidates_token_count,
            "thinking": usage.thoughts_token_count,
        }
    if not response.text:
        result["error"] = "응답 본문이 비어 있음 (차단되었거나 토큰 한도 초과)"
        return result
    try:
        result["data"] = json.loads(response.text)
    except json.JSONDecodeError as e:
        result["error"] = f"JSON 파싱 실패: {e}"
        return result
    if not isinstance(result["data"], dict):
        result["error"] = "JSON 객체가 아님"
        return result
    result["errors"], result["notes"] = check(sample, result["data"])
    return result


def print_result(sample: dict, result: dict) -> None:
    print(f"\n[{sample['id']}] {sample['questType']} · {sample['placeName']}")
    print(f"    그리운 이유: {sample['longingText']}")
    if "seconds" in result:
        tokens = result.get("tokens", {})
        retried = f" · 재시도 {result['retries']}회" if result.get("retries") else ""
        print(f"    {result['seconds']}초 · 토큰 입력 {tokens.get('input')} / 출력 {tokens.get('output')} / 생각 {tokens.get('thinking')}{retried}")
    if "error" in result:
        print(f"    ❌ {result['error']}")
        if result.get("raw"):
            print(f"    원문: {result['raw']}")
        return
    data = result["data"]
    print(f"    제목: {data.get('quest_title')}")
    print(f"    설명: {data.get('quest_description')}")
    print(f"    난이도: {data.get('difficulty')} (예상 {sample.get('expectedDifficulty')}) · {data.get('difficulty_reason')}")
    print(f"    키워드: {data.get('keywords')}")
    for line in result["errors"]:
        print(f"    ❌ {line}")
    for line in result["notes"]:
        print(f"    ⚠️ {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--repeat", type=int, default=1, help="예시마다 몇 번 호출할지 (결과가 흔들리는지 볼 때)")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(HERE.parent / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY가 없어요. 레포 루트의 .env에 넣어 주세요 (.env.example 참고).", file=sys.stderr)
        return 2

    client = make_client(api_key)
    print(f"모델: {args.model} · 예시 {len(SAMPLES)}개 × {args.repeat}회")
    runs = []
    for sample in SAMPLES:
        for _ in range(args.repeat):
            result = call(client, args.model, sample)
            print_result(sample, result)
            runs.append({"sample": sample, **result})

    ok = [r for r in runs if "error" not in r and not r["errors"]]
    matched = [r for r in ok if r["data"]["difficulty"] == r["sample"].get("expectedDifficulty")]
    seconds = [r["seconds"] for r in runs if "seconds" in r]
    retried = [r for r in runs if r.get("retries")]
    print(f"\n== 요약: JSON 형식 통과 {len(ok)}/{len(runs)} · 난이도 예상 일치 {len(matched)}/{len(runs)}", end="")
    print(f" · 재시도한 호출 {len(retried)}/{len(runs)}", end="")
    print(f" · 응답 시간 중앙값 {statistics.median(seconds):.2f}초" if seconds else "")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{args.model}.json"
    out.write_text(json.dumps({"model": args.model, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out.relative_to(HERE.parent)}")
    return 0 if len(ok) == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
