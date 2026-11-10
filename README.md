# 그립데이 — AI

> 떠난 사람이 그리워하는 대구·경북의 장소를, 남아 있는 사람과 여행자가 퀘스트로 대신 다녀와 안부를 전하는 서비스

2026 단디톤(멋쟁이사자처럼 14기 대구·경북 연합 해커톤) 출품작의 AI 기능입니다. 담당: 강영한

## 기능

### 1. 퀘스트 정형화·난이도 판별

장소명, 주소, 그리운 이유, 퀘스트 종류를 받아 LLM을 1회 호출하고 구조화된 JSON을 반환합니다. 백엔드의 `POST /quests/preview`에서 사용합니다.

```json
{
  "quest_title": "○○ 떡볶이집 안부 확인하기",
  "quest_description": "아직 영업 중인지 확인하고, 영업 중이라면 떡볶이를 먹고 가게 사진을 남겨주세요",
  "difficulty": 2,
  "difficulty_reason": "영업 여부 확인 + 음식 구매가 필요",
  "keywords": ["학창시절", "분식", "친구"]
}
```

| 난이도 | 기준 |
|---|---|
| 1 | 방문·촬영만 |
| 2 | 확인·구매 등 행동 1개 추가 |
| 3 | 시간대·구도 등 조건이 까다로움 |

LLM은 백엔드가 직접 호출합니다. 이 레포의 `llm/`에는 백엔드에 넘길 프롬프트(`quest_prompt.md`)와 출력 스키마(`quest_schema.json`), 그리고 이를 시험하는 스크립트를 둡니다.

### 2. 이미지 퀘스트 검증

수행 사진과 기준 사진의 임베딩 코사인 유사도로 같은 장소·같은 구도인지 판별합니다. 백엔드의 `POST /quests/{id}/submissions`에서 사용합니다.

임계값은 같은 장소/다른 장소 사진 10쌍으로 테스트해 정합니다. 위치 조건은 백엔드에서 먼저 확인합니다.

## 임베딩 서버

백엔드에서만 호출하는 내부 API입니다. 유사도 계산은 백엔드에서 합니다.

### `POST /embed`

`multipart/form-data`의 `image` 파트로 사진 1장(최대 10MB)을 보냅니다.

```json
{
  "model": "ViT-B-32",
  "dim": 512,
  "embedding": [0.0123, -0.0456]
}
```

`embedding`은 L2 정규화된 값이라 두 벡터의 내적이 곧 코사인 유사도입니다. 실제 길이는 `dim`과 같습니다. 모델이 바뀌면 `dim`도 바뀔 수 있으므로 512로 고정하지 마세요.

| 상태 | `code` | 의미 |
|---|---|---|
| 400 | `INVALID_IMAGE` | 이미지 해석 실패 |
| 400 | `IMAGE_TOO_LARGE` | 10MB 초과 |
| 500 | `INFERENCE_FAILED` | 추론 오류 |
| 503 | `MODEL_LOADING` | 모델 로딩 중이거나 로딩 실패 |

에러 응답은 `{ "code": ..., "message": ... }` 형식입니다.

### `GET /health`

모델 로딩이 끝나면 200, 그전에는 503을 반환합니다.

```json
{ "status": "ok", "model": "ViT-B-32" }
```

### 환경 변수

| 이름 | 기본값 | 설명 |
|---|---|---|
| `EMBED_MODEL` | `ViT-B-32` | open_clip 모델 이름 |
| `EMBED_PRETRAINED` | `openai` | 가중치 태그 |
| `EMBED_NUM_THREADS` | `2` | 추론에 쓰는 CPU 스레드 수 |

Docker 이미지는 빌드할 때 받은 가중치만 사용합니다. 모델을 바꾸면 이미지를 다시 빌드하고, 저장된 임베딩도 다시 만들어야 합니다.

## 기술 스택

- Python 3.12, FastAPI
- open_clip, PyTorch (CPU)
- CLIP ViT-B-32 (`openai` 가중치)

LLM 모델은 테스트 후 확정합니다. 후보는 Gemini 3.5 Flash-Lite입니다.

## 실행 방법

### Docker

```bash
docker build -t danditon-ai .
docker run --rm -p 8000:8000 danditon-ai
```

실행 메모리는 1.2GB 이상 잡아 주세요.

### 로컬

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000
```

처음 실행하면 가중치를 내려받느라 시간이 걸립니다.

### 테스트

```bash
pytest
```

### 퀘스트 정형화 프롬프트 시험

`llm/samples.json`의 예시 5개로 Gemini를 호출해 JSON 형식, 난이도, 응답 시간을 확인합니다. 결과는 `llm/results/`에 저장되며 커밋하지 않습니다.

```bash
pip install -r llm/requirements.txt
cp .env.example .env    # GEMINI_API_KEY에 개발용 키 입력
python llm/check_quest.py
```

`--model`로 다른 모델을, `--repeat 3`으로 같은 예시를 여러 번 돌려 결과가 흔들리는지 볼 수 있습니다.

## 협업

브랜치와 커밋 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md)를 따릅니다.
