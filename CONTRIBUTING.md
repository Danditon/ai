# 협업 규칙

## 브랜치

| 브랜치 | 용도 | 규칙 |
|---|---|---|
| `main` | 배포·제출용 | 직접 푸시 금지. `develop`에서 PR로만 병합 |
| `develop` | 개발 통합 (기본 브랜치) | 직접 푸시 금지. 작업 브랜치에서 PR로 병합 |
| `feat/기능명` | 새 기능 | `develop`에서 분기 |
| `fix/내용` | 버그 수정 | `develop`에서 분기 |
| `docs/내용` | 문서 | `develop`에서 분기 |
| `chore/내용` | 설정·빌드·기타 | `develop`에서 분기 |

브랜치 이름은 영어 소문자와 하이픈으로 씁니다. 예: `feat/quest-map`, `fix/login-token`

## 작업 순서

```bash
git switch develop
git pull
git switch -c feat/quest-map
# 작업 후
git push -u origin feat/quest-map
```

그다음 GitHub에서 `develop`으로 PR을 열고, 팀원 1명의 확인을 받아 병합합니다.

## 커밋 메시지

`타입: 내용` 형식으로 씁니다.

| 타입 | 용도 |
|---|---|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서 |
| `refactor` | 동작 변화 없는 코드 개선 |
| `test` | 테스트 |
| `chore` | 설정·빌드·기타 |

예: `feat: 퀘스트 지도 핀 표시`

## 주의

- API 키, 비밀번호, `.env` 파일은 커밋하지 않습니다.
- 값이 필요한 환경 변수는 `.env.example`에 이름만 적어 둡니다.
