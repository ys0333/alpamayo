# Codex Harness Engineering Notes

작성일: 2026-04-15

이 문서는 `Codex`를 더 "똑똑하게" 보이게 만드는 핵심이 모델 교체보다 **하네스 설계**에 있다는 전제로 정리한 실무 메모다. 아래 내용은 OpenAI 공식 Codex 문서와 공식 Cookbook만 기준으로 추렸다.

## 1. 가장 큰 레버는 모델보다 하네스다

- `Codex`는 그냥 프롬프트 한 덩어리보다 `AGENTS.md`, `rules`, 툴 정의, shell 정책, patch 형식, 병렬 호출 규칙을 잘 설계했을 때 훨씬 잘 맞춘다.
- 공식 Prompting Guide도 `apply_patch` 구현과 도구 설계를 주요 성능 레버로 본다.
- 특히 coding agent는 "무슨 파일을 읽고", "어떤 툴을 우선 쓰고", "어디까지 자율적으로 해도 되는지"가 분명할수록 성능이 좋아진다.

## 2. `AGENTS.md`는 가장 중요한 MD다

공식 가이드 기준:

- `Codex-cli`는 `AGENTS.md` 파일들을 자동으로 수집해 대화 상단에 주입한다.
- 파일은 `~/.codex`와 repo root에서 현재 작업 디렉터리까지의 경로에서 수집된다.
- 더 깊은 디렉터리의 `AGENTS.md`가 더 상위 지침을 덮어쓴다.

실무 권장:

- repo root `AGENTS.md`에는 전역 규칙만 둔다.
- 하위 디렉터리 `AGENTS.md`에는 그 폴더 전용 규칙만 둔다.
- "설명"보다 "행동 규칙" 위주로 짧게 쓴다.
- 반드시 넣을 항목:
  - 어떤 파일을 먼저 읽을지
  - 어떤 명령/툴을 우선 사용할지
  - 금지할 명령
  - 테스트/검증 기준
  - 응답 스타일

예시:

```md
# AGENTS.md

- Read `README.md` and `docs/architecture.md` before editing.
- Use `rg` for search and `apply_patch` for manual edits.
- Never edit generated files under `build/`.
- Run `pytest tests/unit -q` after Python changes.
- Keep final answers short and include file paths changed.
```

## 3. `rules`로 승인 병목을 줄여라

공식 Rules 문서 기준:

- Codex는 `rules/` 아래 규칙 파일을 startup 시 스캔한다.
- 허용 규칙은 반복 승인 프롬프트를 줄여 속도를 올린다.
- 규칙은 `allow`, `prompt`, `forbidden`을 지원하고, 여러 규칙이 겹치면 더 보수적인 결정이 우선한다.
- `codex execpolicy check`로 규칙을 테스트할 수 있다.

실무 권장:

- 자주 쓰는 안전 명령만 `allow`한다.
- destructive command는 기본 `prompt` 또는 `forbidden`으로 둔다.
- `git`, `rg`, `pytest`, `ls`, `cat`, `sed`, `find`처럼 반복 읽기/검증 계열만 먼저 allow 후보로 본다.

예시 전략:

- `rg`, `rg --files`, `git status`, `git diff`, `pytest tests/unit`
- 배포/삭제/원격 쓰기 계열은 여전히 수동 승인

## 4. 툴 스키마를 모델 친화적으로 만들어라

공식 Prompting Guide 기준:

- `apply_patch`는 공식 구현을 그대로 쓰는 것이 가장 좋다.
- terminal wrapping tool을 만들더라도 이름, 인자, 출력이 원래 명령과 최대한 비슷해야 한다.
- semantic search 같은 custom tool도 가능하지만 추가 튜닝이 더 필요하다.

실무 권장:

- 편집은 `apply_patch`
- 쉘은 `shell` 또는 터미널과 유사한 래퍼
- 계획은 `update_plan`
- git 전용 tool이 있으면 git은 터미널 대신 전용 tool만 쓰게 지시

즉, 좋은 하네스는 "툴이 많다"가 아니라:

- 이름이 명확하고
- 입력 스키마가 단순하고
- 언제 써야 하는지 프롬프트에 분명하며
- 출력 형식이 일관된 상태

여야 한다.

## 5. 파일 읽기는 반드시 병렬화하는 편이 낫다

공식 Prompting Guide 기준:

- parallel tool calling이 가능할 때는 파일 읽기/검색/나열을 최대한 병렬화하라고 권장한다.
- 한 파일씩 순차적으로 읽는 건 불가피할 때만 하라고 되어 있다.

실무 권장:

- 첫 탐색에서 필요한 파일을 한 번에 계획
- `rg`, `ls`, `find`, `sed`, `git show`, `wc`는 한 배치로 묶기
- "파일 읽기 -> 또 읽기 -> 또 읽기" 패턴을 줄이기

이건 latency뿐 아니라 모델의 작업 추적 안정성에도 좋다.

## 6. 툴 출력은 길면 잘라라

공식 Prompting Guide 기준:

- 툴 응답은 대략 `10k tokens` 이내로 자르길 권장한다.
- 잘라야 하면 앞/뒤를 절반씩 남기고 가운데를 생략하라고 한다.

실무 권장:

- `cat` 전체 출력보다 필요한 줄만
- 대형 로그/JSON은 요약 + 핵심 부분만
- 너무 긴 출력은 앞부분/끝부분을 유지

긴 출력이 그대로 들어가면 진짜 중요한 instruction signal이 묻힌다.

## 7. `Compaction`을 써야 긴 세션에서 안 무너진다

공식 Prompting Guide 기준:

- `/responses/compact`는 긴 세션에서 유효 context를 압축해 유지하는 first-class 기능이다.
- 긴 작업, 많은 툴 호출, 장시간 디버깅에서 중요하다.

실무 권장:

- 장기 작업 세션이면 요약을 사람이 직접 넣는 대신 compaction 전략을 쓴다.
- "결정", "보류 중인 이슈", "다음 step", "파일 경로"가 압축 후에도 남도록 설계한다.

## 8. 모델 선택도 하네스 일부다

공식 모델 문서 기준:

- `GPT-5.3-Codex`는 현재 가장 강한 agentic coding 모델 축으로 안내된다.
- `GPT-5.3-Codex-Spark`는 더 빠르지만 덜 강한 near-instant iteration용이다.
- Speed 문서 기준 `Fast mode`는 GPT-5.4를 1.5x 빠르게 하지만 크레딧은 2x 쓴다.

실무 권장:

- 깊은 리팩터링/장거리 작업: 강한 Codex 계열
- 짧은 반복/프로토타입: Spark 계열 또는 fast mode
- "항상 최고 모델"보다, 작업 길이와 승인 병목에 맞춰 고르는 편이 낫다

## 9. 하네스용 MD에 실제로 넣어야 하는 내용

`codex_md` 아래 두는 문서는 설명서보다 **운영 규칙**이어야 한다.

좋은 MD 구성:

1. 프로젝트 컨텍스트
2. 우선 읽을 파일
3. 편집 금지 구역
4. 테스트/검증 루틴
5. 권장 툴
6. 금지 명령
7. 응답 형식
8. PR/commit 기준

나쁜 MD 특징:

- 배경 설명만 길다
- 실행 규칙이 없다
- 어느 파일부터 읽어야 하는지 없다
- 금지 사항이 없다
- 검증 기준이 없다

## 10. 이 repo에 바로 적용하면 좋은 것

- repo root에 짧은 `AGENTS.md` 추가
- `src/alpamayo1_5`, `tools`, `tests`에 폴더별 `AGENTS.md` 분리
- `rules`로 `rg`, `git status`, `git diff`, `pytest` 허용 범위 정리
- 히스토리/작업 로그는 `codex_history/`
- 하네스 문서는 `codex_md/`
- 장기 세션용 요약 템플릿 추가

## 추천 파일 세트

- `AGENTS.md`
- `codex_md/PROJECT_MAP.md`
- `codex_md/EDITING_RULES.md`
- `codex_md/TESTING_RULES.md`
- `codex_md/CODE_REVIEW_RULES.md`
- `codex_md/LONG_RUNNING_TASKS.md`

## 출처

- OpenAI Codex Prompting Guide  
  https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide
- OpenAI Codex Rules  
  https://developers.openai.com/codex/rules
- OpenAI Codex Speed  
  https://developers.openai.com/codex/speed
- OpenAI Codex model pages  
  https://developers.openai.com/api/docs/models/gpt-5.3-codex  
  https://developers.openai.com/api/docs/models/gpt-5.2-codex  
  https://developers.openai.com/api/docs/models/gpt-5.1-codex
