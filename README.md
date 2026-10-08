# UV 가상환경 설정 가이드

## RAG 검색 및 평가 실행

구현된 흐름: 구조 청킹 → Qdrant Dense / BM25 → RRF → LLM 재정렬 → 상위 조문·내부 참조 확장 → 근거 답변.
대상은 **제21311호, 2026-07-21 기준 스냅샷**입니다. 과거 시점의 법률 적용 여부 판정은 지원하지 않습니다.

```bash
uv sync
uv run uvicorn app.main:app --reload
```

`.env`의 `SERVICE_BASE_URL`, `OC_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`,
`EMBEDDING_MODEL`, `QDRANT_URL`을 사용합니다. API 키를 코드나 노트북에 넣지 마세요.

```bash
curl -X POST http://localhost:8000/index
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"고영향 인공지능이란 무엇인가요?"}'
```

`/index`는 명시적으로 호출할 때 해당 컬렉션을 현재 문서에 맞게 갱신합니다.
벡터 차원이 달라지면 기존 컬렉션을 삭제하지 않고 오류를 반환합니다.
모델 또는 endpoint가 바뀌면 재인덱싱해야 합니다. `/ask`는 답변, `is_answerable`,
실제 저장 원문으로 구성한 `sources`, 기준일 `as_of`를 반환합니다.
조문 번호가 명시된 질문은 가지번호까지 구별하며, 다른 법률 이름이 있는 질문을 이 법의 조문으로 강제 연결하지 않습니다.

### 평가 노트북

**`notebook/rag_evaluation.ipynb`**를 프로젝트 `.venv` 커널로 열고 Run All 하세요.
기본값은 저장 결과를 읽으며 외부 API를 다시 호출하지 않습니다.
결과 폴더는 `eval/results/full`(전체), `eval/results/lexical`(BM25)입니다.

재평가는 노트북의 `RUN_EVALUATION=True`, `MODE='full'`로 설정하거나 CLI를 사용합니다.

```bash
# API 없이 BM25: 최초 XML 다운로드에는 법령 API가 필요합니다.
uv run python -m eval.evaluate --lexical-only --output eval/results/lexical

# 45문항 A~F, 답변/LLM judge, 정답 조문 직접 제공 비교
# 설정된 외부 API로 법령·질문을 전송하며 사용 비용이 발생합니다.
uv run python -m eval.evaluate --answers --oracle --output eval/results/full

# 저장 결과만 사용하여 노트북 출력까지 생성
uv run python scripts/execute_eval_notebook.py --results eval/results/full

# 외부 API가 필요 없는 회귀 및 메모리 Qdrant 통합 테스트
uv run python -m unittest discover -s tests -v
```

| 실험 | 청킹 | 검색 |
|---|---|---|
| A | 조 전체 | Dense |
| B | 구조 단위 | Dense |
| C | 구조 단위 | BM25 |
| D | 구조 단위 | Dense + BM25 + RRF |
| E | 구조 단위 | D + LLM 재정렬 |
| F | 구조 단위 | E + 부모·내부 참조 확장 |

BM25는 단어+한국어 2-gram 토크나이저를 사용하는 기준선입니다. 형태소 분석기나 cross-encoder 모델은 설치하지 않습니다.
검색 실험은 법률 **본문**을 대상으로 하며 부칙/제개정이유는 저장하되 자동 검색 대상에서는 제외합니다.
법률 시점 질문과 부칙 전용 검색은 별도 라벨과 라우팅이 필요합니다.
Multi Query는 이번 기본 비교에 포함하지 않습니다.
기본 후보는 각 검색 20개→RRF 30개, 최종 8개 청크, 컨텍스트는 `cl100k_base` 기준 6000 토큰입니다.
상위 문장과 예외를 자르지 않고, 넘치는 청크는 건너뛰며 trace에 기록합니다.

### 평가 지표 해석

- 골든셋 원본 45문항은 수정하지 않았습니다. 정답은 **조문 단위**이고 36문항은 답변 가능, 9문항은 범위 밖입니다.
- Hit/Recall/MRR/nDCG의 K는 **중복 제거한 조문 순위**입니다. nDCG 등급은 정답 2 / 관련 1 / 기타 0입니다.
- 후보 Recall@20·30, 최종 순위 @1·3·5·10, 토큰 예산 내 컨텍스트 Recall·전체 확보율을 따로 표시합니다.
- 범위 밖 문항은 검색 정답 지표 평균에서 제외하고 답변 거절 precision/recall/혼동행렬로 평가합니다.
- `required_evidence`가 없는 세부 근거 지표는 미측정입니다. 조문을 찾았다는 이유만으로 항·호·목을 모두 찾았다고 판정하지 않습니다.
- 인용 원문 일치율은 결정적으로 계산하며, 정답성·관련성·요점 충족·충실성·주장 인용 지원·환각은 LLM judge로 평가합니다.
- 생성과 judge는 동일 설정 모델이므로 독립적 법률 심사가 아닙니다. 조건·예외·법적 강도 및 골든 정답의 완전성은 사람이 검수해야 합니다.
- 순수 거절에 법률 주장이 없으면 faithfulness/환각은 미측정입니다. 각 지표의 측정 문항 수도 저장합니다.
- Oracle은 정답 조문을 직접 제공하는 **생성 진단**이며 검색 성능으로 보고하지 않습니다.
- API 실패를 0점 또는 거절 정답으로 바꾸지 않습니다. 실패 건수와 평가 분모를 함께 확인하세요.
- `.cache/rag/`의 내용 기반 캐시로 재실행 비용을 줄입니다. 기록된 지연시간은 캐시가 섞인 관측값이며 신규 호출 성능 비교로 해석하면 안 됩니다.

산출물: `manifest.json`(설정·해시), `corpus_audit.json`, `retrieval_rows.json/csv`,
`retrieval_details.json`(원문 포함), `retrieval_summary.csv`, `answer_rows.json`,
`answer_summary.csv`, `figures/*.png`. 노트북 생성 소스는 `scripts/build_eval_notebook.py`입니다.

---

`notebook/`과 `app/`이 하나의 Python 가상환경을 공유하도록 구성한다.

## 1. 프로젝트 구조

```text
rag_minipjt/
│
├── .venv/
├── .env
├── .python-version
├── pyproject.toml
├── uv.lock
│
├── notebook/
│   └── test.ipynb
│
└── app/
    └── main.py
```

핵심은 다음과 같다.

- `.venv` : Python 가상환경
- `pyproject.toml` : 사용할 패키지 정의
- `uv.lock` : 실제 설치된 패키지 버전 고정
- `notebook/` : 실험 및 검증 코드
- `app/` : 실제 애플리케이션 코드

---

## 2. 프로젝트 초기화

프로젝트 루트에서 실행한다.

```bash
uv init --bare --python 3.12
```

---

## 3. 가상환경 생성

```bash
uv venv
```

가상환경은 프로젝트 루트의 `.venv/`에 생성된다.

필요한 경우 활성화한다.

```bash
source .venv/bin/activate
```

---

## 4. 필요한 패키지 설치

Qdrant, OpenAI Embedding, Notebook, FastAPI를 한 번에 설치한다.

```bash
uv add   openai   qdrant-client   python-dotenv   jupyter   ipykernel   fastapi   "uvicorn[standard]"
```

패키지를 설치하면 다음 파일이 자동으로 관리된다.

```text
pyproject.toml
uv.lock
```

---

## 5. 환경 복원

다른 PC 또는 수강생 환경에서는 다음 명령만 실행하면 동일한 환경을 만들 수 있다.

```bash
uv sync
```

`uv.lock`에 기록된 버전을 기준으로 패키지가 설치된다.

---

## 6. Notebook 실행

```bash
uv run jupyter lab
```

VS Code에서 `.ipynb`를 사용할 경우 `.venv`의 Python을 Kernel로 선택한다.

필요하면 Kernel을 등록한다.

```bash
uv run python -m ipykernel install --user --name rag-minipjt   --display-name "rag-minipjt"
```

---

## 7. FastAPI 실행

예를 들어 `app/main.py`를 다음과 같이 작성한다.

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "RAG API"
    }
```

실행:

```bash
uv run uvicorn app.main:app --reload
```

브라우저:

```text
http://localhost:8000
```

API 문서:

```text
http://localhost:8000/docs
```

---

## 8. 패키지 추가

새로운 패키지가 필요하면 다음처럼 추가한다.

```bash
uv add 패키지명
```

예:

```bash
uv add langchain langchain-openai langchain-qdrant
```

---

## 9. Git 관리

`.gitignore`에는 다음 항목을 넣는다.

```gitignore
.venv/
.env
__pycache__/
.ipynb_checkpoints/
```

다음 파일은 Git에 포함한다.

```text
pyproject.toml
uv.lock
.python-version
```

---

## 핵심 정리

```text
프로젝트 루트
     │
     ├── pyproject.toml
     ├── uv.lock
     └── .venv
           │
      ┌────┴────┐
      │         │
 notebook/    app/
```

`notebook/`과 `app/`에 각각 별도의 가상환경을 만들지 않고  
**프로젝트 전체에서 하나의 `.venv`를 공유하는 방식이 가장 단순하다.**

---

## 10. common/ 공유 모듈 설정

`notebook/`과 `app/`에서 공통으로 쓰는 설정, Qdrant 클라이언트, 임베딩 로직은 `common/` 패키지로 분리한다.

```text
rag_minipjt/
└── common/
    ├── __init__.py
    ├── config.py      # 환경변수 로딩
    ├── qdrant.py       # Qdrant client 생성
    └── ai_model.py    # LLM / Embedding 모델 생성
```

`notebook/`은 프로젝트 루트가 아닌 `notebook/` 디렉토리에서 커널이 실행되기 때문에, `common`을 그냥 `.venv`에 패키지로 설치해두지 않으면 노트북에서 다음과 같은 오류가 발생한다.

```text
ModuleNotFoundError: No module named 'common'
```

이를 해결하기 위해 프로젝트 자체를 editable 패키지로 설치한다. `pyproject.toml`에 build-system을 추가한다.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["common", "app"]
```

그리고 `common/__init__.py`, `app/__init__.py`를 빈 파일로 만들어 패키지로 인식시킨다.

```bash
uv sync
```

`uv sync`를 실행하면 `rag-minipjt` 프로젝트 자체가 `.venv`에 editable 모드로 설치되어, 노트북이 어느 위치에서 실행되든 다음처럼 공통 모듈을 바로 사용할 수 있다.

```python
import os
from dotenv import load_dotenv
from common.ai_model import get_llm_model, get_embedding_model
from common.qdrant import get_qdrant_client
```

> `common/`의 코드를 수정한 뒤에는 커널만 재시작하면 되고, `uv sync`를 다시 실행할 필요는 없다 (editable 설치이므로 소스 변경이 즉시 반영된다).
