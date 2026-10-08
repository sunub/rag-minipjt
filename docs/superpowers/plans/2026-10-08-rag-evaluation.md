# RAG 검색 및 평가 구현 계획

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 합의된 Hybrid RAG와 golden_set_v2.jsonl 기반 시각적 평가 노트북 구현 및 실행.

**Architecture:** 기존 구조 청킹을 유지하고 동일 문서에 Dense/BM25/RRF/LLM reranking을 적용한다. 검색 trace와 답변을 평가 모듈에 전달하고 JSON/CSV 및 노트북으로 보고한다.

**Tech Stack:** Python, FastAPI, Qdrant, LangChain, pandas, matplotlib, unittest.

**Spec:** 대화에서 승인한 조문 기반 검색·평가 표. 45개 원본 질문과 라벨은 수정하지 않는다.

## Global Constraints
- 골든셋은 조문 수준이다. 세부 근거 라벨 미제공 지표는 미측정으로 표시한다.
- 범위 밖 질문은 검색 Recall 평균에서 제외하고 거절 지표로 별도 평가한다.
- 동일 조문 중복을 제거한 순위, 관련도 2(정답)/1(관련)/0(그 외)를 사용한다.
- 외부 호출 실패를 점수 0 또는 가짜 성공으로 대체하지 않는다.
- 실험용 Qdrant는 로컬 별도 저장소를 사용한다. API 키는 산출물에 기록하지 않는다.

## Review Focus
- 가지번호와 항 없는 호, 긴 텍스트 조각의 근거 과대계산.
- 빈 검색 결과, 빈 정답, 범위 밖 질문의 잘못된 perfect score.
- 문서/모델/질문 변경 시 캐시 재사용, 불완전한 재정렬 응답.
- 외부 법령 인용을 현 법령 조문으로 잘못 확장.
- 인용 ID 위조 및 동시 요청에서 공유 상태 오염.

## Tasks
- [x] 1. unittest로 조문 중복 제거/nDCG/범위 밖 처리, BM25/RRF, 인용 검증 및 토큰 예산 실패 테스트 작성·실행.
- [x] 2. loader와 캐시, Qdrant 저장/검색, BM25/RRF, LLM reranker, 컨텍스트 확장 구현. 원문 보존과 버전·메타데이터 검증.
- [x] 3. grounded pipeline과 POST /ask, 오류 처리·입력 제한 구현.
- [x] 4. eval 모듈에 조문 지표, 선택적 세부 근거 지표, LLM judge, 거절 지표, 원문/인용 무결성 및 실행 manifest 구현.
- [x] 5. notebook/rag_evaluation.ipynb에 데이터 점검·실험·비교 차트·유형별 heatmap·실패 탐색·답변 평가·oracle 비교·내보내기 구성.
- [x] 6. 실제 45문항 실행, 전체 unittest 및 노트북 실행 검증, 결과와 제한 문서화.

## Execution notes
- 현재 디렉터리는 Git 저장소가 아니므로 worktree/commit을 생성하지 않고 현 작업공간에서 구현한다.
- 사용자는 앞선 설계에 동의하고 구현 및 실행을 요청했으므로 추가 승인 없이 진행한다.

- 검증: 독립 검토에서 발견된 임베딩 identity, 평가 실패 분모, 인용 일치, 외부 법률 라우팅 및 노트북 실패 처리 보완.
- 사용자 승인: monogpt.kr로 45개 공개 질문/법령 컨텍스트를 전송하는 전체 API 평가 승인받음.

- 완료 검증: 26개 테스트 통과. 270건 검색 및 81건 답변/judge 평가 오류 0. 노트북은 정상/전체 실패/부분 실패 결과로 실행 검증.
